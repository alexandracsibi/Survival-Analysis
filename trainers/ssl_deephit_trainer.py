import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, Any, Optional, List, Callable
from tqdm import tqdm
import numpy as np
import copy

from .deephit_trainer import DeepHitTrainer
from losses.deephit import deephit_loss

class SSLDeepHitTrainer(DeepHitTrainer):
    """
    Semi-supervised extension of DeepHitTrainer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        discretizer,
        device: str = "cuda",
        alpha: float = 1.0,
        beta: float = 1.0,
        max_pairs: int = 8192,
        metrics_horizon: float = 30.0,
        event_of_interest: int = 1,
        metrics_horizons: Optional[List[float]] = None,
        ssl_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            discretizer=discretizer,
            device=device,
            alpha=alpha,
            beta=beta,
            max_pairs=max_pairs,
            metrics_horizon=metrics_horizon,
            event_of_interest=event_of_interest,
            metrics_horizons=metrics_horizons,
        )

        # SSL hyperparameters
        self.ssl_cfg: Dict[str, Any] = ssl_cfg or {}
        self.warmup_epochs: int = int(self.ssl_cfg.get("warmup_epochs", 30))
        self.ssl_epochs: int = int(self.ssl_cfg.get("ssl_epochs", 50))
        self.lambda_pseudo: float = float(self.ssl_cfg.get("lambda_pseudo", 0.5))
        self.ssl_enabled: bool = bool(self.ssl_cfg.get("enabled", False))
        self.ssl_method: str = str(self.ssl_cfg.get("method", "pseudo_label"))
        self.conf_thresh_start: float = float(
            self.ssl_cfg.get("confidence_threshold_start", self.ssl_cfg.get("confidence_threshold", 0.15))
        )
        self.conf_thresh_end: float = float(
            self.ssl_cfg.get("confidence_threshold_end", self.conf_thresh_start)
        )
        self.conf_schedule: str = str(self.ssl_cfg.get("confidence_schedule", "linear")).lower()
        self.conf_schedule_steps: int = int(self.ssl_cfg.get("conf_schedule_steps", self.ssl_epochs))
        self.temperature: float = float(self.ssl_cfg.get("temperature", 1.0))
        self._ssl_step: int = 0

        self.use_ema: bool = bool(self.ssl_cfg.get("ema_enabled", False))
        self.ema_decay: float = float(self.ssl_cfg.get("ema_decay", 0.999))
        self.ema_model = None
        if self.use_ema:
            self.ema_model = copy.deepcopy(self.model).to(self.device)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    def current_conf_threshold(self) -> float:
        if self.conf_thresh_start == self.conf_thresh_end:
            return float(self.conf_thresh_start)

        steps = int(self.conf_schedule_steps)
        denom = max(1, steps - 1)

        step = min(max(self._ssl_step, 0), denom)
        p = float(step) / float(denom)

        if self.conf_schedule == "linear":
            thr = self.conf_thresh_start + p * (self.conf_thresh_end - self.conf_thresh_start)
        else:
            raise ValueError(f"Unknown confidence_schedule: {self.conf_schedule}")

        return float(thr)

    @torch.no_grad()
    def update_ema(self):
        if not self.use_ema or self.ema_model is None:
            return
        decay = self.ema_decay
        msd = self.model.state_dict()
        esd = self.ema_model.state_dict()
        for k in esd.keys():
            esd[k].mul_(decay).add_(msd[k], alpha=(1.0 - decay))


    def generate_pseudo_dataset(self, unl_loader, forward_fn):
        """
        Produce (x, pseudo_time_idx, pseudo_event) for confident samples.
        Confidence definition:
        conf = p(event=k*) * max_t p(t | event=k*)
        Pseudo-time is converted from bin index -> continuous time using bin centers,
        so it is compatible with deephit_loss (which expects continuous time and encodes it)
        """
        self.model.eval()
        xs, times, events, confidences = [], [], [], []
        conf_thresh = float(self.current_conf_threshold())
        max_size = int(self.ssl_cfg.get("pseudo_refresh_max_size", 0))
        D = None # feature dim for empty dataset fallback

        with torch.no_grad():
            for batch in tqdm(unl_loader, desc="Pseudo-labeling", leave=False):
                x = batch["x"].to(self.device)
                if D is None:
                    D = int(x.shape[1])

                # forward returns (probs, logits), but we recompute probs from logits with temperature
                teacher = self.ema_model if (self.use_ema and self.ema_model is not None) else self.model
                _, logits = forward_fn(teacher, {"x": x})

                B, K, T = logits.shape
                flat = logits.reshape(B, K * T)
                p_flat = torch.softmax(flat / self.temperature, dim=-1)
                probs = p_flat.reshape(B, K, T)  # [B,K,T]

                # 1) event marginal p(k) = sum_t p(k,t)
                p_k = probs.sum(dim=-1)  # [B,K]
                p_event, k_star = torch.max(p_k, dim=-1)  # [B], [B] 0..K-1

                # 2) p(t | k*) = p(k*,t) / p(event=k*)
                p_kstar_t = probs[torch.arange(B, device=self.device), k_star, :]  # [B,T]
                denom = p_event.clamp_min(1e-12).unsqueeze(-1)                     # [B,1]
                p_t_given_k = p_kstar_t / denom                                    # [B,T]

                p_time, t_star = torch.max(p_t_given_k, dim=-1)  # [B], [B] 0..T-1
                # combined confidence
                conf = p_event * p_time  # [B]

                mask = conf >= conf_thresh
                if mask.sum().item() == 0:
                    continue

                # time bin index -> continuous time
                centers = self.discretizer.bin_centers.to(self.device)  # [T]
                t_star_real = centers[t_star]                           # [B]

                xs.append(x[mask].detach().cpu())
                times.append(t_star_real[mask].detach().cpu())
                events.append((k_star[mask].detach().cpu() + 1).long())
                confidences.append(conf[mask].detach().cpu())

        if len(xs) == 0:
            print(f"[SSL] Accepted 0 pseudo-labels (threshold={conf_thresh:.4f}, T={self.temperature})")
            # Return an empty TensorDataset for consistent downstream len() behavior
            if D is None:
                D = 0
            empty_x = torch.empty((0, D), dtype=torch.float32)
            empty_t = torch.empty((0,), dtype=torch.float32)
            empty_e = torch.empty((0,), dtype=torch.long)
            empty_c = torch.empty((0,), dtype=torch.float32)
            self._ssl_step += 1
            return TensorDataset(empty_x, empty_t, empty_e, empty_c)
            
        xs = torch.cat(xs, dim=0)
        times = torch.cat(times, dim=0)         # continuous time
        events = torch.cat(events, dim=0)       # 1..K
        confs = torch.cat(confidences, dim=0)

        # ---- CAP: keep top-confidence pseudo-labels ----
        if max_size > 0 and len(confs) > max_size:
            topk = torch.topk(confs, k=max_size, largest=True).indices
            xs = xs[topk]
            times = times[topk]
            events = events[topk]
            confs = confs[topk]
            print(f"[SSL] Capped pseudo set to top-{max_size} by confidence")

        print(
            f"[SSL] conf stats: "
            f"min={confs.min().item():.4f} "
            f"mean={confs.mean().item():.4f} "
            f"max={confs.max().item():.4f}"
        )
        print(
            f"[SSL] Accepted {len(xs)} pseudo-labels "
            f"(threshold={conf_thresh:.4f}, T={self.temperature})"
        )

        self._ssl_step += 1
        return TensorDataset(xs, times, events, confs)


    def train_ssl_epoch(
        self,
        labeled_loader,
        pseudo_loader,
        forward_fn,
        lambda_pseudo=0.5,
    ):
        """
        One SSL epoch: iterate labeled batches + pseudo-labeled batches.
        Pseudo-labeled times are continuous (bin centers), compatible with deephit_loss.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pseudo_iter = iter(pseudo_loader)

        for batch_lab in tqdm(labeled_loader, desc="SSL-Train", leave=False):
            # 1) Labeled supervised loss
            batch_lab = self._move_to_device(batch_lab)
            probs, logits = forward_fn(self.model, batch_lab)

            loss_labeled = deephit_loss(
                logits=logits,
                time=batch_lab["time"],
                event=batch_lab["event"],
                discretizer=self.discretizer,
                alpha=self.alpha,
                beta=self.beta,
                max_pairs=self.max_pairs,
            )

            # 2) Pseudo-labeled loss
            try:
                x_pseudo, t_pseudo, e_pseudo, _ = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = iter(pseudo_loader)
                x_pseudo, t_pseudo, e_pseudo, _ = next(pseudo_iter)

            x_pseudo = x_pseudo.to(self.device, non_blocking=True)
            t_pseudo = t_pseudo.to(self.device, non_blocking=True)  # continuous time
            e_pseudo = e_pseudo.to(self.device, non_blocking=True)

            probs_p, logits_p = forward_fn(self.model, {"x": x_pseudo})

            loss_pseudo = deephit_loss(
                logits=logits_p,
                time=t_pseudo,              # continuous time (bin center)
                event=e_pseudo.long(),
                discretizer=self.discretizer,
                alpha=self.alpha,
                beta=self.beta,
                max_pairs=self.max_pairs,
            )

            loss = loss_labeled + float(lambda_pseudo) * loss_pseudo

            # Optimize
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.update_ema()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train_epoch(
        self,
        loader: DataLoader,
        forward_fn: Callable[[torch.nn.Module, Dict[str, Any]], torch.Tensor],
    ) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in tqdm(loader, desc="Train", leave=False):
            batch_dev = self._move_to_device(batch)

            out = forward_fn(self.model, batch_dev)
            if isinstance(out, tuple):
                _, logits = out
            else:
                logits = out

            loss = deephit_loss(
                logits=logits,
                time=batch_dev["time"],
                event=batch_dev["event"],
                discretizer=self.discretizer,
                alpha=self.alpha,
                beta=self.beta,
                max_pairs=self.max_pairs,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # EMA teacher update during warmup/supervised training
            self.update_ema()

            bs = int(batch_dev["time"].shape[0])
            total_loss += float(loss.item()) * bs
            n_samples += bs

        return total_loss / max(n_samples, 1)
