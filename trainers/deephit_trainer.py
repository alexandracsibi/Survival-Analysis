from typing import Callable, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses.deephit import deephit_loss, TimeDiscretizer

from metrics.c_index import concordance_index
from metrics.time_dependent_auc import td_auc_uno_ipcw
from metrics.integrated_auc import integrated_auc
from metrics.brier import integrated_brier_score
from metrics.deephit_survival import deephit_survival_at_horizons


def _deephit_cif_scores_at_horizon(
    probs_np: np.ndarray,
    t_index: int,
    event_of_interest: int,
) -> np.ndarray:
    """
    probs_np: [N, K, T] joint pmf over (k,t), with NO censoring channel.
    returns CIF_k(t_index) = sum_{u<=t_index} p(k,u)  for k = event_of_interest
    """
    p = np.asarray(probs_np, dtype=float)
    if p.ndim != 3:
        raise ValueError(f"Expected probs [N,K,T], got {p.shape}")

    N, K, T = p.shape
    if not (0 <= t_index < T):
        raise ValueError(f"t_index out of range: {t_index} for T={T}")

    k_idx = event_of_interest - 1
    if not (0 <= k_idx < K):
        raise ValueError(f"event_of_interest={event_of_interest} incompatible with K={K}")

    pmf_k = p[:, k_idx, :]                 # [N, T]
    cif_k = np.cumsum(pmf_k, axis=1)       # [N, T]
    return cif_k[:, t_index]               # [N]


class DeepHitTrainer:
    """
    Trainer for DeepHit models (single-event or competing risks).
    forward_fn(model, batch_on_device) should return either:
      - logits [B,K,T]
      - or (probs, logits) where probs/logits are [B,K,T]
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        discretizer: TimeDiscretizer,
        device: str = "cuda",
        alpha: float = 1.0,
        beta: float = 1.0,
        max_pairs: int = 8192,
        metrics_horizon: float = 30.0,
        event_of_interest: int = 1,
        metrics_horizons: list[float] | None = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.discretizer = discretizer
        self.device = device

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_pairs = int(max_pairs)

        self.metrics_horizon = float(metrics_horizon)
        self.event_of_interest = int(event_of_interest)

        if metrics_horizons is None:
            t0 = self.metrics_horizon
            self.metrics_horizons = [0.5 * t0, 1.0 * t0, 2.0 * t0]
        else:
            self.metrics_horizons = [float(x) for x in metrics_horizons]

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            out[k] = v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        return out

    # -------------------------
    # training
    # -------------------------
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
                probs, logits = out
            else:
                logits = out
                probs = None  # not needed for training

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

            bs = int(batch_dev["time"].shape[0])
            total_loss += float(loss.item()) * bs
            n_samples += bs

        return total_loss / max(n_samples, 1)

    # -------------------------
    # evaluation
    # -------------------------
    def eval_epoch(
        self,
        loader: DataLoader,
        forward_fn: Callable[[torch.nn.Module, Dict[str, Any]], torch.Tensor],
    ) -> Dict[str, float]:
        """
        Metrics:
          - loss: NLL only (beta=0)
          - c_index: risk=-E[T] (ranking, any-event)
          - td_auc: Uno/IPCW AUC(t0) using CIF_k(t0) as score for event_of_interest
          - iauc: mean AUC over a small horizon grid
          - ibs: integrated Brier score for event-free survival S(t)=P(T>t)
        """
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        all_time: list[np.ndarray] = []
        all_event: list[np.ndarray] = []
        all_risk: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval", leave=False):
                batch_dev = self._move_to_device(batch)

                out = forward_fn(self.model, batch_dev)
                if isinstance(out, tuple):
                    probs, logits = out
                else:
                    logits = out
                    # if forward_fn only returns logits, recompute probs from logits here if you want IBS/AUC
                    # but in your code path, you already return (probs, logits), so we require probs.
                    raise ValueError("DeepHit eval requires forward_fn to return (probs, logits) for AUC/IBS.")

                loss = deephit_loss(
                    logits=logits,
                    time=batch_dev["time"],
                    event=batch_dev["event"],
                    discretizer=self.discretizer,
                    alpha=self.alpha,
                    beta=0.0,
                    max_pairs=0,
                )

                bs = int(batch_dev["time"].shape[0])
                total_loss += float(loss.item()) * bs
                n_samples += bs

                expected_time = self.discretizer.expected_time_from_logits(logits)  # [B]
                risk_score = -expected_time                                        # [B]

                all_time.append(batch_dev["time"].detach().cpu().numpy())
                all_event.append(batch_dev["event"].detach().cpu().numpy())
                all_risk.append(risk_score.detach().cpu().numpy())
                all_probs.append(probs.detach().cpu().numpy())

        metrics: Dict[str, float] = {"loss": total_loss / max(n_samples, 1)}

        if not all_time:
            metrics.update(
                {
                    "c_index": float("nan"),
                    "td_auc": float("nan"),
                    "td_auc_cases": float("nan"),
                    "td_auc_ctrls": float("nan"),
                    "iauc": float("nan"),
                    "ibs": float("nan"),
                }
            )
            return metrics

        time_np = np.concatenate(all_time).astype(float)
        event_np = np.concatenate(all_event).astype(int)
        risk_np = np.concatenate(all_risk).astype(float)
        probs_np = np.concatenate(all_probs, axis=0).astype(float)

        # ---- C-index (any-event indicator) ----
        metrics["c_index"] = float(concordance_index(time_np, risk_np, (event_np > 0).astype(int)))

        # ---- AUC(t0) with CIF scores ----
        t0 = self.metrics_horizon
        eoi = self.event_of_interest
        t_index = int(self.discretizer.transform_time(np.asarray([t0], dtype=float))[0])

        cif_scores_t0 = _deephit_cif_scores_at_horizon(probs_np, t_index=t_index, event_of_interest=eoi)

        td_auc, n_cases, n_ctrl = td_auc_uno_ipcw(
            time=time_np,
            event=event_np,
            scores=cif_scores_t0,     # IMPORTANT: classification score at t0
            t0=t0,
            event_of_interest=eoi,
        )
        metrics["td_auc"] = float(td_auc)
        metrics["td_auc_cases"] = float(n_cases)
        metrics["td_auc_ctrls"] = float(n_ctrl)

        # ---- iAUC over grid (mean of defined AUCs) ----
        # compute CIF scores for each horizon and average AUC(t)
        aucs = []
        for ht in self.metrics_horizons:
            ht = float(ht)
            ht_idx = int(self.discretizer.transform_time(np.asarray([ht], dtype=float))[0])
            scores_ht = _deephit_cif_scores_at_horizon(probs_np, t_index=ht_idx, event_of_interest=eoi)
            auc_ht, _, _ = td_auc_uno_ipcw(time_np, event_np, scores_ht, ht, event_of_interest=eoi)
            if not np.isnan(auc_ht):
                aucs.append(auc_ht)
        metrics["iauc"] = float(np.mean(aucs)) if len(aucs) > 0 else float("nan")

        # ---- IBS (event-free survival) ----
        # For IBS we evaluate survival for "any event" (event>0)
        # DeepHit provides joint pmf over causes, so survival is well-defined:
        #   S(t) = P(T > t) = sum_{u>t} sum_k p(k,u)
        event_any = (event_np > 0).astype(int)
        S = deephit_survival_at_horizons(probs_np, self.discretizer, horizons=self.metrics_horizons)
        ibs, _ = integrated_brier_score(time_np, event_any, S, horizons=self.metrics_horizons)
        metrics["ibs"] = float(ibs)

        # Metadata
        metrics["horizon"] = float(t0)
        metrics["event_of_interest"] = float(eoi)
        metrics["iauc_grid_n"] = float(len(self.metrics_horizons))

        return metrics
