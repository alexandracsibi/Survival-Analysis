from typing import Callable, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses.cox import cox_ph_loss
from metrics.c_index import concordance_index
from metrics.time_dependent_auc import td_auc_uno_ipcw
from metrics.integrated_auc import integrated_auc
from metrics.brier import integrated_brier_score
from metrics.cox_survival import cox_survival_at_horizons


class SupervisedCoxTrainer:
    """
    Trainer for Cox PH models (DeepSurv + multimodal).

    forward_fn(model, batch_on_device) must return log-risk scores of shape [B] (or [B,1]).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        loss_fn: Callable = cox_ph_loss,
        metrics_horizon: float = 30.0, 
        event_of_interest: int = 1,
        metrics_horizons: list[float] | None = None,        # For iAUC/IBS: horizons grid
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.metrics_horizon = float(metrics_horizon)
        self.event_of_interest = int(event_of_interest)

        if metrics_horizons is None:
            # Default grid, can be overrided from config later.
            t0 = self.metrics_horizon
            self.metrics_horizons = [0.5 * t0, 1.0 * t0, 2.0 * t0]
        else:
            self.metrics_horizons = [float(x) for x in metrics_horizons]

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        return out

    @staticmethod
    def _flatten_scores(x: torch.Tensor) -> torch.Tensor:
        return x.view(-1) if x.ndim > 1 else x

    def train_epoch(
        self,
        loader: DataLoader,
        forward_fn: Callable[[torch.nn.Module, Dict[str, Any]], torch.Tensor],
    ) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in tqdm(loader, desc="Train", leave=False):
            batch = self._move_to_device(batch)

            log_risk = self._flatten_scores(forward_fn(self.model, batch))
            loss = self.loss_fn(log_risk, batch["time"], batch["event"])

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            bs = int(batch["time"].shape[0])
            total_loss += float(loss.item()) * bs
            n_samples += bs

        return total_loss / max(n_samples, 1)

    def eval_epoch(
        self,
        loader: DataLoader,
        forward_fn: Callable[[torch.nn.Module, Dict[str, Any]], torch.Tensor],
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0

        all_time: list[np.ndarray] = []
        all_event: list[np.ndarray] = []
        all_risk: list[np.ndarray] = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval", leave=False):
                batch = self._move_to_device(batch)

                log_risk = self._flatten_scores(forward_fn(self.model, batch))
                loss = self.loss_fn(log_risk, batch["time"], batch["event"])

                bs = int(batch["time"].shape[0])
                total_loss += float(loss.item()) * bs
                n_samples += bs

                all_time.append(batch["time"].detach().cpu().numpy())
                all_event.append(batch["event"].detach().cpu().numpy())
                all_risk.append(log_risk.detach().cpu().numpy())

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

        # ---- C-index (ranking) ----
        metrics["c_index"] = float(concordance_index(time_np, risk_np, (event_np > 0).astype(int)))

        # ---- Time-dependent AUC(t0) (Uno/IPCW) ----
        t0 = self.metrics_horizon
        eoi = self.event_of_interest
        td_auc, n_cases, n_ctrl = td_auc_uno_ipcw(
            time=time_np, event=event_np, scores=risk_np, t0=t0, event_of_interest=eoi
        )
        metrics["td_auc"] = float(td_auc)
        metrics["td_auc_cases"] = float(n_cases)
        metrics["td_auc_ctrls"] = float(n_ctrl)

        # ---- Integrated AUC over a small grid ----
        iauc, _ = integrated_auc(
            time=time_np, event=event_np, scores=risk_np, horizons=self.metrics_horizons, event_of_interest=eoi
        )
        metrics["iauc"] = float(iauc)

        # ---- Integrated Brier Score (IBS) ----
        # IBS needs survival probabilities S(t|x) at the same horizons grid.
        event_any = (event_np > 0).astype(int)
        S = cox_survival_at_horizons(time_np, event_any, risk_np, horizons=self.metrics_horizons)
        ibs, _ = integrated_brier_score(time_np, event_any, S, horizons=self.metrics_horizons)
        metrics["ibs"] = float(ibs)

        # Helpful metadata for debugging / thesis tables
        metrics["horizon"] = float(t0)
        metrics["event_of_interest"] = float(eoi)
        metrics["iauc_grid_n"] = float(len(self.metrics_horizons))

        return metrics