from typing import Callable, Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
tqdm.__call__ = lambda *a, **k: a[0]

from trainers.supervised_trainer import SupervisedCoxTrainer
from losses.cox import cox_ph_loss
from losses.graph import graph_consistency_loss
from graph.knn_graph import KNNGraph


class GraphCoxTrainer(SupervisedCoxTrainer):
    """
    Graph-SSL trainer for Cox PH models.

    Uses:
      L = L_sup(labeled) + lambda_graph * L_graph(consistency)

    L_graph uses a teacher cache of risks for ALL train nodes (labeled + unlabeled),
    computed once per epoch with no grad (stopgrad).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        graph: KNNGraph,
        lambda_graph: float = 1.0,
        max_neighbors: int = 10,
        device: str = "cuda",
        loss_fn: Callable = cox_ph_loss,
        metrics_horizon: float = 30.0,
        event_of_interest: int = 1,
        metrics_horizons: list[float] | None = None,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            metrics_horizon=metrics_horizon,
            event_of_interest=event_of_interest,
            metrics_horizons=metrics_horizons,
        )
        self.graph = graph.to(device)  # neighbors to device
        self.lambda_graph = float(lambda_graph)
        self.max_neighbors = int(max_neighbors)

        self._teacher_risk_all: Optional[torch.Tensor] = None

    def _build_teacher_cache(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        forward_fn: Callable[[torch.nn.Module, Dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute teacher risks for ALL nodes once per epoch (stopgrad cache).
        Requires idx in batches and assumes idx ranges [0..N-1] for train split.
        """
        self.model.eval()
        N = int(self.graph.num_nodes)
        risk_all = torch.empty((N,), device=self.device, dtype=torch.float32)

        def fill_from(loader: DataLoader):
            for batch in loader:
                batch = self._move_to_device(batch)
                idx = batch["idx"].view(-1).long()
                with torch.no_grad():
                    r = self._flatten_scores(forward_fn(self.model, batch)).float()
                risk_all[idx] = r

        fill_from(labeled_loader)
        fill_from(unlabeled_loader)

        return risk_all

    def train_epoch(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        forward_fn: Callable[[torch.nn.Module, Dict[str, Any]], torch.Tensor],
    ) -> float:
        # ---- teacher cache (stopgrad) ----
        teacher_risk_all = self._build_teacher_cache(labeled_loader, unlabeled_loader, forward_fn)
        self._teacher_risk_all = teacher_risk_all  # stored for debugging if needed

        # ---- supervised training on labeled batches, with graph regularization ----
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        if hasattr(self, "_printed_losses"):
            delattr(self, "_printed_losses")

        print(f"[GraphCox] teacher_cache | nodes={self.graph.num_nodes} | lambda={self.lambda_graph}")

        for batch in tqdm(labeled_loader, desc="Train(GraphCox)", leave=False):
                    
            batch = self._move_to_device(batch)
            idx = batch["idx"].view(-1).long()
            log_risk = self._flatten_scores(forward_fn(self.model, batch))

            sup_loss = self.loss_fn(log_risk, batch["time"], batch["event"])
            g_loss = graph_consistency_loss(
                student_risk=log_risk,
                batch_idx=idx,
                neighbors=self.graph.neighbors,
                teacher_risk_all=teacher_risk_all,
                max_neighbors=self.max_neighbors,
            )
            if not hasattr(self, "_printed_losses"):
                print(
                    f"[GraphCox] sup_loss={float(sup_loss.item()):.4f} "
                    f"graph_loss={float(g_loss.item()):.4f} "
                    f"lambda={self.lambda_graph}"
                )
                self._printed_losses = True

            loss = sup_loss + (self.lambda_graph * g_loss)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            bs = int(batch["time"].shape[0])
            total_loss += float(loss.item()) * bs
            n_samples += bs

        return total_loss / max(n_samples, 1)
