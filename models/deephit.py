import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import MLPBackbone


class DeepHit(nn.Module):
    """
    DeepHit model for (possibly) competing risks.

    IMPORTANT:
      - logits are [B, K, T]
      - probs are a JOINT distribution p(k,t|x) with softmax over K*T
        (this matches the standard DeepHit formulation used in the improved loss)

    Args:
        backbone: MLPBackbone or similar
        n_times: number of discrete time bins (T)
        n_events: number of event types (K). For binary: K=1; for competing risks: K>1.
    """

    def __init__(self, backbone: MLPBackbone, n_times: int, n_events: int = 1):
        super().__init__()
        if n_times <= 1:
            raise ValueError(f"n_times must be > 1, got {n_times}")
        if n_events <= 0:
            raise ValueError(f"n_events must be >= 1, got {n_events}")

        self.backbone = backbone
        self.n_times = int(n_times)
        self.n_events = int(n_events)
        self.out = nn.Linear(backbone.out_dim, self.n_times * self.n_events)

    def forward(self, x):
        """
        x: [B, d]

        returns:
            probs:  [B, K, T] joint PMF p(k,t|x), softmax over K*T
            logits: [B, K, T] unnormalized scores
        """
        h = self.backbone(x)  # [B, H]
        flat_logits = self.out(h)  # [B, K*T]
        logits = flat_logits.view(-1, self.n_events, self.n_times)  # [B, K, T]

        # joint softmax over (K*T)
        B = logits.shape[0]
        probs = F.softmax(flat_logits, dim=-1).view(B, self.n_events, self.n_times)

        return probs, logits
