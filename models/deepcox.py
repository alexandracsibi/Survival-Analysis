import torch
import torch.nn as nn
from .mlp import MLPBackbone

# DeepSurv = MLP -> scalar log-risk.
# Used for SUPPORT2, MNB, MIMIC-Eye tabular, MIMIC-Eye multimodal (as final head).

class DeepCoxPH(nn.Module):
    """
    DeepSurv-style Cox proportional hazards model.

    backbone: an nn.Module with:
        - forward(x) -> [B, H]
        - .out_dim attribute = H
    """

    def __init__(self, backbone: MLPBackbone):
        super().__init__()
        self.backbone = backbone
        self.risk_head = nn.Linear(backbone.out_dim, 1)

    def forward(self, x):
        """
        x: [B, d] features
        returns: [B] log-risk scores
        """

        h = self.backbone(x)         # [B, H]
        log_risk = self.risk_head(h) # [B, 1]
        return log_risk.squeeze(-1)  # [B]

# log_risk is what the Cox loss uses (partial log-likelihood).
# Higher log_risk → higher instantaneous hazard → worse prognosis.
# C-index will usually be computed from -log_risk or log_risk depending on convention (we can fix later in metrics).