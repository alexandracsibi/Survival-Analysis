import torch
import torch.nn as nn

class MLPBackbone(nn.Module):
    """
    Generic MLP for tabular features.

    - in_dim: number of input features
    - hidden_dims: list like [128, 64]
    - dropout: dropout probability after each hidden layer
    """

    def __init__(self, in_dim, hidden_dims=None, dropout: float = 0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = h

        self.net = nn.Sequential(*layers)
        self.out_dim = last_dim  # needed by heads (DeepCox/DeepHit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim]
        returns: [B, out_dim]
        """
        return self.net(x)

# MLPBackbone is a pure feature extractor.
# self.out_dim tells any head (Cox / DeepHit / multimodal) what its input size is.
# hidden_dims from config (mlp_hidden_dims) plugs in directly.