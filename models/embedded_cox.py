import torch
import torch.nn as nn

from .mlp import MLPBackbone


class EmbeddedCoxPH(nn.Module):
    """
    CoxPH model with learned embeddings for selected categorical features.

    Input x: [B, D], where some columns are integer IDs (categorical),
    others are continuous/binary.
    """

    def __init__(
        self,
        in_dim: int,
        cat_feature_indices,
        cat_cardinalities,
        hidden_dims=(128, 64),
        dropout=0.1,
        embed_dim_rule=None,
    ):
        super().__init__()

        self.cat_feature_indices = list(cat_feature_indices)

        # indices for continuous part (everything not in cat_feature_indices)
        all_idx = set(range(in_dim))
        cont_indices = sorted(all_idx - set(self.cat_feature_indices))
        self.cont_feature_indices = cont_indices

        # default embedding dimension rule if none given
        if embed_dim_rule is None:
            def embed_dim_rule(card):
                # simple heuristic, adjust if you want
                return min(50, max(4, card // 2))

        # build embeddings
        self.embeddings = nn.ModuleList()
        self.embed_dims = []
        for card in cat_cardinalities:
            emb_dim = embed_dim_rule(card)
            self.embed_dims.append(emb_dim)
            self.embeddings.append(nn.Embedding(card, emb_dim))

        # input dim to MLP = cont_dim + sum(embed_dims)
        cont_dim = len(self.cont_feature_indices)
        mlp_in_dim = cont_dim + sum(self.embed_dims)

        self.backbone = MLPBackbone(
            in_dim=mlp_in_dim,
            hidden_dims=list(hidden_dims),
            dropout=dropout,
        )

        # Cox head
        self.risk_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cont = x[:, self.cont_feature_indices].float()

        cat_tensors = []
        for emb, idx in zip(self.embeddings, self.cat_feature_indices):
            cat_idx = x[:, idx].long()
            cat_idx = cat_idx.clamp(min=0, max=emb.num_embeddings - 1)
            cat_tensors.append(emb(cat_idx))

        h = torch.cat([x_cont] + cat_tensors, dim=1) if cat_tensors else x_cont
        z = self.backbone(h)
        log_risk = self.risk_head(z).squeeze(-1)
        return log_risk
