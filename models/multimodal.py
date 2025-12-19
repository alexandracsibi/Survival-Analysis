import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from .mlp import MLPBackbone
from typing import Sequence, Optional

class MultiModalCox(nn.Module):
    """
    Multimodal Cox model:
      - image branch: ResNet18 backbone
      - tabular branch: MLPBackbone
      - fusion: concat(image_feat, tab_feat) -> linear Cox head

    Used for MIMIC-Eye multimodal dataset.
    """

    def __init__(self,
                 num_tab_features: int,
                 img_out_dim: int = 128,
                 tab_hidden_dims: Sequence[int] = (64, 32),
                 tab_dropout: float = 0.1,
                 pretrained: bool = True,
                 weights_path: str | None = None,
                 freeze_cnn: bool = False,
                 cat_feature_indices: Optional[Sequence[int]] = None,
                 cat_cardinalities: Optional[Sequence[int]] = None,
                 ):
        super().__init__()

        # ----- Image branch -----
        if weights_path is not None:
            resnet = resnet18(weights=None)
            state = torch.load(weights_path, map_location="cpu")
            resnet.load_state_dict(state)
        else:
            if pretrained:
                weights = ResNet18_Weights.DEFAULT
            else:
                weights = None
            resnet = resnet18(weights=weights)

        # remove final fully-connected layer -> get [B, 512] features
        modules = list(resnet.children())[:-1]  # global avg pooled feature
        self.cnn = nn.Sequential(*modules)
        cnn_out_dim = resnet.fc.in_features
                
        # Freeze CNN backbone
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.img_proj = nn.Linear(cnn_out_dim, img_out_dim)

        # ----- Tabular branch -----
        # Setup categorical embedding structure if indices are provided
        self.cat_feature_indices = []
        self.cont_feature_indices = list(range(num_tab_features))
        self.embeddings = None
        self.embed_dims = []
        self.cat_cardinalities = []

        if (
            cat_feature_indices is not None
            and cat_cardinalities is not None
            and len(cat_feature_indices) > 0
        ):
            self.cat_feature_indices = list(cat_feature_indices)
            self.cat_cardinalities = [int(c) for c in cat_cardinalities]

            # continuous indices are everything except categorical indices
            all_idx = set(range(num_tab_features))
            self.cont_feature_indices = sorted(all_idx - set(self.cat_feature_indices))

            # build embeddings
            self.embeddings = nn.ModuleList()
            self.embed_dims = []
            for card in self.cat_cardinalities:
                if card <= 0:
                    raise ValueError(f"Invalid categorical cardinality: {card}")
                emb_dim = min(50, max(4, card // 2))
                self.embed_dims.append(emb_dim)
                self.embeddings.append(nn.Embedding(card, emb_dim))

            cont_dim = len(self.cont_feature_indices)
            tab_in_dim = cont_dim + sum(self.embed_dims)
        else:
            # no categorical embeddings used
            tab_in_dim = num_tab_features

        # MLP over (continuous + embedded categorical) tabular features
        self.tab_mlp = MLPBackbone(
            in_dim=tab_in_dim,
            hidden_dims=list(tab_hidden_dims),
            dropout=tab_dropout,
        )

        # ----- Fusion + Cox head -----
        fusion_in_dim = img_out_dim + self.tab_mlp.out_dim
        self.fusion_head = nn.Linear(fusion_in_dim, 1)

    def forward(self, x_tab: torch.Tensor, x_img: torch.Tensor) -> torch.Tensor:
        """
        x_tab: [B, d_tab]  tabular features
        x_img: [B, 3, H, W] CXR images

        returns:
            log_risk: [B]
        """
        # image path
        h_img = self.cnn(x_img)                 # [B, 512, 1, 1]
        h_img = h_img.view(h_img.size(0), -1)   # [B, 512]
        h_img = self.img_proj(h_img)              # [B, img_out_dim]

        # tabular path
        if self.embeddings is not None and len(self.cat_feature_indices) > 0:
            # continuous / binary part
            x_cont = x_tab[:, self.cont_feature_indices]

            # embedded categorical part
            cat_embs = []
            for emb, idx, card in zip(
                self.embeddings,
                self.cat_feature_indices,
                self.cat_cardinalities,
            ):
                cat_idx = x_tab[:, idx].long().clamp(0, card - 1)
                cat_embs.append(emb(cat_idx))

            x_tab_in = torch.cat([x_cont] + cat_embs, dim=1)
        else:
            # no categorical embedding
            x_tab_in = x_tab

        h_tab = self.tab_mlp(x_tab_in)             # [B, tab_out]

        # fusion
        h = torch.cat([h_tab, h_img], dim=1)    # [B, img_out_dim + tab_out]
        log_risk = self.fusion_head(h)          # [B, 1]
        return log_risk.squeeze(-1)             # [B]

# Two branches: 
# CNN (ResNet18) for images -> vector.
# MLPBackbone for tabular features -> vector.
# Concatenate them and feed into a Cox head (log_risk)
# Output is again log_risk, so you can reuse the same Cox loss as for DeepSurv