import os
from typing import Sequence, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from .base import infer_feature_cols


class MIMICEyeMultimodalDataset(Dataset):
    """
    Multimodal MIMIC-Eye dataset:
      - tabular embedded features
      - CXR image at cxr_jpg_path
      - survival target: duration_days, event
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str | None = None,
        feature_cols=None,
        img_size: int = 224,
        float_dtype: torch.dtype = torch.float32,
        on_missing_image: str = "raise",
    ):
        self.csv_path = csv_path
        self.image_root = image_root  # optional override of base dir
        self.float_dtype = float_dtype
        self.on_missing_image = on_missing_image

        df = pd.read_csv(csv_path)

        for col in ["admission_type_id", "admission_location_id", "race_id"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)

        # Determine feature columns: drop targets + path + split
        extra_drop = [
            "cxr_jpg_path",
            "subject_id",
            "hadm_id",
            "study_id",
            "dicom_id",
            "split",
        ]

        if feature_cols is None:
            feature_cols = infer_feature_cols(
                df,
                time_col="duration_days",
                event_cols=["event"],
                extra_drop_cols=extra_drop,
            )
        self.feature_cols = list(feature_cols)

        x_df = df[self.feature_cols].copy()
        for c in x_df.columns:
            # make sure everything is numeric
            x_df[c] = pd.to_numeric(x_df[c], errors="coerce")
        x_df = x_df.fillna(0.0)

        self.x = x_df.to_numpy(dtype=np.float32)
        self.time = pd.to_numeric(df["duration_days"], errors="coerce").fillna(0).to_numpy(np.float32)
        self.event = pd.to_numeric(df["event"], errors="coerce").fillna(0).to_numpy(np.int64)

        self.paths = df["cxr_jpg_path"].astype(str).tolist()

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._img_size = img_size

    def __len__(self) -> int:
        return len(self.time)

    def _resolve_path(self, path: str) -> str:
        """
        Resolve image path stored in CSV to a valid path inside the container.

        Handles:
        1) Absolute paths from old environment
           (e.g. /home/c_gnnca/c_gnn42/alexa_thesis/raw_data/...)
           -> rewrite prefix to /workspace
        2) Relative paths combined with image_root.
        """
        p = path

        # Case 1: old absolute host path, rewrite to /workspace/...
        old_prefix = "/home/c_gnnca/c_gnn42/alexa_thesis"
        new_prefix = "/workspace"
        if p.startswith(old_prefix):
            p = p.replace(old_prefix, new_prefix, 1)

        # Case 2: if still not absolute and image_root is given, join
        if not os.path.isabs(p) and self.image_root is not None:
            p = os.path.join(self.image_root, p)

        return p

    def __getitem__(self, idx: int):
        x = torch.as_tensor(self.x[idx], dtype=self.float_dtype)
        t = torch.as_tensor(self.time[idx], dtype=self.float_dtype)
        e = torch.as_tensor(self.event[idx], dtype=torch.long)

        img_path = self._resolve_path(self.paths[idx])

        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            img_missing = torch.tensor(0, dtype=torch.uint8)

        except Exception as ex:
            if self.on_missing_image == "zeros":
                warnings.warn(f"[MIMIC-Eye multimodal] Missing/unreadable image at idx={idx}: {img_path} ({ex})")
                img = torch.zeros((3, self._img_size, self._img_size), dtype=self.float_dtype)
                img_missing = torch.tensor(1, dtype=torch.uint8)
            else:
                raise RuntimeError(f"Failed to load image idx={idx} path={img_path}. Original error: {ex}") from ex

        return {"x": x, "image": img, "time": t, "event": e, "img_missing": img_missing}


def load_mimiceye_multimodal_split(
    base_dir: str,
    split: str,
    feature_cols: Optional[Sequence[str]] = None,
    image_root: Optional[str] = None,
    img_size: int = 224,
) -> MIMICEyeMultimodalDataset:
    """
    Files:
        base_dir / "multimodal" / "{split}.csv"
    The CSV must have 'cxr_jpg_path' pointing to each CXR.
    """
    csv_path = os.path.join(base_dir, "multimodal", f"{split}.csv")
    return MIMICEyeMultimodalDataset(
        csv_path=csv_path,
        image_root=image_root,
        feature_cols=feature_cols,
        img_size=img_size,
    )


def load_mimiceye_multimodal_all(
    base_dir: str,
    feature_cols: Optional[Sequence[str]] = None,
    image_root: Optional[str] = None,
    img_size: int = 224,
) -> Tuple[MIMICEyeMultimodalDataset,
           MIMICEyeMultimodalDataset,
           MIMICEyeMultimodalDataset]:
    train_ds = load_mimiceye_multimodal_split(
        base_dir, "train", feature_cols, image_root, img_size
    )
    val_ds = load_mimiceye_multimodal_split(
        base_dir, "val", feature_cols, image_root, img_size
    )
    test_ds = load_mimiceye_multimodal_split(
        base_dir, "test", feature_cols, image_root, img_size
    )
    return train_ds, val_ds, test_ds