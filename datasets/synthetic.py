import os
from typing import Sequence, Optional, Tuple

from .base import CSVSurvivalDataset, infer_feature_cols
import pandas as pd


def load_synthetic_split(
    base_dir: str,
    split: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> CSVSurvivalDataset:
    """
    Synthetic DeepHit-style dataset (competing risks).
    Columns:
        - time      : observed time
        - label     : 0=censored, 1,2 = event types
        - features  : already normalized numeric features
    """
    csv_path = os.path.join(base_dir, f"{split}.csv")

    if feature_cols is None:
        df = pd.read_csv(csv_path)
        feature_cols = infer_feature_cols(
            df,
            time_col="time",
            event_cols=["label"],
            extra_drop_cols=["id"],
        )

    return CSVSurvivalDataset(
        csv_path=csv_path,
        time_col="time",
        event_col="label",
        feature_cols=feature_cols,
    )


def load_synthetic_all(
    base_dir: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[CSVSurvivalDataset, CSVSurvivalDataset, CSVSurvivalDataset]:
    train_ds = load_synthetic_split(base_dir, "train", feature_cols)
    val_ds = load_synthetic_split(base_dir, "val", feature_cols)
    test_ds = load_synthetic_split(base_dir, "test", feature_cols)
    return train_ds, val_ds, test_ds
