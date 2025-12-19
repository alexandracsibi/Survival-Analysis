import os
from typing import Sequence, Optional, Tuple

import pandas as pd
from .base import CSVSurvivalDataset, infer_feature_cols


def load_support2_split(
    base_dir: str,
    split: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> CSVSurvivalDataset:
    """
    SUPPORT2 survival dataset (binary event).
    Files:
        base_dir / "{split}.csv"
    Columns:
        - time
        - event (1=event, 0=censored)
        - features: OHE + normalized, numeric
    """
    csv_path = os.path.join(base_dir, f"{split}.csv")

    if feature_cols is None:
        df = pd.read_csv(csv_path)
        feature_cols = infer_feature_cols(
            df,
            time_col="time",
            event_cols=["event"],
        )

    return CSVSurvivalDataset(
        csv_path=csv_path,
        time_col="time",
        event_col="event",
        feature_cols=feature_cols,
    )


def load_support2_all(
    base_dir: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[CSVSurvivalDataset, CSVSurvivalDataset, CSVSurvivalDataset]:
    train_ds = load_support2_split(base_dir, "train", feature_cols)
    val_ds = load_support2_split(base_dir, "val", feature_cols)
    test_ds = load_support2_split(base_dir, "test", feature_cols)
    return train_ds, val_ds, test_ds
