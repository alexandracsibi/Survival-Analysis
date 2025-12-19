import os
from typing import Sequence, Optional, Tuple

import pandas as pd
from .base import CSVSurvivalDataset, infer_feature_cols


def load_mnb_comprisk_split(
    base_dir: str,
    split: str,
    event_col: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> CSVSurvivalDataset:
    """
    MNB contract-level dataset.

    Files:
        base_dir / "{split}.csv"
    Columns:
        - time   : float
        - event  : int (0=censored, 1/2=event types)
        - others : numeric features (already scaled + OHE)
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

def load_mnb_comprisk_all(
    base_dir: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[CSVSurvivalDataset, CSVSurvivalDataset, CSVSurvivalDataset]:
    train_ds    = load_mnb_comprisk_split(base_dir, "train", feature_cols)
    val_ds      = load_mnb_comprisk_split(base_dir, "val", feature_cols)
    test_ds     = load_mnb_comprisk_split(base_dir, "test", feature_cols)
    return train_ds, val_ds, test_ds
