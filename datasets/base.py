import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Sequence


def infer_feature_cols(
    df: pd.DataFrame,
    time_col: str,
    event_cols: Sequence[str],
    extra_drop_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Infer feature columns by dropping target / meta columns.
    Keeps only numeric columns by default.
    """
    drop = set(event_cols) | {time_col}
    if extra_drop_cols is not None:
        drop |= set(extra_drop_cols)
    feature_cols = [
        c for c in df.columns
        if c not in drop and np.issubdtype(df[c].dtype, np.number)
    ]
    return feature_cols


class CSVSurvivalDataset(Dataset):
    """
    Generic tabular survival dataset.

    Returns a dict:
        {
            "x":      [d] float tensor features,
            "time":   scalar float tensor,
            "event":  scalar float or long tensor (as in CSV),
        }
    """

    def __init__(
        self,
        csv_path: str,
        time_col: str,
        event_col: str,
        feature_cols: Optional[Sequence[str]] = None,
        extra_drop_cols: Optional[Sequence[str]] = None,
        float_dtype: torch.dtype = torch.float32,
    ):
        self.csv_path = csv_path
        self.time_col = time_col
        self.event_col = event_col
        self.float_dtype = float_dtype

        df = pd.read_csv(csv_path)
        self.df = df

        if feature_cols is None:
            feature_cols = infer_feature_cols(
                df, time_col=time_col,
                event_cols=[event_col],
                extra_drop_cols=extra_drop_cols,
            )
        self.feature_cols = list(feature_cols)

        self.x = df[self.feature_cols].to_numpy(dtype=np.float32)
        self.time = df[time_col].to_numpy(dtype=np.float32)
        # keep event as int; works for binary and multi-class
        self.event = df[event_col].to_numpy(dtype=np.int64)

        self.indices = np.arange(len(self.df))

    def __len__(self) -> int:
        return len(self.time)

    def __getitem__(self, idx: int):
        return {
            "x": torch.as_tensor(self.x[idx], dtype=self.float_dtype),
            "time": torch.as_tensor(self.time[idx], dtype=self.float_dtype),
            "event": torch.as_tensor(self.event[idx], dtype=torch.long),
            "idx": torch.as_tensor(self.indices[idx], dtype=torch.long),
        }

class UnlabeledSurvivalWrapper(Dataset):
    """
    Wraps a CSVSurvivalDataset but removes time/event labels.
    Returns only x (features).
    Used for pseudo-labeling.
    """
    def __init__(self, base_ds: Dataset):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        item = self.base_ds[idx]
        # Only return features; no labels
        return {
            "x": item["x"],
            "idx": item["idx"],
        }
