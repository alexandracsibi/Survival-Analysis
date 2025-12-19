import os
import json
from typing import Sequence, Optional, Tuple

import pandas as pd
from .base import CSVSurvivalDataset, infer_feature_cols


def load_mimiceye_tabular_split(
    base_dir: str,
    split: str,
    feature_cols: Optional[Sequence[str]] = None,
    cat_feature_names: Optional[Sequence[str]] = None,
    category_mapping_json: Optional[str] = None,
) -> CSVSurvivalDataset:
    csv_path = os.path.join(base_dir, "tabular", f"{split}.csv")

    if feature_cols is None or cat_feature_names is not None:
        df = pd.read_csv(csv_path)
    else:
        df = None

    if feature_cols is None:
        feature_cols = infer_feature_cols(
            df,
            time_col="duration_days",
            event_cols=["event"],
            extra_drop_cols=["split", "subject_id", "hadm_id"],
        )

    ds = CSVSurvivalDataset(
        csv_path=csv_path,
        time_col="duration_days",
        event_col="event",
        feature_cols=feature_cols,
    )

    if cat_feature_names is not None:
        if category_mapping_json is None:
            raise ValueError("category_mapping_json must be provided when using cat embeddings")

        with open(category_mapping_json, "r") as f:
            mappings = json.load(f)

        ds.cat_feature_names = list(cat_feature_names)
        ds.cat_feature_indices = []
        ds.cat_cardinalities = []

        for name in ds.cat_feature_names:
            if name not in feature_cols:
                raise ValueError(f"Categorical feature {name} not in feature_cols")

            # map admission_type_id → admission_type
            base_name = name.replace("_id", "")
            if base_name not in mappings:
                raise ValueError(f"{base_name} not found in category_mappings.json")

            ds.cat_feature_indices.append(feature_cols.index(name))
            ds.cat_cardinalities.append(mappings[base_name]["num_classes"])

    return ds


def load_mimiceye_tabular_all(
    base_dir: str,
    feature_cols: Optional[Sequence[str]] = None,
    cat_feature_names: Optional[Sequence[str]] = None,
    category_mapping_json: Optional[str] = None,
) -> Tuple[CSVSurvivalDataset, CSVSurvivalDataset, CSVSurvivalDataset]:
    train_ds = load_mimiceye_tabular_split(
        base_dir, "train", feature_cols, cat_feature_names, category_mapping_json
    )
    val_ds = load_mimiceye_tabular_split(
        base_dir, "val", feature_cols, cat_feature_names, category_mapping_json
    )
    test_ds = load_mimiceye_tabular_split(
        base_dir, "test", feature_cols, cat_feature_names, category_mapping_json
    )
    return train_ds, val_ds, test_ds