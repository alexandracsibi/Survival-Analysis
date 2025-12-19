#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
# Paths
BASE_DIR = "/home/c_gnnca/c_gnn42/alexa_thesis/data/mimic-eye"
INPUT_CSV = os.path.join(BASE_DIR, "mimic_eye_survival_admissions.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_embed")
MAPPING_JSON = os.path.join(OUTPUT_DIR, "category_mappings.json")

def main():
    print(f"Loading survival dataset from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # -------------------------------------------------
    # 1) Basic cleanup
    # -------------------------------------------------
    drop_cols = [
            "admittime",
            "edregtime",
            "edouttime",
            "endtime",
            "duration_hours",
            "hospital_expire_flag",  # duplicate of event
            "discharge_location", # leaks data
            "patient_folder",
    ]

    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    duration_col = "duration_days"
    event_col = "event"

    # -------------------------------------------------
    # 2) Define which columns are embedded vs OHE
    # -------------------------------------------------

    # Columns for embeddings (more categories, keep original granularity)
    embed_cols = [
        "admission_type",
        "admission_location",
        "race",
    ]
    embed_cols = [c for c in embed_cols if c in df.columns]

    # Small-cardinality categoricals -> OHE
    ohe_cols = [
        "insurance",
        "marital_status",
    ]
    ohe_cols = [c for c in ohe_cols if c in df.columns]

    # Language: binary numeric: ENGLISH vs other
    if "language" in df.columns:
        df["language_english"] = (df["language"] == "ENGLISH").astype("uint8")
    else:
        df["language_english"] = 0

    df.drop(columns=["language"], inplace=True)

    # Fill missing for categorical columns
    for col in embed_cols + ohe_cols:
        df[col] = df[col].fillna("UNKNOWN")

    # -------------------------------------------------
    # 3) Subject-wise train/val/test split
    # -------------------------------------------------
    print("\n--- CREATING SUBJECT-WISE SPLITS ---")
    rng = np.random.RandomState(42)
    unique_subjects = df["subject_id"].unique()
    rng.shuffle(unique_subjects)

    n_subj = len(unique_subjects)
    n_train = int(0.7 * n_subj)
    n_val = int(0.15 * n_subj)

    train_subj = set(unique_subjects[:n_train])
    val_subj = set(unique_subjects[n_train:n_train + n_val])
    test_subj = set(unique_subjects[n_train + n_val:])

    def assign_split(sid):
        if sid in train_subj:
            return "train"
        elif sid in val_subj:
            return "val"
        else:
            return "test"

    df["split"] = df["subject_id"].apply(assign_split)

    print(f"Total subjects: {n_subj}")
    print(f"Train subjects: {len(train_subj)}")
    print(f"Val subjects  : {len(val_subj)}")
    print(f"Test subjects : {len(test_subj)}")
    print("\nSplit distribution by rows:")
    print(df["split"].value_counts())

    # -------------------------------------------------
    # 4) Build integer encodings (mappings) for embed_cols
    # -------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mappings = {}

    for col in embed_cols:
        cats = sorted(df[col].unique())
        mapping = {c: i for i, c in enumerate(cats)}
        mappings[col] = {
            "mapping": mapping,
            "num_classes": len(cats),
        }
        df[f"{col}_id"] = df[col].map(mapping).astype("int64")

        print(f"\nEmbedding column '{col}': {len(cats)} categories → '{col}_id'")

    # Save mappings for model to build embedding layers
    with open(MAPPING_JSON, "w") as f:
        json.dump(mappings, f, indent=2)
    print(f"\nSaved embedding mappings to: {MAPPING_JSON}")

    # -------------------------------------------------
    # 5) One-hot encode small-cardinality categoricals
    # -------------------------------------------------
    if ohe_cols:
        X_ohe = pd.get_dummies(df[ohe_cols], drop_first=False).astype("uint8")
        print(f"\nOHE columns expanded to: {X_ohe.shape[1]} features")
    else:
        X_ohe = pd.DataFrame(index=df.index)

    # -------------------------------------------------
    # 6) Build final DataFrame
    # -------------------------------------------------
    id_cols = ["subject_id", "hadm_id"]
    base_cols = [duration_col, event_col, "split", "language_english"]
    embed_id_cols = [f"{c}_id" for c in embed_cols]

    final_df = pd.concat(
        [
            df[id_cols + base_cols + embed_id_cols],
            X_ohe,
        ],
        axis=1,
    )

    print("\n--- FINAL DF INFO (EMBED VERSION) ---")
    print(final_df.info())

    # -------------------------------------------------
    # 7) Save split CSVs
    # -------------------------------------------------
    for split_name in ["train", "val", "test"]:
        split_df = final_df[final_df["split"] == split_name].copy()
        out_path = os.path.join(OUTPUT_DIR, f"mimic_eye_survival_embed_{split_name}.csv")
        split_df.to_csv(out_path, index=False)
        print(f"Saved {split_name} split to: {out_path}  (rows: {len(split_df)})")

    print("\nDone preprocessing.")


if __name__ == "__main__":
    main()