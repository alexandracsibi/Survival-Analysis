#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

# Paths
BASE_DIR = "/home/c_gnnca/c_gnn42/alexa_thesis/data/mimic-eye"
INPUT_CSV = os.path.join(BASE_DIR, "mimic_eye_survival_admissions.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")


def simplify_race(r):
    if pd.isna(r):
        return "UNKNOWN"
    r = str(r)
    if "WHITE" in r:
        return "WHITE"
    if "BLACK" in r:
        return "BLACK"
    if "HISPANIC" in r or "LATINO" in r:
        return "HISPANIC"
    if "ASIAN" in r:
        return "ASIAN"
    if "UNKNOWN" in r or "UNABLE TO OBTAIN" in r or "PATIENT DECLINED" in r:
        return "UNKNOWN"
    return "OTHER"


def simplify_admission_type(x: str) -> str:
    x = str(x)

    # Emergency / urgent
    if "EMER" in x or "URGENT" in x:
        return "EMERGENCY"

    # Observation-type
    if "OBSERV" in x:
        return "OBSERVATION"

    # Planned / scheduled
    if "ELECTIVE" in x or "SURGICAL SAME DAY" in x:
        return "PLANNED"

    return "OTHER"


def simplify_admission_location(x: str) -> str:
    x = str(x)

    # Emergency / ED arrivals
    if "EMERGENCY" in x or "WALK-IN" in x or "SELF REFERRAL" in x:
        return "ED"

    # Referrals
    if "REFERRAL" in x:
        return "REFERRAL"

    # Transfers
    if "TRANSFER" in x:
        return "TRANSFER"

    # Procedure / ambulatory surgery / PACU
    if "AMBULATORY" in x or "PROCEDURE" in x or "PACU" in x:
        return "PROCEDURE/AMBULATORY"

    if "INFORMATION NOT AVAILABLE" in x:
        return "UNKNOWN"

    return "OTHER"


def main():
    print(f"Loading survival dataset from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # 1) Basic cleanup
    id_cols = ["subject_id", "hadm_id"]

    # Survival targets
    duration = "duration_days"
    event = "event"

    drop_cols = [
        "admittime",
        "edregtime",
        "edouttime",
        "endtime",
        "duration_hours",
        "hospital_expire_flag",  # duplicate of event
        "discharge_location",    # leaks data
        "patient_folder",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # 2) simplify into category groups
    if "race" in df.columns:
        df["race"] = df["race"].apply(simplify_race)

    # language: binary column (1 = ENGLISH, 0 = not English/unknown)
    if "language" in df.columns:
        df["language"] = (df["language"] == "ENGLISH").astype("uint8")

    if "admission_type" in df.columns:
        df["admission_type"] = df["admission_type"].apply(simplify_admission_type)
    if "admission_location" in df.columns:
        df["admission_location"] = df["admission_location"].apply(simplify_admission_location)

    # 3) Categorical columns & missing values
    categorical_cols = [
        "admission_type",
        "admission_location",
        "insurance",
        "marital_status",
        "race",
    ]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    for col in categorical_cols:
        df[col] = df[col].fillna("UNKNOWN")

    # 4) Subject-wise train/val/test split
    print("\n--- CREATING SUBJECT-WISE SPLITS ---")
    rng = np.random.RandomState(42)

    unique_subjects = df["subject_id"].unique()
    rng.shuffle(unique_subjects)

    n_subj = len(unique_subjects)
    n_train = int(0.7 * n_subj)
    n_val = int(0.15 * n_subj)
    # Rest goes to test
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

    # 5) One-hot encode categorical features
    print("\n--- ONE-HOT ENCODING CATEGORICAL FEATURES ---")
    X_cat = pd.get_dummies(df[categorical_cols], drop_first=False)
    X_cat = X_cat.astype("uint8")

    # 6) Build final feature matrix and targets
    feature_cols = list(X_cat.columns)

    final_df = pd.concat(
        [
            df[id_cols],
            df[[duration, event, "split"]],
            df[["language"]] if "language" in df.columns else None,
            X_cat,
        ],
        axis=1,
    )

    print("\n--- FINAL DF INFO ---")
    print(final_df.info())

    # 7) Save train / val / test CSVs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_df = final_df[final_df["split"] == split_name].copy()
        out_path = os.path.join(OUTPUT_DIR, f"mimic_eye_survival_{split_name}.csv")
        split_df.to_csv(out_path, index=False)
        print(f"Saved {split_name} split to: {out_path}  (rows: {len(split_df)})")

    print("\nDone preprocessing.")


if __name__ == "__main__":
    main()
