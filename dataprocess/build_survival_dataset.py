#!/usr/bin/env python3
import os
import pandas as pd

# Root of MIMIC-eye patient folders
BASE_DIR = "/home/c_gnnca/c_gnn42/alexa_thesis/data/mimic-eye/patients"

# Where to save the final survival dataset
OUTPUT_CSV = "/home/c_gnnca/c_gnn42/alexa_thesis/data/mimic-eye/mimic_eye_survival_admissions.csv"


def main():
    patient_folders = [
        d for d in os.listdir(BASE_DIR)
        if d.startswith("patient_") and os.path.isdir(os.path.join(BASE_DIR, d))
    ]

    time_format = "%Y-%m-%d %H:%M:%S"

    all_rows = []
    total_admissions = 0

    print(f"Found {len(patient_folders)} patient folders. Collecting admissions...")

    # 1) COLLECT ALL ADMISSIONS
    for folder in sorted(patient_folders):
        hosp_path = os.path.join(BASE_DIR, folder, "Hosp", "admissions.csv")
        if not os.path.exists(hosp_path):
            continue

        df = pd.read_csv(hosp_path)

        # Parse timestamps
        for col in ["admittime", "dischtime", "deathtime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=time_format, errors="coerce")

        # Keep track of which patient folder this came from (optional but useful)
        df["patient_folder"] = folder

        all_rows.append(df)
        total_admissions += len(df)

    print(f"Total admissions loaded from all patients: {total_admissions}")

    # Concatenate all admissions into a single DataFrame
    full = pd.concat(all_rows, ignore_index=True)
    print(f"Combined DataFrame shape: {full.shape}")
    print("Columns:", list(full.columns))

    # 2) DEFINE SURVIVAL TARGETS: endtime, duration, event
    # Event: in-hospital death (already cleaned inconsistencies)
    # hospital_expire_flag should be 0 or 1, no NaNs
    full["event"] = full["hospital_expire_flag"].astype(int)

    # End of follow-up:
    # for in-hospital deaths, deathtime is set and equals discharge time
    # for others, deathtime is NaN, so we use dischtime
    full["endtime"] = full["deathtime"].fillna(full["dischtime"])

    # Duration in hours and days
    full["duration_hours"] = (full["endtime"] - full["admittime"]).dt.total_seconds() / 3600
    full["duration_days"] = full["duration_hours"] / 24.0

    # 3) BASIC FILTERS: remove invalid / missing durations
    before_filter = len(full)

    # Drop rows where admittime or endtime is missing
    mask_missing_times = full["admittime"].isna() | full["endtime"].isna()
    missing_count = mask_missing_times.sum()
    if missing_count > 0:
        print(f"Dropping {missing_count} rows with missing admittime or endtime.")
    full = full[~mask_missing_times]

    # Drop rows with non-positive duration (<= 0 hours)
    mask_nonpositive = full["duration_hours"] <= 0
    nonpositive_count = mask_nonpositive.sum()
    if nonpositive_count > 0:
        print(f"Dropping {nonpositive_count} rows with non-positive duration (<= 0 hours).")
    full = full[~mask_nonpositive]

    after_filter = len(full)
    print(f"Rows before filtering: {before_filter}")
    print(f"Rows after filtering : {after_filter}")
    print(f"Dropped total        : {before_filter - after_filter}")

    # 4) SOME SUMMARY STATS TO VERIFY
    n_events = int(full["event"].sum())
    n_censored = int(len(full) - n_events)
    median_days = full["duration_days"].median()
    max_days = full["duration_days"].max()
    min_days = full["duration_days"].min()

    print("\n=== SURVIVAL DATASET SUMMARY ===")
    print(f"Total admissions (after filters): {len(full)}")
    print(f"Events (in-hospital deaths)     : {n_events}")
    print(f"Censored                        : {n_censored}")
    print(f"Event rate                      : {n_events / len(full):.3f}")
    print(f"Duration days: min={min_days:.2f}, median={median_days:.2f}, max={max_days:.2f}")

    # 5) SELECT COLUMNS TO SAVE
    # You can adjust this list later as you add features.
    cols_to_keep = [
        "subject_id",
        "hadm_id",
        "admittime",
        "endtime",
        "duration_days",
        "duration_hours",
        "event",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "edregtime",
        "edouttime",
        "hospital_expire_flag",
        "patient_folder",
    ]

    # Keep only columns that actually exist (edregtime/edouttime may be missing in some rows)
    cols_to_keep = [c for c in cols_to_keep if c in full.columns]

    survival_df = full[cols_to_keep].copy()

    print("\nFirst few rows of survival_df:")
    print(survival_df.head())

    # 6) SAVE TO CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    survival_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved survival dataset to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
