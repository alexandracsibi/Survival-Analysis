#!/usr/bin/env python3
import os
import pandas as pd

BASE_DIR = "/home/c_gnnca/c_gnn42/alexa_thesis/data/physionet.org/files/mimic-eye-multimodal-datasets/1.0.0/mimic-eye"

def main():
    # Find all patient folders
    patient_folders = [
        d for d in os.listdir(BASE_DIR)
        if d.startswith("patient_") and os.path.isdir(os.path.join(BASE_DIR, d))
    ]

    for folder in sorted(patient_folders):
        patient_path = os.path.join(BASE_DIR, folder)
        hosp_path = os.path.join(patient_path, "Hosp", "admissions.csv")

        if not os.path.exists(hosp_path):
            continue

        # Load admissions
        df = pd.read_csv(hosp_path)

        # Parse datetimes
        fmt = "%Y-%m-%d %H:%M:%S"
        for col in ["admittime", "dischtime", "deathtime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")

        # Mask for inconsistencies
        mask = (
            (df["hospital_expire_flag"] == 1) &
            (df["deathtime"].notna()) &
            (df["dischtime"].notna()) &
            (df["deathtime"] > df["dischtime"])
        )

        if mask.any():
            print(f"{folder}: fixing {mask.sum()} rows where deathtime > dischtime")

            # Fix: set dischtime = deathtime
            df.loc[mask, "dischtime"] = df.loc[mask, "deathtime"]

            # Save corrected file
            df.to_csv(hosp_path, index=False)

    print("Done fixing all deathtime > dischtime inconsistencies.")

if __name__ == "__main__":
    main()
