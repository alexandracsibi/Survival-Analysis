#!/usr/bin/env python3
import os
import pandas as pd

BASE_DIR = "/home/c_gnnca/c_gnn42/alexa_thesis/data/physionet.org/files/mimic-eye-multimodal-datasets/1.0.0/mimic-eye"


def main():
    patient_folders = [
        d for d in os.listdir(BASE_DIR)
        if d.startswith("patient_") and os.path.isdir(os.path.join(BASE_DIR, d))
    ]

    # Counters for logical inconsistencies
    total_rows = 0
    cnt_flag1_no_death = 0
    cnt_flag0_with_death = 0
    cnt_missing_admittime = 0
    cnt_missing_dischtime = 0
    cnt_discharge_before_admit = 0
    cnt_death_before_admit = 0
    cnt_death_after_discharge_flag1 = 0

    # Counters for duration issues
    zero_duration = 0
    negative_duration = 0
    long_stays = 0  # > 180 days

    print("=== CHECK 1: FLAG==1 but deathtime is NULL ===")

    time_format = "%Y-%m-%d %H:%M:%S"

    for folder in sorted(patient_folders):
        patient_path = os.path.join(BASE_DIR, folder)
        hosp_path = os.path.join(patient_path, "Hosp", "admissions.csv")

        if not os.path.exists(hosp_path):
            # after your cleanup this should not happen, but we keep it safe
            continue

        try:
            df = pd.read_csv(hosp_path)
        except Exception as e:
            print(f"Could not read {hosp_path}: {e}")
            continue

        if "hospital_expire_flag" not in df.columns or "deathtime" not in df.columns:
            print(f"Missing columns in {hosp_path}, skipping.")
            continue

        # parse times for consistency checks
        for col in ["admittime", "dischtime", "deathtime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format=time_format, errors="coerce")

        total_rows += len(df)

        # --------- CHECK: FLAG==1 but deathtime is NULL ----------
        mask_flag1_no_death = (df["hospital_expire_flag"] == 1) & df["deathtime"].isna()
        if mask_flag1_no_death.any():
            cnt = mask_flag1_no_death.sum()
            cnt_flag1_no_death += cnt
            # print each problematic admission so you can fix manually
            for _, row in df[mask_flag1_no_death].iterrows():
                hadm_id = row.get("hadm_id", "NA")
                print(f"{folder}  hadm_id={hadm_id}  [FLAG=1 but deathtime is NULL]")

        # --------- CHECK: FLAG==0 but deathtime is NOT NULL ----------
        mask_flag0_with_death = (df["hospital_expire_flag"] == 0) & df["deathtime"].notna()
        if mask_flag0_with_death.any():
            cnt_flag0_with_death += mask_flag0_with_death.sum()
            for _, row in df[mask_flag0_with_death].iterrows():
                hadm_id = row.get("hadm_id", "NA")
                print(f"{folder}  hadm_id={hadm_id}  [FLAG=0 but deathtime is NOT NULL]")

        # --------- CHECK: missing admittime / dischtime ----------
        if "admittime" in df.columns:
            mask_missing_admit = df["admittime"].isna()
            if mask_missing_admit.any():
                cnt_missing_admittime += mask_missing_admit.sum()
                for _, row in df[mask_missing_admit].iterrows():
                    hadm_id = row.get("hadm_id", "NA")
                    print(f"{folder}  hadm_id={hadm_id}  [MISSING admittime]")
        else:
            cnt_missing_admittime += len(df)

        if "dischtime" in df.columns:
            mask_missing_discharge = df["dischtime"].isna()
            if mask_missing_discharge.any():
                cnt_missing_dischtime += mask_missing_discharge.sum()
                for _, row in df[mask_missing_discharge].iterrows():
                    hadm_id = row.get("hadm_id", "NA")
                    print(f"{folder}  hadm_id={hadm_id}  [MISSING dischtime]")
        else:
            cnt_missing_dischtime += len(df)

        # --------- CHECK: Dischtime < Admittime ----------
        if "admittime" in df.columns and "dischtime" in df.columns:
            mask_discharge_before_admit = df["dischtime"] < df["admittime"]
            if mask_discharge_before_admit.any():
                cnt_discharge_before_admit += mask_discharge_before_admit.sum()
                for _, row in df[mask_discharge_before_admit].iterrows():
                    hadm_id = row.get("hadm_id", "NA")
                    print(f"{folder}  hadm_id={hadm_id}  [DISCHTIME < ADMITTIME]")

        # --------- CHECK: Deathtime before Admittime ----------
        if "admittime" in df.columns and "deathtime" in df.columns:
            mask_death_before_admit = df["deathtime"].notna() & (df["deathtime"] < df["admittime"])
            if mask_death_before_admit.any():
                cnt_death_before_admit += mask_death_before_admit.sum()
                for _, row in df[mask_death_before_admit].iterrows():
                    hadm_id = row.get("hadm_id", "NA")
                    print(f"{folder}  hadm_id={hadm_id}  [DEATHTIME < ADMITTIME]")

        # --------- CHECK: FLAG==1 & Deathtime AFTER Dischtime ----------
        if "dischtime" in df.columns and "deathtime" in df.columns:
            mask_flag1_death_after_discharge = (
                (df["hospital_expire_flag"] == 1)
                & df["deathtime"].notna()
                & df["dischtime"].notna()
                & (df["deathtime"] > df["dischtime"])
            )
            if mask_flag1_death_after_discharge.any():
                cnt_death_after_discharge_flag1 += mask_flag1_death_after_discharge.sum()
                for _, row in df[mask_flag1_death_after_discharge].iterrows():
                    hadm_id = row.get("hadm_id", "NA")
                    print(f"{folder}  hadm_id={hadm_id}  [FLAG=1 & DEATHTIME > DISCHTIME]")

        # =========================
        # DURATION CHECKS SECTION
        # =========================

        # Define survival end time: deathtime if available, else dischtime
        df["endtime"] = df["deathtime"].fillna(df["dischtime"])

        # Duration in hours
        df["duration_hours"] = (df["endtime"] - df["admittime"]).dt.total_seconds() / 3600

        # Zero-duration stays
        mask_zero = df["duration_hours"] == 0
        if mask_zero.any():
            zero_duration += mask_zero.sum()
            for _, row in df[mask_zero].iterrows():
                hadm_id = row.get("hadm_id", "NA")
                print(f"{folder}  hadm_id={hadm_id}  [ZERO DURATION]")

        # Negative-duration stays (should not happen after previous fixes)
        mask_negative = df["duration_hours"] < 0
        if mask_negative.any():
            negative_duration += mask_negative.sum()
            for _, row in df[mask_negative].iterrows():
                hadm_id = row.get("hadm_id", "NA")
                print(f"{folder}  hadm_id={hadm_id}  [NEGATIVE DURATION]")

        # Very long stays: threshold = 180 days
        mask_long = df["duration_hours"] > (180 * 24)
        if mask_long.any():
            long_stays += mask_long.sum()
            for _, row in df[mask_long].iterrows():
                hadm_id = row.get("hadm_id", "NA")
                print(f"{folder}  hadm_id={hadm_id}  [EXTREMELY LONG STAY > 180 days]")

    # =========================
    # FINAL SUMMARY
    # =========================
    print("\n=== SUMMARY OF INCONSISTENCIES ===")
    print(f"Total admissions rows processed            : {total_rows}")
    print(f"FLAG==1 but deathtime is NULL              : {cnt_flag1_no_death}")
    print(f"FLAG==0 but deathtime is NOT NULL          : {cnt_flag0_with_death}")
    print(f"Missing admittime                          : {cnt_missing_admittime}")
    print(f"Missing dischtime                          : {cnt_missing_dischtime}")
    print(f"Dischtime < admittime                      : {cnt_discharge_before_admit}")
    print(f"Deathtime before admittime                 : {cnt_death_before_admit}")
    print(f"FLAG==1 & deathtime AFTER dischtime        : {cnt_death_after_discharge_flag1}")

    print("\n=== DURATION SUMMARY ===")
    print(f"Zero-duration stays                        : {zero_duration}")
    print(f"Negative-duration stays                    : {negative_duration}")
    print(f"Very long stays (>180 days)                 : {long_stays}")


if __name__ == "__main__":
    main()
