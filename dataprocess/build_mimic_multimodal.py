#!/usr/bin/env python3
import os
import pandas as pd

# admission-level survival dataset
SURVIVAL_CSV = "/home/c_gnnca/c_gnn42/alexa_thesis/data/mimic-eye/tabular/mimic_eye_survival_admissions.csv"

# patient_* folders (with CXR-JPG) are
PATIENTS_ROOT = "/home/c_gnnca/c_gnn42/alexa_thesis/raw_data/mimic-eye/patients"

# Official MIMIC-Eye root with spreadsheets/cxr_meta.csv
MIMICEYE_ROOT = "/home/c_gnnca/c_gnn42/alexa_thesis/raw_data/mimic-eye"
CXR_META_CSV = os.path.join(MIMICEYE_ROOT, "spreadsheets", "cxr_meta.csv")

# Output
OUTPUT_DIR = "/home/c_gnnca/c_gnn42/alexa_thesis/data/mimic-eye/multimodal"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "mimic_eye_multimodal_admissions.csv")

# === HELPERS ==============================================================

def normalize_study_time(val):
    """
    StudyTime in cxr_meta looks like: 92717.109, 160736.171, etc.
    We normalize to a string HHMMSS(.fff) with zero-padding, e.g.:

        92717.109  -> "092717.109"
        160736.171 -> "160736.171"

    Returns None if val is missing/NaN.
    """
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None

    if "." in s:
        left, right = s.split(".", 1)
        left = left.zfill(6)
        return f"{left}.{right}"
    else:
        left = s.zfill(6)
        return left


def find_cxr_jpg_path(patient_folder, study_id, dicom_id):
    """
    Build the JPG path for a single admission:

        PATIENTS_ROOT/patient_xxxxx/CXR-JPG/s<study_id>/<dicom_id>.jpg

    Fallbacks:
      - if that exact file does not exist, try any .jpg in s<study_id> whose
        name contains dicom_id.
      - if still nothing, search the entire CXR-JPG subtree for a .jpg that
        contains dicom_id in the filename.

    Returns:
        full path (str) if found, otherwise None.
    """
    study_id_str = str(study_id)
    dicom_id_str = str(dicom_id)

    # Folder is named s<study_id>, add leading 's' if not present
    if study_id_str.startswith("s"):
        study_folder_name = study_id_str
    else:
        study_folder_name = "s" + study_id_str

    base_dir = os.path.join(PATIENTS_ROOT, patient_folder, "CXR-JPG", study_folder_name)
    jpg_path = os.path.join(base_dir, f"{dicom_id_str}.jpg")

    # 1) Exact match
    if os.path.exists(jpg_path):
        return jpg_path

    # 2) Any .jpg in that study folder whose name contains dicom_id
    if os.path.isdir(base_dir):
        candidates = []
        for f in os.listdir(base_dir):
            if not f.lower().endswith(".jpg"):
                continue
            name_no_ext, _ = os.path.splitext(f)
            if dicom_id_str in name_no_ext:
                candidates.append(os.path.join(base_dir, f))
        if candidates:
            candidates.sort()  # deterministic
            return candidates[0]

    # 3) Final fallback: search entire CXR-JPG subtree for dicom_id in filename
    cxr_jpg_root = os.path.join(PATIENTS_ROOT, patient_folder, "CXR-JPG")
    if os.path.isdir(cxr_jpg_root):
        for root, _, files in os.walk(cxr_jpg_root):
            for f in files:
                if not f.lower().endswith(".jpg"):
                    continue
                name_no_ext, _ = os.path.splitext(f)
                if dicom_id_str in name_no_ext:
                    return os.path.join(root, f)

    return None


def match_cxr_to_admissions(surv_sub, cxr_sub):
    """
    For one subject_id:
      surv_sub: admissions (rows from survival df) for this subject
      cxr_sub:  cxr_meta rows for this subject

    Returns:
      DataFrame of cxr_sub rows + a resolved hadm_id column, one row per CXR.
    """
    results = []

    # Ensure times are datetime
    # surv_sub already has admittime / endtime as datetime in main()
    for _, cxr_row in cxr_sub.iterrows():
        dt = cxr_row["study_datetime"]
        if pd.isna(dt):
            # Cannot match without a timestamp -> skip
            continue

        # Admissions where CXR is within [admittime, endtime]
        mask_instay = (surv_sub["admittime"] <= dt) & (surv_sub["endtime"] >= dt)
        candidates = surv_sub[mask_instay].copy()

        if candidates.empty:
            # Fallback: choose admission with closest admittime
            candidates = surv_sub.copy()
            candidates["time_diff"] = (candidates["admittime"] - dt).abs()
        else:
            # among in-stay admissions, choose closest to admittime
            candidates["time_diff"] = (dt - candidates["admittime"]).abs()

        best = candidates.sort_values("time_diff").iloc[0]
        new_row = cxr_row.to_dict()
        new_row["hadm_id"] = int(best["hadm_id"])
        results.append(new_row)

    if not results:
        return pd.DataFrame(columns=list(cxr_sub.columns) + ["hadm_id"])

    return pd.DataFrame(results)


# === MAIN PIPELINE ========================================================

def main():
    # ----------------------------------------------------------------------
    # 1) Load survival data (already built from Hosp/admissions.csv)
    # ----------------------------------------------------------------------
    print(f"Loading survival admissions from: {SURVIVAL_CSV}")
    surv = pd.read_csv(SURVIVAL_CSV)

    required_surv_cols = ["subject_id", "hadm_id", "patient_folder", "admittime", "endtime"]
    for c in required_surv_cols:
        if c not in surv.columns:
            raise ValueError(f"Survival CSV missing required column '{c}'")

    # Normalize ID types
    surv["subject_id"] = surv["subject_id"].astype("int64")
    surv["hadm_id"] = pd.to_numeric(surv["hadm_id"], errors="coerce").astype("Int64")
    surv = surv[surv["hadm_id"].notna()].copy()
    surv["hadm_id"] = surv["hadm_id"].astype("int64")

    # Parse admission times
    for col in ["admittime", "endtime"]:
        surv[col] = pd.to_datetime(surv[col], errors="coerce")

    # Drop rows with missing times (should be none after your cleaning)
    surv = surv[surv["admittime"].notna() & surv["endtime"].notna()].copy()

    # ----------------------------------------------------------------------
    # 2) Load CXR metadata (without hadm_id, we will infer it)
    # ----------------------------------------------------------------------
    print(f"Loading CXR metadata from: {CXR_META_CSV}")
    cxr = pd.read_csv(CXR_META_CSV)

    required_cxr_cols = ["subject_id", "study_id", "dicom_id", "StudyDate", "StudyTime"]
    for c in required_cxr_cols:
        if c not in cxr.columns:
            raise ValueError(f"CXR meta CSV missing required column '{c}'")

    cxr["subject_id"] = cxr["subject_id"].astype("int64")

    # Normalize StudyTime and build study_datetime
    cxr["StudyTime_norm"] = cxr["StudyTime"].apply(normalize_study_time)

    def build_datetime(row):
        if pd.isna(row["StudyDate"]) or row["StudyTime_norm"] is None:
            return pd.NaT
        # StudyDate looks like 21560419
        date_str = str(int(row["StudyDate"]))
        time_str = row["StudyTime_norm"]  # e.g. "092717.109"
        dt_str = f"{date_str} {time_str}"
        # Parse with explicit format
        return pd.to_datetime(dt_str, format="%Y%m%d %H%M%S.%f", errors="coerce")

    cxr["study_datetime"] = cxr.apply(build_datetime, axis=1)

    # Drop CXR rows with no valid datetime
    before_cxr = len(cxr)
    cxr = cxr[cxr["study_datetime"].notna()].copy()
    after_cxr = len(cxr)
    print(f"CXR rows before dropping invalid datetimes: {before_cxr}")
    print(f"CXR rows after dropping invalid datetimes : {after_cxr}")

    # ----------------------------------------------------------------------
    # 3) For each subject, match CXR rows to admissions to get hadm_id
    # ----------------------------------------------------------------------
    matched_rows = []

    surv_grouped = surv.groupby("subject_id")
    cxr_grouped = cxr.groupby("subject_id")

    common_subjects = sorted(set(surv_grouped.groups.keys()) & set(cxr_grouped.groups.keys()))
    print(f"Subjects with both survival + CXR: {len(common_subjects)}")

    for sid in common_subjects:
        surv_sub = surv_grouped.get_group(sid)
        cxr_sub = cxr_grouped.get_group(sid)
        matched = match_cxr_to_admissions(surv_sub, cxr_sub)
        matched_rows.append(matched)

    if matched_rows:
        cxr_with_hadm = pd.concat(matched_rows, ignore_index=True)
    else:
        cxr_with_hadm = pd.DataFrame(columns=list(cxr.columns) + ["hadm_id"])

    print(f"Total CXR rows matched to admissions: {len(cxr_with_hadm)}")

    # In case of multiple CXR per admission, keep earliest study_datetime per (subject_id, hadm_id)
    cxr_with_hadm_sorted = cxr_with_hadm.sort_values(["subject_id", "hadm_id", "study_datetime"])
    cxr_unique = cxr_with_hadm_sorted.groupby(["subject_id", "hadm_id"], as_index=False).first()

    print(f"Unique CXR rows after grouping by (subject_id, hadm_id): {len(cxr_unique)}")

    # ----------------------------------------------------------------------
    # 4) Merge survival + CXR (now we have hadm_id on both sides)
    # ----------------------------------------------------------------------
    multimodal = pd.merge(
        surv,
        cxr_unique,
        on=["subject_id", "hadm_id"],
        how="inner",
        suffixes=("", "_cxr"),
    )

    print(f"\nAfter merging survival + CXR: {multimodal.shape}")

    # ----------------------------------------------------------------------
    # 5) Build CXR JPG paths for each admission
    # ----------------------------------------------------------------------
    def build_path(row):
        return find_cxr_jpg_path(
            patient_folder=row["patient_folder"],
            study_id=row["study_id"],
            dicom_id=row["dicom_id"],
        )

    multimodal["cxr_jpg_path"] = multimodal.apply(build_path, axis=1)

    before_drop = len(multimodal)
    multimodal = multimodal[multimodal["cxr_jpg_path"].notna()].copy()
    after_drop = len(multimodal)

    print(f"\nRows before dropping missing images: {before_drop}")
    print(f"Rows after  dropping missing images: {after_drop}")
    print(f"Dropped due to missing JPG path   : {before_drop - after_drop}")

    # ----------------------------------------------------------------------
    # 6) Arrange columns in a nice order
    # ----------------------------------------------------------------------
    core_cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "endtime",
        "duration_days",
        "event",
    ]
    core_cols = [c for c in core_cols if c in multimodal.columns]

    tabular_cols = [
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "race",
    ]
    tabular_cols = [c for c in tabular_cols if c in multimodal.columns]

    image_cols = ["study_id", "dicom_id", "cxr_jpg_path"]
    image_cols = [c for c in image_cols if c in multimodal.columns]

    meta_cols = ["patient_folder"]
    meta_cols = [c for c in meta_cols if c in multimodal.columns]

    ordered_cols = core_cols + tabular_cols + image_cols + meta_cols
    remaining_cols = [c for c in multimodal.columns if c not in ordered_cols]
    final_cols = ordered_cols + remaining_cols

    multimodal = multimodal[final_cols]

    print("\n=== MULTIMODAL DF PREVIEW ===")
    print(multimodal.head())
    print("\nMultimodal df info:")
    print(multimodal.info())

    # ----------------------------------------------------------------------
    # 7) Save final multimodal CSV
    # ----------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    multimodal.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved multimodal survival dataset to: {OUTPUT_CSV}")
    print(f"Rows with both survival + CXR image path: {len(multimodal)}")


if __name__ == "__main__":
    main()
