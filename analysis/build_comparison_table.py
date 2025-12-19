#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


METRICS = ["test_c_index", "test_td_auc", "test_iauc", "test_ibs"]


# ----------------------------
# Data loading
# ----------------------------
def load_multiseed_csv(run_dir: Path) -> pd.DataFrame | None:
    ms = list((run_dir / "multiseed").glob("*_multiseed_per_seed.csv"))
    if not ms:
        return None
    return pd.read_csv(ms[0])


def load_seed_jsons(run_dir: Path) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(run_dir.glob("seed*")):
        p = seed_dir / "test_metrics.json"
        if not p.exists():
            continue
        with open(p) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def collect_test_metrics(run_dir: Path) -> pd.DataFrame:
    df = load_multiseed_csv(run_dir)
    if df is not None and not df.empty:
        return df
    return load_seed_jsons(run_dir)


# ----------------------------
# Aggregation
# ----------------------------
def aggregate(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    for m in METRICS:
        if m not in df.columns:
            continue
        vals = df[m].dropna().values
        if len(vals) == 0:
            continue
        out[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
        }
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="runs/")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    root = Path(args.root)
    rows: List[Dict[str, object]] = []

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name

        for cond_dir in sorted(dataset_dir.iterdir()):
            if not cond_dir.is_dir():
                continue

            df = collect_test_metrics(cond_dir)
            if df.empty:
                continue

            agg = aggregate(df)
            if not agg:
                continue

            row = {
                "dataset": dataset,
                "condition": cond_dir.name,
                "n_seeds": max(v["n"] for v in agg.values()),
            }

            for m in METRICS:
                if m in agg:
                    row[f"{m}_mean"] = agg[m]["mean"]
                    row[f"{m}_std"] = agg[m]["std"]
                else:
                    row[f"{m}_mean"] = np.nan
                    row[f"{m}_std"] = np.nan

            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["dataset", "condition"])
    out_df.to_csv(args.out, index=False)

    print(f"[OK] Wrote comparison table to: {args.out}")


if __name__ == "__main__":
    main()
