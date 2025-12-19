#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np

def horizon_counts(time, event, t0, event_of_interest):
    time = np.asarray(time, float)
    event = np.asarray(event, int)

    pos = (event == event_of_interest) & (time <= t0)
    neg = time > t0
    evaluable = pos | neg  # exclude censored with time<=t0

    n_eval = int(evaluable.sum())
    n_pos = int(pos.sum())
    pos_rate = (n_pos / n_eval) if n_eval > 0 else np.nan
    return n_eval, n_pos, pos_rate

def describe_split(df, time_col="time", event_col="event"):
    t = pd.to_numeric(df[time_col], errors="coerce")
    e = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)

    desc = t.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    event_counts = e.value_counts(dropna=False).sort_index().to_dict()
    return desc, event_counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="Dataset base dir, e.g. data/MNB or data/mimic-eye")
    ap.add_argument("--subdir", type=str, required=True, help="Subdir containing train/val/test CSVs, e.g. tabular or '.'")
    ap.add_argument("--time_col", type=str, default="time")
    ap.add_argument("--event_col", type=str, default="event")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--horizons", type=str, default="", help="Comma-separated horizons. If empty, auto-generate from train quantiles.")
    ap.add_argument("--max_auto", type=int, default=6, help="How many auto horizons to print if horizons not given.")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    split_paths = {}
    for s in splits:
        p = os.path.join(args.base_dir, args.subdir, f"{s}.csv") if args.subdir != "." else os.path.join(args.base_dir, f"{s}.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split file: {p}")
        split_paths[s] = p

    dfs = {s: pd.read_csv(p) for s, p in split_paths.items()}

    # Basic stats
    print("\n=== BASIC TIME/EVENT STATS ===")
    for s, df in dfs.items():
        desc, counts = describe_split(df, time_col=args.time_col, event_col=args.event_col)
        print(f"\n[{s}] n={len(df)}")
        print("time:", {k: (float(v) if pd.notna(v) else None) for k, v in desc.items() if k in ["min","10%","25%","50%","75%","90%","max","mean"]})
        print("event_counts:", counts)

    # Candidate horizons
    train_df = dfs[splits[0]]
    train_time = pd.to_numeric(train_df[args.time_col], errors="coerce").to_numpy()
    train_time = train_time[np.isfinite(train_time)]
    train_event = pd.to_numeric(train_df[args.event_col], errors="coerce").fillna(0).astype(int).to_numpy()

    if args.horizons.strip():
        horizons = [float(x) for x in args.horizons.split(",")]
    else:
        # auto horizons from quantiles (good default for unknown units/ranges)
        qs = [0.1, 0.25, 0.5, 0.75, 0.9]
        horizons = [float(np.quantile(train_time, q)) for q in qs]
        horizons = sorted(set(horizons))[:args.max_auto]

    print("\n=== HORIZON EVALUATION (train split) ===")
    print(f"time_col={args.time_col} event_col={args.event_col}")
    print("Horizons:", horizons)

    # Determine which event types exist (>0)
    event_types = sorted([k for k in np.unique(train_event) if k > 0])
    if not event_types:
        event_types = [1]

    for t0 in horizons:
        print(f"\n-- t0 = {t0:.6g} --")
        for k in event_types:
            n_eval, n_pos, pos_rate = horizon_counts(train_time, train_event, t0, event_of_interest=k)
            print(f"event_of_interest={k}: n_eval={n_eval} n_pos={n_pos} pos_rate={pos_rate:.6f}")

if __name__ == "__main__":
    main()
