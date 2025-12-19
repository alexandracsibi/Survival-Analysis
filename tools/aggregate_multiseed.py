import os
import json
import glob
import argparse
import numpy as np

METRIC_KEYS = [
    "test_c_index", "test_td_auc", "test_iauc", "test_ibs", "test_loss",
    "val_c_index", "val_td_auc", "val_iauc", "val_ibs", "val_loss",
    "best_epoch",
]

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def agg(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(arr) == 1:
        return {"mean": float(arr.mean()), "std": 0.0, "n": 1}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "n": int(len(arr))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_name", required=True, help="Base experiment name, e.g. mimiceye_multimodal_deepsurv")
    ap.add_argument("--runs_dir", default="runs", help="Runs directory (default: runs)")
    args = ap.parse_args()

    pattern = os.path.join(args.runs_dir, f"{args.base_name}_seed*", "test_metrics.json")
    test_paths = sorted(glob.glob(pattern))
    if not test_paths:
        raise SystemExit(f"No test_metrics.json found with pattern: {pattern}")

    rows = []
    for tp in test_paths:
        run_dir = os.path.dirname(tp)
        seed_str = os.path.basename(run_dir).split("_seed")[-1]
        seed = int(seed_str)
        test = read_json(tp)

        # also try to read best val metrics (optional)
        val_path = os.path.join(run_dir, "best_val_metrics.json")
        val = read_json(val_path) if os.path.exists(val_path) else {}

        row = {"seed": seed, **val, **test}
        rows.append(row)

    # Aggregate
    summary = {
        "experiment": args.base_name,
        "seeds": [r["seed"] for r in rows],
        "metrics": {},
        "per_seed": rows,
    }

    for k in METRIC_KEYS:
        vals = [r.get(k, float("nan")) for r in rows]
        summary["metrics"][k] = agg(vals)

    out_dir = os.path.join(args.runs_dir, f"{args.base_name}_multiseed")
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, f"{args.base_name}_multiseed_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # also write a simple CSV
    out_csv = os.path.join(out_dir, f"{args.base_name}_multiseed_per_seed.csv")
    cols = ["seed"] + [k for k in METRIC_KEYS if k in summary["metrics"]]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    main()
