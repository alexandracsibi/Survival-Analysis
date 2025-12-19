    #!/usr/bin/env python3
import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SEED_RE = re.compile(r"^seed(\d+)$", re.IGNORECASE)


# -----------------------------
# Utilities
# -----------------------------

def safe_read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_first_metrics_csv(seed_dir: Path) -> Optional[Path]:
    # Your per-epoch files look like: *_metrics.csv
    for p in seed_dir.glob("*_metrics.csv"):
        return p
    return None


def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def metric_key_candidates(prefix: str) -> List[str]:
    # helper for schema variability
    # e.g. "test_c_index" or "c_index" etc.
    return [
        prefix,
        prefix.replace("val_", "").replace("test_", ""),
        prefix.replace("val_", "valid_").replace("test_", "test_"),
    ]


def extract_metric(d: dict, key: str) -> Optional[float]:
    if d is None:
        return None
    if key in d:
        try:
            return float(d[key])
        except Exception:
            return None
    # attempt fallback keys
    for cand in metric_key_candidates(key):
        if cand in d:
            try:
                return float(d[cand])
            except Exception:
                pass
    return None


@dataclass
class SeedRun:
    dataset: str
    condition: str
    seed: int
    seed_dir: Path
    metrics_csv: Optional[Path]
    best_val_json: Optional[Path]
    test_json: Optional[Path]

def audit_discovery(runs: List[SeedRun], verbose: bool, strict: bool, max_missing_print: int) -> None:
    if not verbose and not strict:
        return

    df = pd.DataFrame(
        [{
            "dataset": r.dataset,
            "condition": r.condition,
            "seed": r.seed,
            "has_metrics_csv": r.metrics_csv is not None,
            "has_best_val_json": r.best_val_json is not None,
            "has_test_json": r.test_json is not None,
            "seed_dir": str(r.seed_dir),
        } for r in runs]
    )

    print(f"[DISCOVERY] Found {len(df)} seed-runs across "
          f"{df['dataset'].nunique()} datasets × {df['condition'].nunique()} conditions.")

    grp = df.groupby(["dataset", "condition"], dropna=False)
    summary = grp.agg(
        n_seeds=("seed", "nunique"),
        n_runs=("seed", "count"),
        metrics_csv=("has_metrics_csv", "sum"),
        best_val_json=("has_best_val_json", "sum"),
        test_json=("has_test_json", "sum"),
    ).reset_index()

    print("[DISCOVERY] Coverage by dataset/condition:")
    print(summary.to_string(index=False))

    missing = df[
        (~df["has_best_val_json"]) | (~df["has_test_json"])
    ].copy()

    # metrics.csv is optional-ish (only needed for learning curves), so treat separately:
    missing_curves = df[~df["has_metrics_csv"]].copy()

    if not missing.empty:
        print(f"[WARN] Missing best_val_metrics.json and/or test_metrics.json in {len(missing)} runs.")
        to_show = missing.head(max_missing_print)
        for _, row in to_show.iterrows():
            print(f"  - {row['dataset']}/{row['condition']}/seed{row['seed']}: "
                  f"best_val={row['has_best_val_json']} test={row['has_test_json']} "
                  f"dir={row['seed_dir']}")
        if len(missing) > max_missing_print:
            print(f"  ... (and {len(missing)-max_missing_print} more)")

        if strict:
            raise RuntimeError("Strict mode: missing required JSON metrics files.")

    if verbose and not missing_curves.empty:
        print(f"[INFO] Missing *_metrics.csv (learning curves will be skipped) in {len(missing_curves)} runs.")
        to_show = missing_curves.head(max_missing_print)
        for _, row in to_show.iterrows():
            print(f"  - {row['dataset']}/{row['condition']}/seed{row['seed']} dir={row['seed_dir']}")
        if len(missing_curves) > max_missing_print:
            print(f"  ... (and {len(missing_curves)-max_missing_print} more)")


# -----------------------------
# Discovery
# -----------------------------
def discover_runs(root: Path) -> List[SeedRun]:
    runs: List[SeedRun] = []

    # expected: root/<dataset>/<condition>/seedXX
    for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        dataset = dataset_dir.name

        for cond_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            condition = cond_dir.name

            # seed dirs
            seed_dirs = [p for p in cond_dir.iterdir() if p.is_dir() and SEED_RE.match(p.name)]
            for sd in sorted(seed_dirs):
                m = SEED_RE.match(sd.name)
                if not m:
                    continue
                seed = int(m.group(1))

                metrics_csv = find_first_metrics_csv(sd)
                best_val_json = sd / "best_val_metrics.json"
                test_json = sd / "test_metrics.json"

                runs.append(
                    SeedRun(
                        dataset=dataset,
                        condition=condition,
                        seed=seed,
                        seed_dir=sd,
                        metrics_csv=metrics_csv if metrics_csv and metrics_csv.exists() else None,
                        best_val_json=best_val_json if best_val_json.exists() else None,
                        test_json=test_json if test_json.exists() else None,
                    )
                )

    return runs


def discover_multiseed_per_seed_csvs(root: Path) -> Dict[Tuple[str, str], Path]:
    """
    Returns mapping: (dataset, condition) -> multiseed_per_seed.csv path
    Expected: root/<dataset>/<condition>/multiseed/*_per_seed.csv
    """
    out: Dict[Tuple[str, str], Path] = {}
    for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        dataset = dataset_dir.name
        for cond_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
            condition = cond_dir.name
            ms = cond_dir / "multiseed"
            if not ms.exists() or not ms.is_dir():
                continue
            # find a *per_seed.csv
            candidates = list(ms.glob("*_per_seed.csv"))
            if candidates:
                out[(dataset, condition)] = candidates[0]
    return out


# -----------------------------
# Parsing + Tables
# -----------------------------
def build_per_seed_table(runs: List[SeedRun]) -> pd.DataFrame:
    rows = []
    for r in runs:
        best_val = safe_read_json(r.best_val_json) if r.best_val_json else None
        test = safe_read_json(r.test_json) if r.test_json else None

        row = {
            "dataset": r.dataset,
            "condition": r.condition,
            "seed": r.seed,
            "seed_dir": str(r.seed_dir),
        }

        # common metrics you log in JSONs
        for k in ["loss", "c_index", "td_auc", "iauc", "ibs", "best_epoch"]:
            row[f"val_{k}"] = extract_metric(best_val, k if k.startswith("val_") else f"{k}")
            row[f"test_{k}"] = extract_metric(test, k if k.startswith("test_") else f"{k}")

        # Some JSONs include horizon metadata; keep if present
        for meta_k in ["horizon", "event_of_interest", "iauc_grid_n"]:
            row[f"val_{meta_k}"] = extract_metric(best_val, meta_k)
            row[f"test_{meta_k}"] = extract_metric(test, meta_k)

        rows.append(row)

    df = pd.DataFrame(rows)
    # normalize columns: some may be all None
    return df


def aggregate_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_seed.columns if c.startswith("test_") or c.startswith("val_")]
    metric_cols = [c for c in metric_cols if c not in ("val_seed", "test_seed")]

    grp = per_seed.groupby(["dataset", "condition"], dropna=False)
    out_rows = []
    for (ds, cond), g in grp:
        row = {"dataset": ds, "condition": cond, "n_seeds": int(g["seed"].nunique())}
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean()) if np.isfinite(vals.mean()) else np.nan
            row[f"{c}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() >= 2 else np.nan
        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values(["dataset", "condition"])


def merge_with_multiseed_if_available(
    agg_df: pd.DataFrame, multiseed_map: Dict[Tuple[str, str], Path]
) -> pd.DataFrame:
    """
    If multiseed per_seed CSV exists, we can trust its values for n/seeds,
    but your per-seed JSON parsing is already consistent. This function is
    mainly to add a pointer to the multiseed file for traceability.
    """
    agg_df = agg_df.copy()
    agg_df["multiseed_per_seed_csv"] = ""
    for i, row in agg_df.iterrows():
        key = (row["dataset"], row["condition"])
        if key in multiseed_map:
            agg_df.at[i, "multiseed_per_seed_csv"] = str(multiseed_map[key])
    return agg_df


# -----------------------------
# Plots
# -----------------------------
TEST_METRICS = ["test_c_index", "test_td_auc", "test_iauc", "test_ibs", "test_loss"]
VAL_METRICS = ["val_c_index", "val_td_auc", "val_iauc", "val_ibs", "val_loss"]


def plot_condition_bars_for_dataset(agg_df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    ds = agg_df[agg_df["dataset"] == dataset].copy()
    if ds.empty:
        return

    conditions = list(ds["condition"].unique())
    conditions.sort()

    # bar plots per metric
    for m in TEST_METRICS:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col not in ds.columns:
            continue

        means = [float(ds.loc[ds["condition"] == c, mean_col].values[0]) for c in conditions]
        stds = []
        for c in conditions:
            v = ds.loc[ds["condition"] == c, std_col].values
            stds.append(float(v[0]) if len(v) else np.nan)

        x = np.arange(len(conditions))

        plt.figure(figsize=(10, 4))
        plt.title(f"{dataset}: {m} (mean ± std across seeds)")
        plt.bar(x, means)
        # error bars only where std is finite
        yerr = [s if np.isfinite(s) else 0.0 for s in stds]
        plt.errorbar(x, means, yerr=yerr, fmt="none", capsize=4)

        plt.xticks(x, conditions, rotation=20, ha="right")
        plt.ylabel(m)
        plt.tight_layout()
        plt.savefig(out_dir / f"{dataset}__bars__{m}.png", dpi=200)
        plt.close()


def plot_learning_curves(per_seed_df: pd.DataFrame, runs: List[SeedRun], out_dir: Path) -> None:
    # index SeedRun by (dataset, condition, seed)
    run_map: Dict[Tuple[str, str, int], SeedRun] = {(r.dataset, r.condition, r.seed): r for r in runs}

    # group by dataset/condition
    for (ds, cond), g in per_seed_df.groupby(["dataset", "condition"], dropna=False):
        # load epoch-wise metrics.csv for each seed (if exists)
        series = []
        for _, row in g.iterrows():
            seed = int(row["seed"])
            r = run_map.get((ds, cond, seed))
            if r is None or r.metrics_csv is None:
                continue
            dfm = safe_read_csv(r.metrics_csv)
            if dfm is None or dfm.empty:
                continue
            dfm = dfm.copy()
            dfm["seed"] = seed
            series.append(dfm)

        if not series:
            continue

        df_all = pd.concat(series, ignore_index=True)

        # plot core curves: val_c_index, val_td_auc, val_loss + train_loss
        plot_cols = []
        for c in ["train_loss", "val_loss", "val_c_index", "val_td_auc", "val_iauc", "val_ibs"]:
            if c in df_all.columns:
                plot_cols.append(c)

        for c in plot_cols:
            plt.figure(figsize=(10, 4))
            plt.title(f"{ds}/{cond}: {c} over epochs")

            # per-seed lines (thin)
            for seed, dseed in df_all.groupby("seed"):
                if "epoch" not in dseed.columns:
                    continue
                plt.plot(dseed["epoch"].values, dseed[c].values, alpha=0.35)

            # mean line if multiple seeds
            if df_all["seed"].nunique() >= 2 and "epoch" in df_all.columns:
                grp = df_all.groupby("epoch")[c]
                mean = grp.mean()
                std = grp.std(ddof=1)
                plt.plot(mean.index.values, mean.values, linewidth=2.0)
                # std band (only where finite)
                lo = (mean - std).values
                hi = (mean + std).values
                plt.fill_between(mean.index.values, lo, hi, alpha=0.15)

            plt.xlabel("epoch")
            plt.ylabel(c)
            plt.tight_layout()
            safe_name = f"{ds}__{cond}__curve__{c}.png".replace("/", "_")
            plt.savefig(out_dir / safe_name, dpi=200)
            plt.close()


def plot_overall_scatter(agg_df: pd.DataFrame, out_dir: Path) -> None:
    """
    A compact “tradeoff” plot: test_c_index vs test_ibs (lower is better).
    """
    if "test_c_index_mean" not in agg_df.columns or "test_ibs_mean" not in agg_df.columns:
        return

    plt.figure(figsize=(9, 6))
    plt.title("Overall: test_c_index vs test_ibs (means)")

    for _, row in agg_df.iterrows():
        x = row["test_ibs_mean"]
        y = row["test_c_index_mean"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        label = f'{row["dataset"]}/{row["condition"]}'
        plt.scatter([x], [y])
        plt.text(x, y, label, fontsize=8)

    plt.xlabel("test_ibs_mean (lower better)")
    plt.ylabel("test_c_index_mean (higher better)")
    plt.tight_layout()
    plt.savefig(out_dir / "overall__scatter__cindex_vs_ibs.png", dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root folder, e.g. runs/")
    ap.add_argument("--out", type=str, required=True, help="Output folder for plots/tables")
    ap.add_argument("--verbose", action="store_true", help="Print file discovery + schema checks")
    ap.add_argument("--strict", action="store_true", help="Fail if any expected file is missing/unreadable")
    ap.add_argument("--max_missing_print", type=int, default=30, help="Cap missing-file printouts")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    mkdir(out_dir)

    runs = discover_runs(root)
    if not runs:
        print(f"[ERR] No seed runs found under: {root}")
        return

    audit_discovery(runs, verbose=args.verbose, strict=args.strict, max_missing_print=args.max_missing_print)    

    multiseed_map = discover_multiseed_per_seed_csvs(root)

    per_seed = build_per_seed_table(runs)
    agg = aggregate_table(per_seed)
    agg = merge_with_multiseed_if_available(agg, multiseed_map)

    # Save tables
    per_seed.to_csv(out_dir / "per_seed_summary.csv", index=False)
    agg.to_csv(out_dir / "agg_summary_mean_std.csv", index=False)

    # Pretty comparison tables (pivot-like) for quick thesis copy/paste
    # One file per metric
    for m in TEST_METRICS:
        col = f"{m}_mean"
        if col not in agg.columns:
            continue
        pivot = agg.pivot_table(index="dataset", columns="condition", values=col, aggfunc="first")
        pivot.to_csv(out_dir / f"pivot__{m}_mean.csv")

    # Plots: per dataset bars
    for dataset in sorted(agg["dataset"].unique()):
        plot_condition_bars_for_dataset(agg, dataset, out_dir)

    # Plots: learning curves per dataset/condition (from per-epoch metrics.csv)
    plot_learning_curves(per_seed, runs, out_dir)

    # Plot: overall scatter
    plot_overall_scatter(agg, out_dir)

    print(f"[OK] Wrote tables + plots to: {out_dir}")
    print(f"     - {out_dir / 'per_seed_summary.csv'}")
    print(f"     - {out_dir / 'agg_summary_mean_std.csv'}")


if __name__ == "__main__":
    main()

    
