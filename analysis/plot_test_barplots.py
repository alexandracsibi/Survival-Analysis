#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ["test_c_index", "test_ibs"]
LOWER_IS_BETTER = {"test_ibs"}


# ----------------------------
# Data loading
# ----------------------------
def load_from_multiseed_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    return df


def load_from_seed_json(seed_dir: Path) -> Dict[str, float]:
    p = seed_dir / "test_metrics.json"
    if not p.exists():
        return {}
    with open(p) as f:
        d = json.load(f)
    return d


def collect_test_metrics(run_dir: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      seed, test_c_index, test_ibs, ...
    """
    # Preferred: multiseed CSV
    ms = list((run_dir / "multiseed").glob("*_multiseed_per_seed.csv"))
    if ms:
        return load_from_multiseed_csv(ms[0])

    # Fallback: per-seed JSONs
    rows = []
    for seed_dir in sorted(run_dir.glob("seed*")):
        d = load_from_seed_json(seed_dir)
        if d:
            rows.append(d)
    return pd.DataFrame(rows)


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
            "n": len(vals),
        }
    return out


# ----------------------------
# Plotting
# ----------------------------
def plot_dataset_comparison(
    dataset: str,
    agg: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Path,
):
    """
    agg[condition][metric] -> {mean, std, n}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = list(agg.keys())

    for metric in METRICS:
        means, stds, labels = [], [], []

        for cond in conditions:
            if metric not in agg[cond]:
                continue
            means.append(agg[cond][metric]["mean"])
            stds.append(agg[cond][metric]["std"])
            labels.append(cond)

        if not means:
            continue

        # --- pretty bar plot styling ---
        LABEL_MAP = {
            "supervised_100": "Supervised",
            "partial_label": "Partial labels",
            "ssl_pseudolabel": "SSL",
            "graphssl": "Graph-SSL",
        }

        COLOR_MAP = {
            "Supervised": "#7f7f7f",       # neutral gray
            "Partial labels": "#f0ad4e",   # muted orange
            "SSL": "#4c72b0",              # muted blue
            "Graph-SSL": "#55a868",        # muted green
        }

        labels = [LABEL_MAP.get(l, l) for l in labels]
        colors = [COLOR_MAP[l] for l in labels]

        x = np.arange(len(labels))
        width = 0.55  # thinner bars

        plt.figure(figsize=(6, 4))

        plt.bar(
            x,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            error_kw=dict(lw=1.2),
        )

        # y-axis scaling (do NOT start from 0 for C-index)
        ymin = min(means) - 2 * max(stds) - 0.01
        ymax = max(means) + 2 * max(stds) + 0.01
        plt.ylim(ymin, ymax)

        plt.xticks(x, labels, rotation=0)
        plt.ylabel(metric.replace("_", " "))
        plt.title(f"{dataset.replace('_', ' ').title()} — {metric.replace('_', ' ')}")

        # grid: horizontal only
        plt.grid(axis="y", alpha=0.25, linewidth=0.8)
        plt.gca().set_axisbelow(True)

        # remove top/right spines
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # invert IBS axis (lower is better)
        if metric == "test_ibs":
            plt.gca().invert_yaxis()

        plt.tight_layout()
        out_path = out_dir / f"{dataset}__{metric}__bar.png"
        plt.savefig(out_path, dpi=200)
        plt.close()



def plot_multimodal_vs_tabular(
    tab_agg: Dict[str, float],
    mm_agg: Dict[str, float],
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        if metric not in tab_agg or metric not in mm_agg:
            continue

        means = [tab_agg[metric]["mean"], mm_agg[metric]["mean"]]
        stds = [tab_agg[metric]["std"], mm_agg[metric]["std"]]

        labels = ["Tabular", "Multimodal"]
        colors = ["#7f7f7f", "#4c72b0"]  # gray vs blue

        x = np.arange(len(labels))
        width = 0.55

        plt.figure(figsize=(5, 4))

        plt.bar(
            x,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            error_kw=dict(lw=1.2),
        )

        # y-axis scaling
        ymin = min(means) - 2 * max(stds) - 0.01
        ymax = max(means) + 2 * max(stds) + 0.01
        plt.ylim(ymin, ymax)

        plt.xticks(x, labels)
        plt.ylabel(metric.replace("_", " "))
        plt.title(f"MIMIC-Eye Supervised — {metric.replace('_', ' ')}")

        # grid: horizontal only
        plt.grid(axis="y", alpha=0.25, linewidth=0.8)
        plt.gca().set_axisbelow(True)

        # remove top/right spines
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # invert IBS axis
        if metric == "test_ibs":
            plt.gca().invert_yaxis()

        plt.tight_layout()

        out_path = out_dir / f"mimiceye_tabular_vs_multimodal__{metric}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="runs/")
    ap.add_argument("--out", required=True, help="analysis_out/bars")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    # per-dataset bar plots
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        agg_by_cond = {}

        for cond_dir in dataset_dir.iterdir():
            if not cond_dir.is_dir():
                continue

            df = collect_test_metrics(cond_dir)
            if df.empty:
                continue

            agg_by_cond[cond_dir.name] = aggregate(df)

        if agg_by_cond:
            plot_dataset_comparison(dataset, agg_by_cond, out)

    # multimodal vs tabular (supervised only)
    try:
        tab = aggregate(
            collect_test_metrics(root / "mimiceye_tabular" / "supervised_100")
        )
        mm = aggregate(
            collect_test_metrics(root / "mimiceye_multimodal" / "supervised_100")
        )
        plot_multimodal_vs_tabular(tab, mm, out)
    except Exception:
        pass


if __name__ == "__main__":
    main()
