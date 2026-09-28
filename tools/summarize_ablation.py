#!/usr/bin/env python3
"""Summarize ablation results into CSV/Markdown records."""

import argparse
import csv
import glob
import json
import os

import numpy as np


def _read_last_epoch_log(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def _safe_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Summarize RUL ablation runs.")
    parser.add_argument("--results_root", type=str, default="results",
                        help="Root folder containing run result directories")
    parser.add_argument("--pattern", type=str, default="rul_RUL_XJTU_P3_ABL_*",
                        help="Glob pattern under results_root")
    parser.add_argument("--output_csv", type=str,
                        default="records/ablation_phase3_20260215.csv")
    parser.add_argument("--output_md", type=str,
                        default="records/ablation_phase3_20260215.md")
    args = parser.parse_args()

    run_dirs = sorted(glob.glob(os.path.join(args.results_root, args.pattern)))
    if not run_dirs:
        raise FileNotFoundError(f"No result directories match {args.pattern}")

    rows = []
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        metrics_path = os.path.join(run_dir, "metrics.npy")
        if not os.path.exists(metrics_path):
            continue

        mse, mae, rmse, score = np.load(metrics_path).tolist()
        deg_path = os.path.join(run_dir, "degradation_metrics.json")
        deg_metrics = {}
        if os.path.exists(deg_path):
            with open(deg_path) as f:
                deg_metrics = json.load(f)

        last_epoch = _read_last_epoch_log(os.path.join(run_dir, "train_epoch_log.csv"))

        row = {
            "run_name": run_name,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "score": score,
            "mae_deg": _safe_float(deg_metrics.get("mae_deg")),
            "mse_deg": _safe_float(deg_metrics.get("mse_deg")),
            "rmse_deg": _safe_float(deg_metrics.get("rmse_deg")),
            "mono_viol": deg_metrics.get("mono_viol"),
            "mono_total_pairs": deg_metrics.get("mono_total_pairs"),
            "tv_ratio": _safe_float(deg_metrics.get("tv_ratio")),
            "slope_mae": _safe_float(deg_metrics.get("slope_mae")),
            "fpt": deg_metrics.get("fpt"),
            "eof": deg_metrics.get("eof"),
            "feature_w_last": _safe_float(last_epoch.get("feature_w")),
            "fcst_w_last": _safe_float(last_epoch.get("fcst_w")),
            "recon_w_last": _safe_float(last_epoch.get("recon_w")),
            "att_w_last": _safe_float(last_epoch.get("att_w")),
            "temperature_last": _safe_float(last_epoch.get("temperature")),
            "phase3_fusion_alpha_last": _safe_float(last_epoch.get("phase3_fusion_alpha")),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No valid runs found with metrics.npy")

    rows.sort(
        key=lambda x: (
            float("inf") if x["mae_deg"] is None else x["mae_deg"],
            x["mae"],
        )
    )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w") as f:
        f.write("# Phase-3 Ablation Summary\n\n")
        f.write(f"Total runs: {len(rows)}\n\n")
        f.write("| run_name | MAE | MSE | MAE_deg | mono_viol | tv_ratio | slope_mae |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['run_name']} | {row['mae']:.6f} | {row['mse']:.6f} | "
                f"{row['mae_deg'] if row['mae_deg'] is not None else 'NA'} | "
                f"{row['mono_viol'] if row['mono_viol'] is not None else 'NA'} | "
                f"{row['tv_ratio'] if row['tv_ratio'] is not None else 'NA'} | "
                f"{row['slope_mae'] if row['slope_mae'] is not None else 'NA'} |\n"
            )

    print(f"Saved CSV summary: {args.output_csv}")
    print(f"Saved Markdown summary: {args.output_md}")


if __name__ == "__main__":
    main()
