#!/usr/bin/env python3
"""Evaluate degradation-phase behavior for one RUL result directory."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rul import get_fpt_eof


def _infer_bearing_and_time(result_dir, test_npz):
    bearing_path = os.path.join(result_dir, "test_bearing_ids.npy")
    timestep_path = os.path.join(result_dir, "test_timesteps.npy")

    bearing_ids = None
    timesteps = None

    if os.path.exists(bearing_path):
        bearing_ids = np.load(bearing_path, allow_pickle=True).tolist()
    if os.path.exists(timestep_path):
        timesteps = np.load(timestep_path)

    if (bearing_ids is None or timesteps is None) and test_npz and os.path.exists(test_npz):
        data = np.load(test_npz, allow_pickle=True)
        meta = data["meta"]
        if bearing_ids is None:
            bearing_ids = [str(m[0]) for m in meta]
        if timesteps is None:
            timesteps = np.array([int(m[1]) for m in meta], dtype=np.int32)

    if bearing_ids is None:
        bearing_ids = []
    if timesteps is None:
        timesteps = np.arange(len(bearing_ids), dtype=np.int32)

    return bearing_ids, timesteps


def _compute_metrics(pred, true, timesteps, fpt, eof):
    timesteps = np.asarray(timesteps)
    pred = np.asarray(pred)
    true = np.asarray(true)

    deg_mask = timesteps > fpt
    pred_deg = pred[deg_mask]
    true_deg = true[deg_mask]
    time_deg = timesteps[deg_mask]

    if len(pred_deg) == 0:
        return {
            "fpt": int(fpt),
            "eof": int(eof),
            "num_degradation_samples": 0,
            "mae_deg": None,
            "mse_deg": None,
            "rmse_deg": None,
            "mono_viol": 0,
            "mono_total_pairs": 0,
            "tv_pred": 0.0,
            "tv_true": 0.0,
            "tv_ratio": None,
            "slope_mae": None,
        }, time_deg, pred_deg, true_deg

    order = np.argsort(time_deg)
    time_deg = time_deg[order]
    pred_deg = pred_deg[order]
    true_deg = true_deg[order]

    mae_deg = float(np.mean(np.abs(pred_deg - true_deg)))
    mse_deg = float(np.mean((pred_deg - true_deg) ** 2))
    rmse_deg = float(np.sqrt(mse_deg))

    if len(pred_deg) > 1:
        diff_pred = np.diff(pred_deg)
        diff_true = np.diff(true_deg)
        dt = np.maximum(np.diff(time_deg), 1)
        mono_viol = int(np.sum(diff_pred > 0))
        tv_pred = float(np.sum(np.abs(diff_pred)))
        tv_true = float(np.sum(np.abs(diff_true)))
        tv_ratio = float(tv_pred / tv_true) if tv_true > 1e-12 else None
        slope_mae = float(np.mean(np.abs((diff_pred / dt) - (diff_true / dt))))
    else:
        mono_viol = 0
        tv_pred = 0.0
        tv_true = 0.0
        tv_ratio = None
        slope_mae = None

    metrics = {
        "fpt": int(fpt),
        "eof": int(eof),
        "num_degradation_samples": int(len(pred_deg)),
        "mae_deg": mae_deg,
        "mse_deg": mse_deg,
        "rmse_deg": rmse_deg,
        "mono_viol": mono_viol,
        "mono_total_pairs": max(0, len(pred_deg) - 1),
        "tv_pred": tv_pred,
        "tv_true": tv_true,
        "tv_ratio": tv_ratio,
        "slope_mae": slope_mae,
    }
    return metrics, time_deg, pred_deg, true_deg


def _save_zoom_plot(result_dir, time_deg, pred_deg, true_deg, fpt, eof):
    if len(time_deg) == 0:
        return None

    plt.figure(figsize=(10, 4))
    plt.plot(time_deg, true_deg, label="Ground Truth", color="blue", linewidth=2.0)
    plt.plot(time_deg, pred_deg, label="Prediction", color="red", linestyle="--", linewidth=1.5)
    plt.axvline(fpt, color="gray", linestyle=":", linewidth=1.0, label=f"FPT={fpt}")
    plt.axvline(eof, color="black", linestyle=":", linewidth=1.0, label=f"EOF={eof}")
    plt.title("Degradation Region (t > FPT)")
    plt.xlabel("Timestep")
    plt.ylabel("RUL (normalized)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(result_dir, "degradation_zoom.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate degradation-phase metrics for RUL run.")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to one results/<setting> directory")
    parser.add_argument("--test_npz", type=str, default="",
                        help="Optional test.npz path for fallback metadata")
    parser.add_argument("--dataset_name", type=str, default="xjtu",
                        help="Dataset key used in utils.rul.get_fpt_eof")
    parser.add_argument("--bearing_id", type=str, default="",
                        help="Optional explicit bearing ID")
    parser.add_argument("--output_json", type=str, default="",
                        help="Optional custom output json path")
    args = parser.parse_args()

    result_dir = args.result_dir
    pred_path = os.path.join(result_dir, "pred.npy")
    true_path = os.path.join(result_dir, "true.npy")
    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        raise FileNotFoundError(f"Missing pred/true files in {result_dir}")

    pred = np.load(pred_path)
    true = np.load(true_path)
    bearing_ids, timesteps = _infer_bearing_and_time(result_dir, args.test_npz)

    if len(pred) != len(true):
        raise ValueError(f"pred length {len(pred)} != true length {len(true)}")
    if len(timesteps) != len(pred):
        timesteps = np.arange(len(pred), dtype=np.int32)

    bearing_id = args.bearing_id
    if not bearing_id:
        unique_ids = sorted(set(bearing_ids))
        if len(unique_ids) == 1:
            bearing_id = unique_ids[0]
        else:
            raise ValueError(
                "Could not infer a single bearing_id. Provide --bearing_id explicitly."
            )

    fpt_eof_table = get_fpt_eof(args.dataset_name)
    if bearing_id not in fpt_eof_table:
        raise ValueError(f"bearing_id={bearing_id} not found in dataset={args.dataset_name}")
    fpt, eof = fpt_eof_table[bearing_id]

    metrics, time_deg, pred_deg, true_deg = _compute_metrics(pred, true, timesteps, fpt, eof)
    metrics["bearing_id"] = bearing_id
    metrics["dataset_name"] = args.dataset_name

    output_json = args.output_json or os.path.join(result_dir, "degradation_metrics.json")
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2)

    zoom_path = _save_zoom_plot(result_dir, time_deg, pred_deg, true_deg, fpt, eof)

    print("Degradation metrics")
    print(json.dumps(metrics, indent=2))
    print(f"Saved json: {output_json}")
    if zoom_path:
        print(f"Saved plot: {zoom_path}")


if __name__ == "__main__":
    main()
