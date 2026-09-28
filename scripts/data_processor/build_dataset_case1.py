# scripts/data_processor/build_dataset_case1.py
"""
Build dataset for XJTU-SY Case 1 experiments.

This script:
1. Loads existing PSR/PCA metadata (m, tau, PCA, xlim, ylim) from pca_metadata.pkl if available
2. Otherwise computes global PSR parameters + fits PCA, then saves pca_metadata.pkl
3. Builds train/val/test splits with visual (density images) and physics (1D trajectories) inputs

Usage:
    python scripts/data_processor/build_dataset_case1.py --denoise on --image-bins 224
"""

import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils.data_utils import (
    build_bearing_map,
    get_global_psr_params,
    fit_pca_on_healthy_data,
    build_dataset_split,
    build_dataset_split_parallel
)
from utils.denoising import IENEMDATDConfig


# Processing mode
USE_PARALLEL = True  # Set to False for single-threaded (debugging)
N_JOBS = 30          # -1 = use all CPUs, or set specific number


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw_data", "XJTU-SY")
BASE_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_data", "case1")

# Data splits (XJTU-SY Case 1)
CASE1_TRAIN = ["Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing2_1", "Bearing2_2"]
CASE1_VAL = ["Bearing2_3", "Bearing2_4", "Bearing2_5"]
CASE1_TEST = ["Bearing1_1"]

# CASE1_TRAIN = ["Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing2_1", "Bearing2_2"]
# CASE1_VAL = ["Bearing2_3", "Bearing2_4", "Bearing2_5"]
# CASE1_TEST = ["Bearing1_1"]

# Processing parameters
IMAGE_BINS = 224
PHYSICS_POINTS = 2560
SUPPORTED_IMAGE_BINS = (64, 224)
TIME_CHECKPOINTS = (1.0, 0.75, 0.5, 0.25, 0.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Build XJTU-SY Case 1 PSDI dataset.")
    parser.add_argument(
        "--denoise",
        choices=("on", "off"),
        default="on",
        help="Enable or disable IENEMD-ATD denoising.",
    )
    parser.add_argument(
        "--image-bins",
        type=int,
        choices=SUPPORTED_IMAGE_BINS,
        default=IMAGE_BINS,
        help="Density image resolution.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help="Parallel jobs for split building.",
    )
    parser.add_argument(
        "--single-threaded",
        action="store_true",
        help="Use single-threaded split builder.",
    )
    parser.add_argument(
        "--no-anchor",
        action="store_true",
        help="Ablation: use per-file local PCA + auto canvas instead of global anchor.",
    )
    return parser.parse_args()


def build_output_dir(base_out_dir: str, denoise_enabled: bool, image_bins: int, no_anchor: bool = False) -> str:
    denoise_tag = "denoise_on" if denoise_enabled else "denoise_off"
    bins_tag = f"image_bins_{image_bins}_no_anchor" if no_anchor else f"image_bins_{image_bins}"
    return os.path.join(base_out_dir, denoise_tag, bins_tag)


# =============================================================================
# METADATA LOAD/SAVE
# =============================================================================

def load_pca_metadata(meta_path: str):
    """
    Load precomputed PCA + PSR metadata.
    Returns tuple: (pca, xlim, ylim, m, tau, image_bins, physics_points) or None if missing.
    """
    if not os.path.exists(meta_path):
        return None

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    required = ["pca", "xlim", "ylim", "m", "tau", "image_bins", "physics_points"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise KeyError(f"pca_metadata.pkl missing keys: {missing}")

    print(f">>> Loaded existing PCA metadata from: {meta_path}")
    print(
        f"    m={meta['m']}, tau={meta['tau']}, "
        f"image_bins={meta['image_bins']}, physics_points={meta['physics_points']}"
    )

    return (
        meta["pca"],
        meta["xlim"],
        meta["ylim"],
        meta["m"],
        meta["tau"],
        meta["image_bins"],
        meta["physics_points"],
        meta.get("denoise_enabled"),
    )


def save_pca_metadata(
    meta_path: str,
    pca,
    xlim,
    ylim,
    m: int,
    tau: int,
    image_bins: int,
    physics_points: int,
    denoise_enabled: bool,
):
    """Save PCA + PSR metadata for reuse."""
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "pca": pca,
                "xlim": xlim,
                "ylim": ylim,
                "m": m,
                "tau": tau,
                "image_bins": image_bins,
                "physics_points": physics_points,
                "denoise_enabled": denoise_enabled,
            },
            f,
        )
    print(f">>> Metadata saved to: {meta_path}")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_dataset(npz_path: str, out_dir: str) -> None:
    """Verify and visualize a sample from the dataset."""
    if not os.path.exists(npz_path):
        print(f"File not found: {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    P = data["P"]
    meta = data["meta"]

    print(f"\nVerifying {os.path.basename(npz_path)}")
    print(f"  X (Visual): {X.shape}")
    print(f"  P (Physics): {P.shape}")
    print(f"  Samples: {len(meta)}")

    # Visualize random sample
    idx = np.random.randint(0, len(X))
    sample_b = meta[idx][0]
    sample_t = meta[idx][1]

    _, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(X[idx, 0], cmap="inferno", origin="lower")
    axes[0].set_title(f"Visual: {sample_b} (t={sample_t})")
    axes[0].axis("off")

    # Plot the first physics channel (e.g., PC1 trajectory)
    axes[1].plot(P[idx, 0])
    axes[1].set_title("Physics (PC1 trajectory)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(out_dir, f"verify_{os.path.basename(npz_path).replace('.npz', '.png')}")
    plt.savefig(out_png)
    plt.close()
    print(f"  Verification plot saved: {out_png}")


def visualize_psdi_timeline(npz_path: str, out_dir: str, checkpoints=TIME_CHECKPOINTS) -> None:
    """
    Visualize PSDI evolution at normalized RUL checkpoints for each bearing.

    Convention:
        t=1.0 -> start of life (new bearing)
        t=0.0 -> end of life (EOL / damaged)
    The script picks the closest sample to each checkpoint in [0, 1].
    """
    if not os.path.exists(npz_path):
        print(f"File not found for timeline visualization: {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    meta = data["meta"]

    if len(X) == 0 or len(meta) == 0:
        print(f"No data in {npz_path} for timeline visualization.")
        return

    by_bearing = {}
    for idx, row in enumerate(meta):
        b = str(row[0])
        t_idx = int(row[1])
        total = int(row[2])
        if total <= 1:
            rul_norm = 1.0
            rul_steps = 0
        else:
            progress = t_idx / float(total - 1)
            rul_norm = 1.0 - progress
            rul_steps = (total - 1) - t_idx
        by_bearing.setdefault(b, []).append((idx, t_idx, total, rul_norm, rul_steps))

    bearings = sorted(by_bearing.keys())
    n_rows = len(bearings)
    n_cols = len(checkpoints)
    if n_rows == 0:
        print(f"No valid bearing metadata found in {npz_path}.")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.4 * n_rows))
    axes = np.atleast_2d(axes)

    for r, bearing_name in enumerate(bearings):
        entries = by_bearing[bearing_name]
        for c, target in enumerate(checkpoints):
            best = min(entries, key=lambda e: abs(e[3] - target))
            sample_idx, t_idx, total, rul_norm, rul_steps = best
            axes[r, c].imshow(X[sample_idx, 0], cmap="inferno", origin="lower")
            axes[r, c].set_title(
                f"t={target:.2f}\nRUL={rul_steps}/{max(total - 1, 1)} | {100.0 * rul_norm:.1f}%",
                fontsize=9
            )
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(bearing_name, fontsize=9)

    split_name = os.path.basename(npz_path).replace(".npz", "")
    plt.tight_layout()
    timeline_dir = os.path.join(out_dir, "timeline_viz")
    os.makedirs(timeline_dir, exist_ok=True)
    out_png = os.path.join(timeline_dir, f"psdi_timeline_{split_name}.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Timeline visualization saved: {out_png}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    denoise_enabled = args.denoise == "on"
    image_bins = args.image_bins
    use_parallel = USE_PARALLEL and (not args.single_threaded)
    n_jobs = args.n_jobs
    no_anchor = args.no_anchor
    out_dir = build_output_dir(BASE_OUT_DIR, denoise_enabled, image_bins, no_anchor)

    print("=" * 60)
    print("Building XJTU-SY Case 1 Dataset")
    print("=" * 60)
    print(f"Denoising: {'ON' if denoise_enabled else 'OFF'}")
    print(f"Image bins: {image_bins}")
    print(f"Anchoring: {'OFF (ablation: local PCA + auto canvas)' if no_anchor else 'ON (global PCA + fixed canvas)'}")
    print(f"Parallel: {'ON' if use_parallel else 'OFF'}")
    if use_parallel:
        print(f"n_jobs: {n_jobs}")

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Build bearing map (XJTU-SY raw data has 3 condition subfolders; only Case-1
    # bearings, i.e. Bearing1_x / Bearing2_x, are actually used below)
    bearing_map = build_bearing_map(RAW_DATA_DIR)
    print(f"Found {len(bearing_map)} bearings")

    # Denoising config
    den_cfg = IENEMDATDConfig(J=5, enabled=denoise_enabled)

    # -------------------------------------------------------------------------
    # Step 1 & 2: Load existing metadata OR compute and save
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 1-2: Loading PCA metadata (or computing if missing)")
    print("-" * 40)

    meta_path = os.path.join(out_dir, "pca_metadata.pkl")
    loaded = load_pca_metadata(meta_path)

    if loaded is not None:
        pca, xlim, ylim, m, tau, image_bins_meta, physics_points_meta, denoise_meta = loaded

        # Safety checks to avoid silent mismatches
        if image_bins_meta != image_bins:
            raise ValueError(
                f"IMAGE_BINS mismatch: script={image_bins} vs metadata={image_bins_meta}. "
                f"Fix script or regenerate metadata."
            )
        if physics_points_meta != PHYSICS_POINTS:
            raise ValueError(
                f"PHYSICS_POINTS mismatch: script={PHYSICS_POINTS} vs metadata={physics_points_meta}. "
                f"Fix script or regenerate metadata."
            )
        if denoise_meta is not None and denoise_meta != denoise_enabled:
            raise ValueError(
                f"Denoising mismatch: script={denoise_enabled} vs metadata={denoise_meta}. "
                f"Fix script args or regenerate metadata."
            )

        print(f">>> Reusing metadata: m={m}, tau={tau}")

    else:
        print(">>> No existing pca_metadata.pkl found. Computing PSR parameters + PCA...")

        print("\n" + "-" * 40)
        print("Computing PSR parameters (m, tau)")
        print("-" * 40)
        m, tau = get_global_psr_params(
            bearing_map, CASE1_TRAIN, n_samples=6, den_cfg=den_cfg
        )
        print(f"\n>>> Global parameters: m={m}, tau={tau}")

        print("\n" + "-" * 40)
        print("Fitting PCA on healthy data")
        print("-" * 40)
        pca, xlim, ylim = fit_pca_on_healthy_data(
            bearing_map, CASE1_TRAIN, m, tau, den_cfg=den_cfg
        )

        save_pca_metadata(
            meta_path=meta_path,
            pca=pca,
            xlim=xlim,
            ylim=ylim,
            m=m,
            tau=tau,
            image_bins=image_bins,
            physics_points=PHYSICS_POINTS,
            denoise_enabled=denoise_enabled,
        )

    # -------------------------------------------------------------------------
    # Step 3: Build dataset splits
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 3: Building dataset splits")
    print("-" * 40)

    if use_parallel:
        print(f"Mode: PARALLEL (n_jobs={n_jobs})")
        build_func = lambda b, s: build_dataset_split_parallel(
            b, s, bearing_map, pca, xlim, ylim, m, tau, out_dir,
            image_bins, PHYSICS_POINTS, den_cfg, n_jobs=n_jobs, no_anchor=no_anchor
        )
    else:
        print("Mode: SINGLE-THREADED")
        build_func = lambda b, s: build_dataset_split(
            b, s, bearing_map, pca, xlim, ylim, m, tau, out_dir,
            image_bins, PHYSICS_POINTS, den_cfg, no_anchor=no_anchor
        )

    build_func(CASE1_TRAIN, "train")
    build_func(CASE1_VAL, "val")
    build_func(CASE1_TEST, "test")

    # -------------------------------------------------------------------------
    # Step 4: Verify
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Step 4: Verification")
    print("-" * 40)
    train_npz = os.path.join(out_dir, "train.npz")
    val_npz = os.path.join(out_dir, "val.npz")
    test_npz = os.path.join(out_dir, "test.npz")

    visualize_psdi_timeline(train_npz, out_dir)
    visualize_psdi_timeline(val_npz, out_dir)
    visualize_psdi_timeline(test_npz, out_dir)

    print("\n" + "=" * 60)
    print("Dataset building complete!")
    print(f"Output: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
