"""
Compare raw vibration signal vs. PSDI without GAR (per-file local PCA + auto
canvas) vs. PSDI with GAR (global anchor PCA + fixed canvas), across several
real timesteps of ONE bearing lifecycle. All three rows come from the same
real CSV files — nothing synthesized.

Row 1: raw vibration waveform (horizontal channel), straight from the CSV.
Row 2: PSDI computed on-the-fly with no_anchor=True (ablation baseline).
Row 3: PSDI with GAR, taken directly from the already-built processed .npz
       (this is exactly what the teacher/student models are trained on).

Usage:
    python tools/viz_gar_ablation_compare.py \
        --bearing_id Bearing1_1 \
        --npz data/processed_data/case1/denoise_off/image_bins_224/test.npz \
        --raw_dir data/raw_data/XJTU-SY/35Hz12kN/Bearing1_1
"""
import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rul import piecewise_rul, get_fpt_eof
from utils.data_utils import process_single_file
from utils.denoising import IENEMDATDConfig

CHECKPOINTS = [1.0, 0.75, 0.5, 0.25, 0.0]


def parse_args():
    p = argparse.ArgumentParser(description="Raw vs PSDI-no-GAR vs PSDI-with-GAR comparison grid")
    p.add_argument("--npz", type=str,
                    default="data/processed_data/case1/denoise_off/image_bins_224/test.npz")
    p.add_argument("--pca_metadata", type=str,
                    default="data/processed_data/case1/denoise_off/image_bins_224/pca_metadata.pkl")
    p.add_argument("--raw_dir", type=str,
                    default="data/raw_data/XJTU-SY/35Hz12kN/Bearing1_1")
    p.add_argument("--bearing_id", type=str, default="Bearing1_1")
    p.add_argument("--dataset_name", type=str, default="xjtu")
    p.add_argument("--image_bins", type=int, default=224)
    p.add_argument("--out", type=str, default="figures/gar_ablation_compare.png")
    return p.parse_args()


def main():
    args = parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X, meta = data["X"], data["meta"]
    idxs = [i for i, row in enumerate(meta) if str(row[0]) == args.bearing_id]
    idxs.sort(key=lambda i: int(meta[i][1]))
    timesteps = np.array([int(meta[i][1]) for i in idxs])

    fpt, eof = get_fpt_eof(args.dataset_name)[args.bearing_id]
    rul = np.array([piecewise_rul(t, fpt, eof) for t in timesteps])

    with open(args.pca_metadata, "rb") as f:
        pmeta = pickle.load(f)
    pca, xlim, ylim, m, tau = pmeta["pca"], pmeta["xlim"], pmeta["ylim"], pmeta["m"], pmeta["tau"]
    den_cfg = IENEMDATDConfig(enabled=False)  # Case-1 was built with --denoise off

    # Pick the real observation closest to each target RUL checkpoint
    chosen = []
    for target in CHECKPOINTS:
        pos = int(np.argmin(np.abs(rul - target)))
        chosen.append((idxs[pos], int(timesteps[pos]), float(rul[pos])))

    n_cols = len(chosen)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.0 * n_cols, 8.5))

    for c, (npz_idx, t_idx, rul_val) in enumerate(chosen):
        raw_path = os.path.join(args.raw_dir, f"{t_idx + 1}.csv")
        raw = pd.read_csv(raw_path).values[:, 0]  # horizontal channel, real samples

        # Row 1 — raw vibration waveform
        ax = axes[0, c]
        ax.plot(raw, linewidth=0.4, color="tab:blue")
        ax.set_title(f"t={rul_val:.2f}  RUL={100 * rul_val:.1f}%\ntimestep {t_idx}", fontsize=9)
        ax.set_xticks([])
        ax.set_ylim(raw.min() * 1.1 - 1e-6, raw.max() * 1.1 + 1e-6)
        if c == 0:
            ax.set_ylabel("Raw vibration (H)", fontsize=9)

        # Row 2 — PSDI WITHOUT GAR (per-file local PCA + auto canvas)
        imgs_noanchor, _ = process_single_file(
            raw_path, pca=pca, xlim=xlim, ylim=ylim, m=m, tau=tau,
            image_bins=args.image_bins, den_cfg=den_cfg, no_anchor=True,
        )
        ax = axes[1, c]
        ax.imshow(imgs_noanchor[0], cmap="inferno", origin="lower", vmin=0, vmax=1)
        ax.axis("off")
        if c == 0:
            ax.set_ylabel("PSDI — no GAR", fontsize=9)
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])

        # Row 3 — PSDI WITH GAR (global anchor), from the already-built dataset
        ax = axes[2, c]
        ax.imshow(X[npz_idx, 0], cmap="inferno", origin="lower", vmin=0, vmax=1)
        ax.axis("off")
        if c == 0:
            ax.set_ylabel("PSDI — with GAR", fontsize=9)
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{args.bearing_id} — raw signal vs. PSDI without/with Globally Anchored Reconstruction",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
