"""
Animated 2-row figure for ONE real bearing lifecycle (XJTU-SY / Case-1):
  Row 1: full raw vibration waveform (all real samples, concatenated in
         chronological order) with a moving marker over the current window.
  Row 2: PSDI WITH GAR for the current window (loaded directly from the
         already-built processed .npz — exactly what the models train on).

Every frame corresponds to one real observation (one CSV file / one
timestep) already present in the dataset. Nothing is synthesized.

Usage:
    python tools/viz_vibration_psdi_animation.py --bearing_id Bearing1_1
    python tools/viz_vibration_psdi_animation.py --bearing_id Bearing2_5
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
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rul import piecewise_rul, get_fpt_eof

CASE1_SPLIT = {
    "Bearing1_2": "train", "Bearing1_3": "train", "Bearing1_4": "train",
    "Bearing1_5": "train", "Bearing2_1": "train", "Bearing2_2": "train",
    "Bearing2_3": "val", "Bearing2_4": "val", "Bearing2_5": "val",
    "Bearing1_1": "test",
}
CASE1_ROOT = "data/processed_data/case1/denoise_off/image_bins_64"
RAW_ROOT = "data/raw_data/XJTU-SY"


def parse_args():
    p = argparse.ArgumentParser(description="Raw waveform + PSDI(with GAR) animation, one bearing lifecycle")
    p.add_argument("--bearing_id", type=str, default="Bearing1_1")
    p.add_argument("--dataset_name", type=str, default="xjtu")
    p.add_argument("--out_dir", type=str, default="figures")
    p.add_argument("--out_name", type=str, default=None,
                    help="Base output filename (without extension). Defaults to "
                         "'vibration_psdi_rul_evolution' for Bearing1_1, else suffixed with the bearing id.")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--hold_frames", type=int, default=10,
                    help="Extra repeats of the first and last real frame, for a visible pause.")
    return p.parse_args()


def main():
    args = parse_args()
    bearing_id = args.bearing_id

    if bearing_id not in CASE1_SPLIT:
        raise ValueError(f"Unknown bearing '{bearing_id}'. Known: {list(CASE1_SPLIT.keys())}")
    split = CASE1_SPLIT[bearing_id]
    npz_path = os.path.join(CASE1_ROOT, f"{split}.npz")
    condition = "35Hz12kN" if bearing_id.startswith("Bearing1") else "37.5Hz11kN"
    raw_dir = os.path.join(RAW_ROOT, condition, bearing_id)

    out_name = args.out_name or (
        "vibration_psdi_rul_evolution" if bearing_id == "Bearing1_1"
        else f"vibration_psdi_rul_evolution_{bearing_id}"
    )

    # ---- PSDI with GAR, straight from the already-built dataset ----
    data = np.load(npz_path, allow_pickle=True)
    X, meta = data["X"], data["meta"]
    idxs = [i for i, row in enumerate(meta) if str(row[0]) == bearing_id]
    idxs.sort(key=lambda i: int(meta[i][1]))
    timesteps = np.array([int(meta[i][1]) for i in idxs])
    gar_imgs = X[idxs, 0]
    n_frames = len(idxs)

    fpt, eof = get_fpt_eof(args.dataset_name)[bearing_id]
    rul = np.array([piecewise_rul(t, fpt, eof) for t in timesteps])
    initial_rul = rul[0]

    raw_paths = [os.path.join(raw_dir, f"{t + 1}.csv") for t in timesteps]

    # ---- raw waveform, real samples only, concatenated in chronological order ----
    print(f"[{bearing_id}] split={split}  frames={n_frames}  FPT={fpt} EOF={eof}")
    print("Loading raw waveforms ...")
    raw_signals = [pd.read_csv(p).values[:, 0] for p in raw_paths]
    lengths = np.array([len(s) for s in raw_signals])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    full_waveform = np.concatenate(raw_signals)
    print(f"  {n_frames} files, {len(full_waveform):,} samples total")

    # ============================== FIGURE ==============================
    plt.rcParams["figure.facecolor"] = "white"
    fig = plt.figure(figsize=(9, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.8], hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # -- Row 1 (static background, drawn once): full real waveform --
    ax1.plot(full_waveform, linewidth=0.12, color="steelblue", rasterized=True)
    ax1.set_xlim(0, len(full_waveform))
    ymargin = 0.08 * (full_waveform.max() - full_waveform.min())
    ax1.set_ylim(full_waveform.min() - ymargin, full_waveform.max() + ymargin)
    ax1.set_xlabel("Sample index (all real observations, concatenated in time order)")
    ax1.set_ylabel("Vibration amplitude (g)")
    ax1.set_title(f"{bearing_id} — full raw vibration signal (horizontal channel)", fontsize=10)

    fpt_x = offsets[np.searchsorted(timesteps, fpt, side="right") - 1] if fpt >= timesteps[0] else 0
    ax1.axvline(fpt_x, color="gray", linestyle="--", linewidth=1)
    y_top = full_waveform.max() + ymargin
    ax1.text(fpt_x * 0.5, y_top * 0.85, "Healthy state", ha="center", fontsize=8, color="dimgray")
    ax1.text(fpt_x + (len(full_waveform) - fpt_x) * 0.5, y_top * 0.85, "Degradation state",
              ha="center", fontsize=8, color="darkorange")
    ax1.text(len(full_waveform) * 0.97, y_top * 0.85, "Failure", ha="right", fontsize=8, color="firebrick")

    highlight = mpatches.Rectangle((0, ax1.get_ylim()[0]), lengths[0],
                                    ax1.get_ylim()[1] - ax1.get_ylim()[0],
                                    facecolor="red", alpha=0.30, edgecolor="red", linewidth=1.2)
    ax1.add_patch(highlight)

    # -- Row 2: PSDI with GAR, fixed color scale --
    im2 = ax2.imshow(gar_imgs[0], cmap="inferno", origin="lower", vmin=0, vmax=1)
    ax2.axis("off")
    ax2.set_title("PSDI with GAR", fontsize=11, color="black")

    suptitle = fig.suptitle("", fontsize=12, family="monospace", y=0.985)

    frame_order = (
        [0] * args.hold_frames +
        list(range(n_frames)) +
        [n_frames - 1] * args.hold_frames
    )

    def render(k):
        i = frame_order[k]
        highlight.set_x(offsets[i])
        highlight.set_width(lengths[i])
        im2.set_data(gar_imgs[i])
        suptitle.set_text(
            f"Sample: {bearing_id}   |   frame {i}/{n_frames - 1}   |   "
            f"t = {rul[i] / initial_rul:.3f}   |   "
            f"RUL = {rul[i]:.3f}/{initial_rul:.3f} | {100.0 * rul[i]:.1f}%"
        )
        return highlight, im2, suptitle

    anim = FuncAnimation(fig, render, frames=len(frame_order), blit=False)

    os.makedirs(args.out_dir, exist_ok=True)
    gif_path = os.path.join(args.out_dir, f"{out_name}.gif")
    mp4_path = os.path.join(args.out_dir, f"{out_name}.mp4")

    print(f"Writing {gif_path} ...")
    anim.save(gif_path, writer=PillowWriter(fps=args.fps))

    print(f"Writing {mp4_path} ...")
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        from matplotlib.animation import FFMpegWriter
        anim.save(mp4_path, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    except Exception as e:
        print(f"  [WARN] MP4 export failed ({e}); GIF was still saved.")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
