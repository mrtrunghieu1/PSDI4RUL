"""
Animate the real PSDI heatmap sequence of ONE bearing (one lifecycle) from
healthy (RUL=1.0) to failure (RUL~0), using only the actual observations
already stored in a processed .npz dataset. No synthetic frames.

Same preprocessing / rendering convention as the existing static
visualizations (scripts/data_processor/build_dataset_case1.py):
  X[:, 0] channel, cmap="inferno", origin="lower".

RUL is computed with the same piecewise_rul(timestep, fpt, eof) function
used for real training labels (data_provider/bearing_loader.py), not a
simplified index-based proxy.

Usage:
    python tools/viz_one_sample_animation.py \
        --npz data/processed_data/case1/denoise_off/image_bins_224/test.npz \
        --dataset_name xjtu \
        --out_dir figures
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rul import piecewise_rul, get_fpt_eof


def parse_args():
    p = argparse.ArgumentParser(description="Animate one real bearing lifecycle from a processed PSDI .npz")
    p.add_argument("--npz", type=str,
                    default="data/processed_data/case1/denoise_off/image_bins_224/test.npz",
                    help="Path to a processed .npz (must contain X, meta)")
    p.add_argument("--dataset_name", type=str, default="xjtu",
                    help="FPT/EOF table to use: xjtu | phm | phm_c5")
    p.add_argument("--bearing_id", type=str, default=None,
                    help="Which bearing to animate. Defaults to the only/first bearing in the file.")
    p.add_argument("--out_dir", type=str, default="figures")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--hold_last_frames", type=int, default=8,
                    help="Repeat the final real frame N extra times so the failure state is visible on loop.")
    return p.parse_args()


def main():
    args = parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X, meta = data["X"], data["meta"]

    all_bearings = sorted(set(str(row[0]) for row in meta))
    bearing_id = args.bearing_id or all_bearings[0]
    if bearing_id not in all_bearings:
        raise ValueError(f"Bearing '{bearing_id}' not found in {args.npz}. Available: {all_bearings}")
    print(f"Sample (bearing): {bearing_id}  |  available in file: {all_bearings}")

    idxs = [i for i, row in enumerate(meta) if str(row[0]) == bearing_id]
    idxs.sort(key=lambda i: int(meta[i][1]))  # chronological order by real timestep

    timesteps = np.array([int(meta[i][1]) for i in idxs])
    totals = np.array([int(meta[i][2]) for i in idxs])
    imgs = X[idxs, 0]  # real PSDI images, channel 0 (horizontal), same as static viz

    fpt_eof = get_fpt_eof(args.dataset_name)
    if bearing_id not in fpt_eof:
        raise ValueError(f"No FPT/EOF entry for '{bearing_id}' in dataset '{args.dataset_name}'")
    fpt, eof = fpt_eof[bearing_id]
    rul = np.array([piecewise_rul(t, fpt, eof) for t in timesteps])
    initial_rul = rul[0]
    t_norm = rul / initial_rul if initial_rul > 0 else rul

    n_frames = len(idxs)
    print(f"Real observations: {n_frames}  |  FPT={fpt} EOF={eof}  |  "
          f"RUL range: {rul[0]:.3f} -> {rul[-1]:.3f}")

    vmin, vmax = float(imgs.min()), float(imgs.max())  # fixed global color scale
    total = int(totals[0])

    plt.rcParams["figure.facecolor"] = "black"
    fig, ax = plt.subplots(figsize=(6, 6.8), facecolor="black")
    fig.subplots_adjust(top=0.78, bottom=0.02, left=0.02, right=0.98)
    ax.set_facecolor("black")
    im = ax.imshow(imgs[0], cmap="inferno", origin="lower", vmin=vmin, vmax=vmax)
    ax.axis("off")
    title = ax.set_title("", color="white", fontsize=12, family="monospace",
                          loc="center", linespacing=1.6)

    # Repeat the last real frame a few extra times so a looping GIF/MP4
    # visibly holds on the real end-of-life state (no new data, just replay).
    frame_order = list(range(n_frames)) + [n_frames - 1] * max(0, args.hold_last_frames)

    def render(k):
        i = frame_order[k]
        im.set_data(imgs[i])
        title.set_text(
            f"Sample: {bearing_id}\n"
            f"timestep: {timesteps[i]}/{total - 1}\n"
            f"t = {t_norm[i]:.3f}\n"
            f"RUL = {rul[i]:.3f}/{initial_rul:.3f} | {100.0 * rul[i]:.1f}%"
        )
        return im, title

    anim = FuncAnimation(fig, render, frames=len(frame_order), blit=False)

    os.makedirs(args.out_dir, exist_ok=True)
    gif_path = os.path.join(args.out_dir, "one_sample_rul_evolution.gif")
    mp4_path = os.path.join(args.out_dir, "one_sample_rul_evolution.mp4")

    print(f"Writing {gif_path} ...")
    anim.save(gif_path, writer=PillowWriter(fps=args.fps))

    print(f"Writing {mp4_path} ...")
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        from matplotlib.animation import FFMpegWriter
        anim.save(mp4_path, writer=FFMpegWriter(fps=args.fps, bitrate=1800))
    except Exception as e:
        print(f"  [WARN] MP4 export failed ({e}); GIF was still saved.")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
