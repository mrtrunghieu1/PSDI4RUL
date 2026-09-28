# scripts/data_processor/visualize_anchor_ablation.py
"""
Generate ECCV comparison figure: PSDI w/ vs w/o Global Anchoring.

Layout: 2 rows x 5 cols
  Row 0 (top):    PSDI w/o Anchoring  (local PCA + auto canvas)
  Row 1 (bottom): PSDI w/ Anchoring   (global PCA + fixed canvas)
  Cols: t=1.0, 0.75, 0.5, 0.25, 0.0  (start-of-life → end-of-life)

Usage:
    python scripts/data_processor/visualize_anchor_ablation.py \
        --cases 1 \
        --bearing Bearing1_2 \
        --split train \
        --image-bins 64 \
        --denoise on
"""

import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

BASE_OUT_ROOT = os.path.join(PROJECT_ROOT, "data", "processed_data")
PAPER_IMAGES_DIR = os.path.join(PROJECT_ROOT, "paper-images")
TIME_CHECKPOINTS = (1.0, 0.75, 0.5, 0.25, 0.0)
SUPPORTED_CASES = (1, 2, 3, 4, 5)


def parse_cases_arg(cases_arg: str):
    text = (cases_arg or "").strip().lower()
    if not text:
        raise ValueError("Empty --cases. Use e.g. --cases 1 or --cases 1,2,3.")
    if text == "all":
        return list(SUPPORTED_CASES)

    case_ids = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.startswith("case"):
            tok = tok[4:]
        try:
            cid = int(tok)
        except ValueError as e:
            raise ValueError(f"Invalid case token '{tok}' in --cases '{cases_arg}'.") from e
        if cid not in SUPPORTED_CASES:
            raise ValueError(f"Unsupported case: {cid}. Supported: {SUPPORTED_CASES}.")
        case_ids.append(cid)

    if not case_ids:
        raise ValueError(f"No valid case found in --cases '{cases_arg}'.")

    return sorted(set(case_ids))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize anchor ablation comparison.")
    parser.add_argument("--cases", type=str, default="1",
                        help="Cases to run. Examples: '1', '1,2,3', or 'all' (1..5).")
    parser.add_argument("--bearing", type=str, default="Bearing1_2",
                        help="Bearing name to visualize.")
    parser.add_argument("--split", type=str, default="train",
                        choices=("train", "val", "test"),
                        help="Dataset split to load.")
    parser.add_argument("--image-bins", type=int, default=64,
                        choices=(64, 224),
                        help="Image resolution (must match built dataset).")
    parser.add_argument("--denoise", choices=("on", "off"), default="on",
                        help="Denoising setting (must match built dataset).")
    parser.add_argument("--channel", type=int, default=0,
                        help="Which channel to display (0=H, 1=V).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output file path (.png/.pdf). Default: paper-images/anchor_ablation_comparison.png")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory override. Useful for saving to a new folder.")
    parser.add_argument("--pdf-only", action="store_true",
                        help="Save PDF only (no PNG).")
    parser.add_argument("--all-bearings", action="store_true",
                        help="Generate one figure per bearing across all splits and channels.")
    parser.add_argument("--metrics", action="store_true",
                        help="Compute and print Silhouette Score + inter/intra-class distance for each bearing.")
    return parser.parse_args()


def load_npz(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset not found: {npz_path}\nRun build script first.")
    data = np.load(npz_path, allow_pickle=True)
    return data["X"], data["meta"]


def find_samples_at_checkpoints(X, meta, bearing_name, checkpoints, channel):
    """Find the sample index closest to each RUL checkpoint for a given bearing."""
    entries = []
    for idx, row in enumerate(meta):
        b = str(row[0])
        if b != bearing_name:
            continue
        t_idx = int(row[1])
        total = int(row[2])
        if total <= 1:
            rul_norm = 1.0
        else:
            rul_norm = 1.0 - t_idx / float(total - 1)
        entries.append((idx, t_idx, total, rul_norm))

    if not entries:
        raise ValueError(f"Bearing '{bearing_name}' not found in dataset.")

    images = []
    rul_steps_list = []
    for target in checkpoints:
        best = min(entries, key=lambda e: abs(e[3] - target))
        idx, t_idx, total, rul_norm = best
        img = X[idx, channel]
        images.append(img)
        rul_steps = (total - 1) - t_idx
        rul_steps_list.append((rul_steps, total - 1, rul_norm))

    return images, rul_steps_list


def compute_anchor_metrics(X, meta, bearing_name, channel,
                           healthy_thresh=0.75, degraded_thresh=0.25):
    """
    Compute Silhouette Score + inter/intra-class distance for healthy vs degraded images.

    Labels:
        1 = healthy  (rul_norm >= healthy_thresh)
        0 = degraded (rul_norm <= degraded_thresh)
    Samples in between are excluded.

    Returns dict with keys: silhouette, inter_dist, intra_healthy, intra_degraded, n_healthy, n_degraded
    """
    feats, labels = [], []
    for idx, row in enumerate(meta):
        if str(row[0]) != bearing_name:
            continue
        t_idx = int(row[1])
        total = int(row[2])
        rul_norm = 1.0 if total <= 1 else 1.0 - t_idx / float(total - 1)

        if rul_norm >= healthy_thresh:
            labels.append(1)
        elif rul_norm <= degraded_thresh:
            labels.append(0)
        else:
            continue
        feats.append(X[idx, channel].ravel())

    if len(feats) < 4 or len(set(labels)) < 2:
        return None

    F = normalize(np.stack(feats, axis=0).astype(np.float32))
    labels = np.array(labels)

    sil = silhouette_score(F, labels, metric="euclidean")

    healthy_F = F[labels == 1]
    degraded_F = F[labels == 0]

    # Intra-class: mean pairwise distance within class (sampled for speed)
    def mean_pw_dist(A, max_samples=200):
        if len(A) > max_samples:
            idx = np.random.default_rng(42).choice(len(A), max_samples, replace=False)
            A = A[idx]
        diff = A[:, None, :] - A[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        n = len(A)
        return d[np.triu_indices(n, k=1)].mean() if n > 1 else 0.0

    intra_h = mean_pw_dist(healthy_F)
    intra_d = mean_pw_dist(degraded_F)

    # Inter-class: mean distance between centroids
    inter = float(np.linalg.norm(healthy_F.mean(0) - degraded_F.mean(0)))

    return {
        "silhouette": float(sil),
        "inter_dist": inter,
        "intra_healthy": float(intra_h),
        "intra_degraded": float(intra_d),
        "n_healthy": int((labels == 1).sum()),
        "n_degraded": int((labels == 0).sum()),
    }


def print_metrics_table(results: list):
    """Print a formatted comparison table. results = list of dicts."""
    header = f"{'Bearing':<14} {'Mode':<12} {'Silhouette':>10} {'Inter↑':>8} {'Intra-H↓':>10} {'Intra-D↓':>10} {'N-H':>5} {'N-D':>5}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['bearing']:<14} {r['mode']:<12} "
            f"{r['silhouette']:>10.4f} {r['inter_dist']:>8.4f} "
            f"{r['intra_healthy']:>10.4f} {r['intra_degraded']:>10.4f} "
            f"{r['n_healthy']:>5} {r['n_degraded']:>5}"
        )
    print("=" * len(header))


def make_comparison_figure(
    anchored_images, no_anchor_images,
    rul_info_anchored, rul_info_no_anchor,
    checkpoints, bearing_name, out_path, pdf_only=False
):
    n_cols = len(checkpoints)
    fig = plt.figure(figsize=(2.6 * n_cols, 5.8))
    gs = gridspec.GridSpec(
        2, n_cols,
        figure=fig,
        hspace=0.05,
        wspace=0.04,
        top=0.88,
        bottom=0.10,
        left=0.10,
        right=0.98,
    )

    row_labels = ["w/o Anchoring", "w/ Anchoring (ours)"]
    all_images = [no_anchor_images, anchored_images]
    all_rul = [rul_info_no_anchor, rul_info_anchored]

    axes_grid = []
    for r in range(2):
        row_axes = []
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(all_images[r][c], cmap="inferno", origin="lower",
                      vmin=0.0, vmax=1.0, aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_edgecolor("#555555")

            if r == 0:
                target = checkpoints[c]
                rul_steps, total, rul_norm = all_rul[r][c]
                ax.set_title(
                    f"$t$ = {target:.2f}\n({100*rul_norm:.0f}% RUL)",
                    fontsize=9, pad=3
                )
            row_axes.append(ax)
        axes_grid.append(row_axes)

        # Row label: use set_ylabel on leftmost axis
        axes_grid[r][0].set_ylabel(
            row_labels[r], fontsize=9, fontweight="bold",
            rotation=90, labelpad=6
        )

    # Highlight proposed row with orange border
    # for ax in axes_grid[1]:
    #     for spine in ax.spines.values():
    #         spine.set_linewidth(1.5)
            # spine.set_edgecolor("#E67E22")

    # fig.suptitle(
    #     f"PSDI Evolution: {bearing_name}  —  Global Anchoring Ablation",
    #     fontsize=11, fontweight="bold", y=0.97
    # )

    # Bottom caption
    # fig.text(
    #     0.5, 0.02,
    #     "Each column = closest sample to normalized RUL checkpoint  |  "
    #     "Row 1: per-file PCA (degradation erased)  |  Row 2: global anchor (drift visible)",
    #     ha="center", va="bottom", fontsize=7, color="#555555"
    # )

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Export vector PDF only if requested, otherwise export both PNG and PDF.
    stem, ext = os.path.splitext(out_path)
    if ext.lower() in (".png", ".pdf"):
        png_path = stem + ".png"
        pdf_path = stem + ".pdf"
    else:
        png_path = out_path + ".png"
        pdf_path = out_path + ".pdf"

    plt.savefig(pdf_path, bbox_inches="tight")
    if not pdf_only:
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {pdf_path}")
    if not pdf_only:
        print(f"Saved: {png_path}")


def get_bearings_in_split(meta) -> list:
    """Return sorted unique bearing names from meta array."""
    return sorted({str(row[0]) for row in meta})


def main():
    args = parse_args()
    case_ids = parse_cases_arg(args.cases)
    denoise_tag = f"denoise_{args.denoise}"
    bins_tag = f"image_bins_{args.image_bins}"

    # Determine which (split, bearing, channel) combos to generate
    if args.all_bearings:
        default_splits = ("train", "val", "test")
        channels = (0, 1)
        out_root = args.out_dir or os.path.join(PAPER_IMAGES_DIR, "anchor_ablation_all")
        os.makedirs(out_root, exist_ok=True)
    else:
        default_splits = (args.split,)
        channels = (args.channel,)
        out_root = args.out_dir

    metrics_rows = []
    generated = 0

    for case_id in case_ids:
        case_dir = os.path.join(BASE_OUT_ROOT, f"case{case_id}")
        if not os.path.isdir(case_dir):
            print(f"\n[WARN] Skip case{case_id}: missing directory {case_dir}")
            continue

        for split in default_splits:
            anchored_npz = os.path.join(case_dir, denoise_tag, bins_tag, f"{split}.npz")
            no_anchor_npz = os.path.join(case_dir, denoise_tag, f"{bins_tag}_no_anchor", f"{split}.npz")

            if not os.path.exists(anchored_npz):
                print(f"\n[WARN] Skip case{case_id}/{split}: missing anchored npz {anchored_npz}")
                continue
            if not os.path.exists(no_anchor_npz):
                print(f"\n[WARN] Skip case{case_id}/{split}: missing no-anchor npz {no_anchor_npz}")
                continue

            print(f"\nLoading anchored:  {anchored_npz}")
            X_anchored, meta_anchored = load_npz(anchored_npz)

            print(f"Loading no-anchor: {no_anchor_npz}")
            X_no_anchor, meta_no_anchor = load_npz(no_anchor_npz)

            bearings = get_bearings_in_split(meta_anchored) if args.all_bearings else [args.bearing]

            for bearing in bearings:
                for ch in channels:
                    try:
                        anchored_imgs, rul_anchored = find_samples_at_checkpoints(
                            X_anchored, meta_anchored, bearing, TIME_CHECKPOINTS, ch
                        )
                        no_anchor_imgs, rul_no_anchor = find_samples_at_checkpoints(
                            X_no_anchor, meta_no_anchor, bearing, TIME_CHECKPOINTS, ch
                        )
                    except ValueError as e:
                        print(f"  Skipping case{case_id} {bearing} ch{ch}: {e}")
                        continue

                    if args.all_bearings:
                        ext = ".pdf" if args.pdf_only else ".png"
                        fname = f"{split}_{bearing}_ch{ch}{ext}"
                        case_out_dir = os.path.join(out_root, f"case{case_id}")
                        out_path = os.path.join(case_out_dir, fname)
                    else:
                        if args.out:
                            if len(case_ids) > 1:
                                stem, ext = os.path.splitext(args.out)
                                out_path = f"{stem}_case{case_id}{ext}" if ext else f"{args.out}_case{case_id}"
                            else:
                                out_path = args.out
                        else:
                            base_dir = out_root or PAPER_IMAGES_DIR
                            if len(case_ids) > 1:
                                base_dir = os.path.join(base_dir, f"case{case_id}")
                            default_name = "anchor_ablation_comparison.pdf" if args.pdf_only else "anchor_ablation_comparison.png"
                            out_path = os.path.join(base_dir, default_name)

                    make_comparison_figure(
                        anchored_imgs, no_anchor_imgs,
                        rul_anchored, rul_no_anchor,
                        TIME_CHECKPOINTS, f"case{case_id} {bearing} (split={split}, ch={ch})", out_path, pdf_only=args.pdf_only
                    )
                    generated += 1

                    if args.metrics:
                        for X_data, meta_data, mode in [
                            (X_anchored, meta_anchored, "anchored"),
                            (X_no_anchor, meta_no_anchor, "no_anchor"),
                        ]:
                            m = compute_anchor_metrics(X_data, meta_data, bearing, ch)
                            if m is not None:
                                metrics_rows.append({"bearing": f"case{case_id}/{bearing}/ch{ch}", "mode": mode, **m})

    if args.all_bearings:
        print(f"\nAll figures root: {out_root}")
    print(f"Generated figures: {generated}")

    if args.metrics and metrics_rows:
        print_metrics_table(metrics_rows)


if __name__ == "__main__":
    main()
