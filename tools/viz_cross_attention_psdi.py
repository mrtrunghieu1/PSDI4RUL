"""
Visualization: Cross-Attention Patch Importance in Phase 3.

Shows how cross-attention conditions spatial PSDI patches on trajectory dynamics
by comparing one healthy sample vs one degraded sample.

For each sample (healthy / degraded) the script produces 4 separate figures:
  (a) PSDI image (channel 0)
  (b) Patch importance heatmap  (14x14 grid)
  (c) Overlay of PSDI + patch importance
  (d) Attention profile of a peripheral patch over 160 phase tokens

Patch importance:
    A  : [196, 160]  (attention map, averaged over heads by PyTorch)
    patch_weight[p] = max over 160 phase tokens of A[p, :]

Usage (defaults):
    python tools/viz_cross_attention_psdi.py

Custom:
    python tools/viz_cross_attention_psdi.py \\
        --ckpt checkpoints/rul_RUL_PHM_C2_P3_KW7_L2_phase3_dm128_0_dst_mae_base/checkpoint.pth \\
        --npz  data/processed_data/case2/denoise_off/image_bins_224/test.npz \\
        --out_dir figures/ \\
        --peripheral_patch_idx 180
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.psdi_kd.rul_model import RULStudentModel
from utils.rul import piecewise_rul, PHM_FPT_EOF


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_args(**overrides):
    """Minimal namespace with all fields required by RULStudentModel."""
    defaults = dict(
        # vision backbone
        student_vision_type="tiny_vit",
        student_hidden_size=128,
        teacher_hidden_size=768,
        d_model=128,
        # phase 3
        rul_phase=3,
        phase3_qkv_mode="vision_q_phase_kv",
        phase3_residual_fusion=False,
        phase3_fusion_alpha_init=0.7,
        # regularisation / dropout
        dropout=0.1,
        # feature aligner
        num_alignment_scales=4,
        feature_alignment_dropout=0.1,
        # K-window: 1 for single-frame inference
        k_window_size=1,
        temporal_conv_layers=2,
        temporal_conv_kernel=3,
        # degradation weighted forecasting (not used in viz)
        use_degradation_weighted_fcst=False,
        degradation_lambda=2.0,
        degradation_gamma=2.0,
        # device
        use_gpu=False,
        gpu=0,
    )
    defaults.update(overrides)
    import argparse
    return argparse.Namespace(**defaults)


def _load_model(ckpt_path: str) -> RULStudentModel:
    """Build a K=1 Phase-3 student model and load weights from checkpoint."""
    args = _build_args()
    model = RULStudentModel(args)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Checkpoint may have been saved from the Model wrapper (keys prefixed with
    # "model.") or directly from RULStudentModel (no prefix).
    # Strip "model." prefix if present so keys match RULStudentModel directly.
    if all(k.startswith("model.") for k in ckpt.keys()):
        ckpt = {k[len("model."):]: v for k, v in ckpt.items()}

    # strict=False: checkpoint may contain temporal_aggregator keys (K=7)
    # that do not exist in the K=1 model — they are silently skipped.
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    print(f"Loaded checkpoint: {ckpt_path}")
    return model


def _select_samples(meta: np.ndarray, fpt: int, eof: int):
    """
    Return (healthy_idx, degraded_idx) from the npz meta array.

    Healthy  : timestep well before FPT (first ~10% of healthy region).
    Degraded : timestep well after FPT and close to EOF.
    """
    timesteps = meta[:, 1].astype(int)

    healthy_candidates = np.where(timesteps <= fpt)[0]
    degraded_candidates = np.where((timesteps > fpt) & (timesteps < eof))[0]

    if len(healthy_candidates) == 0:
        raise RuntimeError("No healthy samples found (timestep <= FPT)")
    if len(degraded_candidates) == 0:
        raise RuntimeError("No degraded samples found (FPT < timestep < EOF)")

    # Pick a sample from the first quarter of the healthy region
    healthy_idx = healthy_candidates[len(healthy_candidates) // 4]
    # Pick a sample from the last quarter of the degraded region
    degraded_idx = degraded_candidates[3 * len(degraded_candidates) // 4]

    return int(healthy_idx), int(degraded_idx)


def _run_inference(model: RULStudentModel, X_np: np.ndarray, P_np: np.ndarray):
    """
    Run forward pass and return (rul_pred, A).

    A : [196, 160]  cross-attention map (patches × phase tokens)
    """
    X = torch.from_numpy(X_np[None]).float()   # [1, 2, 224, 224]
    P = torch.from_numpy(P_np[None]).float()   # [1, 2, 2560]

    with torch.no_grad():
        rul_pred, _, A = model.forward_with_features(X, P=P, return_attention=True)

    if A is None:
        raise RuntimeError(
            "Attention map is None — ensure model.use_phase3 is True "
            "and P is provided."
        )

    rul_val = float(rul_pred[0, 0].cpu())
    A_np = A[0].cpu().numpy()   # [196, 160]
    return rul_val, A_np


def _patch_importance(A: np.ndarray) -> np.ndarray:
    """
    Compute patch importance using max over phase tokens.

    patch_weight[p] = max(A[p, :])   shape: [196]

    Normalised to [0, 1] and reshaped to [14, 14].
    """
    pw = A.max(axis=-1)   # [196]
    pw = (pw - pw.min()) / (pw.max() - pw.min() + 1e-8)
    return pw.reshape(14, 14)


def _upsample_patch_map(patch_map: np.ndarray, size: int = 224) -> np.ndarray:
    """Nearest-neighbour upsample [14, 14] → [size, size]."""
    factor = size // patch_map.shape[0]
    return np.kron(patch_map, np.ones((factor, factor), dtype=np.float32))


# ── per-sample figure functions ───────────────────────────────────────────────

def _save_psdi(X_np: np.ndarray, rul: float, label: str, out_dir: str):
    """Figure (a): PSDI image, channel 0."""
    fig, ax = plt.subplots(figsize=(5, 5))
    img = ax.imshow(X_np[0], aspect="auto", origin="lower")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"PSDI Image — {label}  (RUL = {rul:.3f})")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Time bin")
    plt.tight_layout()
    path = os.path.join(out_dir, f"cross_attn_{label}_psdi.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_patch_importance(patch_map: np.ndarray, rul: float, label: str, out_dir: str):
    """Figure (b): Patch importance heatmap [14x14]."""
    fig, ax = plt.subplots(figsize=(5, 5))
    img = ax.imshow(patch_map, aspect="equal", origin="upper")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Patch Importance — {label}  (RUL = {rul:.3f})\nmax-attn over 160 phase tokens")
    ax.set_xlabel("Patch col")
    ax.set_ylabel("Patch row")
    # Grid lines at patch boundaries
    for k in range(15):
        ax.axhline(k - 0.5, linewidth=0.3)
        ax.axvline(k - 0.5, linewidth=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"cross_attn_{label}_patch_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_overlay(X_np: np.ndarray, patch_map: np.ndarray, rul: float,
                  label: str, out_dir: str):
    """Figure (c): PSDI image overlaid with patch importance."""
    base = X_np[0]   # [224, 224]
    importance_224 = _upsample_patch_map(patch_map, size=base.shape[0])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(base, aspect="auto", origin="lower")
    overlay = ax.imshow(importance_224, aspect="auto", origin="lower", alpha=0.5)
    fig.colorbar(overlay, ax=ax, fraction=0.046, pad=0.04, label="Patch importance")
    ax.set_title(f"PSDI + Patch Importance Overlay — {label}  (RUL = {rul:.3f})")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Time bin")
    plt.tight_layout()
    path = os.path.join(out_dir, f"cross_attn_{label}_overlay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_phase_profile(A: np.ndarray, patch_idx: int, rul: float,
                        label: str, out_dir: str):
    """Figure (d): Attention of one peripheral patch over 160 phase tokens."""
    profile = A[patch_idx]   # [160]
    # Patch spatial location for the title
    row, col = divmod(patch_idx, 14)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(profile)
    ax.set_xlabel("Phase token index (0–159)")
    ax.set_ylabel("Attention weight")
    ax.set_title(
        f"Attention over 160 Phase Tokens — {label}  (RUL = {rul:.3f})\n"
        f"Patch {patch_idx}  (row={row}, col={col})"
    )
    ax.set_xlim(0, 159)
    plt.tight_layout()
    path = os.path.join(out_dir, f"cross_attn_{label}_phase_profile.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize Phase-3 cross-attention patch importance")
    p.add_argument(
        "--ckpt",
        default="checkpoints/rul_RUL_PHM_C2_P3_KW7_L2_phase3_dm128_0_dst_mae_base/checkpoint.pth",
        help="Path to Phase-3 student checkpoint",
    )
    p.add_argument(
        "--npz",
        default="data/processed_data/case2/denoise_off/image_bins_224/test.npz",
        help="Path to test.npz",
    )
    p.add_argument(
        "--dataset_name",
        default="phm",
        choices=["phm", "xjtu"],
        help="Dataset name (used for FPT/EOF lookup)",
    )
    p.add_argument(
        "--bearing_id",
        default="Bearing1_5",
        help="Bearing ID for FPT/EOF lookup (PHM case2 test bearing)",
    )
    p.add_argument(
        "--out_dir",
        default="figures",
        help="Directory to save output figures",
    )
    p.add_argument(
        "--peripheral_patch_idx",
        type=int,
        default=180,
        help="Patch index (0–195) for the phase-token attention profile plot",
    )
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"Loading data: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)
    X_all = data["X"]   # [N, 2, 224, 224]
    P_all = data["P"]   # [N, 2, 2560]
    meta  = data["meta"]  # [N, 3]

    # ── FPT / EOF ─────────────────────────────────────────────────────────────
    if args.dataset_name == "phm":
        fpt_eof_table = PHM_FPT_EOF
    else:
        from utils.rul import XJTU_FPT_EOF
        fpt_eof_table = XJTU_FPT_EOF

    if args.bearing_id not in fpt_eof_table:
        raise ValueError(
            f"Bearing '{args.bearing_id}' not in FPT/EOF table. "
            f"Available: {list(fpt_eof_table.keys())}"
        )
    fpt, eof = fpt_eof_table[args.bearing_id]
    print(f"Bearing {args.bearing_id}: FPT={fpt}, EOF={eof}")

    # ── select samples ────────────────────────────────────────────────────────
    healthy_idx, degraded_idx = _select_samples(meta, fpt, eof)
    healthy_t  = int(meta[healthy_idx, 1])
    degraded_t = int(meta[degraded_idx, 1])
    healthy_rul  = piecewise_rul(healthy_t,  fpt, eof)
    degraded_rul = piecewise_rul(degraded_t, fpt, eof)

    print(f"Healthy sample  : idx={healthy_idx},  t={healthy_t},  RUL={healthy_rul:.3f}")
    print(f"Degraded sample : idx={degraded_idx}, t={degraded_t}, RUL={degraded_rul:.3f}")

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nBuilding model and loading checkpoint ...")
    model = _load_model(args.ckpt)

    # ── process each sample ───────────────────────────────────────────────────
    samples = [
        ("healthy",  healthy_idx,  healthy_rul),
        ("degraded", degraded_idx, degraded_rul),
    ]

    for label, idx, rul in samples:
        print(f"\n{'='*60}")
        print(f"Sample: {label}  (idx={idx}, RUL={rul:.3f})")
        print(f"{'='*60}")

        X_np = X_all[idx]   # [2, 224, 224]
        P_np = P_all[idx]   # [2, 2560]

        # Inference
        rul_pred, A = _run_inference(model, X_np, P_np)
        print(f"  Model RUL prediction: {rul_pred:.3f}  (ground truth: {rul:.3f})")
        print(f"  Attention map A shape: {A.shape}")   # should be (196, 160)

        # Patch importance [14, 14]
        patch_map = _patch_importance(A)

        # Figures
        _save_psdi(X_np, rul, label, args.out_dir)
        _save_patch_importance(patch_map, rul, label, args.out_dir)
        _save_overlay(X_np, patch_map, rul, label, args.out_dir)
        _save_phase_profile(A, args.peripheral_patch_idx, rul, label, args.out_dir)

    print(f"\nAll figures saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
