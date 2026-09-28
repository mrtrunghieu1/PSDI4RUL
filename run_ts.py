"""
Entry point for SOTA time-series RUL baselines.

Supports: PatchTST, TimesNet, TimeMixer, TimeMixerPP

Usage example (single-model):
    python run_ts.py \\
        --task_name ts_rul --model PatchTST --is_training 1 \\
        --model_id RUL_XJTU_PatchTST \\
        --dataset_name xjtu \\
        --raw_data_path data/raw_data/XJTU-SY \\
        --ts_window_len 2560 --enc_in 2 \\
        --d_model 128 --n_heads 8 --e_layers 3 --d_ff 512 \\
        --patch_len 64 --stride 32 \\
        --batch_size 64 --train_epochs 50 --patience 10 \\
        --learning_rate 0.0001 --lradj cosine \\
        --loss SmoothL1 --use_amp --gpu 0

Benchmark script:
    bash scripts/rul/train_ts_baselines.sh [GPU_ID]
"""

import argparse
import os
import random

import numpy as np
import torch

from exp.exp_ts_rul import Exp_TS_RUL


# ── Helpers ───────────────────────────────────────────────────────────────────

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="SOTA Time-Series RUL Baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Basic config ──────────────────────────────────────────────────────────
    parser.add_argument("--task_name", type=str, default="ts_rul",
                        help="Task name (must be 'ts_rul')")
    parser.add_argument("--is_training", type=int, required=True,
                        help="1 = train+test, 0 = test only")
    parser.add_argument("--model_id", type=str, required=True,
                        help="Experiment identifier (used in setting string)")
    parser.add_argument("--model", type=str, required=True,
                        choices=["PatchTST", "TimesNet", "TimeMixer", "TimeMixerPP"],
                        help="Model architecture")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--dataset_name", type=str, default="xjtu",
                        help="Dataset name for FPT/EOF lookup: xjtu | phm")
    parser.add_argument("--raw_data_path", type=str,
                        default="data/raw_data/XJTU-SY",
                        help="Root directory containing BearingX_Y/ subdirs")
    parser.add_argument("--ts_window_len", type=int, default=2560,
                        help="Window length L (samples extracted from each CSV)")
    parser.add_argument("--ts_window_offset", type=str, default="center",
                        choices=["center", "start", "end"],
                        help="Which part of the raw signal to use as window")
    parser.add_argument("--normalize_signal", type=str2bool, default=True,
                        help="Per-sample z-score normalisation on raw signal")
    parser.add_argument("--use_ts_cache", type=str2bool, default=True,
                        help="Cache preprocessed arrays to disk for fast reload")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader num workers")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/",
                        help="Directory for model checkpoints")

    # ── Model architecture (shared flags — same names as run.py) ─────────────
    parser.add_argument("--enc_in", type=int, default=2,
                        help="Number of input channels (2 for H+V vibration)")
    parser.add_argument("--d_model", type=int, default=128,
                        help="Model embedding dimension")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads (PatchTST)")
    parser.add_argument("--e_layers", type=int, default=3,
                        help="Number of encoder / block layers")
    parser.add_argument("--d_ff", type=int, default=512,
                        help="Feed-forward / inner dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--use_norm", type=int, default=1,
                        help="1 = apply RevIN instance norm inside model, 0 = off")

    # ── PatchTST-specific ─────────────────────────────────────────────────────
    parser.add_argument("--patch_len", type=int, default=64,
                        help="Patch length (PatchTST)")
    parser.add_argument("--stride", type=int, default=32,
                        help="Patch stride (PatchTST)")

    # ── TimesNet-specific ─────────────────────────────────────────────────────
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-K FFT periods (TimesNet / TimeMixer++ DFT decomp)")
    parser.add_argument("--num_kernels", type=int, default=6,
                        help="Number of Inception kernels (TimesNet)")

    # ── TimeMixer / TimeMixer++-specific ──────────────────────────────────────
    parser.add_argument("--down_sampling_layers", type=int, default=3,
                        help="Number of downsampling scales M (TimeMixer/++)")
    parser.add_argument("--down_sampling_window", type=int, default=2,
                        help="Avg-pool downsampling factor (TimeMixer/++)")
    parser.add_argument("--decomp_method", type=str, default="moving_avg",
                        choices=["moving_avg", "dft_decomp"],
                        help="Series decomposition method (TimeMixer/++)")
    parser.add_argument("--moving_avg", type=int, default=25,
                        help="Moving average kernel size for decomposition")
    parser.add_argument("--channel_independence", type=int, default=1,
                        help="1 = each channel processed independently (TimeMixer/++), "
                             "0 = channel-dependent (cross-channel FFN)")
    parser.add_argument("--use_rms_seq", type=str2bool, default=False,
                        help="TimeMixer RMS-sequence mode: each sample is RMS of "
                             "ts_window_len consecutive files (not raw waveform). "
                             "Use with --ts_window_len 20 --down_sampling_layers 2 "
                             "--moving_avg 3")
    parser.add_argument("--use_feat_seq", type=str2bool, default=False,
                        help="TimeMixer multi-feature sequence mode: each sample is "
                             "8 statistical features (RMS, Variance, Peak, Kurtosis, "
                             "Skewness, Crest Factor, Shape Factor, Impulse Factor) "
                             "per sensor over ts_window_len consecutive files. "
                             "enc_in is auto-set to 16 (2 sensors × 8 features). "
                             "Use with --ts_window_len 96 --down_sampling_layers 3 "
                             "--moving_avg 7")

    # ── Optimisation ──────────────────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--train_epochs", type=int, default=50,
                        help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="AdamW weight decay")
    parser.add_argument("--lradj", type=str, default="cosine",
                        choices=["type1", "type2", "cosine"],
                        help="Learning rate adjustment strategy")
    parser.add_argument("--loss", type=str, default="SmoothL1",
                        choices=["SmoothL1", "MSE", "MAE"],
                        help="Training loss function")
    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="Use automatic mixed precision (AMP)")
    parser.add_argument("--itr", type=int, default=1,
                        help="Number of independent experiment repetitions")
    parser.add_argument("--des", type=str, default="baseline",
                        help="Short description tag added to setting string")

    # ── Hardware ──────────────────────────────────────────────────────────────
    parser.add_argument("--use_gpu", type=str2bool, default=True,
                        help="Use GPU if available")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device id")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False,
                        help="Use multiple GPUs via DataParallel")
    parser.add_argument("--devices", type=str, default="0,1,2,3",
                        help="Comma-separated GPU device ids for multi-GPU")

    return parser


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Reproducibility
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = build_parser()
    args = parser.parse_args()

    # Resolve GPU availability
    args.use_gpu = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(d) for d in device_ids]
        args.gpu = args.device_ids[0]
    else:
        args.device_ids = [args.gpu]

    # ts_rul task does not use content/VLM
    args.content = None

    # Alias: seq_len = ts_window_len (TimeMixer internals use configs.seq_len)
    args.seq_len = args.ts_window_len

    # Multi-feature sequence mode: override enc_in to 2 sensors × 8 features = 16
    if args.use_feat_seq:
        args.enc_in = 16

    # Print key settings
    print("=" * 60)
    print(f"  Task:    {args.task_name}")
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {args.dataset_name}  ({args.raw_data_path})")
    print(f"  Window:  L={args.ts_window_len}  offset={args.ts_window_offset}")
    print(f"  d_model: {args.d_model}  e_layers: {args.e_layers}")
    print(f"  GPU:     {'cuda:' + str(args.gpu) if args.use_gpu else 'cpu'}")
    print("=" * 60)

    if args.is_training:
        for ii in range(args.itr):
            setting = "ts_rul_{model_id}_{model}_L{L}_dm{dm}_el{el}_{ii}_{des}".format(
                model_id=args.model_id,
                model=args.model,
                L=args.ts_window_len,
                dm=args.d_model,
                el=args.e_layers,
                ii=ii,
                des=args.des,
            )

            exp = Exp_TS_RUL(args)
            print(f"\n>>> Training: {setting}")
            exp.train(setting)

            print(f"\n>>> Testing:  {setting}")
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        # Test-only mode: load checkpoint and evaluate
        ii = 0
        setting = "ts_rul_{model_id}_{model}_L{L}_dm{dm}_el{el}_{ii}_{des}".format(
            model_id=args.model_id,
            model=args.model,
            L=args.ts_window_len,
            dm=args.d_model,
            el=args.e_layers,
            ii=ii,
            des=args.des,
        )
        exp = Exp_TS_RUL(args)
        print(f"\n>>> Testing (test-only mode): {setting}")
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
