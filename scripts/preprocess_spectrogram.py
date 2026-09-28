"""
preprocess_spectrogram.py

Offline preprocessing script: reads existing PSDI .npz files and generates
new .npz files containing STFT spectrogram or CWT wavelet scalogram images
in the same format, so the same model training pipeline can be used unchanged.

Input  .npz keys:  X [N, 2, H, W], P [N, 2, 2560], meta [N, 3]
Output .npz keys:  X [N, 2, H, W], P [N, 2, 2560], meta [N, 3]
                   (only X is replaced; P and meta are copied as-is)

Output file naming:
    <split>.npz  ->  <split>_stft.npz  or  <split>_wavelet.npz

Usage examples:
    # Generate STFT for all XJTU case1 splits
    python scripts/preprocess_spectrogram.py \\
        --mode stft \\
        --input_dirs data/processed_data/case1/denoise_off/image_bins_224 \\
        --image_size 224

    # Generate Wavelet for PHM case2 + case3 (denoise_on + denoise_off, both resolutions)
    python scripts/preprocess_spectrogram.py \\
        --mode wavelet \\
        --input_dirs \\
            data/processed_data/case2/denoise_on/image_bins_224 \\
            data/processed_data/case3/denoise_on/image_bins_224 \\
        --image_size 224

    # Generate both modes for ALL available directories at once
    python scripts/preprocess_spectrogram.py \\
        --mode both \\
        --auto_discover \\
        --image_size 224

    # Dry-run: show what would be processed without writing files
    python scripts/preprocess_spectrogram.py --mode both --auto_discover --dry_run
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from layers.SpectrogramGenerator import STFTImageGenerator, WaveletImageGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPLITS = ['train', 'val', 'test']


def discover_dirs(root: Path) -> list[Path]:
    """Return all leaf directories that contain at least one split .npz file."""
    found = []
    for dirpath in sorted(root.rglob('*')):
        if not dirpath.is_dir():
            continue
        npz_files = [f for f in dirpath.iterdir() if f.suffix == '.npz' and f.stem in SPLITS]
        if npz_files:
            found.append(dirpath)
    return found


def load_npz(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_npz(path: Path, X: np.ndarray, arrays: dict):
    """Save output .npz preserving P and meta from original."""
    np.savez(
        str(path),
        X=X,
        P=arrays['P'],
        meta=arrays['meta'],
    )


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Per-directory processing
# ---------------------------------------------------------------------------

def process_directory(
    input_dir: Path,
    mode: str,
    image_size: int,
    stft_cfg: dict,
    wavelet_cfg: dict,
    dry_run: bool,
    overwrite: bool,
):
    """
    Process all split files in one directory.

    mode : 'stft' | 'wavelet' | 'both'
    """
    generators = {}
    if mode in ('stft', 'both'):
        generators['stft'] = STFTImageGenerator(image_size=image_size, **stft_cfg)
    if mode in ('wavelet', 'both'):
        generators['wavelet'] = WaveletImageGenerator(image_size=image_size, **wavelet_cfg)

    for split in SPLITS:
        src = input_dir / f"{split}.npz"
        if not src.exists():
            continue

        for tag, gen in generators.items():
            dst = input_dir / f"{split}_{tag}.npz"

            if dst.exists() and not overwrite:
                print(f"  [skip] {dst.relative_to(PROJECT_ROOT)}  (already exists, use --overwrite)")
                continue

            if dry_run:
                print(f"  [dry]  {src.name} -> {dst.name}")
                continue

            # Load
            t0 = time.time()
            arrays = load_npz(src)
            P = arrays['P']           # [N, 2, 2560]
            N = P.shape[0]

            print(f"  [{tag.upper()}] {src.relative_to(PROJECT_ROOT)}  N={N} ...", end='', flush=True)

            # Generate images
            X_new = gen.generate(P)   # [N, 2, image_size, image_size]

            # Save
            save_npz(dst, X_new, arrays)
            elapsed = time.time() - t0
            size_mb = dst.stat().st_size / 1e6
            print(f" done  {fmt_time(elapsed)}  {size_mb:.1f} MB  -> {dst.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate STFT / Wavelet spectrogram datasets from PSDI .npz files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- What to generate ---
    parser.add_argument(
        '--mode', required=True, choices=['stft', 'wavelet', 'both'],
        help="Which spectrogram type to generate."
    )

    # --- Input directories ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_dirs', nargs='+', type=Path,
        help="One or more leaf directories containing split .npz files."
    )
    input_group.add_argument(
        '--auto_discover', action='store_true',
        help="Auto-discover all valid directories under data/processed_data/."
    )

    # --- Output options ---
    parser.add_argument(
        '--image_size', type=int, default=224,
        help="Output image resolution (default: 224)."
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help="Re-generate and overwrite existing output files."
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help="Print what would be done without writing any files."
    )

    # --- STFT options ---
    stft = parser.add_argument_group('STFT options')
    stft.add_argument('--stft_n_fft',      type=int,   default=256,    help="FFT window length (default: 256).")
    stft.add_argument('--stft_hop_length',  type=int,   default=None,   help="Hop length (default: n_fft // 4).")
    stft.add_argument('--stft_window',      type=str,   default='hann', choices=['hann', 'hamming'],
                      help="Window function (default: hann).")
    stft.add_argument('--stft_log_scale',   action='store_true', default=True,
                      help="Apply log1p to STFT magnitude (default: True).")

    # --- Wavelet options ---
    wav = parser.add_argument_group('Wavelet (CWT) options')
    wav.add_argument('--wav_num_scales',      type=int,   default=128,          help="Number of CWT scales (default: 128).")
    wav.add_argument('--wav_scale_min',       type=int,   default=1,            help="Smallest CWT scale (default: 1).")
    wav.add_argument('--wav_wavelet',         type=str,   default='cmor1.5-1.0',
                     help="PyWavelets wavelet name (default: cmor1.5-1.0).")
    wav.add_argument('--wav_log_scale',       action='store_true', default=True,
                     help="Apply log1p to CWT magnitude (default: True).")
    wav.add_argument('--wav_sampling_period', type=float, default=1.0,
                     help="Sampling period in seconds (default: 1.0).")

    return parser.parse_args()


def main():
    args = parse_args()

    # Collect input directories
    if args.auto_discover:
        dataset_root = PROJECT_ROOT / 'dataset' / 'processed_psdi_reconstruction'
        input_dirs = discover_dirs(dataset_root)
        if not input_dirs:
            print(f"No valid directories found under {dataset_root}")
            sys.exit(1)
    else:
        input_dirs = []
        for p in args.input_dirs:
            p = p if p.is_absolute() else PROJECT_ROOT / p
            if not p.exists():
                print(f"[WARNING] Directory not found: {p}")
            else:
                input_dirs.append(p)

    if not input_dirs:
        print("No directories to process. Exiting.")
        sys.exit(1)

    # Build generator configs
    stft_cfg = dict(
        n_fft=args.stft_n_fft,
        hop_length=args.stft_hop_length,
        window=args.stft_window,
        log_scale=args.stft_log_scale,
    )
    wavelet_cfg = dict(
        num_scales=args.wav_num_scales,
        scale_min=args.wav_scale_min,
        wavelet=args.wav_wavelet,
        log_scale=args.wav_log_scale,
        sampling_period=args.wav_sampling_period,
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"Mode          : {args.mode.upper()}")
    print(f"Image size    : {args.image_size}x{args.image_size}")
    print(f"Directories   : {len(input_dirs)}")
    if args.dry_run:
        print(f"DRY RUN       : no files will be written")
    print(f"{'='*60}\n")

    total_t0 = time.time()

    for d in input_dirs:
        print(f"[DIR] {d.relative_to(PROJECT_ROOT)}")
        process_directory(
            input_dir=d,
            mode=args.mode,
            image_size=args.image_size,
            stft_cfg=stft_cfg,
            wavelet_cfg=wavelet_cfg,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"All done in {fmt_time(total_elapsed)}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
