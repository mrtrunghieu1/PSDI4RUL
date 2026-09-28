"""
Visualization: Vibration Time Series + STFT/Wavelet Images across Health/Degradation/Failure states.
Supports:
  1) Raw XJTU-SY CSVs (legacy behaviour)
  2) Preprocessed .npz datasets with keys X/P/meta (e.g. PHM case2 STFT)

Reproduces the figure style:
  - Top panel: full vibration waveform with 3 annotated red-box zones
  - Bottom panel: STFT + Wavelet spectrograms per zone (clear side-by-side comparison)
  - Red arrows connecting each red box to its STFT panel

STFT method mirrors STFTImageGenerator in layers/SpectrogramGenerator.py:
    torch.stft  →  magnitude  →  log1p  →  min-max normalize  →  bilinear resize
Wavelet method mirrors signal_to_wavelet_image in
scripts/data_processor/build_dataset_case2_spectrogram.py:
    pywt.cwt  →  abs  →  log1p  →  min-max normalize  →  bilinear resize

Usage (raw CSV):
    python tools/viz_stft_degradation.py \
        --bearing_dir /path/to/XJTU-SY/35Hz12kN/Bearing1_5 \
        --fpt 25 --eof 52 \
        --output figures/stft_degradation_bearing1_5.png

Usage (preprocessed npz):
    python tools/viz_stft_degradation.py \
        --npz_path data/processed_data/case2_spectrogram/stft/test.npz \
        --bearing Bearing1_5 \
        --channel 0 \
        --output figures/stft_degradation_bearing1_5_case2_from_npz.png
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import scipy.signal as scipy_signal
import torch
import torch.nn.functional as F

# Allow import from project root (for STFTImageGenerator)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
#  STFT/Wavelet transforms
# --------------------------------------------------------------------------- #

def _normalize_map(arr: np.ndarray) -> np.ndarray:
    """Per-sample min-max normalisation to [0, 1].  arr: [H, W]"""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _resize_map(arr: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize of a 2-D float map to (size, size).  arr: [H, W]"""
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
    t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()                                     # [size, size]


def compute_stft_image(signal: np.ndarray,
                       n_fft: int = 256,
                       hop_length: int = None,
                       image_size: int = 224,
                       log_scale: bool = True,
                       normalize: bool = True) -> np.ndarray:
    """
    Convert a 1-D vibration signal to an STFT spectrogram image.

    log_scale  : apply log1p compression (default True)
    normalize  : apply per-image min-max normalisation to [0,1] (default True)

    Returns
    -------
    np.ndarray, shape (image_size, image_size), float32
    """
    if hop_length is None:
        hop_length = n_fft // 4

    window = torch.hann_window(n_fft)
    t = torch.from_numpy(signal.astype(np.float32))

    stft_out = torch.stft(
        t,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
        pad_mode='reflect',
        center=True,
    )
    magnitude = stft_out.abs().numpy()          # [freq_bins, time_frames]

    if log_scale:
        magnitude = np.log1p(magnitude)

    if normalize:
        magnitude = _normalize_map(magnitude)

    magnitude = _resize_map(magnitude, image_size)
    return magnitude


def compute_wavelet_image(signal: np.ndarray,
                          scales: np.ndarray,
                          wavelet: str = 'cmor1.5-1.0',
                          image_size: int = 224,
                          log_scale: bool = True,
                          normalize: bool = True,
                          sampling_period: float = 1.0) -> np.ndarray:
    """
    Convert a 1-D vibration signal to a Wavelet (CWT) scalogram image.

    Mirrors scripts/data_processor/build_dataset_case2_spectrogram.py:
    pywt.cwt -> abs -> log1p -> min-max -> resize.
    """
    try:
        import pywt
        coeffs, _ = pywt.cwt(
            signal.astype(np.float64),
            scales=scales,
            wavelet=wavelet,
            sampling_period=sampling_period,
        )
    except Exception:
        # Fallback: approximate CWT with Complex Morlet via FFT convolution.
        # This keeps the visual workflow usable when pywt is unavailable/incompatible.
        x = signal.astype(np.float64)
        coeffs = np.empty((len(scales), len(x)), dtype=np.complex64)
        w0 = 6.0
        for i, scale in enumerate(scales):
            sigma = max(float(scale), 1.0)
            half_width = max(16, int(round(4.0 * sigma)))
            t = np.arange(-half_width, half_width + 1, dtype=np.float64)
            wave = (np.pi ** -0.25) * np.exp(1j * w0 * t / sigma) * np.exp(-(t ** 2) / (2.0 * sigma ** 2))
            wave = wave / np.sqrt(sigma)
            coeffs[i] = scipy_signal.fftconvolve(x, np.conj(wave[::-1]), mode='same').astype(np.complex64)

    magnitude = np.abs(coeffs).astype(np.float32)

    if log_scale:
        magnitude = np.log1p(magnitude)

    if normalize:
        magnitude = _normalize_map(magnitude)

    magnitude = _resize_map(magnitude, image_size)
    return magnitude


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #

def load_xjtu_bearing(bearing_dir: str):
    """
    Load all CSVs (sorted numerically) from an XJTU-SY bearing directory.

    Returns
    -------
    signal   : 1-D float32 array — horizontal vibration concatenated across files
    n_files  : number of measurement files
    """
    files = sorted(
        [f for f in os.listdir(bearing_dir) if f.endswith('.csv')],
        key=lambda f: int(os.path.splitext(f)[0])
    )
    segments = []
    for fname in files:
        fpath = os.path.join(bearing_dir, fname)
        df = pd.read_csv(fpath, header=0)
        segments.append(df.iloc[:, 0].values.astype(np.float32))

    return np.concatenate(segments), len(files)


def load_npz_bearing(npz_path: str, bearing: str, channel: int):
    """
    Load one bearing sequence from preprocessed npz file.

    npz keys:
      X    [N, 2, H, W]
      P    [N, 2, L]
      meta [N, 3] = [bearing_name, t_idx, total_files]
    """
    data = np.load(npz_path, allow_pickle=True)
    if not all(k in data for k in ('X', 'P', 'meta')):
        raise KeyError(f"{npz_path} must contain X, P, meta")

    X = data['X']
    P = data['P']
    meta = data['meta']

    if X.ndim != 4 or P.ndim != 3:
        raise ValueError(f"Unexpected shapes: X={X.shape}, P={P.shape}")
    if channel < 0 or channel >= X.shape[1]:
        raise ValueError(f"channel={channel} out of range [0, {X.shape[1]-1}]")

    keep = [i for i, m in enumerate(meta) if str(m[0]) == bearing]
    if not keep:
        raise ValueError(f"Bearing '{bearing}' not found in {npz_path}")

    # Sort by t_idx to recover temporal order
    keep = sorted(keep, key=lambda i: int(meta[i][1]))

    X_b = X[keep]  # [n_files, 2, H, W]
    P_b = P[keep]  # [n_files, 2, L]
    t_idx = np.array([int(meta[i][1]) for i in keep], dtype=np.int64)
    n_files = X_b.shape[0]

    signal = np.concatenate([P_b[i, channel].astype(np.float32) for i in range(n_files)])
    return X_b, P_b, t_idx, signal, n_files


# --------------------------------------------------------------------------- #
#  Zone segment selection — pick 1 representative segment per zone
# --------------------------------------------------------------------------- #

def pick_zone_representative(signal: np.ndarray, n_files: int,
                              fpt_file: int, eof_file: int):
    """
    Pick one representative file-segment from each zone (health / degradation / failure).

    Returns
    -------
    dict: zone_name -> 1-D np.ndarray (one file's worth of samples)
    samples_per_file : int
    """
    samples_per_file = len(signal) // n_files

    # Representative file indices
    health_fi      = fpt_file // 2                           # middle of healthy zone
    degradation_fi = (fpt_file + eof_file) // 2             # middle of degradation zone
    failure_fi     = max(eof_file - 2, fpt_file + 1)        # near end (failure)

    def get_seg(fi):
        s = fi * samples_per_file
        e = s + samples_per_file
        return signal[s:e]

    return {
        'health':      get_seg(health_fi),
        'degradation': get_seg(degradation_fi),
        'failure':     get_seg(failure_fi),
    }, samples_per_file


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def pick_zone_indices_by_position(n_files: int,
                                  health_pos: float,
                                  degradation_pos: float,
                                  failure_pos: float):
    """
    Pick representative file indices by life-position fractions in [0,1].
    """
    if n_files < 3:
        raise ValueError(f"Need at least 3 timesteps, got {n_files}")

    positions = {
        'health': _clip01(health_pos),
        'degradation': _clip01(degradation_pos),
        'failure': _clip01(failure_pos),
    }
    idxs = {
        z: int(round(p * (n_files - 1))) for z, p in positions.items()
    }
    if not (idxs['health'] < idxs['degradation'] < idxs['failure']):
        raise ValueError(
            "Need health_pos < degradation_pos < failure_pos "
            f"(got indices {idxs})"
        )
    return idxs


# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #

CMAP_STFT = 'magma'
CMAP_WAVELET = 'viridis'

ZONE_LABELS = {
    'health':      'Health state',
    'degradation': 'Degradation state',
    'failure':     'Failure state',
}

ZONE_ORDER = ['health', 'degradation', 'failure']


# --------------------------------------------------------------------------- #
#  Main figure
# --------------------------------------------------------------------------- #

def make_figure(signal_display: np.ndarray,
                stft_by_zone: dict,
                wavelet_by_zone: dict | None,
                fpt_ratio: float,
                eof_ratio: float,
                output_path: str,
                n_files: int,
                bearing_name: str | None = None,
                stft_image_size: int = 224,
                log_scale: bool = True,
                normalize: bool = True,
                zone_center_ratios: dict | None = None,
                show_boundaries: bool = True,
                caption_override: str | None = None):

    has_wavelet = wavelet_by_zone is not None
    fig = plt.figure(figsize=(15.5, 8.0 if has_wavelet else 6.5))
    fig.patch.set_facecolor('white')

    # ------------------------------------------------------------------ #
    #  TOP: waveform
    # ------------------------------------------------------------------ #
    if has_wavelet:
        ax_w = fig.add_axes([0.06, 0.60, 0.90, 0.33])
        axb_rect = [0.0, 0.0, 1.0, 0.58]
    else:
        ax_w = fig.add_axes([0.06, 0.55, 0.90, 0.38])
        axb_rect = [0.0, 0.0, 1.0, 0.52]

    ax_w.plot(signal_display, color='#1A56B0', linewidth=0.35, alpha=0.9)
    ax_w.set_xlim(0, len(signal_display))
    ax_w.set_ylabel('Amplitude (g)', fontsize=10)
    title = 'Vibration time series'
    if bearing_name:
        title += f' | {bearing_name}'
    ax_w.set_title(title, fontsize=13,
                   fontweight='bold', pad=6)
    ax_w.tick_params(axis='x', labelbottom=False)
    ax_w.spines['top'].set_visible(False)
    ax_w.spines['right'].set_visible(False)

    N = len(signal_display)
    fpt_x = int(fpt_ratio * N)
    eof_x = int(eof_ratio * N)

    if show_boundaries:
        ax_w.axvline(fpt_x, color='#888888', lw=1.2, ls='--', alpha=0.6)
        ax_w.axvline(eof_x, color='#888888', lw=1.2, ls='--', alpha=0.6)

    ylim = ax_w.get_ylim()
    box_h_data = (ylim[1] - ylim[0]) * 0.96
    box_y_data = ylim[0] + (ylim[1] - ylim[0]) * 0.02
    box_w_frac = 0.07          # fraction of total signal length
    box_w_data = int(box_w_frac * N)

    # Red box center positions (in display sample coords)
    if zone_center_ratios is None:
        box_centers = {
            'health':      int(0.18 * fpt_x),
            'degradation': int((fpt_x + eof_x) / 2),
            'failure':     int(eof_x + 0.5 * (N - eof_x)),
        }
        label_x = {
            'health':      fpt_x * 0.35,
            'degradation': (fpt_x + eof_x) / 2,
            'failure':     eof_x + (N - eof_x) * 0.50,
        }
    else:
        box_centers = {
            zone: int(_clip01(zone_center_ratios[zone]) * N) for zone in ZONE_ORDER
        }
        label_x = {
            zone: int(np.clip(cx, 0.10 * N, 0.88 * N))
            for zone, cx in box_centers.items()
        }

    for zone, cx in box_centers.items():
        bx = cx - box_w_data // 2
        rect = mpatches.FancyBboxPatch(
            (bx, box_y_data), box_w_data, box_h_data,
            boxstyle='square,pad=0',
            linewidth=2.0, edgecolor='red', facecolor='none', zorder=5
        )
        ax_w.add_patch(rect)

    # Zone labels
    label_y = ylim[0] + (ylim[1] - ylim[0]) * 0.85
    for zone, lx in label_x.items():
        ax_w.text(lx, label_y, ZONE_LABELS[zone],
                  fontsize=9.5, ha='center', va='top',
                  fontweight='bold',
                  bbox=dict(facecolor='white', edgecolor='none',
                            alpha=0.75, pad=1.5))

    # ------------------------------------------------------------------ #
    #  BOTTOM: spectrogram panels
    # ------------------------------------------------------------------ #
    ax_b = fig.add_axes(axb_rect)
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis('off')

    if has_wavelet:
        tile_w, tile_h = 0.18, 0.33
        stft_y, wav_y = 0.55, 0.14
        tile_configs = {
            'health':      (0.07,  stft_y),
            'degradation': (0.39,  stft_y),
            'failure':     (0.71,  stft_y),
        }

        for zone in ZONE_ORDER:
            ox, oy = tile_configs[zone]

            stft_img = stft_by_zone[zone]
            wav_img = wavelet_by_zone[zone]

            inset_stft = ax_b.inset_axes([ox, oy, tile_w, tile_h], transform=ax_b.transAxes)
            inset_stft.imshow(stft_img, cmap=CMAP_STFT, aspect='auto',
                              origin='lower', vmin=0, vmax=1)
            inset_stft.set_xticks([])
            inset_stft.set_yticks([])
            for spine in inset_stft.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(2.0)
                spine.set_visible(True)

            inset_wav = ax_b.inset_axes([ox, wav_y, tile_w, tile_h], transform=ax_b.transAxes)
            inset_wav.imshow(wav_img, cmap=CMAP_WAVELET, aspect='auto',
                             origin='lower', vmin=0, vmax=1)
            inset_wav.set_xticks([])
            inset_wav.set_yticks([])
            for spine in inset_wav.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(2.0)
                spine.set_visible(True)

            label_cx = ox + tile_w / 2
            ax_b.text(label_cx, wav_y - 0.05, ZONE_LABELS[zone],
                      ha='center', va='top', fontsize=10, fontweight='bold',
                      color='#222222', transform=ax_b.transAxes)

        ax_b.text(0.02, stft_y + tile_h / 2, 'STFT',
                  ha='left', va='center', fontsize=11, fontweight='bold',
                  color='#222222', transform=ax_b.transAxes)
        ax_b.text(0.02, wav_y + tile_h / 2, 'Wavelet',
                  ha='left', va='center', fontsize=11, fontweight='bold',
                  color='#222222', transform=ax_b.transAxes)

        cbar_stft_ax = fig.add_axes([0.955, 0.30, 0.012, 0.24])
        sm_stft = plt.cm.ScalarMappable(cmap=CMAP_STFT, norm=plt.Normalize(vmin=0, vmax=1))
        sm_stft.set_array([])
        cbar_stft = fig.colorbar(sm_stft, cax=cbar_stft_ax)
        cbar_stft.set_label('STFT\nnorm mag', fontsize=8, labelpad=4)
        cbar_stft.ax.tick_params(labelsize=7)

        cbar_wav_ax = fig.add_axes([0.955, 0.04, 0.012, 0.24])
        sm_wav = plt.cm.ScalarMappable(cmap=CMAP_WAVELET, norm=plt.Normalize(vmin=0, vmax=1))
        sm_wav.set_array([])
        cbar_wav = fig.colorbar(sm_wav, cax=cbar_wav_ax)
        cbar_wav.set_label('Wavelet\nnorm mag', fontsize=8, labelpad=4)
        cbar_wav.ax.tick_params(labelsize=7)
    else:
        # Position of each STFT panel (left, bottom) in ax_b fraction space
        tile_w, tile_h = 0.20, 0.56
        tile_configs = {
            'health':      (0.07,  0.20),
            'degradation': (0.40,  0.20),
            'failure':     (0.73,  0.20),
        }

        for zone in ZONE_ORDER:
            ox, oy = tile_configs[zone]
            img = stft_by_zone[zone]

            inset = ax_b.inset_axes([ox, oy, tile_w, tile_h], transform=ax_b.transAxes)
            inset.imshow(img, cmap=CMAP_STFT, aspect='auto',
                         origin='lower', vmin=0, vmax=1)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(2.0)
                spine.set_visible(True)

            label_cx = ox + tile_w / 2
            ax_b.text(label_cx, oy - 0.06, ZONE_LABELS[zone],
                      ha='center', va='top', fontsize=10, fontweight='bold',
                      color='#222222', transform=ax_b.transAxes)

        cbar_ax = fig.add_axes([0.955, 0.04, 0.012, 0.44])
        sm = plt.cm.ScalarMappable(cmap=CMAP_STFT, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Normalised\nmagnitude', fontsize=8, labelpad=4)
        cbar.ax.tick_params(labelsize=7)

    # Caption — reflect which processing steps were applied
    if caption_override is not None:
        caption = caption_override
    else:
        steps = []
        if log_scale:
            steps.append('log1p')
        if normalize:
            steps.append('per-image min-max')
        if not steps:
            steps.append('raw magnitude')
        if has_wavelet:
            caption = 'STFT + Wavelet spectrograms (' + ' · '.join(steps) + ')'
        else:
            caption = 'STFT spectrograms (' + ' · '.join(steps) + ')'
    ax_b.text(0.48, 0.03, caption,
              ha='center', va='bottom', fontsize=10,
              fontstyle='italic', color='#333333',
              transform=ax_b.transAxes)


    plt.savefig(output_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved → {output_path}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--bearing_dir',
        default='/home/ktp_user/Desktop/PhD-Code/TakensNet/data/XJTU-SY/35Hz12kN/Bearing1_5',
        help='Path to XJTU-SY bearing directory (numbered CSVs)')
    parser.add_argument('--npz_path', type=str, default=None,
                        help='If set, load preprocessed npz (X/P/meta) instead of raw CSVs.')
    parser.add_argument('--bearing', type=str, default='Bearing1_5',
                        help='Bearing name inside npz mode (e.g., Bearing1_5).')
    parser.add_argument('--channel', type=int, default=0,
                        help='Channel index for waveform/STFT in npz mode (default 0).')
    parser.add_argument('--health_pos', type=float, default=0.10,
                        help='Life-position for health zone in [0,1] (npz mode).')
    parser.add_argument('--degradation_pos', type=float, default=0.55,
                        help='Life-position for degradation zone in [0,1] (npz mode).')
    parser.add_argument('--failure_pos', type=float, default=0.92,
                        help='Life-position for failure zone in [0,1] (npz mode).')
    parser.add_argument('--fpt', type=int, default=25,
                        help='FPT file index (0-based); default 25 for Bearing1_5')
    parser.add_argument('--eof', type=int, default=52,
                        help='Total file count / EOF index; default 52 for Bearing1_5')
    parser.add_argument('--output', default='figures/stft_degradation_bearing1_5.png')
    parser.add_argument('--subsample', type=int, default=32,
                        help='Downsample factor for waveform display only')
    parser.add_argument('--n_fft', type=int, default=256,
                        help='STFT FFT window length (default 256, same as preprocess_spectrogram.py)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='STFT output image resolution (default 224)')
    parser.add_argument('--no_log', action='store_true',
                        help='Skip log1p compression')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip min-max normalisation')
    parser.add_argument('--wavelet_num_scales', type=int, default=128,
                        help='Number of wavelet scales (default 128).')
    parser.add_argument('--wavelet_scale_min', type=int, default=1,
                        help='Minimum wavelet scale (default 1).')
    parser.add_argument('--wavelet_name', type=str, default='cmor1.5-1.0',
                        help='PyWavelets wavelet name (default cmor1.5-1.0).')
    parser.add_argument('--wavelet_sampling_period', type=float, default=1.0,
                        help='Wavelet sampling period (default 1.0).')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    log_scale = not args.no_log
    normalize = not args.no_normalize
    wavelet_scales = np.arange(
        args.wavelet_scale_min,
        args.wavelet_scale_min + args.wavelet_num_scales,
        dtype=np.float64,
    )

    if args.npz_path:
        print(f"Loading npz: {args.npz_path}")
        X_b, P_b, t_idx, signal, n_files = load_npz_bearing(
            args.npz_path, args.bearing, args.channel
        )
        print(f"  bearing={args.bearing}, channel={args.channel}, timesteps={n_files}")

        zone_idx = pick_zone_indices_by_position(
            n_files=n_files,
            health_pos=args.health_pos,
            degradation_pos=args.degradation_pos,
            failure_pos=args.failure_pos,
        )
        print("  selected timestep indices:", zone_idx)

        stft_by_zone = {
            zone: X_b[idx, args.channel].astype(np.float32)
            for zone, idx in zone_idx.items()
        }
        wavelet_by_zone = {}
        # X already in [0,1] from preprocessing (log1p + per-image min-max).
        for zone in ZONE_ORDER:
            img = stft_by_zone[zone]
            stft_by_zone[zone] = np.clip(img, 0, 1)
            wavelet_by_zone[zone] = compute_wavelet_image(
                signal=P_b[zone_idx[zone], args.channel],
                scales=wavelet_scales,
                wavelet=args.wavelet_name,
                image_size=args.image_size,
                log_scale=log_scale,
                normalize=normalize,
                sampling_period=args.wavelet_sampling_period,
            )

        center_ratios = {z: zone_idx[z] / max(n_files - 1, 1) for z in ZONE_ORDER}
        fpt_ratio = 0.5 * (center_ratios['health'] + center_ratios['degradation'])
        eof_ratio = 0.5 * (center_ratios['degradation'] + center_ratios['failure'])
        signal_display = signal[::args.subsample]

        print("Rendering figure from preprocessed signals (STFT + Wavelet)...")
        make_figure(
            signal_display=signal_display,
            stft_by_zone=stft_by_zone,
            wavelet_by_zone=wavelet_by_zone,
            fpt_ratio=fpt_ratio,
            eof_ratio=eof_ratio,
            output_path=args.output,
            n_files=n_files,
            bearing_name=f"{args.bearing} (Ch-{args.channel})",
            stft_image_size=args.image_size,
            log_scale=True,
            normalize=True,
            zone_center_ratios=center_ratios,
            show_boundaries=False,
            caption_override='STFT + Wavelet from preprocessed npz (log1p · per-image min-max)',
        )
    else:
        print(f"Loading: {args.bearing_dir}")
        signal, n_files = load_xjtu_bearing(args.bearing_dir)
        bearing_name = os.path.basename(os.path.normpath(args.bearing_dir))
        print(f"  {n_files} files, {len(signal):,} total samples")

        fpt_ratio = args.fpt / n_files
        eof_ratio = min(args.eof, n_files) / n_files

        zone_segs, _ = pick_zone_representative(
            signal, n_files, args.fpt, min(args.eof, n_files)
        )

        print(f"Computing STFT + Wavelet images "
              f"(n_fft={args.n_fft}, image_size={args.image_size}, "
              f"log1p={log_scale}, normalize={normalize}, mode=per-image, "
              f"wavelet={args.wavelet_name}, scales={args.wavelet_num_scales})...")
        stft_by_zone = {}
        wavelet_by_zone = {}
        for zone, seg in zone_segs.items():
            stft_img = compute_stft_image(
                seg, n_fft=args.n_fft, image_size=args.image_size,
                log_scale=log_scale, normalize=normalize
            )
            wav_img = compute_wavelet_image(
                signal=seg,
                scales=wavelet_scales,
                wavelet=args.wavelet_name,
                image_size=args.image_size,
                log_scale=log_scale,
                normalize=normalize,
                sampling_period=args.wavelet_sampling_period,
            )
            stft_by_zone[zone] = stft_img
            wavelet_by_zone[zone] = wav_img
            print(f"  [{zone}]  segment shape={seg.shape}  →  STFT {stft_img.shape}, Wavelet {wav_img.shape}")

        signal_display = signal[::args.subsample]

        print("Rendering figure...")
        make_figure(
            signal_display=signal_display,
            stft_by_zone=stft_by_zone,
            wavelet_by_zone=wavelet_by_zone,
            fpt_ratio=fpt_ratio,
            eof_ratio=eof_ratio,
            output_path=args.output,
            n_files=n_files,
            bearing_name=bearing_name,
            stft_image_size=args.image_size,
            log_scale=log_scale,
            normalize=normalize,
        )


if __name__ == '__main__':
    main()
