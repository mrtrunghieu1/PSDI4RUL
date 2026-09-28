"""
SpectrogramGenerator.py

Offline image generators for bearing vibration signals.
Produces STFT spectrogram and CWT scalogram images in the same
[N, 2, H, W] format as PSDI datasets so the same model pipeline
can be used without modification.

Usage (see scripts/preprocess_spectrogram.py):
    from layers.SpectrogramGenerator import STFTImageGenerator, WaveletImageGenerator

    gen = STFTImageGenerator(image_size=224, n_fft=256, hop_length=64)
    X_stft = gen.generate(P)   # P: [N, 2, 2560] -> X_stft: [N, 2, 224, 224]
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_map(arr: np.ndarray) -> np.ndarray:
    """
    Per-sample min-max normalisation to [0, 1].

    arr: [H, W]  (a single 2-D map)
    """
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _resize_map(arr: np.ndarray, size: int) -> np.ndarray:
    """
    Bilinear resize of a single 2-D float map to (size, size).

    arr : [H, W]
    out : [size, size]
    """
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()                              # [size, size]


# ---------------------------------------------------------------------------
# STFT Generator
# ---------------------------------------------------------------------------

class STFTImageGenerator:
    """
    Convert raw vibration signals to STFT magnitude spectrograms.

    Parameters
    ----------
    image_size : int
        Output spatial resolution (default 224 → 224×224).
    n_fft : int
        FFT window length. Controls frequency resolution.
        Typical: 128, 256 (default), 512.
    hop_length : int
        Hop size between windows. Controls time resolution.
        Default: n_fft // 4.
    window : str
        Window function. 'hann' (default) or 'hamming'.
    log_scale : bool
        Apply log1p compression to magnitude (recommended).

    Notes
    -----
    Input  P : [N, 2, signal_len]   (2 vibration channels)
    Output X : [N, 2, image_size, image_size]  float32 in [0, 1]

    Each channel is processed independently:
        signal  →  torch.stft  →  magnitude  →  log1p
                →  min-max normalise  →  bilinear resize  →  channel image
    The two channel images are stacked along dim-1, matching PSDI format.
    """

    def __init__(
        self,
        image_size: int = 224,
        n_fft: int = 256,
        hop_length: int = None,
        window: str = 'hann',
        log_scale: bool = True,
    ):
        self.image_size = image_size
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.log_scale = log_scale

        if window == 'hann':
            self._window = torch.hann_window(n_fft)
        elif window == 'hamming':
            self._window = torch.hamming_window(n_fft)
        else:
            raise ValueError(f"Unsupported window '{window}'. Choose 'hann' or 'hamming'.")

    def _signal_to_map(self, signal: np.ndarray) -> np.ndarray:
        """
        Single 1-D signal  →  normalised [H, W] spectrogram map.

        signal : [signal_len]
        """
        t = torch.from_numpy(signal.astype(np.float32))

        # torch.stft output: [freq_bins, time_frames, 2]  (real, imag)
        stft_out = torch.stft(
            t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self._window,
            return_complex=True,         # -> [freq_bins, time_frames] complex
            pad_mode='reflect',
            center=True,
        )
        magnitude = stft_out.abs().numpy()   # [freq_bins, time_frames]

        if self.log_scale:
            magnitude = np.log1p(magnitude)

        magnitude = _normalize_map(magnitude)
        magnitude = _resize_map(magnitude, self.image_size)
        return magnitude

    def generate(self, P: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        P : np.ndarray, shape [N, 2, signal_len]
            Raw vibration signals (2 sensor channels per sample).

        Returns
        -------
        X : np.ndarray, shape [N, 2, image_size, image_size], dtype float32
        """
        N, C, L = P.shape
        assert C == 2, f"Expected 2 channels, got {C}"

        X = np.zeros((N, 2, self.image_size, self.image_size), dtype=np.float32)
        for i in range(N):
            for c in range(2):
                X[i, c] = self._signal_to_map(P[i, c])
        return X


# ---------------------------------------------------------------------------
# CWT Wavelet Generator
# ---------------------------------------------------------------------------

class WaveletImageGenerator:
    """
    Convert raw vibration signals to CWT (Complex Morlet) scalograms.

    Parameters
    ----------
    image_size : int
        Output spatial resolution (default 224 → 224×224).
    num_scales : int
        Number of wavelet scales (frequency bands).  Default 128.
        Range: [scale_min, scale_min + num_scales - 1].
    scale_min : int
        Smallest scale (highest pseudo-frequency). Default 1.
    wavelet : str
        PyWavelets CWT wavelet name.
        'cmor1.5-1.0' (default)  – Complex Morlet, good for rotating machinery.
        'morl'                   – Real Morlet (faster, slightly less accurate).
        'mexh'                   – Mexican Hat, sensitive to transients.
    log_scale : bool
        Apply log1p compression to magnitude (recommended).
    sampling_period : float
        Sampling period in seconds (1 / sampling_rate).
        Passed to pywt.cwt for proper scale-to-frequency mapping.
        Default 1.0 (scales treated as dimensionless).

    Notes
    -----
    Requires:  pip install PyWavelets

    Input  P : [N, 2, signal_len]
    Output X : [N, 2, image_size, image_size]  float32 in [0, 1]
    """

    def __init__(
        self,
        image_size: int = 224,
        num_scales: int = 128,
        scale_min: int = 1,
        wavelet: str = 'cmor1.5-1.0',
        log_scale: bool = True,
        sampling_period: float = 1.0,
    ):
        self.image_size = image_size
        self.wavelet = wavelet
        self.log_scale = log_scale
        self.sampling_period = sampling_period
        self.scales = np.arange(scale_min, scale_min + num_scales, dtype=np.float64)

        try:
            import pywt
            self._pywt = pywt
        except ImportError:
            raise ImportError(
                "PyWavelets is required for WaveletImageGenerator.\n"
                "Install with:  pip install PyWavelets"
            )

    def _signal_to_map(self, signal: np.ndarray) -> np.ndarray:
        """
        Single 1-D signal  →  normalised [H, W] scalogram map.

        signal : [signal_len]
        """
        # coefficients: [num_scales, signal_len]  complex (for cmor) or real
        coefficients, _ = self._pywt.cwt(
            signal.astype(np.float64),
            scales=self.scales,
            wavelet=self.wavelet,
            sampling_period=self.sampling_period,
        )
        magnitude = np.abs(coefficients).astype(np.float32)  # [num_scales, signal_len]

        if self.log_scale:
            magnitude = np.log1p(magnitude)

        magnitude = _normalize_map(magnitude)
        magnitude = _resize_map(magnitude, self.image_size)
        return magnitude

    def generate(self, P: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        P : np.ndarray, shape [N, 2, signal_len]
            Raw vibration signals (2 sensor channels per sample).

        Returns
        -------
        X : np.ndarray, shape [N, 2, image_size, image_size], dtype float32
        """
        N, C, L = P.shape
        assert C == 2, f"Expected 2 channels, got {C}"

        X = np.zeros((N, 2, self.image_size, self.image_size), dtype=np.float32)
        for i in range(N):
            for c in range(2):
                X[i, c] = self._signal_to_map(P[i, c])
        return X
