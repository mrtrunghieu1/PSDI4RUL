# utils/denoising.py
"""
IENEMD-ATD Denoising for vibration signals.

Reference: Improved Ensemble Noise-assisted Empirical Mode Decomposition
with Adaptive Threshold Denoising.
"""

import numpy as np
from scipy.stats import kurtosis
from dataclasses import dataclass
try:
    from PyEMD import EMD
    HAS_EMD = True
except ImportError:
    HAS_EMD = False

@dataclass
class IENEMDATDConfig:
    """Configuration for IENEMD-ATD denoising."""
    enabled: bool = True
    beta: float = 0.719
    rho95: float = 2.449
    rho99: float = 1.919
    alpha: float = 2.0
    J: int = 5
    top_imfs: int = 5
    eps: float = 1e-12


def _emd_imfs(x: np.ndarray) -> np.ndarray:
    """Extract IMFs using EMD."""
    if not HAS_EMD:
        return np.zeros((0, len(x)))
    try:
        emd = EMD()
        imfs = emd.emd(x)
        if imfs is None or len(imfs) == 0:
            return np.zeros((0, len(x)))
        return np.asarray(imfs, dtype=np.float64)
    except Exception:
        return np.zeros((0, len(x)))


def _noise_energy(imf: np.ndarray, eps: float) -> float:
    """Calculate noise energy of an IMF."""
    med = np.median(np.abs(imf))
    return float((med / 0.6745) ** 2 + eps)


def _extract_inherent_noise(noise_only_imfs: np.ndarray, energies: np.ndarray,
                            signal_length: int, eps: float) -> np.ndarray:
    """Extract inherent noise from noise-only IMFs."""
    if noise_only_imfs.size == 0:
        return np.zeros((signal_length,), dtype=np.float64)

    N = signal_length
    n_hat = np.zeros((N,), dtype=np.float64)

    for l in range(noise_only_imfs.shape[0]):
        c = noise_only_imfs[l]
        L = len(c)
        E_l = float(energies[l])
        cs = np.sort(c ** 2)
        prefix = np.cumsum(cs)
        k = np.arange(L)

        term1 = L - 2 * (k + 1)
        term3 = (L - (k + 1)) * cs
        rli = (term1 + prefix + term3) / L

        lam = int(np.argmin(rli))
        Gamma = float(np.sqrt(E_l * cs[lam] + eps))

        abs_c = np.abs(c)
        out = np.zeros_like(c)
        mask1 = abs_c <= Gamma
        mask2 = (abs_c > Gamma) & (abs_c <= 2 * Gamma)
        out[mask1] = c[mask1]
        out[mask2] = np.sign(c[mask2]) * (2 * Gamma - abs_c[mask2])

        # Handle length mismatch
        if L == N:
            n_hat += out
        elif L < N:
            n_hat[:L] += out
        else:
            n_hat += out[:N]

    return n_hat


def denoise_signal(x: np.ndarray, cfg: IENEMDATDConfig = None) -> np.ndarray:
    """
    Denoise a signal using IENEMD-ATD method.

    Args:
        x: Input signal
        cfg: IENEMD-ATD configuration (uses default if None)

    Returns:
        Denoised signal as float32
    """
    if cfg is None:
        cfg = IENEMDATDConfig()

    x = np.asarray(x, dtype=np.float64).ravel()
    if not cfg.enabled:
        return x.astype(np.float32)
    if not HAS_EMD:
        raise ImportError(
            "PyEMD is required when denoising is enabled. "
            "Please install it manually (e.g., `pip install EMD-signal`)."
        )
    N = len(x)

    if N < 100:
        return x.astype(np.float32)

    # 1. EMD decomposition
    ck = _emd_imfs(x)
    if ck.shape[0] == 0:
        return x.astype(np.float32)

    # 2. Energy & Confidence Interval
    Ek = np.array([_noise_energy(imf, cfg.eps) for imf in ck])
    if len(Ek) < 2:
        return x.astype(np.float32)

    E1 = Ek[0]
    k_idx = np.arange(1, len(Ek) + 1, dtype=np.float64)
    Ehat95 = E1 * (cfg.rho95 ** (-k_idx / cfg.beta))
    Ehat99 = E1 * (cfg.rho99 ** (-k_idx / cfg.beta))

    # 3. Identify Noise IMFs
    def log2(a):
        return np.log(a + cfg.eps) / np.log(2.0)

    lower = log2(Ehat95) - cfg.alpha
    upper = log2(Ehat99) + cfg.alpha
    val_k = log2(Ek)
    noise_mask = (val_k >= lower) & (val_k <= upper)

    # 4. Extract Noise
    n_hat = _extract_inherent_noise(ck[noise_mask], Ek[noise_mask], N, cfg.eps)

    # 5. Ensemble (Noise Assisted)
    imfs_collection = []
    max_modes = 0
    for j in range(cfg.J):
        sign = 1.0 if (j % 2 == 0) else -1.0
        x_j = x + sign * n_hat
        imfs_j = _emd_imfs(x_j)
        if imfs_j.shape[0] > 0:
            imfs_collection.append(imfs_j)
            max_modes = max(max_modes, imfs_j.shape[0])

    if not imfs_collection:
        return x.astype(np.float32)

    # Average IMFs
    ci = np.zeros((max_modes, N), dtype=np.float64)
    for imfs in imfs_collection:
        n_imfs, imf_len = imfs.shape
        copy_len = min(imf_len, N)
        pad = np.zeros((max_modes, N))
        pad[:n_imfs, :copy_len] = imfs[:, :copy_len]
        ci += pad
    ci /= len(imfs_collection)

    # 6. Adaptive Threshold
    K_limit = min(cfg.top_imfs, ci.shape[0])
    denoised_imfs = np.zeros((K_limit, N), dtype=np.float64)
    term_N = np.sqrt(2.0 * np.log(N))

    for i in range(K_limit):
        c = ci[i]
        mu = np.mean(np.abs(c))
        krt = float(np.abs(kurtosis(c, fisher=False)) + cfg.eps)
        Ti = (mu / krt) * (term_N / 0.6745)
        denoised_imfs[i] = c * (np.abs(c) >= Ti)

    return np.sum(denoised_imfs, axis=0).astype(np.float32)
