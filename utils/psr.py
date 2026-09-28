# proposed_method/representation/psr.py
"""
Phase Space Reconstruction (PSR) methods.

Includes:
- Mutual Information method for optimal time delay (tau)
- Cao's method for optimal embedding dimension (m)
- Delay embedding
- Density image generation
"""

import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PSRConfig:
    """Configuration for Phase Space Reconstruction."""
    tau_max: int = 60
    mi_bins: int = 64
    m_max: int = 15
    e1_range: Tuple[float, float] = (0.95, 1.05)
    eps: float = 1e-12


def mutual_information_tau(x: np.ndarray, cfg: PSRConfig = None) -> int:
    """
    Find optimal time delay tau using Mutual Information method.

    Args:
        x: Input signal
        cfg: PSR configuration

    Returns:
        Optimal time delay tau
    """
    if cfg is None:
        cfg = PSRConfig()

    x = np.asarray(x, dtype=np.float64).ravel()
    x = (x - np.mean(x)) / (np.std(x) + cfg.eps)

    I = []
    for tau in range(1, cfg.tau_max + 1):
        s1, s2 = x[:-tau], x[tau:]
        H, _, _ = np.histogram2d(s1, s2, bins=cfg.mi_bins)
        Pxy = H / (np.sum(H) + cfg.eps)
        Px = np.sum(Pxy, axis=1)
        Py = np.sum(Pxy, axis=0)
        P_indep = Px[:, None] * Py[None, :]
        nz_mask = (Pxy > 0) & (P_indep > 0)
        mi_val = np.sum(Pxy[nz_mask] * np.log(Pxy[nz_mask] / P_indep[nz_mask]))
        I.append(mi_val)

    I = np.array(I)
    # Find first local minimum
    for i in range(1, len(I) - 1):
        if I[i - 1] > I[i] and I[i] < I[i + 1]:
            return i + 1
    return int(np.argmin(I) + 1)


def delay_embed(x: np.ndarray, tau: int, m: int) -> np.ndarray:
    """
    Create delay embedding of signal.

    Args:
        x: Input signal
        tau: Time delay
        m: Embedding dimension

    Returns:
        Embedded signal of shape (n_points, m)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x) - (m - 1) * tau
    if n <= 10:
        return np.zeros((0, m))
    out = np.empty((n, m), dtype=np.float64)
    for j in range(m):
        out[:, j] = x[j * tau: j * tau + n]
    return out


def cao_embedding_dimension(x: np.ndarray, tau: int, cfg: PSRConfig = None) -> int:
    """
    Find optimal embedding dimension m using Cao's method.

    Args:
        x: Input signal
        tau: Time delay
        cfg: PSR configuration

    Returns:
        Optimal embedding dimension m
    """
    if cfg is None:
        cfg = PSRConfig()

    x = np.asarray(x, dtype=np.float64).ravel()
    x = (x - np.mean(x)) / (np.std(x) + cfg.eps)
    E = {}

    for d in range(1, cfg.m_max + 2):
        Yd = delay_embed(x, tau, d)
        if Yd.shape[0] < 100:
            E[d] = np.nan
            continue

        # Sampling if too large
        query_pts = Yd
        if len(Yd) > 1000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(Yd), 1000, replace=False)
            query_pts = Yd[idx]
        else:
            idx = np.arange(len(Yd))

        tree = cKDTree(Yd)
        dists, nn_indices = tree.query(query_pts, k=2)
        Rd = dists[:, 1]
        nn_idx = nn_indices[:, 1]

        # Check boundaries for d+1 dimension
        valid_mask = (idx + d * tau < len(x)) & (nn_idx + d * tau < len(x))
        if np.sum(valid_mask) == 0:
            E[d] = np.nan
            continue

        idx_v = idx[valid_mask]
        nn_v = nn_idx[valid_mask]
        Rd_v = Rd[valid_mask] + cfg.eps

        dist_increment = np.abs(x[idx_v + d * tau] - x[nn_v + d * tau])
        Rd1 = np.sqrt(Rd_v ** 2 + dist_increment ** 2)
        a = Rd1 / Rd_v
        E[d] = np.mean(a)

    for d in range(1, cfg.m_max + 1):
        if d not in E or d + 1 not in E or np.isnan(E[d]) or np.isnan(E[d + 1]):
            continue
        E1 = E[d + 1] / (E[d] + cfg.eps)
        if cfg.e1_range[0] <= E1 <= cfg.e1_range[1]:
            return d

    return 5  # Fallback


def points_to_density_image(z2: np.ndarray, bins: int,
                            xlim: Tuple[float, float],
                            ylim: Tuple[float, float]) -> np.ndarray:
    """
    Convert 2D points to density image.

    Args:
        z2: 2D points of shape (n, 2)
        bins: Number of bins for histogram
        xlim: X-axis limits (min, max)
        ylim: Y-axis limits (min, max)

    Returns:
        Normalized density image of shape (bins, bins)
    """
    H, _, _ = np.histogram2d(z2[:, 0], z2[:, 1], bins=bins, range=[xlim, ylim])
    H = np.log1p(H.astype(np.float32))
    denom = H.max() - H.min()
    return (H - H.min()) / denom if denom > 1e-6 else np.zeros_like(H)
