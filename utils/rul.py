"""
Piecewise RUL (Remaining Useful Life) utility functions.

RUL is defined as a piecewise linear function:
    t <= FPT      -> RUL = 1.0  (healthy phase)
    FPT < t < EOF -> RUL = (EOF - t) / (EOF - FPT)  (degradation phase)
    t >= EOF      -> RUL = 0.0  (failure)

where FPT = First Prediction Time, EOF = End of Life.
"""
from typing import Dict, Tuple


# ── XJTU-SY Bearing Dataset FPT/EOF ─────────────────────────────────────────
XJTU_FPT_EOF: Dict[str, Tuple[int, int]] = {
    "Bearing1_1": (71, 123),
    "Bearing1_2": (30, 161),
    "Bearing1_3": (58, 158),
    "Bearing1_4": (26, 122),
    "Bearing1_5": (28, 52),
    "Bearing2_1": (451, 491),
    "Bearing2_2": (42, 161),
    "Bearing2_3": (302, 533),
    "Bearing2_4": (30, 42),
    "Bearing2_5": (119, 339),
    "Bearing3_1": (2387, 2528),
    "Bearing3_3": (339, 372),
}

# ── PHM (PRONOSTIA / FEMTO-ST) Bearing Dataset FPT/EOF ──────────────────────
PHM_FPT_EOF: Dict[str, Tuple[int, int]] = {
    # Condition 1 bearings
    "Bearing1_1": (1489, 2739),
    "Bearing1_2": (826, 870),
    "Bearing1_3": (1432, 2288),
    "Bearing1_4": (1083, 1190),
    "Bearing1_5": (2408, 2452),
    "Bearing1_6": (2407, 2447),
    "Bearing1_7": (2203, 2258),
    # Condition 2 bearings
    "Bearing2_1": (874, 910),
    "Bearing2_2": (745, 796),
    "Bearing2_3": (1935, 1954),
    "Bearing2_4": (739, 750),
    "Bearing2_6": (683, 700),
    "Bearing2_7": (220, 229),
    "Bearing3_2": (1585, 1637)
}

_FPT_EOF_TABLES = {
    "xjtu":   XJTU_FPT_EOF,
    "phm":    PHM_FPT_EOF,
    "phm_c5": PHM_FPT_EOF,  # Case-5 uses same FPT/EOF table as PHM
}


def get_fpt_eof(dataset_name: str) -> Dict[str, Tuple[int, int]]:
    """Return the FPT/EOF lookup table for the given dataset."""
    key = dataset_name.lower()
    if key not in _FPT_EOF_TABLES:
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. "
            f"Available: {list(_FPT_EOF_TABLES.keys())}"
        )
    return _FPT_EOF_TABLES[key]


def piecewise_rul(t: int, fpt: int, eof: int) -> float:
    """Compute piecewise linear RUL at timestep t.

    Returns:
        1.0 if t <= fpt (healthy)
        (eof - t) / (eof - fpt) if fpt < t < eof (degradation)
        0.0 if t >= eof (failure)
    """
    if t <= fpt:
        return 1.0
    if t >= eof:
        return 0.0
    return (eof - t) / (eof - fpt)


def compute_delta_rul(t_start: int, t_end: int, fpt: int, eof: int) -> float:
    """Compute delta RUL = max(0, RUL(t_start) - RUL(t_end)).

    Uses piecewise linear RUL. Always >= 0 since RUL is monotonically
    non-increasing.
    """
    return max(0.0, piecewise_rul(t_start, fpt, eof)
               - piecewise_rul(t_end, fpt, eof))


def compute_delta_max(fpt_eof: Dict[str, Tuple[int, int]],
                      window_size: int,
                      delta_factor: float = 3.0) -> float:
    """Compute the maximum possible delta RUL for output bounding.

    Following TakensNet: delta_max = max_degradation_rate × (K-1) × delta_factor

    For piecewise linear RUL, a fully-degradation window of K steps in the
    fastest-degrading bearing has delta = (K-1) / min(EOF-FPT).

    Args:
        fpt_eof: dict of bearing_id -> (fpt, eof)
        window_size: K (number of consecutive timesteps per window)
        delta_factor: safety multiplier (default 3.0)

    Returns:
        delta_max value for sigmoid scaling
    """
    max_step_delta = 0.0
    for _, (fpt, eof) in fpt_eof.items():
        if eof > fpt:
            max_step_delta = max(max_step_delta, 1.0 / (eof - fpt))
    return max_step_delta * (window_size - 1) * delta_factor


def compute_delta_max_from_data(fpt_eof: Dict[str, Tuple[int, int]],
                                window_size: int,
                                bearing_ids: list = None) -> float:
    """Compute delta_max from actual bearing data (no safety factor).

    This computes the actual maximum delta that any window can produce,
    which is (K-1)/min(EOF-FPT) for the fastest-degrading bearing.
    A small headroom (1.2x) is added to avoid saturation.

    Args:
        fpt_eof: dict of bearing_id -> (fpt, eof)
        window_size: K
        bearing_ids: optional list of bearing IDs to consider (e.g., train only)

    Returns:
        delta_max value
    """
    max_window_delta = 0.0
    items = fpt_eof.items() if bearing_ids is None else \
        ((bid, fpt_eof[bid]) for bid in bearing_ids if bid in fpt_eof)
    for _, (fpt, eof) in items:
        if eof > fpt:
            # Maximum delta for a window fully inside degradation phase
            window_delta = (window_size - 1) / (eof - fpt)
            max_window_delta = max(max_window_delta, window_delta)
    # Small headroom to avoid sigmoid saturation at edges
    return max_window_delta * 1.5
