"""
Raw Vibration Time-Series Dataset for SOTA TS Model Baselines.

Loads raw vibration CSVs from data/raw_data/XJTU-SY/BearingX_Y/t.csv (XJTU format).
Each CSV: header row + 32768 rows × 2 columns (Horizontal, Vertical vibration).

Three modes (controlled by args.use_rms_seq / args.use_feat_seq):

  use_rms_seq=False, use_feat_seq=False (default, waveform mode):
    One sample = one timestep (one CSV file), window of L raw samples extracted.
    x shape: [2, L=2560]

  use_rms_seq=True (RMS sequence mode):
    One sample = RMS(H), RMS(V) of L consecutive CSV files ending at timestep t.
    Each step = [RMS_H, RMS_V] of one file → x shape: [2, L]
    This matches the TimeMixer design intent: L time steps over the
    operational timeline, allowing season/trend decomp to capture
    degradation trends across time.

  use_feat_seq=True (multi-feature sequence mode):
    One sample = F statistical features × 2 sensors of L consecutive files.
    Features (8 per sensor): RMS, Variance, Peak, Kurtosis, Skewness,
                              Crest Factor, Shape Factor, Impulse Factor
    x shape: [2*F, L]  where F=8, so enc_in=16
    Richer than RMS-only: captures frequency/distribution info per timestep
    while still forming a proper operational time series for TimeMixer.

Returns per sample:
    x:    float32 [C, L]    channels-first signal
    rul:  float32 scalar     piecewise RUL in [0, 1]
    meta: dict(bearing_id, timestep, eof)

Split is strictly by bearing identity — zero train/test leakage.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rul import piecewise_rul, get_fpt_eof


# ── Split assignments (match existing .npz splits exactly) ───────────────────
CASE1_SPLITS = {
    "train": ["Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5",
              "Bearing2_1", "Bearing2_2"],
    "val":   ["Bearing2_3", "Bearing2_4", "Bearing2_5"],
    "test":  ["Bearing1_1"],
}

# PHM Case-5 splits — split by condition group
# train: Condition 1 (Bearing1_1..1_7)
# val:   Condition 2 (Bearing2_1..2_4)
# test:  Bearing3_2
PHM_CASE5_SPLITS = {
    "train": ["Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing1_4",
              "Bearing1_5", "Bearing1_6", "Bearing1_7"],
    "val":   ["Bearing2_1", "Bearing2_2", "Bearing2_3", "Bearing2_4"],
    "test":  ["Bearing3_2"],
}

# dataset_name → split dict
SPLIT_REGISTRY = {
    "xjtu":   CASE1_SPLITS,
    "phm_c5": PHM_CASE5_SPLITS,
}


class TSBearingRULDataset(Dataset):
    """Raw vibration dataset for time-series RUL models.

    Args:
        args: Namespace with fields:
            raw_data_path (str):      root dir, e.g. 'data/raw_data/XJTU-SY'
            dataset_name  (str):      'xjtu'
            ts_window_len (int):      window length L, default 2560
            ts_window_offset (str):   'center'|'start'|'end', default 'center'
            normalize_signal (bool):  per-sample z-score, default True
            use_ts_cache (bool):      disk cache for fast reload, default True
        flag (str): 'train' | 'val' | 'test'
    """

    def __init__(self, args, flag: str = "train"):
        self.root = Path(getattr(args, "raw_data_path", "data/raw_data/XJTU-SY"))
        self.dataset_name = getattr(args, "dataset_name", "xjtu").lower()
        # PHM CSVs: no header, 6 cols (timestamp×4, H_vib, V_vib), named acc_NNNNN.csv
        self._is_phm = self.dataset_name.startswith("phm")
        self.L = int(getattr(args, "ts_window_len", 2560))
        self.offset_mode = getattr(args, "ts_window_offset", "center")
        self.normalize = getattr(args, "normalize_signal", True)
        self.use_cache = getattr(args, "use_ts_cache", True)
        self.use_rms_seq = bool(getattr(args, "use_rms_seq", False))
        self.use_feat_seq = bool(getattr(args, "use_feat_seq", False))
        self.flag = flag

        splits = SPLIT_REGISTRY.get(self.dataset_name)
        if splits is None:
            raise ValueError(
                f"Unknown dataset_name '{self.dataset_name}'. "
                f"Available: {list(SPLIT_REGISTRY.keys())}"
            )
        self.bearing_ids = splits[flag]
        self.fpt_eof = get_fpt_eof(self.dataset_name)

        # Try disk cache first
        cache_dir = self.root / ".cache"
        norm_tag = "norm" if self.normalize else "raw"
        if self.use_feat_seq:
            mode_tag = "feat"
        elif self.use_rms_seq:
            mode_tag = "rms"
        else:
            mode_tag = self.offset_mode
        cache_file = cache_dir / f"ts_{flag}_L{self.L}_{mode_tag}_{norm_tag}.npz"

        if self.use_cache and cache_file.exists():
            print(f"[TSBearingRULDataset] Loading cache: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            self.signals = data["signals"]      # [N, C, L]
            self.rul_arr = data["rul"].astype("float32")  # [N]
            self.meta_list = data["meta"].tolist()        # list of dicts
        else:
            if self.use_feat_seq:
                self.signals, self.rul_arr, self.meta_list = self._load_all_bearings_feat()
            elif self.use_rms_seq:
                self.signals, self.rul_arr, self.meta_list = self._load_all_bearings_rms()
            else:
                self.signals, self.rul_arr, self.meta_list = self._load_all_bearings()
            if self.use_cache:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_file,
                    signals=self.signals,
                    rul=self.rul_arr,
                    meta=np.array(self.meta_list, dtype=object),
                )
                print(f"[TSBearingRULDataset] Saved cache: {cache_file}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_csv(self, path: Path) -> np.ndarray:
        """Load one CSV and return float32 [2, L].

        XJTU format: header row + N rows × 2 cols (H, V)
        PHM  format: no header, N rows × 6 cols; cols 4,5 are H, V vibration
        """
        if self._is_phm:
            # Some PHM files use ';' instead of ',' — auto-detect from first line
            with open(path) as _f:
                first = _f.readline()
            delim = ";" if ";" in first else ","
            data = np.loadtxt(str(path), delimiter=delim, dtype=np.float32)
            sig = data[:, 4:6].T  # [2, N_rows] — columns 4,5 = H,V
        else:
            data = np.loadtxt(str(path), delimiter=",", skiprows=1, dtype=np.float32)
            sig = data.T  # [2, N_rows]
        N = sig.shape[1]

        if N <= self.L:
            # Pad by repeating last sample if signal shorter than window
            pad = self.L - N
            sig = np.concatenate([sig, np.repeat(sig[:, -1:], pad, axis=1)], axis=1)
            return sig  # [2, L]

        if self.offset_mode == "center":
            start = (N - self.L) // 2
        elif self.offset_mode == "start":
            start = 0
        else:  # 'end'
            start = N - self.L

        return sig[:, start: start + self.L]  # [2, L]

    def _bearing_dir(self, bearing_id: str) -> Path:
        """Resolve a bearing's folder under self.root.

        PHM raw data is flat (Bearing1_1, ... directly under root). XJTU-SY raw
        data is nested one level under a condition folder per bearing group.
        """
        if self._is_phm:
            return self.root / bearing_id
        condition = "35Hz12kN" if bearing_id.startswith("Bearing1") else "37.5Hz11kN"
        return self.root / condition / bearing_id

    def _load_all_bearings(self):
        signals_list, rul_list, meta_list = [], [], []

        for bearing_id in self.bearing_ids:
            bearing_dir = self._bearing_dir(bearing_id)
            if not bearing_dir.exists():
                raise FileNotFoundError(
                    f"Bearing directory not found: {bearing_dir}\n"
                    f"Check --raw_data_path points to the Case directory."
                )

            # PHM dirs contain both acc_*.csv (vibration) and temp_*.csv (temperature)
            prefix = "acc_" if self._is_phm else ""
            csv_files = sorted(
                [f for f in bearing_dir.iterdir()
                 if f.suffix == ".csv" and f.name.startswith(prefix)],
                key=lambda f: int("".join(filter(str.isdigit, f.stem)) or "0"),
            )
            if len(csv_files) == 0:
                raise FileNotFoundError(f"No CSV files found in {bearing_dir}")

            if bearing_id not in self.fpt_eof:
                raise KeyError(
                    f"No FPT/EOF entry for '{bearing_id}' in "
                    f"dataset '{self.dataset_name}'."
                )
            fpt, eof = self.fpt_eof[bearing_id]

            for csv_path in csv_files:
                t = int("".join(filter(str.isdigit, csv_path.stem)) or "0")  # 1-based
                sig = self._load_csv(csv_path)  # [2, L]

                if self.normalize:
                    for c in range(sig.shape[0]):
                        mu = sig[c].mean()
                        sd = sig[c].std() + 1e-8
                        sig[c] = (sig[c] - mu) / sd

                rul = piecewise_rul(t, fpt, eof)
                signals_list.append(sig)
                rul_list.append(rul)
                meta_list.append({"bearing_id": bearing_id, "timestep": t, "eof": eof})

        signals_arr = np.stack(signals_list, axis=0).astype(np.float32)  # [N, 2, L]
        rul_arr = np.array(rul_list, dtype=np.float32)                    # [N]
        return signals_arr, rul_arr, meta_list

    @staticmethod
    def _extract_features(sig_1d: np.ndarray) -> np.ndarray:
        """Extract 8 time-domain statistical features from a 1-D vibration signal.

        Features (in order):
          0: RMS              = sqrt(mean(x^2))
          1: Variance         = var(x)
          2: Peak             = max(|x|)
          3: Kurtosis         = E[(x-mu)^4] / sigma^4  (Fisher, mean-corrected)
          4: Skewness         = E[(x-mu)^3] / sigma^3
          5: Crest Factor     = Peak / RMS
          6: Shape Factor     = RMS / mean(|x|)
          7: Impulse Factor   = Peak / mean(|x|)

        Returns float32 array of shape [8].
        """
        x = sig_1d.astype(np.float64)
        mu = x.mean()
        xc = x - mu
        sigma = xc.std() + 1e-12

        rms       = float(np.sqrt(np.mean(x ** 2)))
        variance  = float(np.var(x))
        peak      = float(np.max(np.abs(x)))
        kurtosis  = float(np.mean(xc ** 4) / (sigma ** 4))
        skewness  = float(np.mean(xc ** 3) / (sigma ** 3))
        mean_abs  = float(np.mean(np.abs(x))) + 1e-12
        crest_f   = peak / (rms + 1e-12)
        shape_f   = rms / mean_abs
        impulse_f = peak / mean_abs

        return np.array([rms, variance, peak, kurtosis, skewness,
                         crest_f, shape_f, impulse_f], dtype=np.float32)

    # Number of features per sensor channel
    N_FEATURES = 8

    def _load_all_bearings_feat(self):
        """Multi-feature sequence mode.

        For each bearing, builds a feature matrix [T, 2*F] where:
          - T = number of CSV files
          - F = N_FEATURES = 8 per sensor
          - Column order: [feat_H0..feat_H7, feat_V0..feat_V7]

        For timestep t, builds a sliding window of L steps → shape [2*F, L].
        This gives enc_in = 2*F = 16 channels, each a time series of one
        statistical feature over the operational lifetime.
        """
        signals_list, rul_list, meta_list = [], [], []

        for bearing_id in self.bearing_ids:
            bearing_dir = self._bearing_dir(bearing_id)
            if not bearing_dir.exists():
                raise FileNotFoundError(
                    f"Bearing directory not found: {bearing_dir}\n"
                    f"Check --raw_data_path points to the Case directory."
                )

            csv_files = sorted(
                [f for f in bearing_dir.iterdir() if f.suffix == ".csv"],
                key=lambda f: int(f.stem),
            )
            if len(csv_files) == 0:
                raise FileNotFoundError(f"No CSV files found in {bearing_dir}")

            if bearing_id not in self.fpt_eof:
                raise KeyError(
                    f"No FPT/EOF entry for '{bearing_id}' in "
                    f"dataset '{self.dataset_name}'."
                )
            fpt, eof = self.fpt_eof[bearing_id]

            # Build feature matrix [T, 2*F] — one row per CSV file
            feat_matrix = []
            for csv_path in csv_files:
                raw = np.loadtxt(str(csv_path), delimiter=",", skiprows=1, dtype=np.float32)
                # raw: [N_rows, 2]
                feat_h = self._extract_features(raw[:, 0])  # [F]
                feat_v = self._extract_features(raw[:, 1])  # [F]
                feat_matrix.append(np.concatenate([feat_h, feat_v]))  # [2*F]
            feat_matrix = np.array(feat_matrix, dtype=np.float32)  # [T, 2*F]

            # Build one sample per timestep
            T = len(feat_matrix)
            for t_idx in range(T):
                t = int(csv_files[t_idx].stem)

                # Sliding window of length L ending at t_idx (inclusive)
                start = max(0, t_idx - self.L + 1)
                window = feat_matrix[start: t_idx + 1]  # [<=L, 2*F]

                # Pad beginning by repeating first row
                if len(window) < self.L:
                    pad_len = self.L - len(window)
                    pad = np.tile(window[0:1], (pad_len, 1))
                    window = np.vstack([pad, window])  # [L, 2*F]

                sig = window.T  # [2*F, L]

                if self.normalize:
                    for c in range(sig.shape[0]):
                        mu = sig[c].mean()
                        sd = sig[c].std() + 1e-8
                        sig[c] = (sig[c] - mu) / sd

                rul = piecewise_rul(t, fpt, eof)
                signals_list.append(sig)
                rul_list.append(rul)
                meta_list.append({"bearing_id": bearing_id, "timestep": t, "eof": eof})

        signals_arr = np.stack(signals_list, axis=0).astype(np.float32)  # [N, 2*F, L]
        rul_arr = np.array(rul_list, dtype=np.float32)
        return signals_arr, rul_arr, meta_list

    def _load_all_bearings_rms(self):
        """RMS sequence mode: each sample is [2, L] of RMS values over L consecutive files.

        For timestep t (1-indexed), x = [[RMS_H(t-L+1),...,RMS_H(t)],
                                          [RMS_V(t-L+1),...,RMS_V(t)]]
        Steps before the bearing starts are padded by repeating step 1.
        """
        signals_list, rul_list, meta_list = [], [], []

        for bearing_id in self.bearing_ids:
            bearing_dir = self._bearing_dir(bearing_id)
            if not bearing_dir.exists():
                raise FileNotFoundError(
                    f"Bearing directory not found: {bearing_dir}\n"
                    f"Check --raw_data_path points to the Case directory."
                )

            csv_files = sorted(
                [f for f in bearing_dir.iterdir() if f.suffix == ".csv"],
                key=lambda f: int(f.stem),
            )
            if len(csv_files) == 0:
                raise FileNotFoundError(f"No CSV files found in {bearing_dir}")

            if bearing_id not in self.fpt_eof:
                raise KeyError(
                    f"No FPT/EOF entry for '{bearing_id}' in "
                    f"dataset '{self.dataset_name}'."
                )
            fpt, eof = self.fpt_eof[bearing_id]

            # Build RMS matrix [T, 2] for this bearing — one row per CSV file
            rms_matrix = []
            for csv_path in csv_files:
                raw = np.loadtxt(str(csv_path), delimiter=",", skiprows=1, dtype=np.float32)
                # raw: [N_rows, 2]
                rms_h = float(np.sqrt(np.mean(raw[:, 0] ** 2)))
                rms_v = float(np.sqrt(np.mean(raw[:, 1] ** 2)))
                rms_matrix.append([rms_h, rms_v])
            rms_matrix = np.array(rms_matrix, dtype=np.float32)  # [T, 2]

            # Build one sample per timestep
            T = len(rms_matrix)
            for t_idx in range(T):
                t = int(csv_files[t_idx].stem)  # 1-indexed timestep from filename

                # Sliding window of length L ending at t_idx (inclusive)
                start = max(0, t_idx - self.L + 1)
                window = rms_matrix[start: t_idx + 1]  # [<=L, 2]

                # Pad beginning by repeating first row if window shorter than L
                if len(window) < self.L:
                    pad_len = self.L - len(window)
                    pad = np.tile(window[0:1], (pad_len, 1))  # [pad_len, 2]
                    window = np.vstack([pad, window])          # [L, 2]

                sig = window.T  # [2, L]

                if self.normalize:
                    for c in range(2):
                        mu = sig[c].mean()
                        sd = sig[c].std() + 1e-8
                        sig[c] = (sig[c] - mu) / sd

                rul = piecewise_rul(t, fpt, eof)
                signals_list.append(sig)
                rul_list.append(rul)
                meta_list.append({"bearing_id": bearing_id, "timestep": t, "eof": eof})

        signals_arr = np.stack(signals_list, axis=0).astype(np.float32)  # [N, 2, L]
        rul_arr = np.array(rul_list, dtype=np.float32)                    # [N]
        return signals_arr, rul_arr, meta_list

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.rul_arr)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.signals[idx])        # [2, L]  float32
        rul = torch.tensor(self.rul_arr[idx], dtype=torch.float32)
        meta = self.meta_list[idx]
        return x, rul, meta


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", default="data/raw_data/XJTU-SY")
    parser.add_argument("--dataset_name",  default="xjtu")
    parser.add_argument("--ts_window_len", type=int, default=2560)
    parser.add_argument("--ts_window_offset", default="center")
    parser.add_argument("--normalize_signal", type=bool, default=True)
    parser.add_argument("--use_ts_cache",     type=bool, default=False)
    args = parser.parse_args()

    for flag in ("train", "val", "test"):
        ds = TSBearingRULDataset(args, flag=flag)
        x, rul, meta = ds[0]
        print(f"[{flag:5s}] len={len(ds):4d}  "
              f"x={tuple(x.shape)}  rul={rul.item():.4f}  "
              f"bearing={meta['bearing_id']}  t={meta['timestep']}")
        x_last, rul_last, meta_last = ds[-1]
        print(f"         last: rul={rul_last.item():.4f}  "
              f"bearing={meta_last['bearing_id']}  t={meta_last['timestep']}")
