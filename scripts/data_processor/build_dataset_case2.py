# scripts/data_processor/build_dataset_case2.py
"""
Build dataset for PHM 2012 Case 2 experiments.

PHM 2012 Dataset (PRONOSTIA):
- Sampling: 25.6 kHz
- Each file: 2560 samples (0.1 sec)
- Format: 6 columns (4 metadata + 2 channels: horizontal, vertical)

Data split:
- Train: Bearing1_1, Bearing1_2, Bearing1_3, Bearing1_4
- Val: Bearing1_6, Bearing1_7
- Test: Bearing1_5

Usage:
    python scripts/data_processor/build_dataset_case2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.signal import resample
from joblib import Parallel, delayed

from utils.denoising import denoise_signal, IENEMDATDConfig
from utils.psr import (
    PSRConfig, mutual_information_tau, cao_embedding_dimension,
    delay_embed, points_to_density_image
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Paths - PHM 2012 P-bearing1
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw_data", "PHM-2012")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed_data", "case2")

# Data splits
CASE2_TRAIN = ["Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing1_4"]
CASE2_VAL = ["Bearing1_6", "Bearing1_7"]
CASE2_TEST = ["Bearing1_5"]

# Processing parameters
IMAGE_BINS = 64
PHYSICS_POINTS = 2560

# Processing mode
USE_PARALLEL = True
N_JOBS = -1


# =============================================================================
# PHM 2012 SPECIFIC FUNCTIONS
# =============================================================================

def list_phm_bearings(root_dir: str) -> List[str]:
    """List all bearing folders in PHM 2012 directory."""
    bearings = []
    if not os.path.isdir(root_dir):
        return bearings
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and name.startswith("Bearing"):
            # Check if has acc_*.csv files
            if len(glob.glob(os.path.join(p, "acc_*.csv"))) > 0:
                bearings.append(name)
    return bearings


def build_phm_bearing_map(root_dir: str) -> Dict[str, str]:
    """
    Build mapping from bearing name to its folder path.
    For PHM 2012, all bearings are in the same root directory.
    """
    bearing_map = {}
    bearings = list_phm_bearings(root_dir)
    for b in bearings:
        bearing_map[b] = root_dir
    return bearing_map


def read_phm_csv(f_path: str) -> np.ndarray:
    """
    Read PHM 2012 CSV file.

    PHM format: 6 columns
    - Columns 0-3: metadata (hour, minute, second, microsecond)
    - Column 4: horizontal acceleration
    - Column 5: vertical acceleration

    Note: Some bearings use ';' as delimiter, others use ','.
    This function auto-detects the delimiter.

    Returns:
        Array of shape (N, 2) with horizontal and vertical channels
    """
    # Auto-detect delimiter by reading first line
    with open(f_path, 'r') as f:
        first_line = f.readline()

    if ';' in first_line:
        df = pd.read_csv(f_path, header=None, sep=';')
    else:
        df = pd.read_csv(f_path, header=None)

    # Extract only acceleration columns (4 and 5)
    return df.values[:, 4:6].astype(np.float32)


def get_phm_csv_files(bearing_folder: str) -> List[str]:
    """Get sorted list of CSV files for a bearing."""
    csv_files = glob.glob(os.path.join(bearing_folder, "acc_*.csv"))
    # Sort by file number
    csv_files = sorted(csv_files, key=lambda x: int(os.path.basename(x).replace("acc_", "").replace(".csv", "")))
    return csv_files


# =============================================================================
# PSR PARAMETER SEARCH
# =============================================================================

def get_global_psr_params_phm(bearing_map: Dict[str, str],
                              train_bearings: List[str],
                              n_samples: int = 5,
                              den_cfg: IENEMDATDConfig = None,
                              psr_cfg: PSRConfig = None) -> Tuple[int, int]:
    """Calculate global m and tau from PHM 2012 training bearings."""
    if den_cfg is None:
        den_cfg = IENEMDATDConfig(J=5)
    if psr_cfg is None:
        psr_cfg = PSRConfig(tau_max=50, m_max=15)

    taus, ms = [], []

    print(f"Searching global PSR parameters from {n_samples} bearings...")

    count = 0
    for b in tqdm(train_bearings, desc="Searching Params"):
        if count >= n_samples:
            break
        if b not in bearing_map:
            continue

        folder = bearing_map[b]
        csv_files = get_phm_csv_files(os.path.join(folder, b))

        if not csv_files:
            continue

        try:
            # Use first file (healthy baseline)
            raw = read_phm_csv(csv_files[0])[:, 0]  # Horizontal channel
            clean = denoise_signal(raw, den_cfg)

            t = mutual_information_tau(clean, psr_cfg)
            m = cao_embedding_dimension(clean, t, psr_cfg)

            taus.append(t)
            ms.append(m)
            print(f"  {b}: tau={t}, m={m}")
            count += 1
        except Exception as e:
            print(f"  Error on {b}: {e}")
            continue

    final_tau = int(np.median(taus)) if taus else 4
    final_m = int(np.median(ms)) if ms else 5

    return final_m, final_tau


# =============================================================================
# PCA FITTING
# =============================================================================

def fit_pca_on_healthy_data_phm(bearing_map: Dict[str, str],
                                train_bearings: List[str],
                                m: int,
                                tau: int,
                                baseline_frac: float = 0.2,
                                max_files_per_bearing: int = 20,
                                points_per_file: int = 2000,
                                den_cfg: IENEMDATDConfig = None
                                ) -> Tuple[PCA, Tuple[float, float], Tuple[float, float]]:
    """Fit PCA on healthy data from PHM 2012 training bearings."""
    if den_cfg is None:
        den_cfg = IENEMDATDConfig(J=5)

    rng = np.random.default_rng(42)
    all_pts = []

    print("Collecting healthy data for PCA fitting...")

    for b in tqdm(train_bearings, desc="Loading Healthy Data"):
        if b not in bearing_map:
            continue
        folder = bearing_map[b]

        csv_files = get_phm_csv_files(os.path.join(folder, b))

        # Only use early data (healthy phase)
        T_healthy = max(1, int(len(csv_files) * baseline_frac))
        sample_indices = np.linspace(0, T_healthy - 1, min(max_files_per_bearing, T_healthy), dtype=int)

        for idx in sample_indices:
            try:
                f_path = csv_files[idx]
                raw_data = read_phm_csv(f_path)

                for ch in range(raw_data.shape[1]):
                    clean = denoise_signal(raw_data[:, ch], den_cfg)

                    N = len(clean)
                    L = N - (m - 1) * tau
                    if L <= 0:
                        continue

                    R = delay_embed(clean, tau, m)

                    if len(R) > points_per_file:
                        sub_idx = rng.choice(len(R), points_per_file, replace=False)
                        R = R[sub_idx]

                    all_pts.append(R)
            except Exception:
                continue

    if not all_pts:
        raise ValueError("No points collected. Check dataset path.")

    X_all = np.concatenate(all_pts, axis=0)
    print(f"Fitting PCA on {X_all.shape} points...")

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_all)

    q = 0.995
    xlim = (float(np.quantile(Z[:, 0], 1 - q)), float(np.quantile(Z[:, 0], q)))
    ylim = (float(np.quantile(Z[:, 1], 1 - q)), float(np.quantile(Z[:, 1], q)))

    print(f"PCA fitted. Axis limits: X={xlim}, Y={ylim}")
    return pca, xlim, ylim


# =============================================================================
# DATASET BUILDING (PARALLEL)
# =============================================================================

def _process_phm_file_worker(task: Tuple,
                             pca_components: np.ndarray,
                             pca_mean: np.ndarray,
                             xlim: Tuple[float, float],
                             ylim: Tuple[float, float],
                             m: int,
                             tau: int,
                             image_bins: int,
                             physics_points: int,
                             den_j: int) -> Optional[Tuple]:
    """Worker function for parallel processing of PHM 2012 files."""
    f_path, bearing_name, t_idx, total_files = task
    den_cfg = IENEMDATDConfig(J=den_j)

    try:
        raw_vals = read_phm_csv(f_path)
        imgs, phys = [], []

        for ch in range(raw_vals.shape[1]):
            clean = denoise_signal(raw_vals[:, ch], den_cfg)

            N = len(clean)
            L = N - (m - 1) * tau
            if L <= 10:
                imgs.append(np.zeros((image_bins, image_bins), dtype=np.float32))
                phys.append(np.zeros((physics_points,), dtype=np.float32))
                continue

            R = delay_embed(clean, tau, m)

            # Manual PCA transform
            Z = (R - pca_mean) @ pca_components.T

            img = points_to_density_image(Z, image_bins, xlim, ylim)
            imgs.append(img)

            traj_1d = resample(Z[:, 0], physics_points)
            phys.append(traj_1d.astype(np.float32))

        return (
            np.stack(imgs, axis=0),
            np.stack(phys, axis=0),
            [bearing_name, t_idx, total_files]
        )

    except Exception:
        return None


def build_dataset_split_phm_parallel(bearings: List[str],
                                     split_name: str,
                                     bearing_map: Dict[str, str],
                                     pca: PCA,
                                     xlim: Tuple[float, float],
                                     ylim: Tuple[float, float],
                                     m: int,
                                     tau: int,
                                     out_dir: str,
                                     image_bins: int = 64,
                                     physics_points: int = 2560,
                                     den_cfg: IENEMDATDConfig = None,
                                     n_jobs: int = -1) -> None:
    """Build dataset split for PHM 2012 using parallel processing."""
    if den_cfg is None:
        den_cfg = IENEMDATDConfig(J=5)

    # Collect all tasks
    tasks = []
    for b in bearings:
        if b not in bearing_map:
            continue
        folder = bearing_map[b]

        csv_files = get_phm_csv_files(os.path.join(folder, b))

        for t, f in enumerate(csv_files):
            tasks.append((f, b, t, len(csv_files)))

    if not tasks:
        print(f"Warning: No files found for {split_name}")
        return

    save_path = os.path.join(out_dir, f"{split_name}.npz")
    print(f"\nProcessing {split_name} ({len(tasks)} files) -> {save_path}")
    print(f"Using {n_jobs} parallel workers...")

    # Extract PCA components
    pca_components = pca.components_
    pca_mean = pca.mean_

    # Parallel execution
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_process_phm_file_worker)(
            task, pca_components, pca_mean, xlim, ylim,
            m, tau, image_bins, physics_points, den_cfg.J
        ) for task in tasks
    )

    # Filter valid results
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print(f"Warning: No valid data for {split_name}")
        return

    # Aggregate
    X_final = np.stack([r[0] for r in valid_results], axis=0)
    P_final = np.stack([r[1] for r in valid_results], axis=0)
    meta_final = np.array([r[2] for r in valid_results], dtype=object)

    np.savez_compressed(save_path, X=X_final, P=P_final, meta=meta_final)
    print(f"Saved {split_name}.npz | X: {X_final.shape} | P: {P_final.shape}")
    print(f"Success rate: {len(valid_results)}/{len(tasks)} ({100*len(valid_results)/len(tasks):.1f}%)")


def build_dataset_split_phm(bearings: List[str],
                            split_name: str,
                            bearing_map: Dict[str, str],
                            pca: PCA,
                            xlim: Tuple[float, float],
                            ylim: Tuple[float, float],
                            m: int,
                            tau: int,
                            out_dir: str,
                            image_bins: int = 64,
                            physics_points: int = 2560,
                            den_cfg: IENEMDATDConfig = None) -> None:
    """Build dataset split for PHM 2012 (single-threaded version)."""
    if den_cfg is None:
        den_cfg = IENEMDATDConfig(J=5)

    X_list, P_list, meta_list = [], [], []

    save_path = os.path.join(out_dir, f"{split_name}.npz")
    print(f"\nProcessing {split_name} -> {save_path}")

    for b in tqdm(bearings, desc=f"Bearings ({split_name})"):
        if b not in bearing_map:
            continue
        folder = bearing_map[b]

        csv_files = get_phm_csv_files(os.path.join(folder, b))

        for t, f in enumerate(tqdm(csv_files, desc=f"  {b}", leave=False)):
            try:
                raw_vals = read_phm_csv(f)
                imgs, phys = [], []

                for ch in range(raw_vals.shape[1]):
                    clean = denoise_signal(raw_vals[:, ch], den_cfg)

                    N = len(clean)
                    L = N - (m - 1) * tau
                    if L <= 10:
                        imgs.append(np.zeros((image_bins, image_bins), dtype=np.float32))
                        phys.append(np.zeros((physics_points,), dtype=np.float32))
                        continue

                    R = delay_embed(clean, tau, m)
                    Z = pca.transform(R)

                    img = points_to_density_image(Z, image_bins, xlim, ylim)
                    imgs.append(img)

                    traj_1d = resample(Z[:, 0], physics_points)
                    phys.append(traj_1d.astype(np.float32))

                X_list.append(np.stack(imgs, axis=0)[None, ...])
                P_list.append(np.stack(phys, axis=0)[None, ...])
                meta_list.append([b, t, len(csv_files)])

            except Exception:
                continue

    if len(X_list) == 0:
        print(f"Warning: No data found for {split_name}")
        return

    X_final = np.concatenate(X_list, axis=0)
    P_final = np.concatenate(P_list, axis=0)
    meta_final = np.array(meta_list, dtype=object)

    np.savez_compressed(save_path, X=X_final, P=P_final, meta=meta_final)
    print(f"Saved {split_name}.npz | X: {X_final.shape} | P: {P_final.shape}")


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_dataset(npz_path: str) -> None:
    """Verify and visualize a sample from the dataset."""
    if not os.path.exists(npz_path):
        print(f"File not found: {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    P = data['P']
    meta = data['meta']

    print(f"\nVerifying {os.path.basename(npz_path)}")
    print(f"  X (Visual): {X.shape}")
    print(f"  P (Physics): {P.shape}")
    print(f"  Samples: {len(meta)}")

    # Visualize random sample
    idx = np.random.randint(0, len(X))
    sample_b = meta[idx][0]
    sample_t = meta[idx][1]

    _, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(X[idx, 0], cmap='inferno', origin='lower')
    axes[0].set_title(f"Visual: {sample_b} (t={sample_t})")
    axes[0].axis('off')

    axes[1].plot(P[idx, 0])
    axes[1].set_title("Physics (PC1 trajectory)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"verify_{os.path.basename(npz_path).replace('.npz', '.png')}"))
    plt.close()
    print(f"  Verification plot saved.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Building PHM 2012 Case 2 Dataset")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build bearing map
    bearing_map = build_phm_bearing_map(RAW_DATA_DIR)
    print(f"Found {len(bearing_map)} bearings: {list(bearing_map.keys())}")

    # Denoising config
    den_cfg = IENEMDATDConfig(J=5)

    # Step 1: Compute global PSR parameters
    print("\n" + "-" * 40)
    print("Step 1: Computing PSR parameters (m, tau)")
    print("-" * 40)
    m, tau = get_global_psr_params_phm(bearing_map, CASE2_TRAIN, n_samples=5, den_cfg=den_cfg)
    print(f"\n>>> Global parameters: m={m}, tau={tau}")

    # Step 2: Fit PCA on healthy data
    print("\n" + "-" * 40)
    print("Step 2: Fitting PCA on healthy data")
    print("-" * 40)
    pca, xlim, ylim = fit_pca_on_healthy_data_phm(bearing_map, CASE2_TRAIN, m, tau, den_cfg=den_cfg)

    # Save metadata
    meta_path = os.path.join(OUT_DIR, "pca_metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump({
            "pca": pca,
            "xlim": xlim,
            "ylim": ylim,
            "m": m,
            "tau": tau,
            "image_bins": IMAGE_BINS,
            "physics_points": PHYSICS_POINTS,
            "dataset": "PHM2012_Case2"
        }, f)
    print(f"Metadata saved to {meta_path}")

    # Step 3: Build datasets
    print("\n" + "-" * 40)
    print("Step 3: Building dataset splits")
    print("-" * 40)

    if USE_PARALLEL:
        print(f"Mode: PARALLEL (n_jobs={N_JOBS})")
        build_func = lambda b, s: build_dataset_split_phm_parallel(
            b, s, bearing_map, pca, xlim, ylim, m, tau, OUT_DIR,
            IMAGE_BINS, PHYSICS_POINTS, den_cfg, n_jobs=N_JOBS
        )
    else:
        print("Mode: SINGLE-THREADED")
        build_func = lambda b, s: build_dataset_split_phm(
            b, s, bearing_map, pca, xlim, ylim, m, tau, OUT_DIR,
            IMAGE_BINS, PHYSICS_POINTS, den_cfg
        )

    build_func(CASE2_TRAIN, "train")
    build_func(CASE2_VAL, "val")
    build_func(CASE2_TEST, "test")

    # Step 4: Verify
    print("\n" + "-" * 40)
    print("Step 4: Verification")
    print("-" * 40)
    verify_dataset(os.path.join(OUT_DIR, "train.npz"))

    print("\n" + "=" * 60)
    print("Dataset building complete!")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
