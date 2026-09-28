# datasets/data_utils.py
"""
Utility functions for dataset building and loading.

Supports multiprocessing for faster dataset building.
"""

import os
import glob
import numpy as np
import pandas as pd
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


def list_bearings_in_folder(condition_folder: str) -> List[str]:
    """List all bearing folders in a condition folder."""
    out = []
    if not os.path.isdir(condition_folder):
        return out
    for name in sorted(os.listdir(condition_folder)):
        p = os.path.join(condition_folder, name)
        if os.path.isdir(p) and len(glob.glob(os.path.join(p, "*.csv"))) > 0:
            out.append(name)
    return out


def build_bearing_map(root_dir: str,
                      conditions: Tuple[str, ...] = ("35Hz12kN", "37.5Hz11kN", "40Hz10kN")
                      ) -> Dict[str, str]:
    """
    Build a mapping from bearing name to its parent condition folder.

    Args:
        root_dir: Root directory containing condition folders
        conditions: Tuple of condition folder names

    Returns:
        Dict mapping bearing name -> condition folder path
    """
    bearing_map = {}
    for cond in conditions:
        cond_path = os.path.join(root_dir, cond)
        bears = list_bearings_in_folder(cond_path)
        for b in bears:
            bearing_map[b] = cond_path
    return bearing_map


def get_global_psr_params(bearing_map: Dict[str, str],
                          train_bearings: List[str],
                          n_samples: int = 5,
                          den_cfg: IENEMDATDConfig = None,
                          psr_cfg: PSRConfig = None) -> Tuple[int, int]:
    """
    Calculate global m and tau parameters from training bearings.

    Uses median of individual bearing parameters to avoid outliers.

    Args:
        bearing_map: Mapping from bearing name to folder
        train_bearings: List of training bearing names
        n_samples: Number of bearings to sample
        den_cfg: Denoising config
        psr_cfg: PSR config

    Returns:
        Tuple of (m, tau)
    """
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
        csv_files = sorted(glob.glob(os.path.join(folder, b, "*.csv")))
        csv_files = sorted(csv_files, key=lambda x: int(os.path.basename(x).replace(".csv", "")))

        if not csv_files:
            continue

        try:
            raw = pd.read_csv(csv_files[0]).values[:, 0]
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


def fit_pca_on_healthy_data(bearing_map: Dict[str, str],
                            train_bearings: List[str],
                            m: int,
                            tau: int,
                            baseline_frac: float = 0.2,
                            max_files_per_bearing: int = 20,
                            points_per_file: int = 2000,
                            den_cfg: IENEMDATDConfig = None
                            ) -> Tuple[PCA, Tuple[float, float], Tuple[float, float]]:
    """
    Fit PCA on healthy (early) data from training bearings.

    Args:
        bearing_map: Mapping from bearing name to folder
        train_bearings: List of training bearing names
        m: Embedding dimension
        tau: Time delay
        baseline_frac: Fraction of data considered healthy (default 20%)
        max_files_per_bearing: Max files to sample per bearing
        points_per_file: Points to sample per file
        den_cfg: Denoising config

    Returns:
        Tuple of (pca, xlim, ylim)
    """
    if den_cfg is None:
        den_cfg = IENEMDATDConfig(J=5)

    rng = np.random.default_rng(42)
    all_pts = []

    print("Collecting healthy data for PCA fitting...")

    for b in tqdm(train_bearings, desc="Loading Healthy Data"):
        if b not in bearing_map:
            continue
        folder = bearing_map[b]

        csv_files = sorted(glob.glob(os.path.join(folder, b, "*.csv")))
        csv_files = sorted(csv_files, key=lambda x: int(os.path.basename(x).replace(".csv", "")))

        # Only use early data (healthy phase)
        T_healthy = max(1, int(len(csv_files) * baseline_frac))
        sample_indices = np.linspace(0, T_healthy - 1, min(max_files_per_bearing, T_healthy), dtype=int)

        for idx in sample_indices:
            try:
                f_path = csv_files[idx]
                raw_data = pd.read_csv(f_path).values

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

    # Use 99.5% quantile for axis limits
    q = 0.995
    xlim = (float(np.quantile(Z[:, 0], 1 - q)), float(np.quantile(Z[:, 0], q)))
    ylim = (float(np.quantile(Z[:, 1], 1 - q)), float(np.quantile(Z[:, 1], q)))

    print(f"PCA fitted. Axis limits: X={xlim}, Y={ylim}")
    return pca, xlim, ylim


# =============================================================================
# SINGLE-THREADED PROCESSING
# =============================================================================

def process_single_file(f_path: str,
                        pca: PCA,
                        xlim: Tuple[float, float],
                        ylim: Tuple[float, float],
                        m: int,
                        tau: int,
                        image_bins: int = 64,
                        physics_points: int = 2560,
                        den_cfg: IENEMDATDConfig = None,
                        no_anchor: bool = False
                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Process a single CSV file to generate visual and physics inputs.

    Args:
        f_path: Path to CSV file
        pca: Fitted PCA object
        xlim, ylim: Axis limits for density image
        m: Embedding dimension
        tau: Time delay
        image_bins: Size of density image
        physics_points: Length of physics vector
        den_cfg: Denoising config
        no_anchor: If True, use per-file local PCA + auto canvas (ablation baseline)

    Returns:
        Tuple of (images, physics) arrays or None if failed
    """
    if den_cfg is None:
        den_cfg = IENEMDATDConfig(J=5)

    try:
        raw_vals = pd.read_csv(f_path).values.astype(np.float32)
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

            if no_anchor:
                pca_local = PCA(n_components=2).fit(R)
                Z = pca_local.transform(R)
                xlim_use = (float(Z[:, 0].min()), float(Z[:, 0].max()))
                ylim_use = (float(Z[:, 1].min()), float(Z[:, 1].max()))
            else:
                Z = pca.transform(R)
                xlim_use, ylim_use = xlim, ylim

            img = points_to_density_image(Z, image_bins, xlim_use, ylim_use)
            imgs.append(img)

            traj_1d = resample(Z[:, 0], physics_points)
            phys.append(traj_1d.astype(np.float32))

        return np.stack(imgs, axis=0), np.stack(phys, axis=0)

    except Exception:
        return None


def build_dataset_split(bearings: List[str],
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
                        no_anchor: bool = False) -> None:
    """
    Build dataset for a single split (train/val/test) - single threaded version.

    Args:
        bearings: List of bearing names
        split_name: Name of split (train/val/test)
        bearing_map: Mapping from bearing name to folder
        pca: Fitted PCA object
        xlim, ylim: Axis limits
        m: Embedding dimension
        tau: Time delay
        out_dir: Output directory
        image_bins: Size of density image
        physics_points: Length of physics vector
        den_cfg: Denoising config
        no_anchor: If True, use per-file local PCA + auto canvas (ablation baseline)
    """
    if den_cfg is None:
        den_cfg = IENEMDATDConfig(J=5)

    X_list, P_list, meta_list = [], [], []

    save_path = os.path.join(out_dir, f"{split_name}.npz")
    print(f"\nProcessing {split_name} -> {save_path}")

    for b in tqdm(bearings, desc=f"Bearings ({split_name})"):
        if b not in bearing_map:
            continue
        folder = bearing_map[b]

        csv_files = sorted(glob.glob(os.path.join(folder, b, "*.csv")))
        csv_files = sorted(csv_files, key=lambda x: int(os.path.basename(x).replace(".csv", "")))

        for t, f in enumerate(tqdm(csv_files, desc=f"  {b}", leave=False)):
            result = process_single_file(f, pca, xlim, ylim, m, tau,
                                         image_bins, physics_points, den_cfg,
                                         no_anchor=no_anchor)
            if result is not None:
                imgs, phys = result
                X_list.append(imgs[None, ...])
                P_list.append(phys[None, ...])
                meta_list.append([b, t, len(csv_files)])

    if len(X_list) == 0:
        print(f"Warning: No data found for {split_name}")
        return

    X_final = np.concatenate(X_list, axis=0)
    P_final = np.concatenate(P_list, axis=0)
    meta_final = np.array(meta_list, dtype=object)

    np.savez_compressed(save_path, X=X_final, P=P_final, meta=meta_final)
    print(f"Saved {split_name}.npz | X: {X_final.shape} | P: {P_final.shape}")


# =============================================================================
# PARALLEL PROCESSING (MULTIPROCESSING)
# =============================================================================

def _process_file_worker(task: Tuple,
                         pca_components: np.ndarray,
                         pca_mean: np.ndarray,
                         xlim: Tuple[float, float],
                         ylim: Tuple[float, float],
                         m: int,
                         tau: int,
                         image_bins: int,
                         physics_points: int,
                         den_j: int,
                         denoise_enabled: bool = True,
                         no_anchor: bool = False) -> Optional[Tuple]:
    """
    Worker function for parallel processing.

    Note: PCA object cannot be pickled directly, so we pass components and mean.
    """
    f_path, bearing_name, t_idx, total_files = task
    den_cfg = IENEMDATDConfig(J=den_j, enabled=denoise_enabled)

    try:
        raw_vals = pd.read_csv(f_path).values.astype(np.float32)
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

            if no_anchor:
                pca_local = PCA(n_components=2).fit(R)
                Z = pca_local.transform(R)
                xlim_use = (float(Z[:, 0].min()), float(Z[:, 0].max()))
                ylim_use = (float(Z[:, 1].min()), float(Z[:, 1].max()))
            else:
                # Manual PCA transform (since PCA object isn't picklable in some cases)
                Z = (R - pca_mean) @ pca_components.T
                xlim_use, ylim_use = xlim, ylim

            img = points_to_density_image(Z, image_bins, xlim_use, ylim_use)
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


def build_dataset_split_parallel(bearings: List[str],
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
                                 n_jobs: int = -1,
                                 no_anchor: bool = False) -> None:
    """
    Build dataset for a single split using parallel processing.

    Uses joblib for multiprocessing to speed up denoising (the main bottleneck).

    Args:
        bearings: List of bearing names
        split_name: Name of split (train/val/test)
        bearing_map: Mapping from bearing name to folder
        pca: Fitted PCA object
        xlim, ylim: Axis limits
        m: Embedding dimension
        tau: Time delay
        out_dir: Output directory
        image_bins: Size of density image
        physics_points: Length of physics vector
        den_cfg: Denoising config
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        no_anchor: If True, use per-file local PCA + auto canvas (ablation baseline)
    """
    if den_cfg is None:
        den_cfg = IENEMDATDConfig(J=5)

    # Collect all tasks
    tasks = []
    for b in bearings:
        if b not in bearing_map:
            continue
        folder = bearing_map[b]

        csv_files = sorted(glob.glob(os.path.join(folder, b, "*.csv")))
        csv_files = sorted(csv_files, key=lambda x: int(os.path.basename(x).replace(".csv", "")))

        for t, f in enumerate(csv_files):
            tasks.append((f, b, t, len(csv_files)))

    if not tasks:
        print(f"Warning: No files found for {split_name}")
        return

    save_path = os.path.join(out_dir, f"{split_name}.npz")
    print(f"\nProcessing {split_name} ({len(tasks)} files) -> {save_path}")
    print(f"Using {n_jobs} parallel workers...")

    # Extract PCA components for pickling
    pca_components = pca.components_
    pca_mean = pca.mean_

    # Parallel execution with joblib
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_process_file_worker)(
            task, pca_components, pca_mean, xlim, ylim,
            m, tau, image_bins, physics_points, den_cfg.J, den_cfg.enabled,
            no_anchor
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
