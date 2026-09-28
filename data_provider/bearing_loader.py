"""
Bearing RUL Dataset Loader for PSDI (Phase Space Density Images)

Loads preprocessed bearing data in .npz format with:
- X: PSDI images [N, 2, 224, 224]
- P: Phase space features [N, 2, 2560]
- meta: Metadata [bearing_id, timestep, EOF]

RUL is computed using piecewise linear function from utils/rul.py
"""

import os
import sys
import importlib
import numpy as np
import torch
from torch.utils.data import Dataset

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- numpy 2.x -> 1.x unpickle compatibility ---------------------------------
# The .npz files store `meta` as an object array, which is pickled. Files saved
# with numpy >= 2.0 reference the internal module `numpy._core`, which was named
# `numpy.core` in numpy < 2.0. Without this shim, loading such files under an
# older numpy raises `ModuleNotFoundError: No module named 'numpy._core'`.
if not hasattr(np, "_core"):
    class _NumpyCoreCompatFinder:
        """Redirect any `numpy._core[.*]` import to `numpy.core[.*]`."""

        _prefix = "numpy._core"

        def find_module(self, fullname, path=None):
            if fullname == self._prefix or fullname.startswith(self._prefix + "."):
                return self
            return None

        def load_module(self, fullname):
            if fullname in sys.modules:
                return sys.modules[fullname]
            target = "numpy.core" + fullname[len(self._prefix):]
            module = importlib.import_module(target)
            sys.modules[fullname] = module
            return module

    sys.meta_path.insert(0, _NumpyCoreCompatFinder())
# -----------------------------------------------------------------------------

from utils.rul import piecewise_rul, get_fpt_eof


class BearingRULDataset(Dataset):
    """
    Dataset for bearing RUL prediction from PSDI images.

    Args:
        args: Configuration object
        root_path: Root directory containing train.npz, val.npz, test.npz
        flag: 'train', 'val', or 'test'

    Data format:
        - X: [N, 2, 224, 224] PSDI images (float32)
        - P: [N, 2, 2560] Phase space features (float32)
        - meta: [N, 3] Metadata array with [bearing_id, timestep, EOF]

    Returns:
        X: [2, 224, 224] image tensor
        P: [2, 2560] phase space tensor
        rul: scalar RUL value (float)
        meta: dict with bearing_id, timestep, EOF
    """

    def __init__(self, args, root_path, flag='train'):
        super(BearingRULDataset, self).__init__()

        self.args = args
        self.root_path = root_path
        self.flag = flag

        # Dataset configuration
        self.dataset_name = getattr(args, 'dataset_name', 'xjtu')  # 'xjtu' or 'phm'

        # Load FPT/EOF lookup table for RUL computation
        self.fpt_eof_table = get_fpt_eof(self.dataset_name)

        # Load data
        self._load_data()

        print(f"BearingRULDataset [{flag}] initialized:")
        print(f"  Dataset: {self.dataset_name}")
        print(f"  Samples: {len(self)}")
        print(f"  Images shape: {self.X.shape}")
        print(f"  Phase shape: {self.P.shape}")
        print(f"  RUL range: [{self.rul.min():.3f}, {self.rul.max():.3f}]")

    def _load_data(self):
        """Load data from .npz file"""
        # Construct file path
        filename = f"{self.flag}.npz"
        filepath = os.path.join(self.root_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Load npz file
        print(f"Loading data from: {filepath}")
        data = np.load(filepath, allow_pickle=True)

        # Extract arrays
        self.X = data['X']  # [N, 2, 224, 224]
        self.P = data['P']  # [N, 2, 2560]
        self.meta = data['meta']  # [N, 3] object array

        # Validate shapes
        assert self.X.shape[0] == self.P.shape[0] == self.meta.shape[0], \
            "Mismatch in number of samples between X, P, and meta"

        # Compute RUL for all samples
        self.rul = self._compute_rul()

        print(f"Loaded {len(self)} samples from {filepath}")

    def _compute_rul(self):
        """
        Compute RUL for all samples using piecewise linear function.

        Returns:
            rul: [N] array of RUL values in [0, 1]
        """
        rul_values = []

        for i in range(len(self.meta)):
            # Extract metadata
            bearing_id = self.meta[i][0]  # e.g., 'Bearing1_2'
            timestep = int(self.meta[i][1])
            eof = int(self.meta[i][2])

            # Get FPT for this bearing
            if bearing_id not in self.fpt_eof_table:
                raise ValueError(f"Bearing ID '{bearing_id}' not found in FPT/EOF table")

            fpt, eof_table = self.fpt_eof_table[bearing_id]

            # Compute piecewise RUL
            rul = piecewise_rul(timestep, fpt, eof_table)
            rul_values.append(rul)

        rul_array = np.array(rul_values, dtype=np.float32)
        return rul_array

    def __len__(self):
        """Return number of samples"""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get single sample.

        Returns:
            X: [2, 224, 224] image tensor
            P: [2, 2560] phase space tensor
            rul: scalar RUL value
            meta: dict with bearing info
        """
        # Get data
        X = self.X[idx]  # [2, 224, 224]
        P = self.P[idx]  # [2, 2560]
        rul = self.rul[idx]  # scalar

        # Convert to tensors
        X = torch.from_numpy(X).float()
        P = torch.from_numpy(P).float()
        rul = torch.tensor(rul, dtype=torch.float32)

        # Metadata (for debugging/analysis)
        meta = {
            'bearing_id': self.meta[idx][0],
            'timestep': int(self.meta[idx][1]),
            'eof': int(self.meta[idx][2])
        }

        return X, P, rul, meta

    def get_sample_info(self, idx):
        """Get human-readable info for a sample (for debugging)"""
        meta = {
            'bearing_id': self.meta[idx][0],
            'timestep': int(self.meta[idx][1]),
            'eof': int(self.meta[idx][2]),
            'rul': float(self.rul[idx])
        }

        # Get FPT
        bearing_id = meta['bearing_id']
        if bearing_id in self.fpt_eof_table:
            fpt, eof = self.fpt_eof_table[bearing_id]
            meta['fpt'] = fpt
            meta['eof_table'] = eof

        return meta


def test_dataset():
    """Test function for BearingRULDataset"""
    import argparse

    # Create dummy args
    args = argparse.Namespace()
    args.dataset_name = 'xjtu'

    # Test data path
    root_path = 'data/processed_data/case1/denoise_off/image_bins_224'

    # Test loading
    for flag in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Testing {flag} dataset")
        print('='*60)

        try:
            dataset = BearingRULDataset(args, root_path, flag=flag)

            # Get first sample
            X, P, rul, meta = dataset[0]
            print(f"\nFirst sample:")
            print(f"  X shape: {X.shape}, dtype: {X.dtype}")
            print(f"  P shape: {P.shape}, dtype: {P.dtype}")
            print(f"  RUL: {rul:.4f}")
            print(f"  Meta: {meta}")

            # Get sample info
            info = dataset.get_sample_info(0)
            print(f"\nDetailed info:")
            for k, v in info.items():
                print(f"  {k}: {v}")

            # Check batch
            print(f"\nBatch test (first 4 samples):")
            batch_X = torch.stack([dataset[i][0] for i in range(min(4, len(dataset)))])
            batch_P = torch.stack([dataset[i][1] for i in range(min(4, len(dataset)))])
            batch_rul = torch.stack([dataset[i][2] for i in range(min(4, len(dataset)))])
            print(f"  Batch X: {batch_X.shape}")
            print(f"  Batch P: {batch_P.shape}")
            print(f"  Batch RUL: {batch_rul.shape}, values: {batch_rul.numpy()}")

        except Exception as e:
            print(f"Error loading {flag} dataset: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    test_dataset()
