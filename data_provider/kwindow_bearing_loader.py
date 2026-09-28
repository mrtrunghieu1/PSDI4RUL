"""
K-Window Bearing RUL Dataset Loader

Extends BearingRULDataset to return K consecutive PSDI images per sample,
enabling temporal context for smoother RUL predictions.

When K=1, behavior is identical to BearingRULDataset.
"""

import numpy as np
import torch
from collections import defaultdict

from data_provider.bearing_loader import BearingRULDataset


class KWindowBearingRULDataset(BearingRULDataset):
    """
    K-Window dataset that returns K consecutive PSDI images per sample.

    For sample at timestep t, returns images [t-K+1, ..., t].
    Boundary: if t < K-1, pads by repeating the earliest available image.
    Never crosses bearing boundaries.

    Args:
        args: Must have k_window_size attribute
        root_path: Path to train/val/test.npz
        flag: 'train', 'val', or 'test'

    Returns:
        X_window: [K, 2, 224, 224] - K consecutive images
        P_window: [K, 2, 2560] - K consecutive phase vectors
        rul: scalar RUL at timestep t (last image)
        meta: dict with bearing_id (str), timestep (int for last), eof (int)
    """

    def __init__(self, args, root_path, flag='train'):
        super().__init__(args, root_path, flag)
        self.K = getattr(args, 'k_window_size', 1)
        self._build_bearing_index()

        print(f"  K-Window size: {self.K}")
        print(f"  Bearings: {list(self.bearing_groups.keys())}")

    def _build_bearing_index(self):
        """
        Build per-bearing index mapping for efficient K-window retrieval.

        Creates:
            bearing_groups: {bearing_id: [idx0, idx1, ...]} sorted by timestep
            sample_to_position: {idx: (bearing_id, position_in_group)}
        """
        # Group sample indices by bearing_id
        groups = defaultdict(list)
        for idx in range(len(self.meta)):
            bid = self.meta[idx][0]
            ts = int(self.meta[idx][1])
            groups[bid].append((ts, idx))

        # Sort each group by timestep and store
        self.bearing_groups = {}
        self.sample_to_position = {}

        for bid, ts_idx_list in groups.items():
            ts_idx_list.sort(key=lambda x: x[0])  # sort by timestep
            sorted_indices = [idx for _, idx in ts_idx_list]
            self.bearing_groups[bid] = sorted_indices

            for pos, idx in enumerate(sorted_indices):
                self.sample_to_position[idx] = (bid, pos)

    def __getitem__(self, idx):
        """
        Get K-window sample ending at idx.

        Returns:
            X_window: [K, 2, 224, 224] tensor
            P_window: [K, 2, 2560] tensor
            rul: scalar RUL at timestep t (last image)
            meta: dict with bearing_id, timestep, eof
        """
        bid, pos = self.sample_to_position[idx]
        group_indices = self.bearing_groups[bid]

        # Collect K indices within same bearing
        window_indices = []
        for k in range(self.K):
            # Position for k-th element: pos - (K-1) + k
            src_pos = pos - (self.K - 1) + k
            # Clamp to [0, pos] to handle boundary (pad with earliest)
            src_pos = max(0, src_pos)
            window_indices.append(group_indices[src_pos])

        # Stack X and P for K timesteps
        X_list = []
        P_list = []
        for wi in window_indices:
            X_list.append(self.X[wi])
            P_list.append(self.P[wi])

        X_window = torch.from_numpy(np.stack(X_list, axis=0)).float()  # [K, 2, 224, 224]
        P_window = torch.from_numpy(np.stack(P_list, axis=0)).float()  # [K, 2, 2560]

        # RUL target: last timestep only
        rul = torch.tensor(self.rul[idx], dtype=torch.float32)

        # Meta: use last timestep info
        meta = {
            'bearing_id': self.meta[idx][0],
            'timestep': int(self.meta[idx][1]),
            'eof': int(self.meta[idx][2])
        }

        return X_window, P_window, rul, meta
