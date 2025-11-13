import warnings
from pathlib import Path
from typing import Optional, Tuple

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset
from trident.Converter import OPENSLIDE_EXTENSIONS, PIL_EXTENSIONS

from nova.utils.deterministic import _set_deterministic

# Constants
VALID_SLIDE_EXTS = OPENSLIDE_EXTENSIONS.union(PIL_EXTENSIONS)


__all__ = [
    "H5Dataset",
]


class H5Dataset(Dataset):
    """
    PyTorch Dataset for loading per-slide patch features and labels from HDF5 files.

    Each item yields patch features, label, and slide ID for a single slide (not grouped by case).

    Args:
        feats_path (str): Path to directory containing .h5 feature files named by slide_id.
        df_path (str): Path to CSV file with metadata, including slide_id and fold columns.
        split (str): Data split to use ('train' or 'test').
        fold_idx (int): Fold index for filtering by fold column.
        sample (bool): If True (and split is 'train'), randomly samples num_features patches per slide.
        label_column (str): Base name of label column (expects '{label_column}_cats' in DataFrame).
        num_features (Optional[int]): Number of patches to sample per slide, if sampling.
        seed (int): Random seed for reproducibility when sampling patches.

    Returns:
        Each __getitem__ yields (features, label, slide_id):
            features (torch.Tensor): Patch features tensor of shape [num_patches, feature_dim].
            label (torch.Tensor): Slide label as a float scalar.
            slide_id (str): Unique slide identifier.

    Notes:
        - Issues a warning if sampling is requested on 'test' split, and disables sampling.
        - Raises AssertionError if a slide's .h5 feature file is missing.
        - Does not save any files or state; all loading is performed at access time.
        - Expects a '{label_column}_cats' integer-coded label column in the input CSV.
        - raises an assertion error if the split is not train or test
        - Features must be stored in 'features' key in the HDF5 file.
        - The dataset does not support loading multiple slides per case; each slide is treated independently.
    """

    def __init__(
        self,
        feats_path: str,
        df_path: str,
        split: str,
        fold_idx: int,
        sample: bool,
        label_column: str,
        num_features: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        assert split in ['train', 'test'], "split must be one of ['train', 'test']"

        if sample and split == 'test':
            warnings.warn("Sampling is not allowed during test split. Setting sample to False.")
            sample = False

        self.fold_idx = fold_idx
        df = pd.read_csv(df_path)
        fold_col = f"fold_{fold_idx}"
        if fold_col not in df.columns:
            raise ValueError(f"Column '{fold_col}' not found in the dataframe at {df_path}.")

        # Filter for the current split
        self.df = df[df[fold_col] == split].reset_index(drop=True)

        self.feats_path = Path(feats_path)
        self.num_features = num_features
        self.sample = sample
        self.split = split
        self.label_column = f"{label_column}_cats"
        self.seed = seed
        _set_deterministic(self.seed)

    def __len__(self) -> int:
        """
        Returns the number of slides in the selected split and fold.

        Returns:
            int: Number of slides available in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Loads patch features and label for a single slide by index.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]:
                - features: Patch feature tensor for the slide.
                - label: Label for this slide.
                - slide_id: Slide identifier as a string.

        Raises:
            AssertionError: If the feature .h5 file does not exist for the slide.
        """
        row = self.df.iloc[idx]
        slide_id = row['slide_id']
        slide_file = self.feats_path / f"{slide_id}.h5"
        assert slide_file.exists(), f"Feature file not found: {slide_file}"

        with h5py.File(slide_file, "r") as f:
            features_dataset = f["features"]
            features = torch.from_numpy(features_dataset[:])  # type: ignore
        label = torch.tensor(row[self.label_column], dtype=torch.float32)

        # Optional: sample patches if needed
        if self.split == 'train' and self.sample and self.num_features is not None:
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(self.seed))[
                    : self.num_features
                ]
            else:
                indices = torch.randint(
                    num_available, (self.num_features,), generator=torch.Generator().manual_seed(self.seed)
                )  # Oversampling
            features = features[indices]

        return features, label, slide_id
