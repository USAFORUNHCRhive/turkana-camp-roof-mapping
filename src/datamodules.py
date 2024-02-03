# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
This module contains the ChipSegmentationDataModule class which is used for loading and 
preprocessing chip segmentation datasets.

The ChipSegmentationDataModule class extends the PyTorch LightningDataModule and provides 
methods for setting up and loading training, validation, and test datasets. It also 
provides methods for plotting samples from the datasets.

Classes:
    ChipSegmentationDataModule: A class for loading and preprocessing chip segmentation datasets.

Functions:
    plot_transforms(N=3): Plots N samples from the training set with 'train' and 'eval' 
    transforms applied.
    train_dataloader(): Returns a DataLoader for the training set.
    val_dataloader(): Returns a DataLoader for the validation set.
    test_dataloader(): Returns a DataLoader for the test set.
    plot(sample): Plots a sample from the training set.

Variables:
    SEGMENTATION_CMAP: A colormap for segmentation masks.
"""
from typing import Callable, Dict, List

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from torch.utils.data import DataLoader

from src.datasets import ChipDataset, preprocess

SEGMENTATION_CMAP = colors.ListedColormap(
    [(1, 1, 1), (0.5, 0.5, 0.5), (0, 0, 1), (1, 0, 0), (1, 1, 0)]
)


class ChipSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_chip_paths: List[str],
        val_chip_paths: List[str],
        test_chip_paths: List[str],
        mask_dir: str,
        transforms: Dict[str, Callable] = {
            "train": preprocess,
            "eval": preprocess,
        },
        batch_size: int = 64,
        num_workers: int = 6,
    ):
        """
        Initialize a DataModule for chip segmentation.

        Args:
            train_chip_paths (List[str]): Paths to training chip files.
            val_chip_paths (List[str]): Paths to validation chip files.
            test_chip_paths (List[str]): Paths to test chip files.
            transforms (Dict[str, Callable]): Dictionary of transformations to apply to the datasets.
            batch_size (int): Batch size for the DataLoader.
            num_workers (int): Number of workers for the DataLoader.
        """
        super().__init__()
        self.train_chip_paths = train_chip_paths
        self.val_chip_paths = val_chip_paths
        self.test_chip_paths = test_chip_paths
        self.mask_dir = mask_dir

        if not all(key in transforms for key in ["train", "eval"]):
            raise ValueError("Transforms should include both 'train' and 'eval' keys.")
        self.transforms = transforms

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Setup the dataset for each split (train, validation, test).
        """
        self.train_ds = ChipDataset(
            self.train_chip_paths,
            self.mask_dir,
            transforms=self.transforms["train"],
        )
        self.val_ds = ChipDataset(
            self.val_chip_paths,
            self.mask_dir,
            transforms=self.transforms["eval"],
        )
        self.test_ds = ChipDataset(
            self.test_chip_paths,
            self.mask_dir,
            transforms=self.transforms["eval"],
        )

    def plot_transforms(self, N=3):
        """
        Plots N samples from the training set with 'train' and 'eval' transforms applied.

        Args:
            N (int, optional): Number of samples to plot. Defaults to 3.

        Returns:
            matplotlib.figure.Figure: A matplotlib Figure object with the plotted images and masks.
        """
        self.setup()
        # Get N random samples from the training set
        N = min(N, len(self.train_ds))
        sample_indices = np.random.permutation(len(self.train_ds))[:N]
        # Create datasets with 'train' and 'eval' transforms applied
        datasets = {
            "eval": ChipDataset(
                [self.train_chip_paths[i] for i in sample_indices],
                self.mask_dir,
                transforms=self.transforms["eval"],
            ),
            "train": ChipDataset(
                [self.train_chip_paths[i] for i in sample_indices],
                self.mask_dir,
                transforms=self.transforms["train"],
            ),
        }

        fig, axs = plt.subplots(N, 4)
        for i in range(N):
            for j, (key, ds) in enumerate(datasets.items()):
                transformed_sample = ds[i]  # Sample with transforms applied
                axs[i, j * 2].imshow(transformed_sample["image"].permute(1, 2, 0))
                axs[i, j * 2].set_title(f"Image ({key.capitalize()})")
                axs[i, j * 2 + 1].imshow(
                    transformed_sample["mask"],
                    cmap=SEGMENTATION_CMAP,
                    vmin=0,
                    vmax=4,
                )
                axs[i, j * 2 + 1].set_title(f"Mask ({key.capitalize()})")

        # Hide x and y labels
        for ax in axs.flat:
            ax.label_outer()
        return fig

    def train_dataloader(self):
        """
        Return DataLoader for the training set.
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Return DataLoader for the validation set.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        """
        Return DataLoader for the test set.
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def plot(self, sample):
        """
        Plot a sample from the training set.
        """
        return self.train_ds.plot(sample)
