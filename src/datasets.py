# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Module datasets.py contains classes and functions defining PyTorch datasets for 
geospatial machine learning tasks.

Functions:
    preprocess(sample): Preprocesses a sample by normalizing the image data to [0, 1] 
        and removing extra dimensions from the mask data.

Classes:
    ChipDataset: A PyTorch Dataset for handling geospatial data chips and their 
        corresponding masks. This class works with a list of paths to image chips and 
        a directory containing corresponding mask files.

    SingleRasterDataset: A subclass of RasterDataset that creates a dataset from a 
        single TIFF file. Unlike the parent class, which operates on a directory of 
        TIFF files, this class works with a single file.

    RoofClfPreSampledDataset: A PyTorch Dataset for roof classification. This 
        dataset is pre-sampled, with samples already prepared and stored in a 
        DataFrame. The dataset reads image data from provided paths and applies 
        optional transformations.

    RoofClfDynamicDataset: A PyTorch Dataset for dynamic roof classification with 
        geospatial data. This dataset creates roof classification samples from a 
        vector file with polygons representing detected buildings.
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.io
import shapely.geometry.point
import torch
from torch.utils.data import Dataset
from torchgeo.datasets import NonGeoDataset, RasterDataset


def preprocess(
    sample: Dict[str, Union[torch.Tensor, torch.Tensor]]
) -> Dict[str, Union[torch.Tensor, torch.Tensor]]:
    """
    Preprocesses a sample by normalizing the image data to [0, 1] and removing extra
    dimensions from the mask data.

    Args:
        sample (Dict[str, Union[torch.Tensor, torch.Tensor]]): A dictionary containing
            the image and mask data as torch.Tensors.

    Returns:
        Dict[str, Union[torch.Tensor, torch.Tensor]]: The preprocessed sample.
    """
    if "image" in sample:
        # Check if the image is in the range 0-255
        if sample["image"].max() > 1:
            sample["image"] = (
                sample["image"] / 255.0
            ).float()  # inputs are normalized to [0, 1]
        else:
            sample["image"] = sample["image"].float()
    if "mask" in sample:
        sample["mask"] = sample["mask"].squeeze().long()

    return sample


class ChipDataset(NonGeoDataset):
    """
    A PyTorch Dataset for handling geospatial data chips and their corresponding masks.

    This class is designed to work with a list of paths to image chips and a directory
    containing corresponding mask files. The image chips should be raster files with
    three bands (RGB) and float values. The mask files should have the same name as the
    image chips and should be single-band raster files with integer values corresponding
    to the class labels.

    Attributes:
        chip_paths (list): A list of paths to the image chips.
        mask_dir (Path): A Path object pointing to the directory containing the masks.
        transforms (callable, optional): Optional transforms to be applied on a sample.

    Methods:
        __len__(): Returns the number of image chips.
        _load_image(index): Loads the image data for the chip at the given index.
        _get_mask_path(chip_path): Returns the path to the mask file corresponding to
                                   the given chip file.
        _load_target(index): Loads the mask data for the chip at the given index.
        __getitem__(index): Returns the image and mask data for the chip at the given
                            index, with transforms applied if specified.
        plot(sample, show_titles=False, **kwargs): Plots the image and mask data for a
                                                    sample, and optionally a prediction.
    """

    def __init__(
        self,
        chip_paths: List[str],
        mask_dir: Union[str, Path],
        transforms: Optional[Callable] = None,
    ):
        self.chip_paths = chip_paths
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.chip_paths)

    def _load_image(self, index: int) -> torch.Tensor:
        """
        Loads the image data for the chip at the given index.

        Args:
            index (int): Index of the chip to load.

        Returns:
            torch.Tensor: A tensor containing the image data.
        """
        chip_path = self.chip_paths[index]
        with rasterio.open(chip_path) as image_data:
            image_array = image_data.read()[0:3, :, :].astype(float)
        image_tensor = torch.from_numpy(image_array)
        return image_tensor

    def _get_mask_path(self, chip_path: Union[str, Path]) -> str:
        """
        Get the path to the mask corresponding to a chip.

        Args:
            chip_path (Union[str, Path]): Path to the chip.

        Returns:
            str: Path to the mask.
        """
        chip_path = Path(chip_path)
        mask_path = self.mask_dir / chip_path.name
        return str(mask_path)

    def _load_target(self, index):
        mask_path = self._get_mask_path(self.chip_paths[index])
        with rasterio.open(mask_path) as target_data:
            target_array = target_data.read(1).astype(np.int32)
        # NOTE: MOVE THE FOLLOWING HACK TO AFTER TRANSFORMS
        # target_array[target_array == 0] = 1  # HACK: replace nodata with background
        target_tensor = torch.from_numpy(target_array)
        return target_tensor

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, torch.Tensor]]:
        """
        Returns the image and mask data for the chip at the given index, with transforms
        applied if specified.

        Args:
            index (int): Index of the chip to return.

        Returns:
            dict: A dictionary containing the image and mask data as torch.Tensors.
        """
        image = self._load_image(index)
        label = self._load_target(index)
        sample = {"image": image, "mask": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        # HACK: replace nodata with background
        mask = sample["mask"]
        mask[mask == 0] = 1
        sample["mask"] = mask

        return sample

    def plot(self, sample, show_titles=False, **kwargs):
        """
        Plots the image, mask, and optionally the prediction from a sample.

        Args:
            sample (dict): A dictionary containing the image and mask data as
                        torch.Tensors, and optionally the prediction data.
            show_titles (bool, optional): Whether to display titles on the plots.
                        Defaults to False.
            **kwargs: Arbitrary keyword arguments for the imshow function.

        Returns:
            matplotlib.figure.Figure: The figure object containing the plots.
        """

        def plot_image(ax, image, title=None):
            ax.imshow(image, **kwargs)
            ax.axis("off")
            if title and show_titles:
                ax.set_title(title)

        if "prediction" in sample:
            prediction = sample["prediction"]
            n_cols = 3
        else:
            n_cols = 2
        image, mask = sample["image"], sample["mask"]

        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, n_cols * 5))
        plot_image(axs[0], image.permute(1, 2, 0), "Image")
        plot_image(axs[1], mask, "Mask")

        if "prediction" in sample:
            plot_image(axs[2], prediction, "Prediction")

        return fig


class SingleRasterDataset(RasterDataset):
    """
    A subclass of RasterDataset that creates a dataset from a single TIFF file.
    Unlike the parent class, which operates on a directory of TIFF files, this class
    works with a single file.
    """

    def __init__(self, fn, transforms=None):
        fn_path = Path(fn)
        self.filename_regex = fn_path.name
        super().__init__(root=str(fn_path.parent), transforms=transforms)


class RoofClfPreSampledDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.transforms = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]["image"]
        with rasterio.open(image_path) as image_data:
            image_array = image_data.read().astype(float)
        image_tensor = torch.from_numpy(image_array)
        if self.transforms:
            image_array = self.transforms(image_tensor)
        label = self.df.iloc[idx]["label"]
        return image_array, label

    def set_transforms(self, transforms):
        self.transforms = transforms

    def plot_sample(self, idx, show_mask=True):
        sample_raster, sample_label = self[idx]
        sample_img = sample_raster[0:3, :, :].numpy()
        if show_mask:
            sample_mask = sample_raster[3, :, :].numpy()
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(sample_img.transpose(1, 2, 0))
        if show_mask:
            ax.imshow(sample_mask, alpha=0.5)
        ax.set_title("Label: {}".format(sample_label))
        ax.set_axis_off()


class RoofClfDynamicDataset(Dataset):
    """A PyTorch Dataset for dynamic roof classification with geospatial data.

    This dataset dynamically creates roof classification samples from a vector file
    with polygons representing detected buildings. For each building, it extracts a
    normalized imagery chip, generates a mask from the polygon, and adds the mask as an
    extra channel to the chip. This allows model inference with a single 4-channel chip
    input.

    Attributes:
        imagery_file (str): Path to the imagery file.
        polygons_file (str): Path to the polygons file.
        polygons (GeoDataFrame): GeoDataFrame containing the polygons.
        mask_channel (int): Channel to use for the mask.
        normalization_value (Union[int, float]): Value to use for normalization.
    """

    def __init__(
        self,
        imagery_file: str,
        polygons_file: str,
        area_threshold: Union[int, float] = 1,
        mask_channel: int = 3,
        normalization_value: Union[int, float] = 255.0,
    ) -> None:
        """Initialize the dataset with an imagery file and polygons file.

        Args:
            imagery_file (str): Path to the imagery file.
            polygons_file (str): Path to the polygons file.
            area_threshold (Union[int, float], optional): Threshold for filtering
                polygons, in the units of the CRS of polygons. Defaults to 1.
            mask_channel (int, optional): Channel to use for the mask. Defaults to 3.
            normalization_value (Union[int, float], optional): Value to use for
                normalization. Defaults to 255 assuming image data ranges from 0 to 255.
        """
        self.validate_files(imagery_file, polygons_file)

        self.imagery_file: str = imagery_file
        self.polygons_file: str = polygons_file
        self.polygons: gpd.GeoDataFrame = gpd.read_file(polygons_file)
        self.filter_polygons(area_threshold)
        self.mask_channel: int = mask_channel
        self.normalization_value: Union[int, float] = normalization_value

    def validate_files(self, imagery_file: str, polygons_file: str) -> None:
        """Validate that the imagery and polygons files exist and have the same CRS.

        Args:
            imagery_file (str): Path to the imagery file.
            polygons_file (str): Path to the polygons file.
        """
        imagery_path = Path(imagery_file)
        polygons_path = Path(polygons_file)
        assert imagery_path.is_file(), f"{imagery_file} does not exist"
        assert polygons_path.is_file(), f"{polygons_file} does not exist"

        with rasterio.open(imagery_file) as src:
            imagery_crs = src.crs
        polygons_crs = gpd.read_file(polygons_file).crs
        assert (
            imagery_crs == polygons_crs
        ), f"CRS mismatch: {imagery_crs} != {polygons_crs}"

    def filter_polygons(self, area_threshold: Union[int, float]) -> None:
        """Filter out polygons smaller than a given area threshold.

        Args:
            area_threshold (Union[int, float]): Threshold for filtering polygons.
        """
        self.polygons["area"] = self.polygons.area
        self.polygons = self.polygons[
            self.polygons["area"] > area_threshold
        ].reset_index(drop=True)
        self.centroids = self.polygons.geometry.centroid

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.centroids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return the chip data and mask for a given index.

        Args:
            idx (int): Index of the sample to return.

        Returns:
            Tensor: Chip data and mask for the given index.
        """
        with rasterio.open(self.imagery_file) as src:
            bbox = self.get_chip_bbox_from_point(self.centroids[idx], src)
            window = rasterio.windows.from_bounds(*bbox.bounds, transform=src.transform)
            chip_data = src.read(window=window, boundless=True, fill_value=src.nodata)
            chip_data = chip_data / self.normalization_value
            mask_data = rasterio.features.rasterize(
                [self.polygons.geometry[idx]],
                out_shape=(chip_data.shape[1], chip_data.shape[2]),
                transform=src.window_transform(window),
                fill=0,
                all_touched=True,
            )
            chip_data[self.mask_channel, :, :] = mask_data
        return torch.from_numpy(chip_data).float()

    def get_chip_bbox_from_point(
        self,
        point: shapely.geometry.Point,
        raster: rasterio.io.DatasetReader,
        chip_size: int = 256,
    ):
        # NOTE: differs from geo_utils.get_chip_bbox_from_point in that it returns a
        # bounding box even if the point is too close to the edge of the raster

        # Get the raster's transform
        transform = raster.transform

        # Check if the transform is an Affine transform
        if isinstance(transform, rasterio.transform.Affine):
            transformer = rasterio.transform.AffineTransformer(transform)
        else:
            raise NotImplementedError("Only Affine Transforms are supported")

        # Convert the point's coordinates to row and column indices in the raster
        point_row, point_col = transformer.rowcol(point.x, point.y)

        # Calculate the row and column indices of the top-left corner of the chip
        chip_row_min = point_row - chip_size // 2
        chip_col_min = point_col - chip_size // 2

        # Calculate the (x, y) coordinates of the top-left corner of the chip
        chip_x_min, chip_y_max = transformer.xy(chip_row_min, chip_col_min)

        # Calculate the (x, y) coordinates of the bottom-right corner of the chip
        chip_x_max, chip_y_min = transformer.xy(
            chip_row_min + chip_size, chip_col_min + chip_size
        )

        # Create a box from the coordinates and return it
        return shapely.geometry.box(chip_x_min, chip_y_min, chip_x_max, chip_y_max)

    def plot(self, idx: int) -> None:
        """Plot the raster data and mask for a given index."""
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(self[idx][: self.mask_channel, :, :].permute(1, 2, 0))
        ax[1].imshow(self[idx][self.mask_channel, :, :])
        plt.show()
