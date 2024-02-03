# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Geospatial data processing functions.

This module provides utility functions to:

Functions:
    concat_geo_files: 
        Load and concatenate geospatial files.
    exclude_points_within_buffer: 
        Exclude points within a buffer distance of any geometry.
    filter_points_to_raster: 
        Filter points to the extent of a raster.
    get_chip_bbox_from_point: 
        Generate a bounding box for a raster image chip centered on a point.
    windowed_raster_read: 
        Perform a windowed read of raster data within a bounding box.

The module uses geopandas, rasterio, shapely, and pandas libraries.
"""
from pathlib import Path
from typing import List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely.geometry


def concat_geo_files(
    paths: List[str], target_crs: Union[str, dict]
) -> gpd.GeoDataFrame:
    """
    Loads multiple geospatial files, projects them to a target CRS, and concatenates
    them.

    Args:
        paths (List[str]): A list of file paths to the geospatial files.
        target_crs (Union[str, dict]): The target coordinate reference system. Can be a
        string or a dictionary.

    Returns:
        gpd.GeoDataFrame: The concatenated GeoDataFrame.

    Raises:
        TypeError: If paths is not a list, all paths are not strings, or
            target_crs is not a string or a dictionary.
        FileNotFoundError: If one or more files do not exist.
    """
    # Input checking
    if not isinstance(paths, list):
        raise TypeError("Paths must be a list.")
    if not all(isinstance(path, str) for path in paths):
        raise TypeError("All paths must be strings.")
    if not all(Path(path).exists() for path in paths):
        raise FileNotFoundError("One or more files do not exist.")
    if not isinstance(target_crs, (str, dict)):
        raise TypeError("Target CRS must be a string or a dictionary.")

    # Main function logic
    return pd.concat([gpd.read_file(path).to_crs(target_crs) for path in paths])


def exclude_points_within_buffer(
    points_df: gpd.GeoDataFrame,
    geometry_df: gpd.GeoDataFrame,
    buffer_distance: float = 20,
) -> gpd.GeoDataFrame:
    """
    Exclude points in a GeoDataFrame that fall within a certain distance of any
    geometry in another GeoDataFrame.

    Args:
        points_df (gpd.GeoDataFrame): A GeoDataFrame of points.
        geometry_df (gpd.GeoDataFrame): A GeoDataFrame of geometries.
        buffer_distance (float, optional): The buffer distance. The units of this
            distance depend on the coordinate reference system (CRS) of the
            GeoDataFrames. If the CRS is geographic (longitude/latitude), the units
            are in degrees. If the CRS is projected, the units are typically in
            meters or feet. Defaults to 20.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame of points that do not fall within the buffer
            distance of any geometry.
    """
    # Reset index of points_df and store original index in a new column
    points_with_original_index = points_df.reset_index().rename(
        columns={"index": "original_index"}
    )

    # Create a copy of geometry_df and buffer the geometries
    buffered_geometry = geometry_df.copy()
    buffered_geometry["geometry"] = buffered_geometry.buffer(
        buffer_distance, cap_style=3
    )

    # Perform a spatial join between points_with_original_index and buffered_geometry
    # to find the points that fall within the buffer
    points_within_buffer = gpd.sjoin(
        points_with_original_index,
        buffered_geometry,
        how="inner",
        predicate="within",
    )

    # Exclude the points that fall within the buffer
    points_excluded = points_with_original_index.loc[
        ~points_with_original_index.index.isin(points_within_buffer.index)
    ]

    # Restore the original index of points_excluded
    points_excluded = points_excluded.set_index("original_index")

    # Remove the original_index column if it exists and return the result
    if "original_index" in points_excluded.columns:
        points_excluded = points_excluded.drop(columns="original_index")

    return points_excluded


def filter_points_to_raster(
    points: gpd.GeoDataFrame, raster: rasterio.io.DatasetReader
) -> gpd.GeoDataFrame:
    """
    Filter points to the extent of a raster.
    """
    # Reproject points to raster CRS if they don't match
    if points.crs != raster.crs:
        points = points.to_crs(raster.crs)

    # Get the raster extent
    xmin, ymin, xmax, ymax = raster.bounds

    # Filter the points to the extent
    points_filtered = points.cx[xmin:xmax, ymin:ymax].copy()

    return points_filtered


def get_chip_bbox_from_point(
    point: shapely.geometry.Point,
    raster: rasterio.io.DatasetReader,
    chip_size: int = 256,
):
    """
    Generates a bounding box (bbox) for a raster image chip, centered on a given point.

    The point is provided in geospatial coordinates and the chip size is defined in
    raster pixels. The function converts the geospatial coordinates of the center point
    to pixel indices, calculates pixel indices for the chip corners, and then converts
    these indices back to geospatial coordinates. The result is a bbox in geospatial
    coordinates representing the chip's extent.

    Args:
        point (shapely.geometry.Point): The geospatial coordinates of the center point
            of the chip.
        raster (rasterio.io.DatasetReader): The raster image from which the chip will
            be extracted.
        chip_size (int, optional): The size of the chip, defined as the length of each
            side in raster pixels. Defaults to 256.

    Raises:
        NotImplementedError: If the raster's transform is not an Affine transform. This
            function currently supports only Affine transforms.
        ValueError: If the calculated chip would fall outside the raster image. This
            can occur if the center point is too close to the edge for the given chip
            size.

    Returns:
        shapely.geometry.Polygon: A bbox, represented as a polygon, that defines the
            geospatial extent of the chip.
    """
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

    # Check if the chip would fall outside the raster
    if (
        chip_row_min < 0
        or chip_col_min < 0
        or chip_row_min + chip_size > raster.height
        or chip_col_min + chip_size > raster.width
    ):
        raise ValueError("Point is too close to the edge of the raster")

    # Calculate the (x, y) coordinates of the top-left corner of the chip
    chip_x_min, chip_y_max = transformer.xy(chip_row_min, chip_col_min)

    # Calculate the (x, y) coordinates of the bottom-right corner of the chip
    chip_x_max, chip_y_min = transformer.xy(
        chip_row_min + chip_size, chip_col_min + chip_size
    )

    # Create a box from the coordinates and return it
    return shapely.geometry.box(chip_x_min, chip_y_min, chip_x_max, chip_y_max)


def windowed_raster_read(
    raster: Union[str, rasterio.io.DatasetReader],
    bbox: Tuple[float, float, float, float],
) -> Tuple[np.array, rasterio.windows.Window]:
    """
    Perform a windowed read of raster data within a bounding box and return the data
    and its window transform.

    Args:
        raster (str or rasterio.io.DatasetReader): Path to the raster file or an open rasterio file.
        bbox (tuple): Bounding box (x_min, y_min, x_max, y_max).

    Returns:
        tuple: A tuple containing the raster data within the bounding box (np.array) and
               its window transform (rasterio.windows.Window).
    """
    if isinstance(raster, str):
        with rasterio.open(raster) as src:
            window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
            window_transform = rasterio.windows.transform(window, src.transform)
            return src.read(window=window), window_transform
    else:
        window = rasterio.windows.from_bounds(*bbox, transform=raster.transform)
        window_transform = rasterio.windows.transform(window, raster.transform)
        return raster.read(window=window), window_transform
