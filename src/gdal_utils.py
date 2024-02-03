# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
This module provides utility functions for working with GDAL.

Functions:
    check_file_exists:
        Checks if a given file exists.
    create_vrt_file:
        Creates a VRT file from a given list of input files.
    create_cog_file:
        Creates a COG file from a VRT file.
    check_valid_data_in_roi:
        Checks if a region of interest in a raster contains valid data in any of the
        specified bands.
    clip_raster_to_bbox:
        Clips a raster to the bounding box of a geographic polygon and saves the result.

Usage:
    from gdal_utils import check_file_exists, create_vrt_file, create_cog_file,
                           check_valid_data_in_roi, clip_raster_to_bbox

    check_file_exists('path/to/file')
    create_vrt_file('path/to/input_file_list', 'output_file_name')
    create_cog_file('path/to/input_file', 'output_cog_file_name')
    check_valid_data_in_roi('path/to/raster', roi_geom, bands)
    clip_raster_to_bbox('path/to/input.tif', clip_polygon, 'path/to/output.tif',
                        '32636', overwrite=True)
"""
import subprocess
from pathlib import Path
from typing import List, Union

import numpy as np
import rasterio
import rasterio.mask
from shapely.geometry import MultiPolygon, Polygon, mapping


def check_file_exists(filename: str) -> None:
    """Check if a file exists."""
    if not Path(filename).is_file():
        raise FileNotFoundError(f"File '{filename}' does not exist.")


def create_vrt_file(
    input_file_list: str, output_file_name: str, verbose: bool = False
) -> str:
    """
    Creates a VRT file from the given input file list.

    Args:
        input_file_list (str): The path to the input file list.
        output_file_name (str): The name of the output file.
        verbose (bool): If True, print statements.

    Returns:
        str: The name of the created VRT file.
    """
    # Check if the input file list exists
    check_file_exists(input_file_list)

    # Check if the files in the input file list exist and are compatible with
    # gdalbuildvrt
    with open(input_file_list, "r") as f:
        for line in f:
            file_path = line.strip()
            check_file_exists(file_path)
            command = ["gdalinfo", file_path]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                raise ValueError(
                    f"File {file_path} is not compatible with gdalbuildvrt."
                )

    vrt_fn = str(Path(output_file_name).resolve().with_suffix(".vrt"))
    command = ["gdalbuildvrt", vrt_fn, "-input_file_list", input_file_list]
    if verbose:
        print(" ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if verbose:
        print(f"VRT file {vrt_fn} created.")
    return vrt_fn, result


def create_cog_file(vrt_fn: str, output_file_name: str, verbose: bool = False) -> None:
    """
    Creates a COG file from the given VRT file.

    Args:
        vrt_fn (str): The name of the VRT file.
        output_file_name (str): The name of the output file.
        verbose (bool): If True, print statements.
    """

    # Check that vrt_fn points to an existing file
    check_file_exists(vrt_fn)

    cog_fn = output_file_name
    command = [
        "gdal_translate",
        "-of", "COG",
        "-co", "BIGTIFF=YES",
        "-co", "NUM_THREADS=ALL_CPUS",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        vrt_fn,
        cog_fn,
    ]
    if verbose:
        print(" ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if verbose:
        print(f"COG file {cog_fn} created.")
    return result


def check_valid_data_in_roi(
    raster_fn: str,
    roi_geom: Polygon | MultiPolygon,
    bands: Union[int, List[int]] = 0,
) -> bool:
    """
    Check if a region of interest in a raster contains valid data in any of the
    specified bands.

    This function checks if any of the specified bands of the raster contain data
    other than the raster's nodata value within the region of interest. If multiple
    bands are specified, the function returns True if any band contains valid data.

    Args:
        raster_fn (str): The filename of the raster to be processed.
        roi_geom (shapely.geometry): The geometry specifying the region of
            interest in the raster to be checked.
        bands (int or list of int, optional): The band or bands to check.
            Defaults to 0, which means only the first band will be checked.

    Returns:
        bool: True if valid data is found in any of the specified bands within the
            region of interest, False otherwise.
    """
    with rasterio.open(raster_fn, "r") as raster:
        mask = [mapping(roi_geom)]
        out_image, _ = rasterio.mask.mask(raster, mask, crop=True)
        out_image = out_image.astype("uint8")
        nodata = raster.nodata

        # If a single band is specified, convert it to a list for consistency
        if isinstance(bands, int):
            bands = [bands]

        # Check each specified band
        for b in bands:
            band_data = out_image[b, :, :]
            if nodata is not None:
                # If nodata value is defined, check for data not equal to nodata
                if np.any(band_data != nodata):
                    return True
            else:
                # If nodata value is not defined, check if there's any non-zero data
                if np.any(band_data != 0):
                    return True

        # If no valid data is found in any of the specified bands, return False
        return False


def clip_raster_to_bbox(
    src_path: str,
    clip_polygon: Polygon | MultiPolygon,
    dst_path: str,
    epsg: str,
    overwrite: bool = False,
):
    """
    Clip a raster to the bounding box of a geographic polygon and save the result.

    Args:
        src_path (str): The file path of the raster to be cropped.
        clip_polygon (geom): The polygon whose bounding box is used to clip the raster.
        dst_path (str): The file path for the output raster.
        epsg (str): The EPSG code (without the "EPSG:" prefix) for the CRS.
        overwrite (bool, optional): A flag indicating whether to overwrite existing
            files. Defaults to False.

    Example:
        >>> from shapely.geometry import Polygon
        >>> clip_polygon = Polygon([(0, 0), (1, 1), (1, 0)])
        >>> clip_raster_to_bbox('path/to/input.tif', clip_polygon,
                                'path/to/output.tif', '32636', overwrite=True)
    """
    dst_path_obj = Path(dst_path)
    dst_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if dst_path_obj.exists() and not overwrite:
        print(f"Image {dst_path} already exists. To overwrite, set overwrite=True.")
        return

    if not check_valid_data_in_roi(src_path, clip_polygon):
        raise ValueError(f"Imagery at {src_path} is empty within the provided polygon.")

    bbox = clip_polygon.bounds
    command = [
        "gdalwarp",
        "-of", "COG",
        "-ot", "Byte",
        "-dstnodata", "0",
        "-co", "BIGTIFF=YES",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-te", str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]),
        "-te_srs", f"EPSG:{epsg}",
        src_path,
        dst_path,
    ]
    subprocess.run(command)
    print(f"Done with image {dst_path}")
