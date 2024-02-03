# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
This script provides functionality to convert a raster image into a vector format
based on a specified raster value. It takes as input a raster image and a raster
value representing a class of interest. The script then polygonizes the raster
image, creating polygons where the raster value matches the class of interest.
The resulting polygons are saved to a specified output path.

The script can be run from the command line with required arguments for the input
raster path, output vector path, and raster value. An optional argument can be
provided to enable or disable verbose output.

Example:
    python polygonize_predictions.py --raster-path /path/to/raster.tif \
        --vector-path /path/to/output.gpkg --raster-value 3 --verbose True

Attributes:
    raster_path (str): Path to the input raster image.
    vector_path (str): Path to save the output vector file.
    raster_value (int): Raster value to polygonize representing one class of interest.
    verbose (bool): Whether to print verbose outputs. Default is True.
"""

import argparse
import numpy as np
import rasterio
import rasterio.features
import fiona
import time


def polygonize_raster(raster_path, vector_path, raster_value=3, verbose=True):
    """
    https://github.com/rasterio/rasterio/blob/main/examples/rasterio_polygonize.py
    """
    tic = time.time()
    with rasterio.open(raster_path) as src:
        image = src.read(1)
        transform = src.transform
    if verbose:
        print(f"Finished reading raster in {time.time()-tic:0.2f} seconds")
    if raster_value is not None:
        mask = (image == raster_value).astype(np.uint8)
    else:
        mask = None

    tic = time.time()
    results = (
        {"properties": {"raster_val": v}, "geometry": s}
        for i, (s, v) in enumerate(
            rasterio.features.shapes(image, mask=mask, transform=transform)
        )
    )
    if verbose:
        print(f"Finished polygonizing raster in {time.time()-tic:0.2f} seconds")

    tic = time.time()
    with fiona.open(
        vector_path,
        "w",
        driver="GPKG",
        crs=src.crs,
        schema={"properties": [("raster_val", "int")], "geometry": "Polygon"},
    ) as dst:
        dst.writerecords(results)
    if verbose:
        print(f"Finished saving vector in {time.time()-tic:0.2f} seconds")


def get_args():
    parser = argparse.ArgumentParser(description="", add_help=True)
    parser.add_argument(
        "--raster-path",
        type=str,
        required=True,
        help="Path to raster to polygonize",
    )
    parser.add_argument(
        "--vector-path",
        type=str,
        required=True,
        help="Path to save polygonized outputs to",
    )
    parser.add_argument(
        "--raster-value",
        type=int,
        required=True,
        help="Raster value to polygonize representing one class of interest",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Whether to print verbose outputs",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    polygonize_raster(
        args.raster_path,
        args.vector_path,
        raster_value=args.raster_value,
        verbose=args.verbose,
    )
