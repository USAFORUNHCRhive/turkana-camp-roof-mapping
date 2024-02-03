# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
get_raster_footprint.py

This script uses GDAL to calculate and save the boundary footprint of a geospatial
raster image. It takes an input raster file, creates a temporary alpha band using the
gdalwarp tool, and then uses the gdal_polygonize.py tool to calculate the boundary
footprint of the image, which is saved as a shapefile at the specified output path.

Usage:
    python get_raster_footprint.py -f input_raster_file -o output_shapefile_path

Arguments:
    -f, --input_raster_file: Path to the input raster file.
    -o, --output_shapefile_path: Path to save the output shapefile.

This script is based on a solution posted on Stack Overflow:
https://gis.stackexchange.com/questions/61512/calculating-image-boundary-footprint-of-satellite-images-using-open-source-too
"""

import pathlib
import argparse
import subprocess
import logging


def main(input_raster_file: str, output_shapefile_path: str):
    """
    This script calculates the boundary footprint of satellite images.
    It uses the gdalwarp and gdal_polygonize.py tools from the GDAL library.

    https://gis.stackexchange.com/questions/61512/calculating-image-boundary-footprint-of-satellite-images-using-open-source-too
    """
    output_file = pathlib.Path(output_shapefile_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    input_file = pathlib.Path(input_raster_file)
    tmpfile = input_file.with_name(
        input_file.stem + "_alphaband" + input_file.suffix
    )

    try:
        command = [
            "gdalwarp",
            "-dstnodata",
            "0",
            "-dstalpha",
            "-of",
            "GTiff",
            str(input_file),
            str(tmpfile),
        ]
        logging.debug(" ".join(command))
        subprocess.run(command)

        command = [
            "gdal_polygonize.py",
            str(tmpfile),
            "-b",
            "4",
            "-f",
            "ESRI Shapefile",
            str(output_file),
        ]
        logging.debug(" ".join(command))
        subprocess.run(command)
    finally:
        if tmpfile.exists():
            tmpfile.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        description="Calculate the boundary footprint of satellite images."
    )
    parser.add_argument(
        "-f",
        "--input_raster_file",
        help="Path to the input raster file.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_shapefile_path",
        help="Path to save the output shapefile.",
        required=True,
    )
    args = parser.parse_args()
    main(args.input_raster_file, args.output_shapefile_path)
