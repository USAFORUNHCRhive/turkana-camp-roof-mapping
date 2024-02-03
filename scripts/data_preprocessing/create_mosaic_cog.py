# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
This script creates a mosaic COG file from a list of raster files.

The process involves two steps/functions:
1. gdalbuildvrt: This command is used to create a VRT (Virtual Raster) file
   referencing all the input raster files. This is like creating a virtual
   mosaic of all the input files.
2. gdal_translate: This command is then used to convert the VRT file into a
   single COG (Cloud Optimized GeoTIFF) file. This is like rendering the
   virtual mosaic into a single, efficient GeoTIFF file that can be easily
   used on the cloud.

Usage:
    python create_mosaic_cog.py --input_file_list /path/to/input/file_list.txt
                  --output_file_name /path/to/output/file.tif
                  [--root_dir /path/to/root/directory]
                  [--verbose]

Required arguments:
    -i, --input_file_list  Path to the text file containing the list of input files.
    -o, --output_file_name Name of the output file.

Optional arguments:
    -r, --root_dir  Root directory of the input files. If provided, a temporary text
                    file with the absolute paths to the input files is created.
    -v, --verbose   Print verbose output.

Example:
    python create_mosaic_cog.py -i /path/to/input/file_list.txt
                  -o /path/to/output/file.tif -r /path/to/root/directory -v
"""

import os
import sys
import argparse
from src.gdal_utils import check_file_exists, create_vrt_file, create_cog_file


def create_mosaic_cog(
    input_file_list: str, output_file_name: str, verbose: bool
) -> None:
    """
    Creates a mosaic Cloud Optimized Geotiff (COG) file from a list of raster files.

    Args:
        input_file_list (str): Path to the input file list.
        output_file_name (str): Path to the output file.
        verbose (bool): If True, print verbose output.

    Returns:
        None
    """
    vrt_fn, _ = create_vrt_file(input_file_list, output_file_name, verbose)
    create_cog_file(vrt_fn, output_file_name, verbose)

    # Clean up the temporary VRT file if it exists
    if os.path.exists(vrt_fn):
        os.remove(vrt_fn)
    if verbose:
        print("Temporary VRT file removed.")


def main():
    parser = argparse.ArgumentParser(
        description="This script creates a mosaic COG from a list of raster files."
    )
    parser.add_argument(
        "-i",
        "--input-file-list",
        help="Path to the text file containing the list of input files.",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--root-dir",
        help="Root directory of the input files.",
        default="",
    )
    parser.add_argument(
        "-o",
        "--output-file-name",
        help="Name of the output file.",
        required=True,
    )
    parser.add_argument(
        "-v", "--verbose", help="Print verbose output.", action="store_true"
    )
    args = vars(parser.parse_args())

    # Check if the input file list exists
    try:
        check_file_exists(args["input_file_list"])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # If a root directory is provided, create a temporary text file with the
    # absolute paths to the input files
    input_files = args["input_file_list"]
    if args["root_dir"]:
        input_files = os.path.splitext(args["input_file_list"])[0] + "_abs.txt"
        with open(args["input_file_list"], "r") as f:
            with open(input_files, "w") as f2:
                for line in f:
                    f2.write(
                        os.path.join(args["root_dir"], line.strip()) + "\n"
                    )

    # Create the mosaic COG file
    create_mosaic_cog(
        input_files,
        args["output_file_name"],
        args["verbose"],
    )

    # Clean up the temporary text file if it was created
    if args["root_dir"]:
        try:
            os.remove(input_files)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
