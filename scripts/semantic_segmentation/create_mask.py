# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
This script creates a raster mask from a GeoJSON/GeoPackage label file.

The script reads an image file and a GeoJSON/GeoPackage label file, and creates a
raster mask where each class from the label file is burned into the raster in the order
specified by the user. If no order is provided, the classes are burned in alphabetical
order. The raster mask is saved to a user-specified output file.

Usage:
        python create_mask.py -i image.tif -l labels.gpkg -o mask.tif

        # With optional label order
        python create_mask.py -i image.tif -l labels.gpkg -o mask.tif -d class1 class2

        # With a configuration file
        python create_mask.py -c config.yml

Example configuration file (config.yml):
        image-file: image.tif
        label-file: labels.gpkg
        label-column: class
        label-order: ["class1", "class2"]
        output-file: mask.tif
"""
import configargparse
import pathlib
import geopandas as gpd
import rasterio
import subprocess
from typing import List, Optional


def get_raster_properties(raster_path: str) -> tuple:
    """
    Get the properties of a raster dataset.

    This function returns the CRS, bounds, width, and height of a raster dataset.

    Args:
        raster_path (str): The path to the raster dataset.

    Returns:
        tuple: A tuple containing the CRS, bounds, width, and height of the raster
            dataset.

    Example:
        raster_path = "path/to/raster.tif"
        crs, bounds, width, height = get_raster_properties(raster_path)
    """
    with rasterio.open(raster_path) as f:
        crs = f.crs
        bounds = f.bounds
        width = f.width
        height = f.height

    return crs, bounds, width, height


def _validate_inputs(
    image_crs: str,
    label_crs: str,
    label_list: Optional[List[str]],
    label_df: gpd.GeoDataFrame,
    label_column: str = "class",
):
    if label_crs != image_crs:
        raise ValueError("CRS mismatch between image raster and label file.")
    if label_list and set(label_list) != set(label_df[label_column].unique()):
        raise ValueError(
            "The ordered label list must contain all the labels in the label file."
        )


def _build_command(
    label_file: str,
    output_file: str,
    where_clause: str,
    burn_value: int,
    left: float,
    bottom: float,
    right: float,
    top: float,
    width: int,
    height: int,
):
    """
    Builds a GDAL command to create a new raster from a GeoJSON/GeoPackage label file.
    file.

    The new raster, same size as the input image, is initialized to 0 (nodata value).
    Then, polygons in the label file with the first class label are burned into
    the raster with the value specified in the class_to_idx dictionary.

    Args:
        label_file (str): Path to the GeoJSON/GeoPackage label file.
        output_file (str): Path to the output raster file.
        where_condition (str): SQL-like where condition to select polygons.
        burn_value (str): Value to burn into the selected polygons.
        left (float): Left bound of the output raster.
        bottom (float): Bottom bound of the output raster.
        right (float): Right bound of the output raster.
        top (float): Top bound of the output raster.
        width (int): Width of the output raster.
        height (int): Height of the output raster.

    Returns:
        List[str]: The GDAL command as a list of strings.
    """
    return [
        "gdal_rasterize",
        "-q",  # be quiet about it
        "-ot",
        "Byte",  # the output dtype of the raster should be uint8
        "-a_nodata",
        "0",  # nodata value should be "0", represents not-labeled
        "-init",
        "0",  # initialize all values to 0
        "-burn",
        str(burn_value),
        # burn the first class value into GeoJSON polygons with the first class label
        "-of",
        "GTiff",  # the output should be a GeoTIFF
        "-co",
        "TILED=YES",
        # tile the output, similar to COGs -- https://www.cogeo.org/ --
        # this is important for fast windowed reads
        "-co",
        "BLOCKXSIZE=512",  # this is important for fast windowed reads
        "-co",
        "BLOCKYSIZE=512",  # this is important for fast windowed reads
        "-co",
        "INTERLEAVE=PIXEL",  # this is important for fast windowed reads
        "-where",
        where_clause,  # f"class='{classes[0]}'",
        # burn in values for polygons where the class label is the first class label
        "-te",
        str(left),
        str(bottom),
        str(right),
        str(top),
        # the output GeoTIFF should cover the same bounds as the input image
        "-ts",
        str(width),
        str(height),
        # the output GeoTIFF should have the same height and width as the input image
        "-co",
        "COMPRESS=LZW",  # compress it
        "-co",
        "PREDICTOR=2",  # compress it good
        "-co",
        "BIGTIFF=YES",  # just incase the image is bigger than 4GB
        label_file,
        output_file,
    ]


def _execute_gdal_command(command: List[str]) -> None:
    """Executes a GDAL command and raises an exception if it fails."""
    try:
        result = subprocess.call(command)
        if result != 0:
            raise RuntimeError(f"GDAL command failed with exit code {result}")
    except Exception as e:
        raise RuntimeError("Failed to execute GDAL command") from e


def create_mask(
    image_file: str,
    label_file: str,
    output_file: str,
    label_order: Optional[List[str]] = None,
    label_column: str = "class",
):
    """
    Creates a raster mask from a GeoJSON/GeoPackage label file.

    This function reads an image file and a label file, and creates a raster mask where
    each class from the label file is burned into the raster in the order specified by
    `label_order`. If `label_order` is not provided, the classes are burned in
    alphabetical order. The raster mask is saved to `output_file`.

    Args:
        image_file (str): Path to the input image file.
        label_file (str): Path to the GeoJSON/GeoPackage label file.
        output_file (str): Path to the output raster file.
        label_order (List[str], optional): List of class labels in the order they
            should be burned into the raster. If not provided, labels are burned in
            alphabetical order. Defaults to None.
        label_column (str, optional): Name of the column in the label file that contains
            the class labels. Defaults to "class".
    """
    image_crs, bounds, width, height = get_raster_properties(image_file)
    left, bottom, right, top = bounds
    label_df = gpd.read_file(label_file)
    label_crs = label_df.crs
    _validate_inputs(image_crs, label_crs, label_order, label_df, label_column)

    classes = (
        label_order
        if label_order
        else sorted(label_df[label_column].value_counts().index.values.tolist())
    )
    class_to_idx = {
        class_name: i + 1  # we add one to reserve class 0 as the nodata class
        for i, class_name in enumerate(classes)
    }
    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    command = _build_command(
        label_file,
        output_file,
        f"{label_column}='{classes[0]}'",
        class_to_idx[classes[0]],
        left,
        bottom,
        right,
        top,
        width,
        height,
    )
    print(" ".join(command))
    _execute_gdal_command(command)
    print(
        f"Mask file {output_file} created with class 0 (nodata) and "
        f"{class_to_idx[classes[0]]} ({classes[0]})"
    )

    # this burns in the class values for each other class in place
    for i in range(1, len(classes)):
        command = [
            "gdal_rasterize",
            "-q",  # be quiet about it
            "-b",
            "1",  # burn into band 1
            "-burn",
            str(class_to_idx[classes[i]]),
            "-where",
            f"{label_column}='{classes[i]}'",
            label_file,
            output_file,
        ]
        print(" ".join(command))
        _execute_gdal_command(command)
        print(f"Class {class_to_idx[classes[i]]} ({classes[i]}) added to mask.")


def main():
    parser = configargparse.ArgParser(
        description="Creates a mask for an image from a GeoJSON/GeoPackage label file."
    )
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        type=str,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-i",
        "--image-file",
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "-l",
        "--label-file",
        required=True,
        help="Path to the GeoJSON/GeoPackage label file.",
    )
    parser.add_argument(
        "--label-column",
        default="class",
        help=(
            "Name of the column in the label file that contains "
            "the class labels."
        ),
    )
    parser.add_argument(
        "-d",
        "--label-order",
        nargs="+",
        default=[],
        help=(
            "Optional list of class labels, in the order they should be "
            "burned into the raster. If not provided, labels will be "
            "burned in alphabetical order."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="Path to the output raster file.",
    )
    parser.add_argument(
        "-r",
        "--root-dir",
        help="Root directory of the project, to prepend to paths specified in"
        "the configuration file.",
        default="",
    )
    args = vars(parser.parse_args())
    # prepend root directory to paths in config file
    if args["root_dir"] != "" and args["config"] is not None:
        for key in [
            "image_file",
            "label_file",
            "output_file",
        ]:
            if args[key] is not None:
                args[key] = str(
                    pathlib.Path(args["root_dir"]) / pathlib.Path(args[key])
                )
    create_mask(
        args["image_file"],
        args["label_file"],
        args["output_file"],
        args["label_order"],
        args["label_column"],
    )


if __name__ == "__main__":
    main()
