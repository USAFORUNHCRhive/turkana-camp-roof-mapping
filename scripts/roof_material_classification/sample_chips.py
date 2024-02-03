# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
This script processes geospatial data to create chips for roof material classification.

Each chip is a small image cut from a raster file. The first three channels of the chip
contain the RGB values of the raster file, and the fourth channel contains a mask of a
single building within the image whose roof material is to be classified.

The script takes several command-line arguments:
- `--polygons`: Path to the polygon annotations file. Must have a column named
    'roof:material' with roof material labels.
- `--imagery_dir`: Directory of imagery files.
- `--data_crs`: CRS of the imagery files. Defaults to "EPSG:32636".
- `--chip_size`: Size of image chips. Defaults to 256.
- `--output_dir`: Directory to save image chips.

The script can also read these arguments from a config file if `--config` is provided
with the path to the config file.

Example:
        python sample_chips.py --polygons buildings.gpkg --imagery_dir imagery/
        --output_dir chips/
"""
from pathlib import Path

import configargparse
import geopandas as gpd
import rasterio
import rasterio.features
from tqdm import tqdm

from src.geo_utils import (
    filter_points_to_raster,
    get_chip_bbox_from_point,
    windowed_raster_read,
)
from src.utils import next_path, is_missing_data


def _process_building_polygons(polygons):
    """
    Processes building polygons DataFrame.
    Filters out rows with NaN values in 'building' and 'roof:material' columns and
    maps the 'roof:material' column values to a predefined set of categories.

    Args:
        polygons (pd.DataFrame): DataFrame containing building polygons data.

    Returns:
        pd.DataFrame: DataFrame with filtered and mapped 'roof:material' values.
    """
    roof_material_mapping = {
        "metal sheet": "metal_sheet",
        "corrugated": "metal_sheet",
        "Plastic": "plastic",
        "grass": "thatch",
        "concrete": "other",  # only 23 examples of concrete
        "wood": "other",  # only 2 examples of wood
        "mud": "other",  # only 1 example of mud
    }
    polygons = polygons[
        ~polygons["building"].isna() & ~polygons["roof:material"].isna()
    ].copy()
    polygons["roof:material"] = [
        roof_material_mapping[val]
        if val in roof_material_mapping.keys()
        else val
        for val in polygons["roof:material"]
    ]
    return polygons


def _create_chip(box, src, polygons):
    # read the image data in the chip bounding box
    chip_data, chip_transform = windowed_raster_read(src, box.bounds)
    # if missing data, return None
    if is_missing_data(chip_data, missing_threshold=0.1):
        return None, None
    # create a mask for the building polygon
    mask_data = rasterio.features.rasterize(
        polygons,
        out_shape=(chip_data.shape[1], chip_data.shape[2]),
        transform=chip_transform,
        fill=0,
        all_touched=True,
    )
    chip_data[3, :, :] = mask_data  # add mask to alpha channel
    return chip_data, chip_transform


def main(args):
    polygons = gpd.read_file(args["polygons"]).to_crs(args["data_crs"])
    buildings = _process_building_polygons(polygons)
    buildings["centroid"] = buildings.geometry.centroid
    buildings = buildings.rename(columns={"geometry": "polygon"})
    buildings = buildings.set_geometry("centroid")

    raster_files = list(Path(args["imagery_dir"]).glob("*.tif"))
    for raster_file in tqdm(raster_files, desc="Raster Files", position=0):
        with rasterio.open(raster_file) as src:
            # get the buildings whose centroids are in the raster
            raster_buildings = filter_points_to_raster(buildings, src)
            if len(raster_buildings) == 0:
                continue
            for _, row in tqdm(
                raster_buildings.iterrows(),
                desc="Buildings with Centroid in Raster",
                position=1,
                leave=False,
            ):
                # get the chip bounding box from the building centroid
                try:
                    box = get_chip_bbox_from_point(
                        row["centroid"], src, chip_size=args["chip_size"]
                    )
                except ValueError:
                    continue
                chip_data, chip_transform = _create_chip(
                    box, src, [row["polygon"]]
                )
                if chip_data is None:
                    continue
                # save the chip
                chip_pattern = str(
                    Path(args["output_dir"])
                    / row["roof:material"]  # roof material class subdirectory
                    / f"{Path(raster_file).stem}_chip_%s.tif"
                )
                chip_path = next_path(chip_pattern)
                Path(chip_path).parent.mkdir(parents=True, exist_ok=True)
                chip_meta = src.meta.copy()
                chip_meta.update(
                    {
                        "driver": "GTiff",
                        "height": args["chip_size"],
                        "width": args["chip_size"],
                        "transform": chip_transform,
                    }
                )
                with rasterio.open(chip_path, "w", **chip_meta) as dst:
                    dst.write(chip_data)


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add_argument("--config", is_config_file=True)
    parser.add_argument(
        "--polygons",
        type=str,
        required=True,
        help="Polygon annotations file. Must have 'roof:material' column with labels.",
    )
    parser.add_argument(
        "--imagery-dir",
        type=str,
        required=True,
        help="Directory of imagery files.",
    )
    parser.add_argument(
        "--data-crs",
        type=str,
        default="EPSG:32636",
        help="CRS of the imagery files.",
    )
    parser.add_argument(
        "--chip-size", type=int, default=256, help="Size of image chips."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save image chips.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="",
        help="Root directory of project, to append to relative paths in config.",
    )
    args = vars(parser.parse_args())
    # prepend root directory to paths in config file
    if args["root_dir"] and args["config"]:
        for key in ["polygons", "imagery_dir", "output_dir"]:
            if args[key]:
                args[key] = str(Path(args["root_dir"]) / Path(args[key]))
    main(args)
