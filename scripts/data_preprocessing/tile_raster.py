# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
This script clips a raster into tiles based on a grid. It can either load an existing
grid from a file or create a new grid over the raster. The resulting tiles are saved to
the specified output directory.

Usage:
    python tile_raster.py -i /path/to/raster -t /path/to/output

Required arguments:
    -i, --input-raster  Path to the geospatial raster file to be tiled.
    -t, --tiles-output  Directory where the tiles will be saved.

Optional arguments:
    -c, --config        Path to the configuration file.
    --grid-init-file    Path to the file used to initialize the grid.
    -g, --grid-output   Directory where the grid will be saved.
    -s, --tile-size     Size of the tiles in meters. Default is 500.
    --overwrite-tiles   If True, existing tiles will be overwritten. Default is False.
    --save-grid-image   Path to save an image of the grid and raster. Default is None.
    -r, --root-dir      Root directory of the project, to prepend to paths specified in
                        the configuration file.

Example:
    python tile_raster.py --input-raster /path/to/raster.tif
                 --tiles-output /path/to/output
                 [--grid-init-file /path/to/grid.shp]
                 [--grid-output /path/to/grid]
                 [--tile-size 500]
                 [--overwrite-tiles True]
                 [--save-grid-image /path/to/grid.png]
                 [--root-dir /path/to/root/directory]
"""

from pathlib import Path
import configargparse
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show
import shapely
import shapely.geometry
from tqdm import tqdm
from src.gdal_utils import check_valid_data_in_roi, clip_raster_to_bbox


def create_tiling_grid(
    raster_path: str,
    tile_size: float,
) -> gpd.GeoDataFrame:
    """
    Create a grid of square tiles over a raster dataset.

    This function generates a grid of square tiles of a specified size over the
    provided raster dataset. The grid is returned as a GeoDataFrame, where each
    row represents a tile and contains its geometric information.

    Args:
        raster_path (str): The path to the raster dataset. This raster will be
            used as the base for creating the grid of tiles.
        tile_size (float): The edge length of each square tile, in units of the
            raster_path CRS. Each tile will be a square with this length as its side.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame representing the grid of tiles.
            Each row in the GeoDataFrame represents a tile and contains its
            geometric information.

    Example:
        raster_path = "path/to/raster.tif"
        tile_size = 100
        grid = create_tiling_grid(raster_path, tile_size)
        print(grid.head())
    """
    with rasterio.open(raster_path) as f:
        raster_crs = f.crs
        left, bottom, right, top = f.bounds

        grid_cells = []
        for x0 in tqdm(np.arange(left, right, tile_size)):
            for y0 in np.arange(bottom, top, tile_size):
                x1 = x0 + tile_size
                y1 = y0 + tile_size
                grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
        grid_cells_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"])
        grid_cells_gdf["id"] = range(len(grid_cells_gdf))
        grid_cells_gdf.set_crs(raster_crs, inplace=True)

    return grid_cells_gdf


def clip_raster_to_tiles(
    raster_path: str,
    tile_extents: gpd.GeoDataFrame,
    tile_output_dir: str,
    overwrite: bool,
):
    """
    Clip raster data to tiles.

    This function clips a raster into tiles using a GeoDataFrame of tile extents.
    It saves the tiles to an output directory and returns a list of created
    tile paths and IDs of empty tiles.

    Args:
        raster_path (str): The path to the raster file.
        tile_extents (gpd.GeoDataFrame): A GeoDataFrame of tile extent vectors.
        tile_output_dir (str): The directory where the tiles should be saved.
        overwrite (bool): If True, existing tiles will be overwritten.

    Returns:
        list: A list of file paths to the created tiles.
        list: A list of IDs of tiles that contained no valid data in the raster.
    """
    print("Clipping raster to tiles...")
    empty_tile_ids = []
    created_tiles = []
    # check that raster and tile CRS match
    raster_crs = rasterio.open(raster_path).crs
    tile_crs = tile_extents.crs
    if raster_crs != tile_crs:
        raise ValueError(
            "The CRS of the raster and the CRS of the tile extents do not match."
        )
    epsg_code = raster_crs.to_epsg()
    for idx, row in tqdm(tile_extents.iterrows(), total=tile_extents.shape[0]):
        valid_patch = check_valid_data_in_roi(raster_path, row.geometry, 0)
        if not valid_patch:
            empty_tile_ids.append(row.id)
            print(
                f"Raster at {raster_path} is empty within the provided polygon."
            )
        else:
            output_file_name = f"{Path(raster_path).stem}_tile_{row.id}.tif"
            clipped_raster_path = Path(tile_output_dir) / output_file_name
            clip_raster_to_bbox(
                raster_path,
                row.geometry,
                clipped_raster_path,
                epsg_code,
                overwrite=overwrite,
            )
            created_tiles.append(str(clipped_raster_path))
    return created_tiles, empty_tile_ids


def parse_args():
    parser = configargparse.ArgParser(
        description=(
            "This script is used to clip a raster into tiles based on a grid. "
            "It can either load an existing grid from a file or create a new grid "
            "over the raster. The resulting tiles are saved to the specified output "
            "directory."
        )
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
        "--input-raster",
        required=True,
        type=str,
        help="Path to the geospatial raster file to be tiled.",
    )
    parser.add_argument(
        "--grid-init-file",
        type=str,
        help="Path to the file used to initialize the grid. If not provided, "
        "a new grid will be created over the raster.",
        default=None,
    )
    parser.add_argument(
        "-g",
        "--grid-output",
        type=str,
        help="Directory where the grid will be saved. If not provided, "
        "the grid will not be saved.",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--tiles-output",
        required=True,
        type=str,
        help="Directory where the tiles will be saved.",
    )
    parser.add_argument(
        "-s",
        "--tile-size",
        type=int,
        help="Size of the tiles in meters. Default is 500. "
        "Each tile will be a square with this length on each side.",
        default=500,
    )
    parser.add_argument(
        "--overwrite-tiles",
        type=bool,
        help="If True, existing tiles will be overwritten. Default is False.",
        default=False,
    )
    parser.add_argument(
        "--save-grid-image",
        help="Path to save an image of the grid and raster. Default is None, "
        "and the image will not be saved.",
        default=None,
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
            "input_raster",
            "grid_init_file",
            "grid_output",
            "tiles_output",
            "save_grid_image",
        ]:
            if args[key] is not None:
                args[key] = str(Path(args["root_dir"]) / Path(args[key]))
    return args


def main():
    args = parse_args()

    # load tiling grid or create a new one
    if args["grid_init_file"]:
        # load grid initialization file
        print("Loading grid initialization file...")
        tiling_grid_gdf = gpd.read_file(args["grid_init_file"])
    else:
        # create a grid over the raster
        print("Creating grid over raster...")
        tiling_grid_gdf = create_tiling_grid(
            args["input_raster"], tile_size=args["tile_size"]
        )

    # clip raster to tiles
    _, empty_tile_ids = clip_raster_to_tiles(
        args["input_raster"],
        tiling_grid_gdf,
        args["tiles_output"],
        args["overwrite_tiles"],
    )
    tiling_grid_gdf.drop(
        tiling_grid_gdf[tiling_grid_gdf["id"].isin(empty_tile_ids)].index,
        inplace=True,
    )

    # save grid
    if args["grid_output"]:
        grid_filename = f"{Path(args['input_raster']).stem}_grid.shp"
        Path(args["grid_output"]).mkdir(parents=True, exist_ok=True)
        tiling_grid_gdf.to_file(Path(args["grid_output"]) / grid_filename)

    # save image of grid
    if args["save_grid_image"]:
        Path(Path(args["save_grid_image"]).parent).mkdir(
            parents=True, exist_ok=True
        )
        with rasterio.open(args["input_raster"]) as f:
            raster = f.read()
            raster_transform = f.transform
        fig, ax = plt.subplots(ncols=1, nrows=1)
        show(raster, transform=raster_transform, ax=ax)
        tiling_grid_gdf.plot(ax=ax, facecolor="none", edgecolor="grey")
        plt.savefig(
            args["save_grid_image"], bbox_inches="tight", transparent=True
        )


if __name__ == "__main__":
    main()
