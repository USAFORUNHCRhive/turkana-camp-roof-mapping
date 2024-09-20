# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from pathlib import Path
from typing import List

import configargparse
import geopandas as gpd
import pandas as pd
import rasterio
from tqdm import tqdm

from src.geo_utils import (
    concat_geo_files,
    exclude_points_within_buffer,
    filter_points_to_raster,
    get_chip_bbox_from_point,
    windowed_raster_read,
)
from src.utils import is_missing_data, next_path


def _filter_dataframe(df, conditions):
    """Used to filter polygon annotations for solar and building polygons.

    Helper function to filter a dataframe based on a dictionary of conditions.
    Each key in the dictionary is a column name, and the value is a function that takes
    a Series and returns a boolean Series.
    """
    for column, condition in conditions.items():
        df = df[condition(df[column])].copy()
    return df


def _sample_diverse_values(df, column, num_to_sample, seed):
    """Used to sample a diverse set of building polygons.

    Helper function to sample a diverse set of rows based on a dataframe column.
    It samples all rpws for values that have fewer than num_to_sample rows,
    and samples num_to_sample rows of each value that has at least num_to_sample rows.

    """
    val_counts = df[column].value_counts()
    few_sample_types = val_counts[val_counts < num_to_sample].index

    # Sample all values of types that have fewer than num_to_sample instances
    few_samples = df[df[column].isin(few_sample_types)]

    # Sample num_to_sample values of each type that has at least num_to_sample instances
    many_samples = df[
        df[column].isin(val_counts.index.difference(few_sample_types))
    ]
    many_samples = many_samples.groupby(column).sample(
        n=num_to_sample, random_state=seed
    )

    return pd.concat([few_samples, many_samples])


def _sample_points_from_footprints(
    footprint_paths, crs, num_to_sample, seed, footprint_threshold=10
):
    """Used to sample backgrround points at random from a region."""
    # Load all footprints, treating multi-polygons as separate polygons
    all_footprints = (
        concat_geo_files(footprint_paths, crs)
        .explode(index_parts=True)
        .reset_index(drop=True)
    )

    # Filter out small polygons and dissolve the rest into a single geometry
    all_footprints = all_footprints.loc[
        all_footprints.area >= footprint_threshold
    ]
    all_footprints = all_footprints.dissolve()

    # Sample points from the combined footprints
    footprint_sample = all_footprints.sample_points(
        num_to_sample, rng=seed
    ).explode(index_parts=False)

    return gpd.GeoDataFrame(geometry=footprint_sample)


def _get_candidate_points(seed):
    """
    Helper function to get candidate points for chip sampling.
    It loads all polygon annotations and filters for solar and building polygons. Then,
    it removes building polygons that contain solar polygons (since these will already
    be included in the solar polygons). Next, it subsamples a diverse set of building
    polygons (since there are many more building polygons than solar polygons). Then,
    it gets centroids of these solar and building polygons.

    To get background points, it samples points from footprints and also loads
    manually-selected background points. It combines the footprint points and manual
    points, making sure to exclude points that are too close to solar or building
    centroids.

    Returns:
    tuple: A tuple containing the solar points, building points, and background points.
    """
    print("Getting candidate points for chip sampling...")
    POLYGON_ANNOTATIONS_PATH = (
        "../../data/interim/annotations/version1updated-polygon.gpkg"
    )
    FOOTPRINT_PATHS = [
        "../../data/interim/mosaic_cog_footprints/kakuma_15.shp",
        "../../data/interim/mosaic_cog_footprints/kakuma_17.shp",
        "../../data/interim/mosaic_cog_footprints/kalobeyei_01.shp",
        "../../data/interim/mosaic_cog_footprints/kalobeyei_03.shp",
    ]
    BACKGROUND_ANNOTATIONS_PATHS = [
        "../../data/raw/annotations/car_locations.shp",
        "../../data/raw/annotations/varied_background_locations.shp",
    ]
    DATA_CRS = "EPSG:32636"

    # Load all polygon annotations
    polygons = gpd.read_file(POLYGON_ANNOTATIONS_PATH).to_crs(DATA_CRS)

    # Filter for solar and building polygons
    solar_conditions = {
        "power": lambda x: x == "generator",
        "generator:source": lambda x: x == "solar",
    }
    building_conditions = {"building": lambda x: ~x.isna()}

    solar = _filter_dataframe(polygons, solar_conditions)
    bldg = _filter_dataframe(polygons, building_conditions)
    print(f"Number of SOLAR polygons: {len(solar)}")

    # Get building polygons that do NOT contain solar polygons
    bldg_contains_solar = gpd.sjoin(
        bldg, solar, how="inner", predicate="contains"
    )
    bldg_no_solar = bldg[~bldg.index.isin(bldg_contains_solar.index)].copy()

    # Subsample diverse set of building polygons
    bldg_no_solar = _sample_diverse_values(
        bldg_no_solar, "building", num_to_sample=150, seed=seed
    )
    print(f"Number of BUILDING polygons: {len(bldg_no_solar)}")

    # Get centroids of solar and building polygons
    solar["geometry"] = solar.centroid
    bldg_no_solar["geometry"] = bldg_no_solar.centroid

    # Sample points from footprints
    footprint_points = _sample_points_from_footprints(
        FOOTPRINT_PATHS, DATA_CRS, num_to_sample=400, seed=seed
    )
    # Load manually-selected background points
    manual_points = concat_geo_files(BACKGROUND_ANNOTATIONS_PATHS, DATA_CRS)
    # Combine footprint points and manual points
    background = pd.concat([footprint_points, manual_points]).reset_index(
        drop=True
    )
    background["background_idx"] = background.index
    # Exclude points that are too close to solar or building centroids
    background = exclude_points_within_buffer(
        background, pd.concat([solar, bldg_no_solar]), 20
    )
    print(f"Number of BACKGROUND points: {len(background)}")

    return solar, bldg_no_solar, background


def _get_valid_chip_extents(
    points: gpd.GeoDataFrame,
    raster_files: List[str],
    existing_chips: gpd.GeoDataFrame = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries()
    ),
) -> gpd.GeoDataFrame:
    """Used to get non-overlapping and data-containing chip extents."""

    def is_overlapping(box, chips):
        return chips.intersects(box).any()

    def get_image_chip(box, src):
        window = rasterio.windows.from_bounds(
            *box.bounds, transform=src.transform
        )
        return src.read(window=window)

    chips = gpd.GeoDataFrame(crs=points.crs, columns=["geometry"])
    for raster_file in tqdm(raster_files):
        with rasterio.open(raster_file) as src:
            centroids_in_raster = filter_points_to_raster(points, src)
            if len(centroids_in_raster) == 0:
                continue
            for point in centroids_in_raster.geometry:
                try:
                    box = get_chip_bbox_from_point(point, src)
                    if is_overlapping(box, chips) or is_overlapping(
                        box, existing_chips
                    ):
                        continue
                    image_chip = get_image_chip(box, src)
                    if is_missing_data(image_chip, missing_threshold=0.1):
                        continue
                    else:
                        box_gdf = gpd.GeoDataFrame(
                            {"geometry": box}, index=[0], crs=chips.crs
                        )
                        chips = pd.concat([chips, box_gdf], ignore_index=True)
                except ValueError:
                    # Error occurs when the point is too close to the edge of the raster
                    continue
    return chips


def _find_raster_file_for_bbox(bbox, raster_files):
    """
    Find the raster file that contains the bounding box
    """
    bbox_left, bbox_bottom, bbox_right, bbox_top = bbox
    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            bounds = src.bounds
        if (
            (bounds.left < bbox_left)
            & (bounds.right > bbox_right)
            & (bounds.top > bbox_top)
            & (bounds.bottom < bbox_bottom)
        ):
            return raster_file
    return None


# SEED = 42
# CHIP_SIZE = 256


def sample_chips(args):
    # Get only the filenames that are common to both the imagery and mask directories
    mask_files_set = set(
        f.stem for f in Path(args.input_mask_dir).rglob("*.tif")
    )
    imagery_files_set = set(
        f.stem for f in Path(args.input_imagery_dir).rglob("*.tif")
    )
    common_files_stem = mask_files_set & imagery_files_set
    mask_files = [
        str(f)
        for f in Path(args.input_mask_dir).glob("*.tif")
        if f.stem in common_files_stem
    ]
    imagery_files = [
        str(f)
        for f in Path(args.input_imagery_dir).glob("*.tif")
        if f.stem in common_files_stem
    ]
    assert len(mask_files) == len(
        imagery_files
    ), "Number of files do not match."

    # Get candidate points for chip sampling
    (
        solar_points,
        bldg_points_no_solar,
        background_points,
    ) = _get_candidate_points(args.seed)

    # Get valid chip extents
    print("Getting valid chip extents from candidate points...")
    # - solar
    solar_chips = _get_valid_chip_extents(solar_points, imagery_files)
    # - building
    bldg_chips = _get_valid_chip_extents(
        bldg_points_no_solar, imagery_files, solar_chips
    )
    if len(bldg_chips) > len(solar_chips):
        bldg_chips = bldg_chips.sample(
            len(solar_chips),
            random_state=args.seed,
        )  # sample same number of chips as solar
    # - background
    background_chips = _get_valid_chip_extents(
        background_points, imagery_files, pd.concat([solar_chips, bldg_chips])
    )
    if len(background_chips) > len(solar_chips):
        background_chips = background_chips.sample(
            len(solar_chips),
            random_state=args.seed,
        )  # sample same number of chips as solar
    
    candidate_chips = pd.concat(
        [solar_chips, bldg_chips, background_chips], ignore_index=True
    )
    print(f"Number of candidate chips: {len(candidate_chips)}")
    print(f"\t- candidate solar chips: {len(solar_chips)}")
    print(f"\t- candidate building chips: {len(bldg_chips)}")
    print(f"\t- candidate background chips: {len(background_chips)}")

    # crop imagery and masks to chips and save
    print("Cropping imagery and masks to chips...")
    rows_to_remove = []
    for idx, chip in tqdm(
        candidate_chips.iterrows(), total=len(candidate_chips)
    ):
        chip_bbox = chip.geometry.bounds
        # find the raster file that contains the chip
        imagery_file = _find_raster_file_for_bbox(chip_bbox, imagery_files)
        if imagery_file is None:
            rows_to_remove.append(idx)
            print(f"Could not find raster file for chip {idx}")
            continue
        # read the image chip and check its size
        image_chip, image_chip_transform = windowed_raster_read(
            imagery_file, chip_bbox
        )
        assert (
            image_chip.shape[1] == args.chip_size
            and image_chip.shape[2] == args.chip_size
        ), "Image chip is not 256x256"
        # read the mask chip and check its size
        mask_file = imagery_file.replace(
            "images", "semantic_segmentation/masks"
        )
        mask_chip, mask_chip_transform = windowed_raster_read(
            mask_file, chip_bbox
        )
        assert (
            mask_chip.shape[1] == args.chip_size
            and mask_chip.shape[2] == args.chip_size
        ), "Mask chip is not 256x256"

        # get the paths to save the chips
        image_chip_pattern = str(
            Path(
                args.chip_output_dir,
            )
            / "images"
            / f"{Path(imagery_file).stem}_chip_%s.tif"
        )
        image_chip_path = next_path(image_chip_pattern)
        mask_chip_path = image_chip_path.replace("/images", "/masks")
        # make directories if they don't exist
        Path(image_chip_path).parent.mkdir(parents=True, exist_ok=True)
        Path(mask_chip_path).parent.mkdir(parents=True, exist_ok=True)

        # write the chip to disk
        with rasterio.open(imagery_file) as src, rasterio.open(
            mask_file
        ) as mask_src:
            chip_meta = src.meta.copy()
            chip_meta.update(
                {
                    "driver": "GTiff",
                    "height": args.chip_size,
                    "width": args.chip_size,
                    "transform": image_chip_transform,
                }
            )
            with rasterio.open(image_chip_path, "w", **chip_meta) as dst:
                dst.write(image_chip)

            chip_mask_meta = mask_src.meta.copy()
            chip_mask_meta.update(
                {
                    "driver": "GTiff",
                    "height": args.chip_size,
                    "width": args.chip_size,
                    "transform": mask_chip_transform,
                }
            )
            with rasterio.open(mask_chip_path, "w", **chip_mask_meta) as dst:
                dst.write(mask_chip)
        candidate_chips.at[idx, "chip_filename"] = image_chip_path

    # remove rows that failed
    print(f"Number of rows to remove: {len(rows_to_remove)}")
    candidate_chips = candidate_chips.drop(rows_to_remove).reset_index(
        drop=True
    )
    print(len(candidate_chips))
    # save the chip locations
    Path(args.chip_locations_save_path).parent.mkdir(
        parents=True, exist_ok=True
    )
    candidate_chips.to_file(args.chip_locations_save_path, driver="GPKG")


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(description="TODO.")
    parser.add_argument(
        "--config",
        is_config_file=True,
        default="../../config/semantic_segmentation/sample_chips.yml",
        help="YAML config file specifying any arguments.",
    )
    parser.add_argument(
        "--chip-size", type=int, default=256, help="Size of the chip in pixels."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generation.",
    )
    parser.add_argument(
        "--chip-locations-save-path",
        type=str,
        required=True,
        help="File to save chip locations to.",
    )
    parser.add_argument(
        "--input-imagery-dir",
        type=str,
        required=True,
        help="Directory containing imagery files.",
    )
    parser.add_argument(
        "--input-mask-dir",
        type=str,
        required=True,
        help="Directory containing masks corresponding to each imagery file.",
    )
    parser.add_argument(
        "--chip-output-dir",
        type=str,
        required=True,
        help="Directory to save chips, with subdirectories for images/ and masks/.",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="",
        help="Root directory of project, to append to relative paths in config.",
    )
    args = parser.parse_args()
    args.input_imagery_dir = Path(args.root_dir) / args.input_imagery_dir
    args.input_mask_dir = Path(args.root_dir) / args.input_mask_dir
    args.chip_output_dir = Path(args.root_dir) / args.chip_output_dir
    args.chip_locations_save_path = (
        Path(args.root_dir) / args.chip_locations_save_path
    )
    sample_chips(args)
