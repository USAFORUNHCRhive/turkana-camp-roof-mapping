# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Processes geospatial data to create semantic segmentation masks.

The main function, `process_annotations`, performs the following steps:
1. Filters raw annotations for polygons and multipolygons, then saves the result.
2. Loads polygon annotations and filters for solar generators and buildings.
3. Creates buffer zones around buildings for a separate class in the mask.
4. Loads as well as samples background point annotations.
8. Combines all annotations geometries into a single GeoDataFrame.

Example:
    $ python collect_annotations.py \
        --config ../../config/examples/semantic_segmentation/collect_annotations.yaml

Attributes:
    TARGET_CRS (str): The target Coordinate Reference System (CRS).
    GEOM_TYPES (list[str]): The geometry types to extract.
    BUFFER_DISTANCE (float): Distance for creating buffer zones around polygons.

Functions:
    extract_features_by_geometry(input_file: fiona.Collection,
                                 geom_types: List[str]) -> List[dict]
    buffer_polygon(geom: Polygon | MultiPolygon,
                   buffer_distance: float) -> Polygon | MultiPolygon
    process_annotations(raw_annotations_path: str,
                        polygon_annotations_path: str,
                        background_annotations_path: str,
                        crs: str,
                        geom_types: List[str],
                        buffer_distance: float) -> gpd.GeoDataFrame
"""
from pathlib import Path
from typing import List

import configargparse
import fiona
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm

TARGET_CRS = "EPSG:32636"
GEOM_TYPES = ["Polygon", "MultiPolygon"]
BUFFER_DISTANCE = 0.25


def extract_features_by_geometry(
    input_file: fiona.Collection,
    geom_types: List[str] = GEOM_TYPES,
) -> List[dict]:
    """
    Extracts geospatial features based on geometry type.

    Args:
        input_file (fiona.Collection): The input geospatial data.
        geom_types (List[str], optional): The geometry types to extract. Defaults to
        ["Polygon", "MultiPolygon"].

    Returns:
        List[dict]: The extracted features.
    """
    features_to_write = []
    for feature in tqdm(input_file):
        if feature["geometry"]["type"] in geom_types:
            features_to_write.append(feature)
    return features_to_write


def buffer_polygon(
    geom: Polygon | MultiPolygon, buffer_distance: float = BUFFER_DISTANCE
) -> Polygon | MultiPolygon:
    """
    Creates a buffer zone around a polygon.

    Args:
        geom (Polygon | MultiPolygon): The input geometry.
        buffer_distance (float, optional): The buffer distance. Defaults to 0.25.

    Returns:
        Polygon | MultiPolygon: The buffered geometry.
    """
    if isinstance(geom, MultiPolygon):
        return MultiPolygon(
            [polygon.exterior.buffer(buffer_distance) for polygon in geom.geoms]
        )
    elif isinstance(geom, Polygon):
        return geom.exterior.buffer(buffer_distance)
    else:
        raise TypeError("Geometry must be a Polygon or MultiPolygon")


def process_annotations(
    raw_annotations_path: str,
    polygon_annotations_path: str,
    background_annotations_path: str,
    crs: str = TARGET_CRS,
    geom_types: List[str] = GEOM_TYPES,
    buffer_distance: float = BUFFER_DISTANCE,
) -> gpd.GeoDataFrame:
    """
    Processes annotation data to help create semantic segmentation masks.

    This function performs the following steps:
    1. Loads raw annotations and filters for polygons/multipolygons.
    2. Filters polygon annotations for solar generators and buildings.
    3. Creates buffer zones around buildings for a separate class in the mask.
    4. Loads background annotations.
    5. Adds class labels.
    6. Combines all annotations into a single GeoDataFrame.

    Args:
        raw_annotations_path (str): Path to the raw annotations file.
        polygon_annotations_path (str): Path to the polygon annotations file.
        background_annotations_path (str): Path to the background annotations file.
        crs (str, optional): The target Coordinate Reference System (CRS).
            Defaults to TARGET_CRS.
        geom_types (List[str], optional): The geometry types to extract.
            Defaults to GEOM_TYPES.
        buffer_distance (float, optional): Distance for creating buffer zones
            around polygons. Defaults to BUFFER_DISTANCE.

    Returns:
        gpd.GeoDataFrame: The processed annotations, including class labels and
            geometries.
    """
    # Load raw annotations and filter for only polygons/multipolygons
    # and save the result to polygon_annotations_path
    with fiona.open(raw_annotations_path, "r") as input_file:
        features = extract_features_by_geometry(input_file, geom_types)
        input_file_schema = input_file.schema
        input_file_crs = input_file.crs
    # Create directories if they don't exist
    Path(polygon_annotations_path).parent.mkdir(parents=True, exist_ok=True)
    with fiona.open(
        polygon_annotations_path,
        "w",
        driver="GPKG",
        crs=input_file_crs,
        schema=input_file_schema,
    ) as output_file:
        output_file.writerecords(features)

    # Load polygon annotations and filter for solar generators and buildings
    polygons_gdf = gpd.read_file(polygon_annotations_path).to_crs(crs)
    solar_gdf = polygons_gdf[
        (polygons_gdf["power"] == "generator")
        & (polygons_gdf["generator:source"] == "solar")
    ].copy()
    buildings_gdf = polygons_gdf[polygons_gdf["building"].notnull()].copy()
    # Create buffer zones around buildings to use as a separate class in the
    # segmentation mask (may help the model learn to separate closeby buildings)
    buildings_gdf["boundary"] = buildings_gdf["geometry"].apply(
        lambda x: buffer_polygon(x, buffer_distance)
    )
    building_boundaries_gdf = gpd.GeoDataFrame(
        geometry=buildings_gdf["boundary"], crs=crs
    )
    # Load background annotations
    background_gdf = gpd.read_file(background_annotations_path).to_crs(crs)

    # Add class labels
    solar_gdf["class"] = "solar"
    buildings_gdf["class"] = "building"
    building_boundaries_gdf["class"] = "building_boundary"
    # background polygons already have a class label, just check that it's correct
    assert background_gdf["class"].unique() == ["background"]

    # Combine annotations into a single GeoDataFrame
    segmentation_classes_gdf = pd.concat(
        [background_gdf, building_boundaries_gdf, buildings_gdf, solar_gdf]
    )
    segmentation_classes_gdf = segmentation_classes_gdf[["class", "geometry"]]
    segmentation_classes_gdf.reset_index(drop=True, inplace=True)
    return segmentation_classes_gdf


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="Configuration file for the script.",
    )
    parser.add_argument(
        "--object-annotations",
        required=True,
        help=(
            "Raw object annotations file. Expects 'building', 'generator',"
            " and 'generator:source' tags."
        ),
    )
    parser.add_argument(
        "--polygon-annotations",
        required=True,
        type=str,
        help="File to save polygon annotations to.",
    )
    parser.add_argument(
        "--background-annotations",
        required=True,
        type=str,
        help="Raw background annotations file. Expects 'background' tag.",
    )
    parser.add_argument(
        "--save-path",
        required=True,
        type=str,
        help="File to save processed annotations to.",
    )
    parser.add_argument(
        "--crs",
        type=str,
        default="EPSG:32636",
        help="The target Coordinate Reference System (CRS)",
    )
    parser.add_argument(
        "--geom-types",
        nargs="+",
        default=["Polygon", "MultiPolygon"],
        help="The geometry types to extract",
    )
    parser.add_argument(
        "--buffer-distance",
        type=float,
        default=0.25,
        help="Buffer distance around polygons, in CRS units.",
    )
    parser.add_argument(
        "-r",
        "--root-dir",
        type=str,
        default="",
        help="Root directory of the project, to prepend to paths specified in"
        "the configuration file.",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="",
        help="Directory where raw data is stored.",
    )
    args = vars(parser.parse_args())
    # prepend project root / data root directory to paths in config file
    # inputs are loaded from folders relative to the raw data directory
    if args["raw_data_dir"] != "" and args["config"] is not None:
        args["object_annotations"] = str(
            Path(args["raw_data_dir"]) / Path(args["object_annotations"])
        )
        args["background_annotations"] = str(
            Path(args["raw_data_dir"]) / Path(args["background_annotations"])
        )
    # outputs are saved to folders relative to the root directory
    if args["root_dir"] != "" and args["config"] is not None:
        args["polygon_annotations"] = str(
            Path(args["root_dir"]) / Path(args["polygon_annotations"])
        )
        args["save_path"] = str(Path(args["root_dir"]) / Path(args["save_path"]))

    # Process annotations
    segmentation_classes_gdf = process_annotations(
        args["object_annotations"],
        args["polygon_annotations"],
        args["background_annotations"],
        crs=args["crs"],
        geom_types=args["geom_types"],
        buffer_distance=args["buffer_distance"],
    )
    # Save annotations
    Path(args["save_path"]).parent.mkdir(parents=True, exist_ok=True)
    segmentation_classes_gdf.to_file(
        args["save_path"],
        driver="GPKG",
    )
