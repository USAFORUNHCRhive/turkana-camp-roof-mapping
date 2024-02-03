# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
This script performs inference for roof material classification on a set of images.

The script loads a trained model from a checkpoint, and applies it to a set of images
specified by the user. For each image, it uses a vector file of detected buildings to
dynamically generate samples for inference. The model's predictions, along with the
prediction entropy and margin, are saved to a GeoPackage file.

Example usage:
    python inference.py --checkpoint-path /path/to/checkpoint.pth \
                        --image-dir /path/to/images \
                        --building-detections-dir /path/to/detections \
                        --output-dir /path/to/output \
                        --batch-size 64 \
                        --num-workers 12 \
                        --device cuda:0
"""
from pathlib import Path

import configargparse
import geopandas as gpd
import numpy as np
import torch
from tqdm import tqdm

from src.datasets import RoofClfDynamicDataset
from src.models import CustomResNet18
from src.utils import load_model


def prediction_entropy(class_probs):
    """
    Compute prediction entropy for a given array of class probabilities.

    Parameters:
    - class_probs (numpy.ndarray): 2D array where each row represents a sample and
                                   each column represents the probability of a class.

    Returns:
    - numpy.ndarray: 1D array containing the prediction entropy for each sample.
    """
    eps = np.finfo(float).eps  # Small epsilon to avoid log(0) issues

    # Ensure the input is a NumPy array
    class_probs = np.array(class_probs)

    # Calculate entropy for each sample
    entropies = -np.sum(class_probs * np.log2(np.clip(class_probs, eps, 1.0)), axis=1)

    return entropies


def prediction_margin(class_probs):
    """
    Compute prediction margin for a given array of class probabilities.

    Parameters:
    - class_probs (numpy.ndarray): 2D array where each row represents a sample and
                                   each column represents the probability of a class.

    Returns:
    - numpy.ndarray: 1D array containing the prediction margin for each sample.
    """
    # Ensure the input is a NumPy array
    class_probs_copy = np.array(class_probs).copy()

    # Find the indices of the two highest probabilities for each sample
    max_indices = np.argmax(class_probs, axis=1)
    max_probs = class_probs[np.arange(class_probs.shape[0]), max_indices]
    class_probs_copy[:, max_indices] = -np.inf  # Set max probabilities to -inf
    second_max_indices = np.argmax(class_probs_copy, axis=1)
    second_max_probs = class_probs_copy[
        np.arange(class_probs_copy.shape[0]), second_max_indices
    ]

    # Calculate margin for each sample
    margins = max_probs - second_max_probs

    return margins


def inference(clf_checkpoint, imagery, detections, outfile, args):
    # Load model
    model = load_model(
        clf_checkpoint,
        CustomResNet18,
        inference=True,
        device=args.device,
    )
    # Load data
    ds = RoofClfDynamicDataset(
        imagery,
        detections,
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    # Run inference
    preds = []
    for batch in tqdm(dl, total=len(dl), desc="Inference", leave=False, position=1):
        batch = batch.to(args.device)
        pred = model(batch)
        pred = torch.softmax(pred, dim=1)
        preds.append(pred.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    # Save predictions
    label_to_class = {0: "metal_sheet", 1: "plastic", 2: "thatch", 3: "other"}
    preds_gdf = gpd.GeoDataFrame(
        {
            "geometry": ds.centroids,
            "p_metal_sheet": preds[:, 0],
            "p_plastic": preds[:, 1],
            "p_thatch": preds[:, 2],
            "p_other": preds[:, 3],
            "entropy": prediction_entropy(preds),
            "margin": prediction_margin(preds),
            "building_material": [label_to_class[x] for x in np.argmax(preds, axis=1)],
        }
    )
    preds_gdf.to_file(outfile, driver="GPKG")


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description="Inference script for roof material classification"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model checkpoint to use for inference",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Path to the directory containing the images to run inference on",
    )
    parser.add_argument(
        "--building-detections-dir",
        type=str,
        required=True,
        help="Path to the directory containing vector files of detected buildings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size to use for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers to use for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference",
    )

    args = parser.parse_args()
    args.device = torch.device(args.device)

    # Create directories if they don't exist
    args.output_dir = (
        Path(args.checkpoint_path).parent.parent
        / "inference"
        / Path(args.checkpoint_path).stem
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over images
    images = sorted(Path(args.image_dir).glob("*.tif"))
    for image in tqdm(images, desc="Images", position=0):
        # Get image name
        image_name = image.stem
        # Get building detections
        detections_dir = Path(args.building_detections_dir)
        detections = list(detections_dir.glob(f"*{image_name}*_buildings.gpkg"))
        assert (
            len(detections) == 1
        ), f"Found {len(detections)} building detections files for {image_name}"
        detections = detections[0]
        # Run inference
        outfile = Path(args.output_dir) / f"{image_name}_predictions.gpkg"
        inference(args.checkpoint_path, image, detections, outfile, args)
