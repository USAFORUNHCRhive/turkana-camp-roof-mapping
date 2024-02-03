# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
General utility functions and classes.

Functions:
    next_path(path_pattern): 
        Finds the next free path in a file list.
    is_missing_data(data_array, ignore_channels, nodata_value, missing_threshold):
        Checks if missing data in a numpy ndarray exceeds a threshold.
    split_chips(tile_split_file, chip_dir, chip_suffix): 
        Splits chips into train, validation, and test sets.
    set_seed(seed): 
        Sets the seed for random numbers in Python, NumPy, and PyTorch.
    load_model(checkpoint_path, ModelClass, inference, device): 
        Loads a model from a checkpoint.    
    get_output_shape(model, input_shape): 
        Computes the output shape of a PyTorch model.
    fit(model, dataloader, criterion, optimizer, device): 
        Trains a model for one epoch.    
    validate(model, dataloader, criterion, device): 
        Validates a model for one epoch.    
    predict(model, dataloader, device): 
        Makes predictions using a model.

Classes:
    TrainingConfig: 
        A class for training settings. 
    MetricLogger: 
        A class for logging metrics during model training.
"""
import random
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Type, Union

import numpy as np
import torch
import yaml
from lightning.pytorch import LightningModule
from torch.nn import Module
from tqdm import tqdm


def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while Path(path_pattern % i).exists():
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if Path(path_pattern % c).exists() else (a, c)

    return path_pattern % b


def is_missing_data(
    data_array: np.ndarray,
    ignore_channels: List[int] = [3],
    nodata_value: int = 0,
    missing_threshold: float = 0.5,
):
    """
    Checks if the fraction of missing data in a numpy ndarray representing raster
    data exceeds a specified threshold.

    Args:
        data_array (numpy.ndarray): The image data to check.
        ignore_channels (list of int, optional): Channels to ignore. Defaults to [3].
        nodata_value (int, optional): Value representing no data. Defaults to 0.
        missing_threshold (float, optional): Threshold for fraction of missing data.
                                             Defaults to 0.5.

    Returns:
        bool: True if missing data fraction exceeds threshold, else False.

    This function denotes pixels that are missing data in all non-ignored channels as
    "no data" pixels. If the fraction of "no data" pixels exceeds the threshold, the
    function returns True.
    """
    data_channels = list(set(range(data_array.shape[0])) - set(ignore_channels))
    nodata_pixels = (data_array[data_channels, :, :] == nodata_value).all(axis=0)
    return nodata_pixels.mean() > missing_threshold


def split_chips(
    tile_split_file: str, chip_dir: str, chip_suffix: str = ".tif"
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Splits the chips into train, validation, and test sets based on the tile split file.

    Args:
        tile_split_file (str): Path to the YAML file containing the tile split.
        chip_dir (str): Directory containing the chip files.
        chip_suffix (str, optional): Suffix of the chip files. Defaults to ".tif".

    Raises:
        ValueError: If a chip belongs to more than one set.

    Returns:
        Tuple[List[Path], List[Path], List[Path]]: Lists of chips for the train,
        validation, and test sets.
    """
    # Open and read the tile split yaml file
    with open(tile_split_file, "r") as f:
        tile_split = yaml.safe_load(f)

    # Get all the chip files from the chip directory
    chips = list(Path(chip_dir).rglob(f"*{chip_suffix}"))

    # Initialize a dictionary to hold the chips for each set (train, val, test)
    chip_sets = {key: [] for key in tile_split.keys()}

    # Assign each chip to the appropriate set
    for chip in chips:
        # Find the set(s) that the chip belongs to
        chip_sets_keys = [
            key
            for key in chip_sets.keys()
            if any(tile + "_" in str(chip) for tile in tile_split[key])
        ]

        # If the chip belongs to more than one set, raise an error
        if len(chip_sets_keys) > 1:
            raise ValueError("Chip is in more than one set!")
        # If the chip belongs to one set, add it to that set
        elif chip_sets_keys:
            chip_sets[chip_sets_keys[0]].append(chip)

    # Check that the sets are disjoint (i.e., they don't share any chips)
    for key1, key2 in combinations(chip_sets.keys(), 2):
        assert set(chip_sets[key1]).isdisjoint(
            chip_sets[key2]
        ), f"Sets {key1} and {key2} are not disjoint!"

    # Return the chips for the train, val, and test sets
    return (
        chip_sets["train_tiles"],
        chip_sets["val_tiles"],
        chip_sets["test_tiles"],
    )


def set_seed(seed):
    """
    Sets the seed for generating random numbers in Python, NumPy, and PyTorch.

    This function sets the seed for the built-in Python random module, NumPy's random
    module, and PyTorch's random number generator for both CPU and CUDA (if available).

    Args:
        seed (int): The seed value to be set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # In general seed PyTorch operations
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            # If you are using CUDA on more than 1 GPU, seed them all
            torch.cuda.manual_seed_all(seed)
        else:
            # If you are using CUDA on 1 GPU, seed it
            torch.cuda.manual_seed(seed)
    # Disable inbuilt cudnn auto-tuner that finds the best algorithm for your hardware.
    # torch.backends.cudnn.benchmark = False # but this might be slow down training
    # Certain operations in Cudnn are not deterministic, this forces them to behave!
    # torch.backends.cudnn.deterministic = True # but this might slow down training


def load_model(
    checkpoint_path: str,
    ModelClass: Type[Union[Module, LightningModule]],
    inference: bool = True,
    device: torch.device = None,
) -> Union[Module, LightningModule]:
    """
    Load a model from a checkpoint.

    Supports both PyTorch and PyTorch Lightning models. For Lightning models,
    the model is extracted from the LightningModule.

    If `inference` is True, the model is set to evaluation mode and parameters
    are frozen. Otherwise, the model is in training mode with trainable
    parameters.

    The model is moved to the specified device, or to the GPU if available and
    no device is specified.

    Args:
        checkpoint_path (str): Path to the checkpoint.
        ModelClass (Type[Union[Module, LightningModule]]): Model class to load.
        inference (bool, optional): If True, prepare the model for inference.
                                    Defaults to True.
        device (torch.device, optional): Device to move the model to.
                                    Defaults to None.

    Returns:
        Union[Module, LightningModule]: The loaded model.

    Raises:
        FileNotFoundError: If no checkpoint file is found.
        Exception: If an error occurs while loading the state dict.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path}")

    if hasattr(ModelClass, "load_from_checkpoint"):
        # PyTorch Lightning model
        model = ModelClass.load_from_checkpoint(checkpoint_path).to(device)
        if inference:
            model.freeze()
            model.eval()
    else:
        # Standard PyTorch model
        model = ModelClass().to(device)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            raise Exception(
                "Error loading model state dict from checkpoint at "
                f"{checkpoint_path}: {e}"
            )
        if inference:
            for param in model.parameters():
                param.requires_grad = False
            model = model.eval()

    return model


def get_output_shape(model, input_shape=(1, 4, 256, 256)):
    """
    Computes the output shape of a PyTorch model.

    This function creates a random tensor with the input shape, passes it through
    the model, and returns the output shape.

    Args:
        model (torch.nn.Module): The model to compute the output shape for.
        input_shape (tuple, optional): Shape of the input tensor. Defaults to
        (1, 4, 256, 256).

    Returns:
        torch.Size: The shape of the output tensor for an input of the specified shape.
    """
    x = torch.randn(*input_shape)
    out = model(x)
    return out.shape


class TrainingConfig:
    """
    A configuration class for training settings.

    This class holds the configuration parameters for a training session, including
    the number of epochs, batch size, device to use for training, and the directory
    to save checkpoints.

    Attributes:
        epochs (int): Number of epochs for training.
        batch_size (int): Size of each training batch.
        device (str): Device to use for training ('cpu' or 'cuda').
        checkpoint_dir (str): Directory to save training checkpoints.
    """

    def __init__(self, epochs, batch_size, device, checkpoint_dir):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.checkpoint_dir = checkpoint_dir


class MetricLogger(object):
    """
    A simple class for logging metrics during model training.

    This class is used to track and update metrics such as loss and accuracy during
    the training of a model. It calculates the average of the metric over time.

    Attributes:
        val (float): Most recent value of the metric.
        sum (float): Sum of the metric values.
        count (int): Number of times the metric has been updated.
        avg (float): Average of the metric values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fit(model, dataloader, criterion, optimizer, device):
    model.train()
    metric_logger = MetricLogger()
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        images = batch[0].to(device).float()
        labels = batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        metric_logger.update(loss.item(), images.size(0))
    return metric_logger.avg


def validate(model, dataloader, criterion, device):
    model.eval()
    metric_logger = MetricLogger()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            images = batch[0].to(device).float()
            labels = batch[1].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            metric_logger.update(loss.item(), images.size(0))
    return metric_logger.avg


def predict(model, dataloader, device):
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            images = batch[0].to(device).float()
            labels = batch[1].to(device)

            outputs = model(images)
            all_outputs.append(outputs)
            all_labels.append(labels)
    preds = torch.cat(all_outputs).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    return preds, labels
