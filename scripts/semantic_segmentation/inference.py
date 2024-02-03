# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import time
from contextlib import contextmanager
from pathlib import Path

import configargparse
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm import tqdm

from src.datasets import preprocess, SingleRasterDataset
from src.models import CustomLogSemanticSegmentation
from src.utils import load_model


@contextmanager
def timer(message):
    """
    Timer function to calculate the time taken by a code block.

    Args:
        message (str): The message to print along with time taken.
    """
    start = time.time()
    yield
    print(f"{message}: {time.time() - start:.2f} seconds")


class SemanticSegmentationTiffPredictor:
    """
    A class used to perform semantic segmentation on TIFF images.

    Attributes:
        checkpoint (str): The path to the model checkpoint file.
        device (torch.device): The device (CPU or GPU) to use for inference.
        model (torch.nn.Module): The loaded model for inference.
        patch_size (int): The size of the patches to use for inference.
        padding (int): The padding to use for the patches.
        stride (int): The stride to use for the patches.
        batch_size (int): The batch size to use for inference.
    """

    def __init__(self, checkpoint, device=None):
        """
        Constructs all the necessary attributes for the
        SemanticSegmentationTiffPredictor object.

        Args:
            checkpoint (str): The path to the model checkpoint file.
            device (torch.device, optional): The device (CPU or GPU) to use for
                                        inference. Defaults to None, i.e. the device is
                                        chosen automatically based on availability.
        """
        self.checkpoint = checkpoint
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = load_model(
            self.checkpoint,
            CustomLogSemanticSegmentation,
            inference=True,
            device=self.device,
        )
        self.patch_size = 256
        self.padding = 16
        self.stride = self.patch_size - self.padding * 2
        self.batch_size = 64

    def load_data(self, image_path):
        """
        Loads the image data from the specified path.

        Args:
            image_path (str): The path to the image file.

        Returns:
            Tuple[SingleRasterDataset, DataLoader]: The loaded dataset and the
                                                    DataLoader for the dataset.
        """
        dataset = SingleRasterDataset(image_path, transforms=preprocess)
        sampler = GridGeoSampler(
            dataset, size=self.patch_size, stride=self.stride
        )
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=24,
            collate_fn=stack_samples,
        )
        return dataset, dataloader

    def run_inference(self, image_path, verbose=True):
        """
        Runs inference on the image at the specified path.

        Args:
            image_path (str): The path to the image file.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            Tuple[np.array, dict]: The output of the inference and the profile of the
                                   image file.
        """
        with rasterio.open(image_path) as f:
            input_height, input_width = f.shape
            profile = f.profile
            transform = profile["transform"]
        _, dataloader = self.load_data(image_path)

        # Run inference
        with timer("Finished running model in"):
            if verbose:
                print(f"Input size: {input_height} x {input_width}")
            assert self.patch_size <= input_height
            assert self.patch_size <= input_width
            output = np.zeros((input_height, input_width), dtype=np.uint8)

            for batch in tqdm(dataloader):
                images = batch["image"][:, 0:3, :, :].to(self.device)
                bboxes = batch["bbox"]
                with torch.inference_mode():
                    predictions = self.model(images)
                    predictions = predictions.argmax(axis=1).cpu().numpy()

                for i in range(len(bboxes)):
                    bb = bboxes[i]

                    left, top = ~transform * (bb.minx, bb.maxy)
                    right, bottom = ~transform * (bb.maxx, bb.miny)
                    left, right, top, bottom = (
                        int(np.round(left)),
                        int(np.round(right)),
                        int(np.round(top)),
                        int(np.round(bottom)),
                    )
                    if (
                        abs(self.patch_size - (right - left)) > 0
                        and abs(self.patch_size - (right - left)) < 5
                    ):
                        right = left + self.patch_size

                    if (
                        abs(self.patch_size - (bottom - top)) > 0
                        and abs(self.patch_size - (bottom - top)) < 5
                    ):
                        bottom = top + self.patch_size

                    assert right - left == self.patch_size
                    assert bottom - top == self.patch_size

                    output[
                        top + self.padding : bottom - self.padding,
                        left + self.padding : right - self.padding,
                    ] = predictions[i][
                        self.padding : -self.padding,
                        self.padding : -self.padding,
                    ]
        return output, profile

    def save_predictions(self, output, save_path, profile, verbose=True):
        """
        Saves the prediction results to the specified path.

        Args:
            output (np.array): The output of the inference.
            save_path (str): The path to save the prediction results to.
            profile (dict): The profile of the image file.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        """
        with timer("Finished saving predictions in"):
            profile["driver"] = "GTiff"
            profile["count"] = 1
            profile["dtype"] = "uint8"
            profile["compress"] = "lzw"
            profile["predictor"] = 2
            profile["nodata"] = 0
            profile["blockxsize"] = 512
            profile["blockysize"] = 512
            profile["tiled"] = True
            profile["interleave"] = "pixel"
            with rasterio.open(save_path, "w", **profile) as f:
                f.write(output, 1)


def get_args():
    parser = configargparse.ArgumentParser(description="", add_help=True)
    parser.add_argument(
        "--config", is_config_file=True, help="Config file path."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Path to the directory containing the images to predict.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the directory to save the predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference.",
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
            "image_dir",
            "checkpoint",
            "output_dir",
        ]:
            if args[key] is not None:
                args[key] = str(Path(args["root_dir"]) / Path(args[key]))
    return args


if __name__ == "__main__":
    args = get_args()
    predictor = SemanticSegmentationTiffPredictor(
        args["checkpoint"], device=args["device"]
    )
    image_paths = list(Path(args["image_dir"]).glob("*.tif"))
    for image_path in image_paths:
        output, profile = predictor.run_inference(image_path, verbose=True)
        save_path = (
            Path(args["output_dir"])
            / f"{Path(args['checkpoint']).stem}_{Path(image_path).name}"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        predictor.save_predictions(output, save_path, profile, verbose=True)
