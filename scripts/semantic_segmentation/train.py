# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import json
import pickle
import time
from pathlib import Path

import configargparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import Trainer, loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.transforms import (  # GaussianBlur,
    ColorJitter,
    Compose,
    RandomHorizontalFlip,
    RandomRotation,
)
from tqdm import tqdm

from src.datamodules import ChipSegmentationDataModule
from src.datasets import ChipDataset, preprocess
from src.models import CustomLogSemanticSegmentation
from src.utils import set_seed, split_chips  # , load_model


def augmentation_transform(sample):
    image_transforms = Compose(
        [
            # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            # GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ]
    )
    affine_transforms = Compose([RandomHorizontalFlip(), RandomRotation(180)])
    if "image" in sample:
        image = sample["image"]
        image = (image / 255.0).float()
        image = image_transforms(image)
        sample["image"] = image
    if "image" in sample and "mask" in sample:
        image = sample["image"]
        mask = sample["mask"]
        stacked = torch.cat([image, mask.unsqueeze(0)], dim=0)
        stacked = affine_transforms(stacked)
        image, mask = stacked[0:3], stacked[3]
        sample["image"] = image
        sample["mask"] = mask.squeeze(0).long()
    return sample


def model_inference(
    dataset, model, device, image_key=None, batch_size=64, num_workers=6
):
    """
    Perform inference on a dataset using a given model.

    Args:
        dataset (torch.utils.data.Dataset): Dataset for inference.
        model (torch.nn.Module): Model for inference.
        device (torch.device): Device to run the model on.
        image_key (str, optional): Key to access image data if batch is a dict.
                                   Defaults to None.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 64.
        num_workers (int, optional): Number of worker processes for data loading.
                                     Defaults to 6.

    Returns:
        numpy.ndarray: Concatenated predictions from the model on the dataset.
    """
    dl = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    predictions = []
    for batch in tqdm(dl):
        with torch.inference_mode():
            if image_key:
                input_data = batch[image_key]
            else:
                input_data = batch
            prediction = model(input_data.to(device))
            predictions.append(prediction.argmax(axis=1).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Create directories
    base_path = (
        Path(args.output_dir) / "semantic_segmentation" / f"{args.run_name}"
    )
    base_path.mkdir(parents=True, exist_ok=True)
    (base_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base_path / "logs").mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(base_path / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # Initialize the data module
    train_chips, val_chips, test_chips = split_chips(
        args.tile_split_file, args.chip_dir, strict=args.strict_splits
    )
    datamodule = ChipSegmentationDataModule(
        train_chips,
        val_chips,
        test_chips,
        args.mask_dir,
        transforms={"train": augmentation_transform, "eval": preprocess},
        batch_size=args.batch_size,
    )
    fig = datamodule.plot_transforms()
    fig.savefig(base_path / "transforms.png")
    plt.close(fig)

    # Initialize the model and task
    class_names = ["background", "building", "building_boundary", "solar"]
    task = CustomLogSemanticSegmentation(
        model="unet",
        backbone="resnext50_32x4d",
        weights="imagenet",  # use pretrained imagenet weights
        in_channels=3,
        num_classes=len(class_names) + 1,  # +1 for 0 "not labeled" class
        loss="ce",
        class_weights=args.class_weights,
        ignore_index=0,  # class 0 represents "not labeled" in the label masks
        learning_rate=args.learning_rate,
        learning_rate_schedule_patience=10,
        train_metrics_file=base_path / "train_metrics.csv",
        val_metrics_file=base_path / "val_metrics.csv",
        test_metrics_file=base_path / "test_metrics.csv",
    )

    # Define callbacks and loggers
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=base_path / "checkpoints",
        save_top_k=1,
        save_last=True,
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )
    tb_logger = loggers.TensorBoardLogger(
        save_dir=base_path / "logs" / "lightning_logs",
        name="tb_logs",
    )
    csv_logger = loggers.CSVLogger(
        save_dir=base_path / "logs" / "lightning_logs",
        name="csv_logs",
    )

    # Determine the device to use
    if args.gpu_id == -1 or not torch.cuda.is_available():
        accelerator = "cpu"
        devices = 1
        device = torch.device("cpu")
    else:
        accelerator = "gpu"
        devices = [args.gpu_id]
        device = torch.device(f"cuda:{args.gpu_id}")

    # Create a PyTorch Lightning trainer
    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=[tb_logger, csv_logger],
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
    )

    # Train the model and test with the best model
    _ = trainer.fit(model=task, datamodule=datamodule)

    # Load the best model from the checkpoint test
    best_model_path = checkpoint_callback.best_model_path
    # FIXME: When using our load_model utility function, the trainer.test() call
    #        silently moves the model from the specified device to CPU for some
    #        reason. This is doesn't happen when using the Lightning
    #        load_from_checkpoint, or when trainer.test() is not called.
    # best_model = load_model(
    #     best_model_path,
    #     CustomLogSemanticSegmentation,
    #     inference=True,
    #     device=torch.device(f"cuda:{args.gpu_id}"),
    # )
    best_model = CustomLogSemanticSegmentation.load_from_checkpoint(
        best_model_path
    )
    print(f"Loaded best model from {best_model_path}")

    # Test using the best model
    _ = trainer.test(model=best_model, datamodule=datamodule)

    # Predict on train, val, and test sets using the best model
    # NOTE: When using our custom load_model function with inference=True,
    #       these lines are not needed.
    best_model.freeze()
    best_model = best_model.eval().to(device)

    results = {
        "train_chips": dict(),
        "val_chips": dict(),
        "test_chips": dict(),
    }
    # train chip inference
    train_ds = ChipDataset(
        train_chips, args.mask_dir, transforms=preprocess
    )  # turn off augmentations for train
    train_predictions = model_inference(
        train_ds, best_model, device, image_key="image"
    )
    for chip, prediction in zip(
        datamodule.train_ds.chip_paths, train_predictions
    ):
        results["train_chips"][chip] = prediction
    # val chip inference
    val_predictions = model_inference(
        datamodule.val_ds, best_model, device, image_key="image"
    )
    for chip, prediction in zip(datamodule.val_ds.chip_paths, val_predictions):
        results["val_chips"][chip] = prediction
    # test chip inference
    test_predictions = model_inference(
        datamodule.test_ds, best_model, device, image_key="image"
    )
    for chip, prediction in zip(
        datamodule.test_ds.chip_paths, test_predictions
    ):
        results["test_chips"][chip] = prediction

    # save results
    save_path = base_path / f"{Path(best_model_path).stem}_chip_inference.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    # parser = configargparse.ArgParser(default_config_files=["/etc/app/config.txt"])
    parser = configargparse.ArgParser()
    parser.add_argument(
        "--config", is_config_file=True, help="config file path"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-version", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chip-dir", type=str, required=True)
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--tile-split-file", type=str, required=True)
    parser.add_argument("--class-weights", nargs="*", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--strict-splits", action="store_true")
    args = parser.parse_args()
    run_timestamp = "{}".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
    args.run_name = f"{args.exp_version}-{run_timestamp}"

    main(args)
