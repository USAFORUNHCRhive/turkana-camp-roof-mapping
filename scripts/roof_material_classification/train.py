# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import configargparse
import pickle
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import (
    ColorJitter,
    Compose,
    GaussianBlur,
    RandomHorizontalFlip,
    RandomRotation,
)

from src.datasets import RoofClfPreSampledDataset
from src.models import CustomResNet18
from src.utils import (
    fit,
    predict,
    set_seed,
    split_chips,
    TrainingConfig,
    validate,
)


def _train_transform(sample):
    image_transforms = Compose(
        [
            GaussianBlur(kernel_size=3, sigma=(0.05, 1.0)),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ]
    )
    affine_transforms = Compose([RandomHorizontalFlip(), RandomRotation(180)])
    image = (sample[0:3, :, :] / 255.0).float()
    image = image_transforms(image)
    mask = sample[3, :, :]
    stacked = torch.cat([image, mask.unsqueeze(0)], dim=0)
    stacked = affine_transforms(stacked)
    return stacked


def _eval_transform(sample):
    image = (sample[0:3, :, :] / 255.0).float()
    mask = sample[3, :, :]
    stacked = torch.cat([image, mask.unsqueeze(0)], dim=0)
    return stacked


def _create_dataset_and_dataloader(df, transform, shuffle=False, batch_size=64):
    dataset = RoofClfPreSampledDataset(df)
    dataset.set_transforms(transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataset, dataloader


def train(
    model, train_loader, val_loader, criterion, optimizer, scheduler, config
):
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    for epoch in range(config.epochs):
        train_loss = fit(
            model, train_loader, criterion, optimizer, config.device
        )
        train_losses.append(train_loss)
        val_loss = validate(model, val_loader, criterion, config.device)
        val_losses.append(val_loss)
        print(
            f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save model checkpoint
            model_path = (
                config.checkpoint_dir
                / f"epoch={epoch}_val_loss={val_loss:.4f}.ckpt"
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)
            print(f"Saved model at {model_path}")
    return train_losses, val_losses


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Create directories
    base_path = (
        Path(args.output_dir)
        / "roof_material_classification"
        / f"{args.run_name}"
    )
    base_path.mkdir(parents=True, exist_ok=True)
    (base_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=torch.device(
            f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        ),
        checkpoint_dir=base_path / "checkpoints",
    )

    # get data
    data_dir = Path(args.data_dir)
    label_to_class = {i: c.name for i, c in enumerate(data_dir.iterdir())}
    class_to_label = {c: i for i, c in label_to_class.items()}
    num_classes = len(label_to_class)
    # samples are organized into folders by class
    df = pd.DataFrame(
        [
            [data_dir / c / f.name, l]
            for c, l in class_to_label.items()
            for f in (data_dir / c).iterdir()
        ],
        columns=["image", "label"],
    )
    train_chips, val_chips, test_chips = split_chips(
        args.tile_split_file, args.data_dir
    )
    train_df = df[df["image"].isin(train_chips)]
    val_df = df[df["image"].isin(val_chips)]
    test_df = df[df["image"].isin(test_chips)]

    # create dataloaders
    _, train_dl = _create_dataset_and_dataloader(
        train_df, _train_transform, shuffle=True, batch_size=args.batch_size
    )
    _, val_dl = _create_dataset_and_dataloader(
        val_df, _eval_transform, batch_size=args.batch_size
    )
    _, test_dl = _create_dataset_and_dataloader(
        test_df, _eval_transform, batch_size=args.batch_size
    )

    # create model accepting 4-channel input
    model = CustomResNet18(num_channels=4, num_classes=num_classes)
    model = model.to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=5, verbose=True
    )
    # class weights
    if args.class_weights is None:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        cw = torch.tensor(args.class_weights, dtype=torch.float).to(cfg.device)
        criterion = torch.nn.CrossEntropyLoss(weight=cw)

    losses = train(model, train_dl, val_dl, criterion, opt, scheduler, cfg)
    test_loss = validate(model, test_dl, criterion, cfg.device)
    test_preds, test_labels = predict(model, test_dl, cfg.device)
    print(
        f"Test Accuracy: {accuracy_score(test_labels, test_preds.argmax(axis=1)):.4f}"
    )

    metrics = {
        "train_losses": losses[0],
        "val_losses": losses[1],
        "test_loss": test_loss,
        "test_accuracy": accuracy_score(test_labels, test_preds.argmax(axis=1)),
        "test_confusion_matrix": confusion_matrix(
            test_labels, test_preds.argmax(axis=1)
        ),
        "test_predictions": test_preds,
        "test_labels": test_labels,
        "class_to_label": class_to_label,
        "label_to_class": label_to_class,
        "class_weights": args.class_weights,
    }
    metrics_f = base_path / "metrics.pkl"
    with open(metrics_f, "wb") as f:
        pickle.dump(metrics, f)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "--config",
        is_config_file=True,
        help="Configuration file for training run.",
    )
    parser.add_argument("--exp-version", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--tile-split-file", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--class-weights", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    run_timestamp = "{}".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
    args.run_name = f"{args.exp_version}-{run_timestamp}"
    if args.class_weights is not None:
        args.class_weights = [float(w) for w in args.class_weights.split(",")]
    main(args)
