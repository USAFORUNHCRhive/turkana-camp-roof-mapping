# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
This module contains two classes: CustomLogSemanticSegmentation and CustomResNet18.

CustomLogSemanticSegmentation is a subclass of SemanticSegmentationTask from the torchgeo.trainers module. It overrides 
the on_train_epoch_end, on_validation_epoch_end, and on_test_epoch_end methods to log epoch level metrics for training, 
validation, and testing respectively. The metrics are logged both to a file and using the log_dict method of the superclass.

CustomResNet18 is a subclass of torch.nn.Module and represents a custom ResNet18 model for multi-channel, multi-class 
classification. It modifies the first convolutional layer to accept a custom number of input channels and the final fully 
connected layer to output a custom number of classes. The forward method is overridden to define the forward pass of the 
network.
"""
import json

import torch
from torchgeo.trainers import SemanticSegmentationTask
from torchvision.models import resnet18


class CustomLogSemanticSegmentation(SemanticSegmentationTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_metrics_file = kwargs["train_metrics_file"]
        self.val_metrics_file = kwargs["val_metrics_file"]
        self.test_metrics_file = kwargs["test_metrics_file"]

    def on_train_epoch_end(self) -> None:
        """Logs epoch level training metrics."""
        train_epoch_end_metrics = self.train_metrics.compute()
        with open(self.train_metrics_file, "a") as f:
            f.write(
                json.dumps({k: v.item() for k, v in train_epoch_end_metrics.items()})
            )
            f.write("\n")
        self.log_dict(train_epoch_end_metrics)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Logs epoch level validation metrics."""
        val_epoch_end_metrics = self.val_metrics.compute()
        with open(self.val_metrics_file, "a") as f:
            f.write(json.dumps({k: v.item() for k, v in val_epoch_end_metrics.items()}))
            f.write("\n")
        self.log_dict(val_epoch_end_metrics)
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Logs epoch level test metrics."""
        test_epoch_end_metrics = self.test_metrics.compute()
        with open(self.test_metrics_file, "a") as f:
            f.write(
                json.dumps({k: v.item() for k, v in test_epoch_end_metrics.items()})
            )
            f.write("\n")
        self.log_dict(test_epoch_end_metrics)
        self.test_metrics.reset()


class CustomResNet18(torch.nn.Module):
    """
    A custom ResNet18 model for multi-channel, multi-class classification.

    This class extends the PyTorch ResNet18 model to support input with a custom
    number of channels and output for a custom number of classes.

    Args:
        num_channels (int, optional): Number of input channels. Defaults to 4.
        num_classes (int, optional): Number of output classes. Defaults to 4.
    """

    def __init__(self, num_channels=4, num_classes=4):
        super(CustomResNet18, self).__init__()
        self.net = resnet18(pretrained=True)
        weight = self.net.conv1.weight.clone()
        self.net.conv1 = torch.nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.net.conv1.weight[:, :3] = weight
            self.net.conv1.weight[:, 3] = self.net.conv1.weight[:, 0]
        self.net.fc = torch.nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        return self.net(x)
