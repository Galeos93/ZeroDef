"""Module that contains various models' architectures."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class CNNModel(nn.Module):
    """Custom CNN model."""

    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(102400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TimmResNetV2(nn.Module):
    """ResNet V2 50 model from `timm` module"""

    def __init__(
        self,
        n_classes=5,
        pretrained=True,
        net_name="resnetv2_50x1_bitm_in21k",
    ):
        super().__init__()

        self.model = timm.create_model(
            net_name, pretrained=pretrained, num_classes=n_classes
        )
        print(self.model)

    def forward(self, x):
        output = self.model(x)
        return output


class TimmEfficientNetB4(nn.Module):
    """EfficientNetB4 model from `timm` module"""

    def __init__(
        self,
        n_classes=5,
        pretrained=True,
        net_name="efficientnet_b4",
    ):
        super().__init__()

        self.model = timm.create_model(
            net_name, pretrained=pretrained, num_classes=n_classes
        )
        print(self.model)

    def forward(self, x):
        output = self.model(x)
        return output


class TimmEfficientNetB0(nn.Module):
    """EfficientNetB0 model from `timm` module"""

    def __init__(
        self,
        n_classes=5,
        pretrained=True,
        net_name="efficientnet_b0",
    ):
        super().__init__()

        self.model = timm.create_model(
            net_name, pretrained=pretrained, num_classes=n_classes
        )
        print(self.model)

    def forward(self, x):
        output = self.model(x)
        return output
