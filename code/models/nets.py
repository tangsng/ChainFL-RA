# -*- coding: utf-8 -*-
"""Model definitions for federated experiments."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """LeNet-5 variant for 28x28 grayscale inputs (MNIST / Fashion-MNIST)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SmallCNN(nn.Module):
    """Compact CNN for CIFAR-10 (32x32x3)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 8
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)   # 4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_model(dataset: str) -> nn.Module:
    if dataset in ("mnist", "fmnist"):
        return LeNet()
    if dataset == "cifar10":
        return SmallCNN()
    raise ValueError(dataset)
