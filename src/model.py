"""Convolutional neural network used for image classification.

Architecture (input: 3 x H x W, output: ``num_classes`` logits)::

    Block 1: 2x [Conv 3->32  + BN + ReLU]  -> MaxPool -> Dropout2d
    Block 2: 2x [Conv 32->64 + BN + ReLU]  -> MaxPool -> Dropout2d
    Block 3: 2x [Conv 64->128+ BN + ReLU]  -> MaxPool -> Dropout2d
    Block 4: 2x [Conv 128->256+ BN + ReLU] -> MaxPool -> Dropout2d
    Head   : AdaptiveAvgPool -> Dropout -> Linear -> ReLU -> Dropout -> Linear

The classifier head is built automatically, so the same code works for any
number of classes and any input image size.
"""

from __future__ import annotations

import torch
from torch import nn

ARCHITECTURE_NAME = "Week2CNN"


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch norm + ReLU, then pooling and dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageClassifier(nn.Module):
    """A compact, beginner-friendly CNN that adapts to ``num_classes``."""

    def __init__(self, num_classes: int, dropout: float = 0.5, image_size: int = 224) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.num_classes = num_classes
        self.image_size = image_size

        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=0.10),
            ConvBlock(32, 64, dropout=0.15),
            ConvBlock(64, 128, dropout=0.20),
            ConvBlock(128, 256, dropout=0.25),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Determine the flattened feature size with a dummy forward pass so the
        # classifier works for any input resolution.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feature_dim: int = int(self.features(dummy).flatten(1).shape[1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch_size, num_classes)``."""
        x = self.features(x)
        return self.classifier(x)


def create_model(
    num_classes: int,
    dropout: float = 0.5,
    image_size: int = 224,
) -> ImageClassifier:
    """Factory that keeps model creation in a single place."""
    return ImageClassifier(num_classes=num_classes, dropout=dropout, image_size=image_size)


def get_device(requested: str | None = None) -> torch.device:
    """Return the device to use.

    Uses CUDA automatically when available, otherwise CPU. ``requested`` can be
    ``"cpu"``, ``"cuda"`` or ``"mps"`` to force a specific device.
    """
    if requested and requested.lower() != "auto":
        device = torch.device(requested.lower())
        if device.type == "cuda" and not torch.cuda.is_available():
            print("[!] CUDA requested but not available - falling back to CPU.")
            return torch.device("cpu")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    """Short, human readable device description."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"CUDA GPU: {torch.cuda.get_device_name(index)}"
    if device.type == "mps":
        return "Apple Metal Performance Shader (MPS)"
    return "CPU"


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters (trainable only by default)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def check_forward_pass(
    model: nn.Module,
    image_size: int,
    device: torch.device,
) -> torch.Size:
    """Run a dummy batch through the model to verify tensor dimensions."""
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 3, image_size, image_size, device=device)
        return tuple(model(dummy).shape)  # type: ignore[return-value]