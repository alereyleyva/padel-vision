"""TrackNet v2 PyTorch model — VGG16 encoder with U-Net decoder for ball tracking.

Input:  (B, 9, H, W)  — 3 consecutive RGB frames concatenated channel-wise
Output: (B, 3, H, W)  — 3 heatmaps (one per input frame), values in [0, 1] via sigmoid

Based on the TrackNetV2-pytorch implementation (https://github.com/ChgygLin/TrackNetV2-pytorch).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → ReLU → BatchNorm2d."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.relu(self.conv(x)))


class TrackNet(nn.Module):
    """TrackNet v2: VGG16 encoder with U-Net-style decoder and skip connections.

    Processes 3 consecutive frames to produce ball position heatmaps.
    """

    def __init__(self, input_channels: int = 9, out_channels: int = 3) -> None:
        super().__init__()

        # Encoder (VGG16-style)
        self.enc1 = nn.Sequential(ConvBlock(input_channels, 64), ConvBlock(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(ConvBlock(64, 128), ConvBlock(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(ConvBlock(128, 256), ConvBlock(256, 256), ConvBlock(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = nn.Sequential(ConvBlock(256, 512), ConvBlock(512, 512), ConvBlock(512, 512))
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512), ConvBlock(512, 512))

        # Decoder with skip connections
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec4 = nn.Sequential(ConvBlock(512 + 512, 256), ConvBlock(256, 256), ConvBlock(256, 256))

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = nn.Sequential(ConvBlock(256 + 256, 128), ConvBlock(128, 128), ConvBlock(128, 128))

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = nn.Sequential(ConvBlock(128 + 128, 64), ConvBlock(64, 64))

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = nn.Sequential(ConvBlock(64 + 64, 32), ConvBlock(32, 32))

        # Output
        self.output_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.sigmoid(self.output_conv(d1))


def tracknet_inference_transform(
    frame_triplet: list[torch.Tensor],
    input_size: tuple[int, int] = (288, 512),
) -> torch.Tensor:
    """Prepare 3 consecutive frames for TrackNet inference.

    Args:
        frame_triplet: List of 3 tensors, each (3, H, W) in [0, 1] RGB format.
        input_size: Target (height, width) for the model.

    Returns:
        Tensor of shape (1, 9, H, W) ready for model input.
    """
    import torchvision.transforms.functional as transforms_f

    resized = [transforms_f.resize(f, list(input_size)) for f in frame_triplet]
    return torch.cat(resized, dim=0).unsqueeze(0)


def extract_ball_position(
    heatmap: torch.Tensor,
    confidence_threshold: float = 0.5,
    original_size: tuple[int, int] | None = None,
) -> tuple[float, float, float] | None:
    """Extract ball (x, y, confidence) from a TrackNet heatmap.

    Args:
        heatmap: Tensor of shape (H, W) with values in [0, 1].
        confidence_threshold: Minimum heatmap value to consider as ball detection.
        original_size: If provided, (width, height) to scale coordinates back.

    Returns:
        (x, y, confidence) in original frame coordinates, or None if no ball detected.
    """
    import numpy as np

    heat_np = heatmap.detach().cpu().numpy()
    binary = (heat_np > confidence_threshold).astype(np.uint8) * 255

    contours, _ = cv2_find_contours(binary)
    if not contours:
        return None

    largest = max(contours, key=lambda c: cv2_contour_area(c))

    moments = cv2_moments(largest)
    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    if original_size is not None:
        orig_w, orig_h = original_size
        h, w = heat_np.shape
        cx = cx * orig_w / w
        cy = cy * orig_h / h

    confidence = float(heat_np.max())
    return (float(cx), float(cy), confidence)


def cv2_find_contours(binary: np.ndarray) -> tuple[list, Any]:
    """Wrapper for cv2.findContours that handles API differences."""
    result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 2:
        return result[0], result[1]
    return result[1], result[2]


def cv2_contour_area(contour: np.ndarray) -> float:
    """Wrapper for cv2.contourArea."""
    return cv2.contourArea(contour)


def cv2_moments(contour: np.ndarray) -> dict:
    """Wrapper for cv2.moments."""
    return cv2.moments(contour)
