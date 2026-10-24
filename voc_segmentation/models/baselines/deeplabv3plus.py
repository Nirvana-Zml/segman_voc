"""A compact DeepLabV3+ implementation with a torchvision ResNet-50 encoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        dilation_rates: Sequence[int] = (12, 24, 36),
    ) -> None:
        super().__init__()
        branches: list[nn.Module] = [ConvNormAct(in_channels, out_channels, 1)]
        branches.extend(
            ConvNormAct(
                in_channels,
                out_channels,
                3,
                padding=rate,
                dilation=rate,
            )
            for rate in dilation_rates
        )
        self.branches = nn.ModuleList(branches)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNormAct(in_channels, out_channels, 1),
        )
        self.project = nn.Sequential(
            ConvNormAct(out_channels * (len(dilation_rates) + 2), out_channels, 1),
            nn.Dropout2d(0.1),
        )

    def forward(self, features: Tensor) -> Tensor:
        outputs = [branch(features) for branch in self.branches]
        pooled = self.pool(features)
        pooled = F.interpolate(
            pooled, size=features.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.project(torch.cat([*outputs, pooled], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int = 21, encoder_pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if encoder_pretrained else None
        encoder = resnet50(
            weights=weights,
            replace_stride_with_dilation=[False, True, True],
        )
        self.stem = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
        )
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.aspp = ASPP(2048)
        self.low_level_projection = ConvNormAct(256, 48, 1)
        self.decoder = nn.Sequential(
            ConvNormAct(304, 256, 3, padding=1),
            ConvNormAct(256, 256, 3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, image: Tensor) -> Tensor:
        input_size = image.shape[-2:]
        features = self.stem(image)
        low_level = self.layer1(features)
        features = self.layer2(low_level)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.aspp(features)
        low_level = self.low_level_projection(low_level)
        features = F.interpolate(
            features, size=low_level.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = self.decoder(torch.cat([features, low_level], dim=1))
        return F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )


def build_deeplabv3plus(
    num_classes: int = 21, encoder_pretrained: bool = False
) -> DeepLabV3Plus:
    return DeepLabV3Plus(
        num_classes=num_classes, encoder_pretrained=encoder_pretrained
    )
