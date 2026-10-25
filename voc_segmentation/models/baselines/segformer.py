"""SegFormer baseline with a resource-safe MiT-B1 encoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.encoders import MixTransformer, build_mit_b1


class LinearProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(in_channels, out_channels)

    def forward(self, features: Tensor) -> Tensor:
        batch, _, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.projection(tokens)
        return tokens.transpose(1, 2).reshape(batch, -1, height, width)


class SegFormerHead(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        *,
        decoder_channels: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            LinearProjection(channels, decoder_channels) for channels in in_channels
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * 4, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, features: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        target_size = features[0].shape[-2:]
        projected: list[Tensor] = []
        for feature, projection in zip(features, self.projections):
            output = projection(feature)
            if output.shape[-2:] != target_size:
                output = F.interpolate(
                    output, size=target_size, mode="bilinear", align_corners=False
                )
            projected.append(output)
        fused = self.fuse(torch.cat(projected[::-1], dim=1))
        return self.classifier(self.dropout(fused))


class SegFormer(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 21,
        encoder: str = "mit_b1",
        decoder_channels: int = 256,
        dropout: float = 0.1,
        encoder_weights_path: Path | str | None = None,
    ) -> None:
        super().__init__()
        if encoder != "mit_b1":
            raise ValueError("the 8 GB-safe baseline requires encoder='mit_b1'")
        self.encoder: MixTransformer = build_mit_b1(encoder_weights_path)
        self.decode_head = SegFormerHead(
            self.encoder.feature_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, image: Tensor) -> Tensor:
        input_size = image.shape[-2:]
        logits = self.decode_head(self.encoder(image))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


def build_segformer(
    *,
    num_classes: int = 21,
    encoder: str = "mit_b1",
    decoder_channels: int = 256,
    dropout: float = 0.1,
    encoder_weights_path: Path | str | None = None,
) -> SegFormer:
    return SegFormer(
        num_classes=num_classes,
        encoder=encoder,
        decoder_channels=decoder_channels,
        dropout=dropout,
        encoder_weights_path=encoder_weights_path,
    )
