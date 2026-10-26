"""Global-context, gated-fusion, and boundary-aware GCB-Seg modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _group_count(channels: int, maximum: int = 32) -> int:
    groups = min(maximum, channels)
    while channels % groups or (channels // groups < 2 and groups > 1):
        groups -= 1
    return groups


class ConvNormActDrop(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        dropout: float = 0.1,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )


class DepthwiseSeparableBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            ConvNormActDrop(
                in_channels,
                in_channels,
                kernel_size=3,
                dilation=dilation,
                groups=in_channels,
                dropout=dropout,
            ),
            ConvNormActDrop(
                in_channels,
                out_channels,
                kernel_size=1,
                dropout=dropout,
            ),
        )


class GlobalContextModule(nn.Module):
    """Five-branch context aggregation over the deepest encoder feature."""

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int = 256,
        branch_channels: int = 64,
        dilations: Sequence[int] = (1, 3, 6),
        use_global_pooling: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if tuple(dilations) != (1, 3, 6):
            raise ValueError("GCB-Seg requires context dilations (1, 3, 6)")
        self.use_global_pooling = use_global_pooling
        self.local_branches = nn.ModuleList(
            [
                ConvNormActDrop(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    dropout=dropout,
                ),
                *[
                    DepthwiseSeparableBlock(
                        in_channels,
                        branch_channels,
                        dilation=dilation,
                        dropout=dropout,
                    )
                    for dilation in dilations
                ],
            ]
        )
        if use_global_pooling:
            self.global_branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvNormActDrop(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    dropout=dropout,
                ),
            )
        else:
            self.global_branch = None
        branch_count = len(self.local_branches) + int(use_global_pooling)
        self.fuse = ConvNormActDrop(
            branch_count * branch_channels,
            out_channels,
            kernel_size=1,
            dropout=dropout,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = [branch(inputs) for branch in self.local_branches]
        if self.global_branch is not None:
            pooled = self.global_branch(inputs)
            outputs.append(
                F.interpolate(
                    pooled,
                    size=inputs.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return self.fuse(torch.cat(outputs, dim=1))


class GatedFusionStage(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        use_gate: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_gate = use_gate
        self.gate = nn.Conv2d(channels * 2, channels, 1) if use_gate else None
        self.refine = nn.Sequential(
            DepthwiseSeparableBlock(channels, channels, dropout=dropout),
            DepthwiseSeparableBlock(channels, channels, dropout=dropout),
        )

    def forward(self, projected: Tensor, deeper: Tensor) -> Tensor:
        upsampled = F.interpolate(
            deeper,
            size=projected.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if self.gate is None:
            fused = projected + upsampled
        else:
            gate = torch.sigmoid(self.gate(torch.cat((projected, upsampled), dim=1)))
            fused = projected + gate * upsampled
        return self.refine(fused)


class GatedFeatureFusion(nn.Module):
    """Top-down fusion from F4 to encoder stages C3, C2, and C1."""

    def __init__(
        self,
        encoder_channels: Sequence[int] = (64, 128, 320),
        *,
        channels: int = 256,
        use_gate: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 3:
            raise ValueError("GFF requires C1, C2, and C3 channel counts")
        self.projections = nn.ModuleList(
            ConvNormActDrop(
                in_channels,
                channels,
                kernel_size=1,
                dropout=dropout,
            )
            for in_channels in encoder_channels
        )
        self.stages = nn.ModuleList(
            GatedFusionStage(
                channels,
                use_gate=use_gate,
                dropout=dropout,
            )
            for _ in range(3)
        )

    def forward(
        self, c1: Tensor, c2: Tensor, c3: Tensor, f4: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        p1, p2, p3 = (
            projection(feature)
            for projection, feature in zip(self.projections, (c1, c2, c3))
        )
        f3 = self.stages[0](p3, f4)
        f2 = self.stages[1](p2, f3)
        f1 = self.stages[2](p1, f2)
        return f1, f2, f3


class BoundaryAwareBranch(nn.Module):
    """Fuse C1 and upsampled C2 into boundary features and a boundary logit."""

    def __init__(
        self,
        c1_channels: int = 64,
        c2_channels: int = 128,
        *,
        boundary_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvNormActDrop(
                c1_channels + c2_channels,
                boundary_channels,
                kernel_size=3,
                dropout=dropout,
            ),
            ConvNormActDrop(
                boundary_channels,
                boundary_channels,
                kernel_size=3,
                dropout=dropout,
            ),
        )
        self.classifier = nn.Conv2d(boundary_channels, 1, 1)

    def forward(self, c1: Tensor, c2: Tensor) -> tuple[Tensor, Tensor]:
        c2 = F.interpolate(
            c2, size=c1.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.features(torch.cat((c1, c2), dim=1))
        return features, self.classifier(features)
