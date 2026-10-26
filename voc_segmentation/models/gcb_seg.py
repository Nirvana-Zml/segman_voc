"""GCB-Seg: global-context and boundary-aware semantic segmentation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.encoders import MixTransformer, build_mit_b1
from models.baselines.segformer import SegFormerHead
from models.modules import (
    BoundaryAwareBranch,
    ConvNormActDrop,
    GatedFeatureFusion,
    GlobalContextModule,
)


class GCBSeg(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 21,
        encoder: str = "mit_b1",
        decoder_channels: int = 256,
        context_branch_channels: int = 64,
        boundary_channels: int = 64,
        global_context_dilations: Sequence[int] = (1, 3, 6),
        use_global_pooling: bool = True,
        use_gated_fusion: bool = True,
        feature_fusion_mode: str = "gated",
        use_boundary_branch: bool = True,
        auxiliary_head: bool = True,
        dropout: float = 0.1,
        encoder_weights_path: Path | str | None = None,
    ) -> None:
        super().__init__()
        if encoder != "mit_b1":
            raise ValueError("the verified 8 GB-safe GCB-Seg profile requires mit_b1")
        self.encoder: MixTransformer = build_mit_b1(encoder_weights_path)
        c1_channels, c2_channels, c3_channels, c4_channels = (
            self.encoder.feature_channels
        )
        self.global_context = GlobalContextModule(
            c4_channels,
            out_channels=decoder_channels,
            branch_channels=context_branch_channels,
            dilations=global_context_dilations,
            use_global_pooling=use_global_pooling,
            dropout=dropout,
        )
        if feature_fusion_mode not in {"linear", "gated"}:
            raise ValueError("feature_fusion_mode must be 'linear' or 'gated'")
        if feature_fusion_mode == "linear" and use_boundary_branch:
            raise ValueError("the linear A1 fusion does not support the boundary branch")
        self.feature_fusion_mode = feature_fusion_mode
        self.feature_fusion = (
            GatedFeatureFusion(
                (c1_channels, c2_channels, c3_channels),
                channels=decoder_channels,
                use_gate=use_gated_fusion,
                dropout=dropout,
            )
            if feature_fusion_mode == "gated"
            else None
        )
        self.linear_head = (
            SegFormerHead(
                (c1_channels, c2_channels, c3_channels, decoder_channels),
                decoder_channels=decoder_channels,
                num_classes=num_classes,
                dropout=dropout,
            )
            if feature_fusion_mode == "linear"
            else None
        )
        self.use_boundary_branch = use_boundary_branch
        if use_boundary_branch:
            self.boundary_branch = BoundaryAwareBranch(
                c1_channels,
                c2_channels,
                boundary_channels=boundary_channels,
                dropout=dropout,
            )
            main_channels = decoder_channels + boundary_channels
        else:
            self.boundary_branch = None
            main_channels = decoder_channels
        self.main_head = (
            nn.Sequential(
                ConvNormActDrop(
                    main_channels,
                    decoder_channels,
                    kernel_size=3,
                    dropout=dropout,
                ),
                nn.Conv2d(decoder_channels, num_classes, 1),
            )
            if feature_fusion_mode == "gated"
            else None
        )
        self.use_auxiliary_head = auxiliary_head
        if auxiliary_head:
            self.auxiliary_head = nn.Sequential(
                ConvNormActDrop(
                    decoder_channels,
                    decoder_channels,
                    kernel_size=3,
                    dropout=dropout,
                ),
                nn.Conv2d(decoder_channels, num_classes, 1),
            )
        else:
            self.auxiliary_head = None

    def forward(self, image: Tensor) -> Tensor | dict[str, Tensor | None]:
        input_size = image.shape[-2:]
        c1, c2, c3, c4 = self.encoder(image)
        f4 = self.global_context(c4)
        edge_logits: Tensor | None = None
        if self.linear_head is not None:
            main_features = self.linear_head((c1, c2, c3, f4))
            auxiliary_features = f4
        else:
            if self.feature_fusion is None or self.main_head is None:
                raise RuntimeError("gated GCB-Seg decoder is incomplete")
            f1, _, f3 = self.feature_fusion(c1, c2, c3, f4)
            auxiliary_features = f3
            if self.boundary_branch is not None:
                boundary_features, edge_logits = self.boundary_branch(c1, c2)
                boundary_features = F.interpolate(
                    boundary_features,
                    size=f1.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                main_features = self.main_head(
                    torch.cat((f1, boundary_features), dim=1)
                )
            else:
                main_features = self.main_head(f1)
        main_logits = F.interpolate(
            main_features, size=input_size, mode="bilinear", align_corners=False
        )

        if not self.training:
            return main_logits
        aux_logits = (
            F.interpolate(
                self.auxiliary_head(auxiliary_features),
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
            if self.auxiliary_head is not None
            else None
        )
        if edge_logits is not None:
            edge_logits = F.interpolate(
                edge_logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        return {
            "logits": main_logits,
            "aux_logits": aux_logits,
            "edge_logits": edge_logits,
        }


def build_gcb_seg(**kwargs: object) -> GCBSeg:
    return GCBSeg(**kwargs)
