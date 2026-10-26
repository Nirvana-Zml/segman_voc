"""GCB-Seg semantic, Lovasz, boundary, and auxiliary losses."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def semantic_boundary_targets(
    labels: Tensor, *, ignore_index: int = 255, max_void_width: int = 3
) -> tuple[Tensor, Tensor]:
    if labels.ndim != 3:
        raise ValueError("labels must have shape [B,H,W]")
    if max_void_width < 0:
        raise ValueError("max_void_width must be nonnegative")
    valid = labels != ignore_index
    boundary = torch.zeros_like(valid)

    horizontal_valid = valid[:, :, 1:] & valid[:, :, :-1]
    horizontal_edge = (labels[:, :, 1:] != labels[:, :, :-1]) & horizontal_valid
    boundary[:, :, 1:] |= horizontal_edge
    boundary[:, :, :-1] |= horizontal_edge

    vertical_valid = valid[:, 1:, :] & valid[:, :-1, :]
    vertical_edge = (labels[:, 1:, :] != labels[:, :-1, :]) & vertical_valid
    boundary[:, 1:, :] |= vertical_edge
    boundary[:, :-1, :] |= vertical_edge

    height, width = labels.shape[-2:]
    for distance in range(2, max_void_width + 2):
        if distance < width:
            left = labels[:, :, :-distance]
            right = labels[:, :, distance:]
            void_bridge = torch.ones_like(left, dtype=torch.bool)
            for offset in range(1, distance):
                void_bridge &= (
                    labels[:, :, offset : width - distance + offset] == ignore_index
                )
            edge = (
                (left != right)
                & (left != ignore_index)
                & (right != ignore_index)
                & void_bridge
            )
            boundary[:, :, :-distance] |= edge
            boundary[:, :, distance:] |= edge
        if distance < height:
            top = labels[:, :-distance, :]
            bottom = labels[:, distance:, :]
            void_bridge = torch.ones_like(top, dtype=torch.bool)
            for offset in range(1, distance):
                void_bridge &= (
                    labels[:, offset : height - distance + offset, :] == ignore_index
                )
            edge = (
                (top != bottom)
                & (top != ignore_index)
                & (bottom != ignore_index)
                & void_bridge
            )
            boundary[:, :-distance, :] |= edge
            boundary[:, distance:, :] |= edge

    boundary = F.max_pool2d(
        boundary.unsqueeze(1).to(torch.float32), kernel_size=3, stride=1, padding=1
    )
    valid = valid.unsqueeze(1)
    boundary = boundary.masked_fill(~valid, 0.0)
    return boundary, valid


def _lovasz_gradient(sorted_foreground: Tensor) -> Tensor:
    count = sorted_foreground.numel()
    foreground_count = sorted_foreground.sum()
    intersection = foreground_count - sorted_foreground.cumsum(0)
    union = foreground_count + (1.0 - sorted_foreground).cumsum(0)
    gradient = 1.0 - intersection / union
    if count > 1:
        gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
    return gradient


def _flat_lovasz(probabilities: Tensor, labels: Tensor) -> Tensor:
    class_losses: list[Tensor] = []
    for class_index in range(probabilities.shape[1]):
        foreground = (labels == class_index).to(probabilities.dtype)
        if foreground.sum() == 0:
            continue
        errors = (foreground - probabilities[:, class_index]).abs()
        sorted_errors, permutation = torch.sort(errors, descending=True)
        sorted_foreground = foreground[permutation]
        class_losses.append(
            torch.dot(sorted_errors, _lovasz_gradient(sorted_foreground))
        )
    if not class_losses:
        return probabilities.sum() * 0.0
    return torch.stack(class_losses).mean()


def lovasz_softmax_loss(
    logits: Tensor, labels: Tensor, *, ignore_index: int = 255
) -> Tensor:
    if logits.ndim != 4 or labels.ndim != 3:
        raise ValueError("logits and labels must have shapes [B,C,H,W] and [B,H,W]")
    if logits.shape[0] != labels.shape[0] or logits.shape[-2:] != labels.shape[-2:]:
        raise ValueError("logits and labels have incompatible shapes")
    probabilities = logits.softmax(dim=1)
    sample_losses: list[Tensor] = []
    for sample_probabilities, sample_labels in zip(probabilities, labels):
        flat_labels = sample_labels.reshape(-1)
        valid = flat_labels != ignore_index
        if not valid.any():
            continue
        flat_probabilities = sample_probabilities.permute(1, 2, 0).reshape(
            -1, probabilities.shape[1]
        )
        sample_losses.append(
            _flat_lovasz(flat_probabilities[valid], flat_labels[valid])
        )
    if not sample_losses:
        return logits.sum() * 0.0
    return torch.stack(sample_losses).mean()


def boundary_bce_dice_loss(
    logits: Tensor,
    targets: Tensor,
    valid: Tensor,
    *,
    pos_weight: float | None = None,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    if logits.shape != targets.shape or logits.shape != valid.shape:
        raise ValueError("boundary logits, targets, and valid mask must match")
    if not valid.any():
        raise ValueError("boundary loss received no valid pixels")
    positive_weight = (
        torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        if pos_weight is not None
        else None
    )
    bce_map = F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none", pos_weight=positive_weight
    )
    bce = bce_map[valid].mean()
    probabilities = logits.sigmoid()[valid]
    valid_targets = targets.to(logits.dtype)[valid]
    dice = 1.0 - (
        2.0 * (probabilities * valid_targets).sum() + epsilon
    ) / (probabilities.sum() + valid_targets.sum() + epsilon)
    return 0.5 * (bce + dice), bce, dice


class GCBSegLoss(nn.Module):
    def __init__(
        self,
        *,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        lovasz_weight: float = 0.5,
        edge_weight: float = 0.2,
        aux_weight: float = 0.4,
        label_smoothing: float = 0.05,
        edge_pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.edge_weight = edge_weight
        self.aux_weight = aux_weight
        self.edge_pos_weight = edge_pos_weight
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index, label_smoothing=label_smoothing
        )

    def forward(
        self, outputs: Mapping[str, Tensor | None], labels: Tensor
    ) -> dict[str, Tensor]:
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("GCB-Seg outputs require main logits")
        ce = self.cross_entropy(logits, labels)
        lovasz = lovasz_softmax_loss(
            logits, labels, ignore_index=self.ignore_index
        )

        aux_logits = outputs.get("aux_logits")
        if self.aux_weight and aux_logits is None:
            raise ValueError("auxiliary logits are required when aux_weight is nonzero")
        aux = (
            self.cross_entropy(aux_logits, labels)
            if aux_logits is not None
            else logits.sum() * 0.0
        )

        edge_logits = outputs.get("edge_logits")
        if self.edge_weight and edge_logits is None:
            raise ValueError("edge logits are required when edge_weight is nonzero")
        if edge_logits is not None:
            edge_targets, edge_valid = semantic_boundary_targets(
                labels, ignore_index=self.ignore_index
            )
            edge, edge_bce, edge_dice = boundary_bce_dice_loss(
                edge_logits,
                edge_targets,
                edge_valid,
                pos_weight=self.edge_pos_weight,
            )
        else:
            edge = logits.sum() * 0.0
            edge_bce = edge
            edge_dice = edge

        total = (
            self.ce_weight * ce
            + self.lovasz_weight * lovasz
            + self.edge_weight * edge
            + self.aux_weight * aux
        )
        return {
            "total": total,
            "ce": ce,
            "lovasz": lovasz,
            "edge": edge,
            "edge_bce": edge_bce,
            "edge_dice": edge_dice,
            "aux": aux,
        }
