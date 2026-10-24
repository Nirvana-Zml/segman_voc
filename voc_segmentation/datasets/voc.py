"""Pascal VOC2012 datasets and synchronized segmentation transforms."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as F

from project_contract import IGNORE_INDEX


@dataclass(frozen=True)
class SegmentationSample:
    sample_id: str
    image_path: Path
    mask_path: Path
    source: str


def _normalize_image(image: Image.Image, mean: tuple[float, ...], std: tuple[float, ...]) -> Tensor:
    return F.normalize(F.pil_to_tensor(image).float().div_(255.0), mean, std)


class SegmentationTrainTransform:
    """Synchronized random scale/crop/flip with image-only color jitter."""

    def __init__(
        self,
        crop_size: tuple[int, int] = (384, 384),
        scale_range: tuple[float, float] = (0.5, 2.0),
        horizontal_flip_probability: float = 0.5,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        color_jitter: ColorJitter | None = None,
    ) -> None:
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.horizontal_flip_probability = horizontal_flip_probability
        self.mean = mean
        self.std = std
        self.color_jitter = color_jitter or ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        scale = random.uniform(*self.scale_range)
        height = max(1, round(image.height * scale))
        width = max(1, round(image.width * scale))
        image = F.resize(image, [height, width], InterpolationMode.BILINEAR)
        mask = F.resize(mask, [height, width], InterpolationMode.NEAREST)

        crop_height, crop_width = self.crop_size
        pad_right = max(0, crop_width - width)
        pad_bottom = max(0, crop_height - height)
        if pad_right or pad_bottom:
            image = F.pad(image, [0, 0, pad_right, pad_bottom], fill=0)
            mask = F.pad(mask, [0, 0, pad_right, pad_bottom], fill=IGNORE_INDEX)

        max_top = image.height - crop_height
        max_left = image.width - crop_width
        top = random.randint(0, max_top) if max_top else 0
        left = random.randint(0, max_left) if max_left else 0
        image = F.crop(image, top, left, crop_height, crop_width)
        mask = F.crop(mask, top, left, crop_height, crop_width)

        if random.random() < self.horizontal_flip_probability:
            image = F.hflip(image)
            mask = F.hflip(mask)
        image = self.color_jitter(image)
        return _normalize_image(image, self.mean, self.std), F.pil_to_tensor(mask).squeeze(0).long()


class SegmentationEvalTransform:
    """Deterministic conversion that preserves original spatial dimensions."""

    def __init__(
        self,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        return _normalize_image(image, self.mean, self.std), F.pil_to_tensor(mask).squeeze(0).long()


class VOCSegmentationDataset(Dataset[dict[str, Tensor | str | tuple[int, int]]]):
    """Official VOC train/val or a provenance-audited trainaug manifest."""

    def __init__(
        self,
        voc_root: Path | str,
        split: str,
        transform: Callable[[Image.Image, Image.Image], tuple[Tensor, Tensor]],
        *,
        trainaug_manifest: Path | str | None = None,
        sbd_root: Path | str | None = None,
        sbd_mask_root: Path | str | None = None,
    ) -> None:
        self.voc_root = Path(voc_root)
        self.split = split
        self.transform = transform
        if split == "trainaug":
            if trainaug_manifest is None or sbd_root is None or sbd_mask_root is None:
                raise ValueError("trainaug requires manifest, sbd_root, and sbd_mask_root")
            self.samples = self._load_trainaug(
                Path(trainaug_manifest), Path(sbd_root), Path(sbd_mask_root)
            )
        elif split in {"train", "val"}:
            ids_path = self.voc_root / "ImageSets/Segmentation" / f"{split}.txt"
            sample_ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.samples = [
                SegmentationSample(
                    sample_id=sample_id,
                    image_path=self.voc_root / "JPEGImages" / f"{sample_id}.jpg",
                    mask_path=self.voc_root / "SegmentationClass" / f"{sample_id}.png",
                    source="voc",
                )
                for sample_id in sample_ids
            ]
        else:
            raise ValueError(f"unsupported split: {split}")

    def _load_trainaug(
        self, manifest: Path, sbd_root: Path, sbd_mask_root: Path
    ) -> list[SegmentationSample]:
        samples: list[SegmentationSample] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item["source"] == "voc":
                image_path = self.voc_root / item["image"]
                mask_path = self.voc_root / item["mask"]
            elif item["source"] == "sbd":
                image_path = sbd_root / item["image"]
                mask_path = sbd_mask_root / Path(item["mask"]).name
            else:
                raise ValueError(f"unsupported source in manifest: {item['source']}")
            samples.append(
                SegmentationSample(item["id"], image_path, mask_path, item["source"])
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | tuple[int, int]]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image_file:
            image = image_file.convert("RGB")
        with Image.open(sample.mask_path) as mask_file:
            mask_indices = np.asarray(mask_file, dtype=np.uint8).copy()
        if mask_indices.ndim != 2:
            raise ValueError(f"mask must contain one index channel: {sample.mask_path}")
        mask = Image.fromarray(mask_indices, mode="L")
        original_size = (image.height, image.width)
        image_tensor, mask_tensor = self.transform(image, mask)
        return {
            "id": sample.sample_id,
            "source": sample.source,
            "image": image_tensor,
            "mask": mask_tensor,
            "original_size": original_size,
        }
