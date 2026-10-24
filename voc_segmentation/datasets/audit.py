"""Deterministic integrity audit for official VOC2012 and test400 data."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


CHINA_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")
VALID_LABELS = frozenset((*range(21), 255))
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


@dataclass(frozen=True)
class AuditExpectations:
    train_count: int = 1464
    val_count: int = 1449
    test_count: int = 400
    overlay_count: int = 20
    seed: int = 42


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sequence_sha256(values: Iterable[str]) -> str:
    normalized = "\n".join(values) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def voc_palette() -> np.ndarray:
    palette = np.zeros((21, 3), dtype=np.uint8)
    for label in range(21):
        value = label
        for shift in range(8):
            palette[label, 0] |= ((value >> 0) & 1) << (7 - shift)
            palette[label, 1] |= ((value >> 1) & 1) << (7 - shift)
            palette[label, 2] |= ((value >> 2) & 1) << (7 - shift)
            value >>= 3
    return palette


def render_overlay(image_path: Path, mask_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
    palette = voc_palette()
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask <= 20
    colored[valid] = palette[mask[valid]]
    colored[~valid] = (127, 127, 127)
    colored_image = Image.fromarray(colored, mode="RGB")
    overlay = Image.blend(image, colored_image, alpha=0.45)
    canvas = Image.new("RGB", (image.width * 3, image.height))
    canvas.paste(image, (0, 0))
    canvas.paste(colored_image, (image.width, 0))
    canvas.paste(overlay, (image.width * 2, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def audit_dataset(
    voc_root: Path,
    test_root: Path,
    output_dir: Path,
    expectations: AuditExpectations = AuditExpectations(),
) -> dict[str, Any]:
    voc_root = voc_root.resolve()
    test_root = test_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = voc_root / "JPEGImages"
    mask_dir = voc_root / "SegmentationClass"
    split_dir = voc_root / "ImageSets/Segmentation"
    errors: list[str] = []

    required = [
        voc_root,
        image_dir,
        mask_dir,
        split_dir / "train.txt",
        split_dir / "val.txt",
        test_root,
    ]
    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        errors.extend(f"missing required path: {path}" for path in missing_required)

    train_ids = read_ids(split_dir / "train.txt") if (split_dir / "train.txt").is_file() else []
    val_ids = read_ids(split_dir / "val.txt") if (split_dir / "val.txt").is_file() else []
    train_counter = Counter(train_ids)
    val_counter = Counter(val_ids)
    train_duplicates = sorted(key for key, count in train_counter.items() if count > 1)
    val_duplicates = sorted(key for key, count in val_counter.items() if count > 1)
    overlap = sorted(set(train_ids) & set(val_ids))

    if len(train_ids) != expectations.train_count:
        errors.append(f"train count {len(train_ids)} != {expectations.train_count}")
    if len(val_ids) != expectations.val_count:
        errors.append(f"val count {len(val_ids)} != {expectations.val_count}")
    if train_duplicates:
        errors.append(f"train contains {len(train_duplicates)} duplicate IDs")
    if val_duplicates:
        errors.append(f"val contains {len(val_duplicates)} duplicate IDs")
    if overlap:
        errors.append(f"train/val overlap contains {len(overlap)} IDs")

    class_pixels = np.zeros(22, dtype=np.int64)
    class_images = np.zeros(22, dtype=np.int64)
    invalid_samples: list[str] = []
    dimension_mismatches: list[str] = []
    unreadable_images: list[str] = []
    missing_images: list[str] = []
    missing_masks: list[str] = []

    for split_name, sample_ids in (("train", train_ids), ("val", val_ids)):
        for sample_id in sample_ids:
            image_path = image_dir / f"{sample_id}.jpg"
            mask_path = mask_dir / f"{sample_id}.png"
            if not image_path.is_file():
                missing_images.append(f"{split_name}:{sample_id}")
                continue
            if not mask_path.is_file():
                missing_masks.append(f"{split_name}:{sample_id}")
                continue
            try:
                with Image.open(image_path) as image:
                    image_size = image.size
                    image.verify()
                with Image.open(mask_path) as mask_image:
                    mask_size = mask_image.size
                    mask = np.asarray(mask_image, dtype=np.uint8)
            except (OSError, ValueError) as exc:
                unreadable_images.append(f"{split_name}:{sample_id}:{exc}")
                continue
            if image_size != mask_size:
                dimension_mismatches.append(
                    f"{split_name}:{sample_id}:image={image_size},mask={mask_size}"
                )
            unique, counts = np.unique(mask, return_counts=True)
            invalid = sorted(int(value) for value in unique if int(value) not in VALID_LABELS)
            if invalid:
                invalid_samples.append(f"{split_name}:{sample_id}:{invalid}")
            for value, count in zip(unique, counts, strict=True):
                label = int(value)
                if label in VALID_LABELS:
                    index = 21 if label == 255 else label
                    class_pixels[index] += int(count)
                    class_images[index] += 1

    test_files = sorted(
        path for path in test_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ) if test_root.is_dir() else []
    test_stems = [path.stem for path in test_files]
    test_duplicate_stems = sorted(
        key for key, count in Counter(test_stems).items() if count > 1
    )
    test_unreadable: list[str] = []
    for path in test_files:
        try:
            with Image.open(path) as image:
                image.verify()
        except OSError as exc:
            test_unreadable.append(f"{path.name}:{exc}")
    if len(test_files) != expectations.test_count:
        errors.append(f"test count {len(test_files)} != {expectations.test_count}")
    if test_duplicate_stems:
        errors.append(f"test contains {len(test_duplicate_stems)} duplicate stems")

    detail_groups = {
        "missing_images": missing_images,
        "missing_masks": missing_masks,
        "unreadable_images_or_masks": unreadable_images,
        "dimension_mismatches": dimension_mismatches,
        "invalid_label_samples": invalid_samples,
        "test_unreadable": test_unreadable,
    }
    for name, values in detail_groups.items():
        if values:
            errors.append(f"{name}: {len(values)}")

    overlay_candidates = [
        sample_id for sample_id in dict.fromkeys((*train_ids, *val_ids))
        if (image_dir / f"{sample_id}.jpg").is_file()
        and (mask_dir / f"{sample_id}.png").is_file()
    ]
    overlay_count = min(expectations.overlay_count, len(overlay_candidates))
    selected = random.Random(expectations.seed).sample(overlay_candidates, overlay_count)
    overlay_dir = output_dir / "overlays"
    overlay_errors: list[str] = []
    for sample_id in selected:
        try:
            render_overlay(
                image_dir / f"{sample_id}.jpg",
                mask_dir / f"{sample_id}.png",
                overlay_dir / f"{sample_id}.jpg",
            )
        except (OSError, ValueError) as exc:
            overlay_errors.append(f"{sample_id}:{exc}")
    if overlay_errors:
        errors.append(f"overlay generation failures: {len(overlay_errors)}")

    rows = []
    for index in range(22):
        label = 255 if index == 21 else index
        rows.append(
            {
                "label": label,
                "image_count": int(class_images[index]),
                "pixel_count": int(class_pixels[index]),
            }
        )
    with (output_dir / "class_pixel_statistics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=("label", "image_count", "pixel_count"))
        writer.writeheader()
        writer.writerows(rows)

    invalid_lines = []
    for name, values in detail_groups.items():
        invalid_lines.extend(f"{name}\t{value}" for value in values)
    invalid_lines.extend(f"overlay_error\t{value}" for value in overlay_errors)
    (output_dir / "invalid_samples.txt").write_text(
        "\n".join(invalid_lines) + ("\n" if invalid_lines else ""), encoding="utf-8"
    )

    summary: dict[str, Any] = {
        "ok": not errors,
        "timestamp": datetime.now(CHINA_TIME).isoformat(timespec="seconds"),
        "voc_root": str(voc_root),
        "test_root": str(test_root),
        "expectations": asdict(expectations),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "train_val_overlap": len(overlap),
        "train_duplicate_count": len(train_duplicates),
        "val_duplicate_count": len(val_duplicates),
        "test_count": len(test_files),
        "test_duplicate_stem_count": len(test_duplicate_stems),
        "train_ids_sha256": sequence_sha256(train_ids),
        "val_ids_sha256": sequence_sha256(val_ids),
        "test_names_sha256": sequence_sha256(path.name for path in test_files),
        "checked_labeled_samples": len(train_ids) + len(val_ids),
        "overlay_count": len(selected) - len(overlay_errors),
        "overlay_sample_ids": selected,
        "errors": errors,
        "details": {name: len(values) for name, values in detail_groups.items()},
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
