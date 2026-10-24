"""SBD provenance, category-mask conversion, and leak-free trainaug assembly."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.io import loadmat

from .audit import VALID_LABELS, read_ids, sequence_sha256


CHINA_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")
SBD_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
SBD_EXPECTED_MD5 = "82b4d87ceb2ed10f6038a1cba92111cb"
SBD_CITATION = "Hariharan et al., Semantic Contours from Inverse Detectors, ICCV 2011"


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_sbd_segmentation(path: Path) -> np.ndarray:
    payload = loadmat(path)
    if "GTcls" not in payload:
        raise ValueError("missing GTcls")
    gtcls = payload["GTcls"]
    if gtcls.shape != (1, 1) or "Segmentation" not in (gtcls.dtype.names or ()):
        raise ValueError("unexpected GTcls structure")
    mask = np.asarray(gtcls[0, 0]["Segmentation"], dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"segmentation must be 2D, got {mask.shape}")
    return mask


def prepare_trainaug(
    voc_root: Path,
    sbd_root: Path,
    archive_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    voc_root = voc_root.resolve()
    sbd_root = sbd_root.resolve()
    archive_path = archive_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    md5 = hash_file(archive_path, "md5")
    sha256 = hash_file(archive_path, "sha256")
    provenance = {
        "ok": md5 == SBD_EXPECTED_MD5,
        "timestamp": datetime.now(CHINA_TIME).isoformat(timespec="seconds"),
        "source_url": SBD_URL,
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "md5": md5,
        "expected_md5": SBD_EXPECTED_MD5,
        "sha256": sha256,
        "citation": SBD_CITATION,
        "license_note": (
            "The archive README requests citation but contains no separate license. "
            "The user-designated course plan explicitly permits SBD for this project."
        ),
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not provenance["ok"]:
        raise ValueError(f"SBD MD5 mismatch: {md5} != {SBD_EXPECTED_MD5}")

    voc_split = voc_root / "ImageSets/Segmentation"
    voc_train = set(read_ids(voc_split / "train.txt"))
    voc_val = set(read_ids(voc_split / "val.txt"))
    sbd_train = set(read_ids(sbd_root / "train.txt"))
    sbd_val = set(read_ids(sbd_root / "val.txt"))
    sbd_all = sbd_train | sbd_val
    train_aug = sorted((sbd_all - voc_val) | voc_train)
    sbd_extra = sorted(set(train_aug) - voc_train)

    mask_output = output_dir / "SegmentationClassAug"
    mask_output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    invalid_label_samples: list[str] = []
    dimension_mismatches: list[str] = []
    converted = 0

    for sample_id in sbd_extra:
        image_path = sbd_root / "img" / f"{sample_id}.jpg"
        mat_path = sbd_root / "cls" / f"{sample_id}.mat"
        output_path = mask_output / f"{sample_id}.png"
        if not image_path.is_file() or not mat_path.is_file():
            errors.append(f"missing SBD image or mask: {sample_id}")
            continue
        try:
            mask = load_sbd_segmentation(mat_path)
            with Image.open(image_path) as image:
                image_size = image.size
                image.verify()
        except (OSError, ValueError) as exc:
            errors.append(f"unreadable SBD sample {sample_id}: {exc}")
            continue
        mask_size = (mask.shape[1], mask.shape[0])
        if image_size != mask_size:
            dimension_mismatches.append(
                f"{sample_id}:image={image_size},mask={mask_size}"
            )
            continue
        invalid = sorted(int(value) for value in np.unique(mask) if int(value) not in VALID_LABELS)
        if invalid:
            invalid_label_samples.append(f"{sample_id}:{invalid}")
            continue
        Image.fromarray(mask, mode="L").save(output_path)
        converted += 1

    manifest_path = output_dir / "train_aug.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for sample_id in train_aug:
            if sample_id in voc_train:
                record = {
                    "id": sample_id,
                    "source": "voc",
                    "image": f"JPEGImages/{sample_id}.jpg",
                    "mask": f"SegmentationClass/{sample_id}.png",
                }
            else:
                record = {
                    "id": sample_id,
                    "source": "sbd",
                    "image": f"img/{sample_id}.jpg",
                    "mask": f"SegmentationClassAug/{sample_id}.png",
                }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "train_aug.txt").write_text(
        "\n".join(train_aug) + "\n", encoding="utf-8"
    )

    overlap = sorted(set(train_aug) & voc_val)
    duplicates = len(train_aug) - len(set(train_aug))
    ok = bool(
        len(train_aug) == 10582
        and len(voc_train) == 1464
        and len(sbd_extra) == 9118
        and converted == 9118
        and not overlap
        and duplicates == 0
        and not errors
        and not invalid_label_samples
        and not dimension_mismatches
    )
    summary: dict[str, Any] = {
        "ok": ok,
        "timestamp": datetime.now(CHINA_TIME).isoformat(timespec="seconds"),
        "train_aug_count": len(train_aug),
        "voc_train_count": len(voc_train),
        "voc_val_count": len(voc_val),
        "sbd_all_count": len(sbd_all),
        "sbd_val_overlap_count": len(sbd_all & voc_val),
        "sbd_extra_count": len(sbd_extra),
        "converted_mask_count": converted,
        "train_val_overlap": len(overlap),
        "duplicate_count": duplicates,
        "invalid_label_count": len(invalid_label_samples),
        "dimension_mismatch_count": len(dimension_mismatches),
        "error_count": len(errors),
        "train_aug_ids_sha256": sequence_sha256(train_aug),
        "errors": errors,
        "invalid_label_samples": invalid_label_samples,
        "dimension_mismatches": dimension_mismatches,
    }
    (output_dir / "trainaug_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
