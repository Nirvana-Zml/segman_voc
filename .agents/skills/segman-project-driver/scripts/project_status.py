#!/usr/bin/env python3
"""Summarize durable completion evidence for the SegMAN VOC2012 project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "SegMAN-main").is_dir() and (candidate / ".git").exists():
            return candidate
    raise RuntimeError("could not locate repository root containing SegMAN-main")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def relative(paths: Iterable[Path], root: Path) -> list[str]:
    return [str(path.relative_to(root)).replace("\\", "/") for path in paths]


def jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
    except (OSError, UnicodeError):
        return


def loss_bounds(logs: Iterable[Path]) -> tuple[float | None, float | None]:
    losses: list[float] = []
    for path in logs:
        for record in jsonl_records(path):
            value = record.get("loss")
            if record.get("mode") == "train" and isinstance(value, (int, float)):
                losses.append(float(value))
    if not losses:
        return None, None
    return losses[0], losses[-1]


def validation_metrics(logs: Iterable[Path]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    metric_keys = ("mIoU", "mAcc", "aAcc")
    for path in logs:
        for record in jsonl_records(path):
            metrics = {
                key: record[key]
                for key in metric_keys
                if isinstance(record.get(key), (int, float))
            }
            if metrics:
                found.append({"log": str(path), **metrics})
    return found


def build_status(repo: Path) -> dict[str, Any]:
    app = repo / "SegMAN-main"
    work = app / "work_dirs"

    docs = app / "project_docs"
    required_docs = [
        docs / "README.md",
        docs / "STATUS.md",
        docs / "INDEX.md",
        docs / "templates/STEP_TEMPLATE.md",
        *[docs / "milestones" / f"{number:02d}-{name}.md" for number, name in (
            (0, "agent-and-documentation"),
            (1, "source-and-tests"),
            (2, "data-audit"),
            (3, "segman-s-smoke"),
            (4, "segman-t-overfit"),
            (5, "formal-training"),
            (6, "model-selection"),
            (7, "test400-inference"),
            (8, "report-and-package"),
        )],
    ]
    missing_docs = [path for path in required_docs if not path.is_file()]
    step_records = sorted((docs / "steps").glob("[0-9][0-9][0-9][0-9]-*.md")) \
        if (docs / "steps").is_dir() else []
    documentation = {
        "state": "complete" if not missing_docs and step_records else "pending",
        "missing": relative(missing_docs, repo),
        "step_record_count": len(step_records),
        "latest_step": relative(step_records[-1:], repo),
    }

    required = [
        app / "segmentation/local_configs/segman/voc/segman_t_voc12.py",
        app / "segmentation/local_configs/segman/voc/segman_s_voc12.py",
        app / "segmentation/tools/audit_voc.py",
        app / "segmentation/tools/predict_test400.py",
        app / "segmentation/tools/verify_predictions.py",
        app / "tests/test_voc_configs.py",
        app / "tests/test_voc_dataset.py",
        app / "tests/test_prediction_format.py",
    ]
    missing = [path for path in required if not path.is_file()]
    source = {
        "state": "complete" if not missing else "pending",
        "missing": relative(missing, repo),
        "note": "Run container unit tests to verify behavior; file presence is structural evidence only.",
    }

    audit_path = work / "data_audit.json"
    audit = load_json(audit_path)
    audit_ok = bool(
        audit
        and audit.get("ok") is True
        and audit.get("split_counts", {}).get("train") == 1464
        and audit.get("split_counts", {}).get("val") == 1449
        and audit.get("train_val_overlap") == 0
        and audit.get("invalid_labels") == []
        and audit.get("test", {}).get("count") == 400
    )
    data = {
        "state": "complete" if audit_ok else "pending",
        "evidence": relative([audit_path], repo) if audit_path.is_file() else [],
        "errors": audit.get("errors", []) if audit else ["missing or unreadable audit JSON"],
    }

    smoke_dir = work / "smoke_s"
    smoke_checkpoints = sorted(smoke_dir.glob("iter_*.pth"))
    smoke_logs = sorted(smoke_dir.glob("*.log.json"))
    smoke_events = sorted(smoke_dir.glob("tf_logs/events.out.tfevents.*"))
    smoke_first, smoke_last = loss_bounds(smoke_logs)
    smoke = {
        "state": "complete" if smoke_checkpoints and smoke_logs and smoke_events else "pending",
        "checkpoints": relative(smoke_checkpoints, repo),
        "initial_loss": smoke_first,
        "final_loss": smoke_last,
    }

    overfit_dirs = sorted(path for path in work.glob("overfit*") if path.is_dir())
    overfit_checkpoints = sorted(
        path for directory in overfit_dirs for path in directory.glob("iter_100.pth")
    )
    overfit_logs = sorted(
        path for directory in overfit_dirs for path in directory.glob("*.log.json")
    )
    overfit_first, overfit_last = loss_bounds(overfit_logs)
    material_drop = bool(
        overfit_first is not None
        and overfit_last is not None
        and overfit_last < overfit_first * 0.5
    )
    overfit = {
        "state": "complete" if overfit_checkpoints and material_drop else "pending",
        "checkpoints": relative(overfit_checkpoints, repo),
        "initial_loss": overfit_first,
        "final_loss": overfit_last,
        "material_loss_drop": material_drop,
        "submission_eligible": False,
    }

    diagnostic_prefixes = ("smoke", "overfit")
    formal_dirs = sorted(
        path
        for path in work.iterdir()
        if path.is_dir() and not path.name.lower().startswith(diagnostic_prefixes)
    ) if work.is_dir() else []
    formal_logs = sorted(
        path for directory in formal_dirs for path in directory.rglob("*.log.json")
    )
    formal_metrics = validation_metrics(formal_logs)
    best_checkpoints = sorted(
        path for directory in formal_dirs for path in directory.rglob("best_mIoU*.pth")
    )
    restorable = sorted(
        path
        for directory in formal_dirs
        for pattern in ("latest.pth", "iter_*.pth")
        for path in directory.rglob(pattern)
    )
    formal_state = "complete" if formal_metrics and best_checkpoints else (
        "partial" if formal_logs or restorable else "pending"
    )
    formal = {
        "state": formal_state,
        "best_checkpoints": relative(best_checkpoints, repo),
        "restorable_checkpoints": relative(restorable, repo),
        "validation_records": formal_metrics[-5:],
    }

    prediction_dir = app / "predictions/test400"
    prediction_files = sorted(prediction_dir.glob("*.png")) if prediction_dir.is_dir() else []
    validation_path = work / "prediction_validation.json"
    prediction_validation = load_json(validation_path)
    predictions_ok = bool(
        prediction_validation
        and prediction_validation.get("ok") is True
        and prediction_validation.get("prediction_count") == 400
    )
    predictions = {
        "state": "complete" if predictions_ok else (
            "partial" if prediction_files else "pending"
        ),
        "png_count": len(prediction_files),
        "validation_report": (
            relative([validation_path], repo) if validation_path.is_file() else []
        ),
    }

    report_candidates = sorted(repo.glob("report*.docx")) + sorted(
        app.glob("report*.docx")
    )
    submission_candidates = sorted(repo.glob("*/*.zip"))
    deliverables = {
        "state": "complete" if report_candidates and submission_candidates else "pending",
        "reports": relative(report_candidates, repo),
        "archives": relative(submission_candidates, repo),
        "note": "Inspect archive contents and real student naming before treating this gate as complete.",
    }

    stages = {
        "documentation_log": documentation,
        "source_structure": source,
        "data_audit": data,
        "segman_s_smoke": smoke,
        "segman_t_overfit": overfit,
        "formal_training_and_validation": formal,
        "test400_predictions": predictions,
        "report_and_package": deliverables,
    }
    order = list(stages)
    next_stage = next(
        (name for name in order if stages[name]["state"] != "complete"), None
    )
    return {
        "repository": str(repo),
        "overall_state": "complete" if next_stage is None else "incomplete",
        "highest_verified_stage": (
            order[order.index(next_stage) - 1]
            if next_stage is not None and order.index(next_stage) > 0
            else None
        ),
        "next_stage": next_stage,
        "stages": stages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        help="Repository root. Auto-detected from this script when omitted.",
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve() if args.repo else find_repo_root(Path(__file__).resolve())
    result = build_status(repo)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=None if args.compact else 2,
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
