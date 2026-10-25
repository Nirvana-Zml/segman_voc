#!/usr/bin/env python3
"""Summarize durable evidence for the standalone VOC2012 project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() and (candidate / "voc_segmentation").is_dir():
            return candidate
    raise RuntimeError("could not locate repository root containing voc_segmentation")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def relative(paths: Iterable[Path], root: Path) -> list[str]:
    return [str(path.relative_to(root)).replace("\\", "/") for path in paths]


def completed_result(path: Path) -> bool:
    value = load_json(path)
    return bool(value and (value.get("ok") is True or value.get("status") == "complete"))


def build_status(repo: Path) -> dict[str, Any]:
    app = repo / "voc_segmentation"
    docs = app / "project_docs"
    artifacts = app / "artifacts"
    runs = app / "runs"

    milestone_names = (
        "00-project-reset",
        "01-environment-and-source",
        "02-data-audit-and-sbd",
        "03-cnn-baseline",
        "04-transformer-baseline",
        "05-gcb-diagnostics-and-ablations",
        "06-formal-training-and-selection",
        "07-test400-inference",
        "08-report-and-package",
    )
    required_docs = [
        docs / "README.md",
        docs / "STATUS.md",
        docs / "INDEX.md",
        docs / "templates/STEP_TEMPLATE.md",
        *[docs / "milestones" / f"{name}.md" for name in milestone_names],
    ]
    missing_docs = [path for path in required_docs if not path.is_file()]
    steps = sorted((docs / "steps").glob("[0-9][0-9][0-9][0-9]-*.md")) if (docs / "steps").is_dir() else []
    documentation = {
        "state": "complete" if not missing_docs and steps else "pending",
        "missing": relative(missing_docs, repo),
        "active_step_count": len(steps),
        "latest_step": relative(steps[-1:], repo),
        "legacy_ledger_commit": "111f8a0",
    }

    required_source = [
        repo / "Dockerfile",
        repo / ".dockerignore",
        app / "README.md",
        app / "requirements.txt",
        app / "project_contract.py",
        app / "configs/deeplabv3plus_r50.yaml",
        app / "configs/segformer_b1.yaml",
        app / "configs/gcb_seg_b1_384.yaml",
        app / "tests/test_project_contract.py",
        app / "tests/test_config_contract.py",
    ]
    missing_source = [path for path in required_source if not path.is_file()]
    environment_report = artifacts / "environment.json"
    environment_value = load_json(environment_report)
    environment_ok = bool(
        environment_value
        and environment_value.get("ok") is True
        and environment_value.get("cuda_allocation_ok") is True
    )
    environment = {
        "state": "complete" if environment_ok else ("structural" if not missing_source else "pending"),
        "missing": relative(missing_source, repo),
        "report": relative([environment_report], repo) if environment_report.is_file() else [],
    }

    audit_path = artifacts / "data_check/split_summary.json"
    audit = load_json(audit_path)
    data_ok = bool(
        audit
        and audit.get("ok") is True
        and audit.get("train_count") == 1464
        and audit.get("val_count") == 1449
        and audit.get("train_val_overlap") == 0
        and audit.get("test_count") == 400
    )
    sbd_provenance_path = artifacts / "sbd/provenance.json"
    trainaug_path = artifacts / "sbd/trainaug_summary.json"
    sbd_provenance = load_json(sbd_provenance_path)
    trainaug = load_json(trainaug_path)
    sbd_ok = bool(
        sbd_provenance
        and sbd_provenance.get("ok") is True
        and sbd_provenance.get("sha256")
        and trainaug
        and trainaug.get("ok") is True
        and trainaug.get("train_aug_count") == 10582
        and trainaug.get("train_val_overlap") == 0
    )
    data = {
        "state": "complete" if data_ok and sbd_ok else ("partial" if data_ok else "pending"),
        "audit": relative([audit_path], repo) if audit_path.is_file() else [],
        "sbd_present": (repo / "data/sbd/benchmark_RELEASE").is_dir(),
        "sbd_provenance": relative([sbd_provenance_path], repo) if sbd_provenance_path.is_file() else [],
        "trainaug_summary": relative([trainaug_path], repo) if trainaug_path.is_file() else [],
    }

    stage_files = {
        "cnn_baseline": runs / "deeplabv3plus_r50_voc_train/result.json",
        "transformer_baseline": runs / "segformer/result.json",
        "gcb_diagnostics_and_ablations": artifacts / "gcb_diagnostics.json",
        "formal_training_and_selection": artifacts / "model_selection.json",
    }
    stage_values = {
        name: {
            "state": "complete" if completed_result(path) else "pending",
            "evidence": relative([path], repo) if path.is_file() else [],
        }
        for name, path in stage_files.items()
    }

    prediction_report = artifacts / "prediction_validation.json"
    prediction = load_json(prediction_report)
    prediction_ok = bool(
        prediction
        and prediction.get("ok") is True
        and prediction.get("prediction_count") == 400
    )
    predictions = {
        "state": "complete" if prediction_ok else "pending",
        "report": relative([prediction_report], repo) if prediction_report.is_file() else [],
    }

    reports = sorted(repo.glob("report*.docx")) + sorted(app.glob("report*.docx"))
    packages = sorted(repo.glob("*.zip"))
    package = {
        "state": "complete" if reports and packages else "pending",
        "reports": relative(reports, repo),
        "packages": relative(packages, repo),
    }

    stages = {
        "documentation": documentation,
        "environment_and_source": environment,
        "data_audit_and_sbd": data,
        **stage_values,
        "test400_predictions": predictions,
        "report_and_package": package,
    }
    order = list(stages)
    next_stage = next((name for name in order if stages[name]["state"] != "complete"), None)
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
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve() if args.repo else find_repo_root(Path(__file__).resolve())
    result = build_status(repo)
    print(json.dumps(result, ensure_ascii=False, indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
