#!/usr/bin/env python3
"""Create an immutable project step record and append it to the index."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


CHINA_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")
STATUS_LABELS = {
    "passed": "Passed",
    "failed": "Failed",
    "blocked": "Blocked",
    "in-progress": "In progress",
    "information": "Information",
}


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "SegMAN-main").is_dir() and (candidate / ".git").exists():
            return candidate
    raise RuntimeError("could not locate repository root containing SegMAN-main")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "step"


def bullets(values: Iterable[str], empty: str = "None.") -> str:
    items = [value.strip() for value in values if value.strip()]
    return "\n".join(f"- `{item}`" for item in items) if items else empty


def prose_bullets(values: Iterable[str], empty: str = "None.") -> str:
    items = [value.strip() for value in values if value.strip()]
    return "\n".join(f"- {item}" for item in items) if items else empty


def next_sequence(step_dir: Path) -> int:
    values: list[int] = []
    for path in step_dir.glob("[0-9][0-9][0-9][0-9]-*.md"):
        try:
            values.append(int(path.name[:4]))
        except ValueError:
            continue
    return max(values, default=0) + 1


def render(args: argparse.Namespace, sequence: int, timestamp: str) -> str:
    return f"""# Step {sequence:04d}: {args.title}

- Time: {timestamp}
- Milestone: `{args.milestone}`
- Status: **{STATUS_LABELS[args.status]}**

## Objective and outcome

{args.summary.strip()}

## Commands

{bullets(args.command)}

## Changed files

{bullets(args.changed)}

## Evidence

{bullets(args.evidence)}

## Decisions and assumptions

{prose_bullets(args.decision)}

## Next action

{args.next_action.strip()}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="Repository root; auto-detected by default.")
    parser.add_argument("--milestone", required=True, help="Milestone identifier, such as 05-formal-training.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--slug", help="Optional ASCII filename slug.")
    parser.add_argument("--status", required=True, choices=tuple(STATUS_LABELS))
    parser.add_argument("--summary", required=True)
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--changed", action="append", default=[])
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--decision", action="append", default=[])
    parser.add_argument("--next", dest="next_action", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve() if args.repo else find_repo_root(Path(__file__).resolve())
    docs = repo / "SegMAN-main" / "project_docs"
    step_dir = docs / "steps"
    index_path = docs / "INDEX.md"
    if not step_dir.is_dir() or not index_path.is_file():
        raise RuntimeError("project_docs package is missing steps/ or INDEX.md")

    sequence = next_sequence(step_dir)
    timestamp = datetime.now(CHINA_TIME).isoformat(timespec="seconds")
    filename = f"{sequence:04d}-{slugify(args.slug or args.title)}.md"
    record_path = step_dir / filename
    content = render(args, sequence, timestamp)
    if args.dry_run:
        print(content)
        return 0
    if record_path.exists():
        raise RuntimeError(f"refusing to overwrite existing record: {record_path}")

    record_path.write_text(content, encoding="utf-8", newline="\n")
    index_entry = (
        f"- [{sequence:04d} — {args.title}](steps/{filename}) — "
        f"{STATUS_LABELS[args.status]}, {timestamp}\n"
    )
    with index_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(index_entry)
    print(record_path.relative_to(repo).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
