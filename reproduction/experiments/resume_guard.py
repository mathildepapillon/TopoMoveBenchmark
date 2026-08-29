"""Idempotent-resume helpers for campaign runners.

Every runner writes its cell's JSONL once, at the end, so a file that exists
and parses with the expected row count IS a completed cell. Runners call
``cell_complete`` first and exit 0 on a hit, which makes whole-array
resubmissions (rescue reruns, the campaign sweeper) safe no-ops for
completed cells.

Multi-seed runners (sweep, menu) additionally journal each finished seed to
``<out>.partial`` so a mid-cell kill (e.g. the 90-min cap) only costs the
unfinished seeds on the rescue rerun.
"""

from __future__ import annotations

import json
from pathlib import Path


def cell_complete(path: Path, min_rows: int = 1,
                  required_key: str = "flops") -> bool:
    """True iff ``path`` holds >= min_rows parseable rows with required_key."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        rows = [json.loads(line)
                for line in path.read_text().splitlines() if line.strip()]
    except (json.JSONDecodeError, OSError):
        return False
    return (len(rows) >= min_rows
            and all(required_key in r for r in rows))


def partial_path(out_path: Path) -> Path:
    return Path(str(out_path) + ".partial")


def load_partial(out_path: Path) -> list[dict]:
    """Rows journaled by a previous (killed) run of this cell."""
    p = partial_path(out_path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # truncated tail from a mid-write kill; drop it
    return rows


def append_partial(out_path: Path, row: dict) -> None:
    with open(partial_path(out_path), "a") as fh:
        fh.write(json.dumps(row) + "\n")


def finalize(out_path: Path, rows: list[dict]) -> None:
    """Write the final JSONL and clear the journal."""
    with open(out_path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    partial_path(out_path).unlink(missing_ok=True)
