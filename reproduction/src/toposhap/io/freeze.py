"""Results freeze: content-hash manifest over data/frozen/.

The freeze (target ~Aug 22) is a hard line: after it, every table and figure
in the paper regenerates from data/frozen/ alone, and the manifest pins what
"frozen" means. ``scripts/freeze_results.py`` drives this; the paper's
results_macros.tex records the manifest hash it was generated from.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from toposhap.io.results import FROZEN_DIR

MANIFEST = FROZEN_DIR / "MANIFEST.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(note: str = "") -> dict:
    """Hash every file under data/frozen/ into MANIFEST.json."""
    files = sorted(
        p
        for p in FROZEN_DIR.rglob("*")
        if p.is_file() and p.name not in {"MANIFEST.json", ".gitkeep"}
    )
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "files": {
            str(p.relative_to(FROZEN_DIR)): {
                "sha256": _sha256(p),
                "bytes": p.stat().st_size,
            }
            for p in files
        },
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()
    ).hexdigest()
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_manifest() -> list[str]:
    """Return a list of drift descriptions (empty = frozen state intact)."""
    if not MANIFEST.exists():
        return ["no MANIFEST.json — results not frozen yet"]
    manifest = json.loads(MANIFEST.read_text())
    problems = []
    for rel, meta in manifest["files"].items():
        p = FROZEN_DIR / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif _sha256(p) != meta["sha256"]:
            problems.append(f"content drift: {rel}")
    known = set(manifest["files"])
    for p in FROZEN_DIR.rglob("*"):
        if p.is_file() and p.name not in {"MANIFEST.json", ".gitkeep"}:
            rel = str(p.relative_to(FROZEN_DIR))
            if rel not in known:
                problems.append(f"unfrozen extra file: {rel}")
    return problems
