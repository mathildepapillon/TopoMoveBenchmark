#!/usr/bin/env python
"""Place and verify the old-cluster result files (see data/INGEST.md).

Modes:
  --staging DIR   copy/rename files found under DIR into data/frozen/
                  according to the inventory map (searches recursively by
                  basename, so a raw rsync of the old results directories
                  works as input)
  --verify        run every schema probe the analysis code depends on;
                  exit non-zero listing what is missing or malformed

The inventory here mirrors data/INGEST.md; keep them in sync.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / "data" / "frozen"

# basename (as produced on the old cluster) -> frozen destination
FILE_MAP = {
    "analysis12.json": "analysis12.json",
    "cost_wallclock12.json": "cost_wallclock12.json",
    "menu12.json": "menu12.json",
    "cheap_sim12.json": "cheap_sim12.json",
    "menu_extraction12.json": "menu_extraction12.json",
    "smokegate12.json": "smokegate12.json",
    "pooled.json": "pooled.json",
    "cold_compare11.json": "cold_compare11.json",
}

# directory-name fragments -> frozen destination dirs (copied wholesale);
# EIDs matched on their unique prefixes to survive transcription slips
DIR_MAP = {
    "[record-id]": "marker_arm14",  # #14 artifact root (partial: was in flight)
    "[record-id]": "hopse13",  # #13 artifact root
    "exp_01kzs88s": "reference/exp1_worktree",  # validated topobench.explain
    "exp_01kzzch6": "reference/synthesis_report",  # #10 synthesis + claim ledger
    "exp1_worktree": "reference/exp1_worktree",  # pod-bundle naming
    "exp10_worktree": "reference/synthesis_report",  # pod-bundle naming
    "figure-sources": "reference/figure-sources",
    "fig-a09699c2f95b": "reference/figure-sources/fig-a09699c2f95b",
    "lab-uploads": "reference/uploads",
}


def stage(staging: Path) -> None:
    placed = []
    for path in staging.rglob("*"):
        if path.is_file() and path.name in FILE_MAP:
            dest = FROZEN / FILE_MAP[path.name]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            placed.append(f"{path.name} -> {dest.relative_to(REPO)}")
    for path in staging.rglob("*"):
        if path.is_dir():
            for frag, destname in DIR_MAP.items():
                if frag in path.name:
                    dest = FROZEN / destname
                    shutil.copytree(path, dest, dirs_exist_ok=True)
                    placed.append(
                        f"{path.name}/ -> {dest.relative_to(REPO)}/"
                    )
                    break
    if not placed:
        print("nothing recognised under", staging)
        sys.exit(1)
    print("\n".join(placed))


def verify() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from toposhap.io import results as R

    problems: list[str] = []

    probes = [
        ("analysis12.json", R.load_analysis12),
        ("cost_wallclock12.json", R.load_cost_wallclock12),
        ("menu12.json", R.load_menu12),
        ("cheap_sim12.json", R.load_cheap_sim12),
        ("pooled.json", R.load_pooled9),
        ("cold_compare11.json", R.load_cold_compare11),
        ("marker_arm14/", R.load_marker_arm14),
    ]
    for name, loader in probes:
        try:
            loader()
            print(f"  ok       {name}")
        except FileNotFoundError:
            problems.append(f"missing: {name}")
            print(f"  MISSING  {name}")
        except (KeyError, json.JSONDecodeError, Exception) as e:  # noqa: BLE001
            problems.append(f"malformed: {name} ({e})")
            print(f"  BAD      {name}: {e}")

    # deep probes on the handoff's exact schema paths
    try:
        pooled = R.load_pooled9()
        ds = next(iter(pooled["retraining_pooled"]))
        table = pooled["retraining_pooled"][ds]["values_mean"]
        if len(table) != 64:
            problems.append(
                f"pooled.json[{ds}].values_mean has {len(table)} masks, "
                "expected 64 (6-space landscape)"
            )
    except FileNotFoundError:
        pass

    try:
        sim = R.load_cheap_sim12()
        ds = next(iter(sim["per_dataset"]))
        seed = next(iter(sim["per_dataset"][ds]["per_seed"]))
        node = sim["per_dataset"][ds]["per_seed"][seed]
        if "permutation" not in node:
            problems.append(
                "cheap_sim12.json missing per_seed[seed].permutation[budget]"
            )
    except (FileNotFoundError, KeyError, StopIteration):
        pass

    if problems:
        print(f"\n{len(problems)} problem(s):")
        print("\n".join(f"  - {p}" for p in problems))
        return 1
    print("\nall probes passed — ready for scripts/freeze_results.py")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staging", type=Path)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.staging:
        stage(args.staging)
    if args.verify:
        sys.exit(verify())
    if not args.staging and not args.verify:
        ap.print_help()


if __name__ == "__main__":
    main()
