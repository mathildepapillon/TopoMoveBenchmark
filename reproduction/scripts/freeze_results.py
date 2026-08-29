#!/usr/bin/env python
"""Freeze (or audit) the results state under data/frozen/.

    python scripts/freeze_results.py --note "results freeze Aug 22"
    python scripts/freeze_results.py --check

After freezing, every paper table/figure regenerates from data/frozen/ alone;
scripts/make_tables.py stamps the manifest hash into results_macros.tex.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toposhap.io import verify_manifest, write_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--note", default="")
    ap.add_argument("--check", action="store_true",
                    help="verify the frozen state instead of re-freezing")
    args = ap.parse_args()

    if args.check:
        problems = verify_manifest()
        if problems:
            print("\n".join(f"DRIFT: {p}" for p in problems))
            sys.exit(1)
        print("frozen state intact")
        return

    manifest = write_manifest(note=args.note)
    print(
        f"froze {len(manifest['files'])} files; "
        f"manifest sha256 {manifest['manifest_sha256'][:16]}…"
    )


if __name__ == "__main__":
    main()
