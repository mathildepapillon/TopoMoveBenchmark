#!/usr/bin/env python
"""Final campaign assembly: freeze whatever cells exist, write the marker.

    python experiments/assemble_campaign.py

Runs assemble_mantra.py and assemble_hopse_k3.py with --allow-missing (the
campaign must record partial state, not die on it), then writes
results/CAMPAIGN_DONE with a completed/missing inventory. Exits 0 even on
partial — the marker's ``status:`` line distinguishes COMPLETE from PARTIAL.

Any future session: read results/CAMPAIGN_DONE (state) and
results/CAMPAIGN_STATUS.md (job DAG + rescue commands) to know everything.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from campaign_missing import k3_cells, mantra_cells, missing  # noqa: E402

EXPECTED_PARQUETS = {  # name -> expected rows
    "anchored3_exact_mantra.parquet": 3,
    "ladder_b2_mantra.parquet": 3,
    "ladder_b2_hopse_mantra.parquet": 3,
    "hopse_sweep_mantra.parquet": 36,
    "gccn_menu_mantra.parquet": 18,
    "hopse_k3_rung.parquet": 21,
}

#: MANTRA parquets must carry non-null balanced-accuracy columns (backfilled
#: by mantra_balanced_eval.py; plain accuracy cannot adjudicate the
#: cross-family contrast on this class-imbalanced task).
BALANCED_PARQUETS = [
    "anchored3_exact_mantra.parquet",
    "ladder_b2_mantra.parquet",
    "ladder_b2_hopse_mantra.parquet",
    "hopse_sweep_mantra.parquet",
    "gccn_menu_mantra.parquet",
]


def main() -> None:
    py = sys.executable
    for script in ["assemble_mantra.py", "assemble_hopse_k3.py"]:
        rc = subprocess.run(
            [py, str(REPO / "experiments" / script), "--allow-missing"],
            cwd=REPO).returncode
        print(f"{script} rc={rc}")

    import pandas as pd
    lines = []
    complete = True
    lines.append("parquets:")
    for name, expected in EXPECTED_PARQUETS.items():
        path = REPO / "data" / "frozen" / name
        if not path.exists():
            complete = False
            lines.append(f"  data/frozen/{name} MISSING (expected "
                         f"{expected} rows)")
            continue
        df = pd.read_parquet(path)
        n = len(df)
        ok = n == expected
        note = ""
        if name in BALANCED_PARQUETS:
            missing_bal = (
                "val_balanced_accuracy" not in df
                or "test_balanced_accuracy" not in df
                or int(df["val_balanced_accuracy"].isna().sum()
                       + df["test_balanced_accuracy"].isna().sum()) > 0)
            if missing_bal:
                ok = False
                note = " BALANCED-FIELDS-MISSING"
        complete &= ok
        lines.append(f"  data/frozen/{name} rows={n}/{expected}"
                     f"{'' if ok else ' INCOMPLETE'}{note}")

    cb = REPO / "results" / "mantra_prepare_logs" / "class_balance.json"
    if cb.exists():
        lines.append(f"class balance record: {cb.relative_to(REPO)} present")
    else:
        complete = False
        lines.append("class balance record: MISSING "
                     "(results/mantra_prepare_logs/class_balance.json)")
    rc = subprocess.run(
        [py, str(REPO / "experiments" / "mantra_balanced_eval.py"),
         "--check"], cwd=REPO).returncode
    lines.append(f"balanced-eval backfill check: "
                 f"{'complete' if rc == 0 else 'PENDING CELLS'}")
    complete &= rc == 0

    m_mantra = missing(mantra_cells())
    m_k3 = missing(k3_cells())
    complete &= not m_mantra and not m_k3
    lines.append("missing cells:")
    if not m_mantra and not m_k3:
        lines.append("  none")
    for label, cells, miss in [("mantra_arrays", mantra_cells(), m_mantra),
                               ("hopse_k3_array", k3_cells(), m_k3)]:
        for i, path, rows in cells:
            if i in miss:
                lines.append(f"  {label}[{i}] -> {path.relative_to(REPO)}")

    marker = REPO / "results" / "CAMPAIGN_DONE"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    marker.write_text(
        f"status: {'COMPLETE' if complete else 'PARTIAL'}\n"
        f"generated: {stamp}\n" + "\n".join(lines) + "\n")
    print(marker.read_text())


if __name__ == "__main__":
    main()
