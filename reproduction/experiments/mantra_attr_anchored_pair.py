"""Cell attributions for the anchored-k3 GCCN on the matched MANTRA pair.

Talk-figure companion to experiments/mantra_attribution_figure.py: the
anchored arm (seed 43, mask 1029 = up_adjacency-0, down_incidence-1,
2-down_incidence-2) is the exemplar whose neighborhoods carry vertex,
edge, AND triangle signal to the rank-0 head, so attribution should span
all three ranks. Same matched test pair (indices 73 / 243), same
parity-proven game semantics (zeros baseline), same 512 passes.

Post-freeze, presentation-only: writes results/mantra_attr_fig/
anchored_pair.json (results/ is not under the freeze manifest).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import os

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")

from toposhap.patches import apply_all  # noqa: E402

PATCHES = apply_all()

import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from mantra_attribution_figure import (  # noqa: E402
    explain_complex,
    single_batch,
)
from mantra_balanced_eval import (  # noqa: E402
    DS,
    R,
    find_ckpt,
    gccn_compose,
    load_ckpt,
)
from toposhap.neighborhoods import prune_backbone_  # noqa: E402

SEED = 43
MASK = 1029
PAIR = {"non_orientable_true_torus": 243, "orientable_role_klein": 73}
# NOTE role semantics follow mantra_attr_fig.json: test index 73 is the
# NON-orientable (Klein) complex, 243 the orientable torus.


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = drv.build_pipeline(gccn_compose(SEED, "anchpair"))
    row = json.loads(
        (R / "anchored3_exact"
         / f"anchored3_{DS}_seed{SEED}.jsonl").read_text())
    assert row["mask"] == MASK, row["mask"]
    model = copy.deepcopy(pipe.model)
    prune_backbone_(model.backbone.backbone, MASK)
    ckpt = find_ckpt(
        R / "anchored3_exact" / "runs" / f"{DS}_seed{SEED}"
        / "continue" / "checkpoints",
        row.get("selected_epoch_in_continuation"))
    load_ckpt(model, ckpt)
    model.to(device).eval()

    loader = pipe.datamodule.test_dataloader()
    out = {"seed": SEED, "mask": MASK, "ckpt": ckpt.name,
           "note": "margin = logit[majority=non-orientable] - logit[minority]"}
    # majority class index measured as 0 in the frozen attribution run
    pos_cls, neg_cls = 0, 1
    for name, idx in (("torus", 243), ("klein", 73)):
        batch = single_batch(loader, idx).to(device)
        batch["model_state"] = "test"
        print(f"explaining anchored / {name} (idx {idx}) ...", flush=True)
        out[name] = explain_complex(model, batch, pos_cls, neg_cls)
        out[name]["test_index"] = idx
    path = R / "mantra_attr_fig" / "anchored_pair.json"
    path.write_text(json.dumps(out))
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
