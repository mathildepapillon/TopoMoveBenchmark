"""Native-readout CWN on MANTRA + participation-game pair attributions.

The menu expresses CWN's neighborhood set under the shared NoReadOut
rank-0 head, which orphans its edge computation. This trains the same
backbone with CWN's neighborhoods and the NATIVE PropagateSignalDown
readout (higher-rank signal reaches the output), then explains the
matched pair under the participation game. Post-freeze, talk-only.
Writes results/mantra_attr_fig/cwn_native_pair.json.
"""

from __future__ import annotations

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
from mantra_attribution_figure import single_batch  # noqa: E402
from mantra_balanced_eval import DS, R  # noqa: E402
from toposhap.cells import CellPlayer  # noqa: E402
from toposhap.cells.participation import (  # noqa: E402
    CellParticipationGame,
    pool_faithful_margin,
)
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.shapley.sampled import sampled_shapley  # noqa: E402
from toposhap.vocabulary import mask_to_coalition  # noqa: E402

import sys as _sys

SEED = int(_sys.argv[1]) if len(_sys.argv) > 1 else 43
MASK, PASSES = 26, 512


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = R / "cwn_native" / f"{DS}_seed{SEED}"
    cfg = drv.compose_config(
        dataset=DS, seed=SEED, neighborhoods=list(mask_to_coalition(MASK)),
        output_dir=str(out_dir),
        model=f"{drv.model_domain(DS)}/{drv.RECORD_MODEL.split('/', 1)[1]}",
        extra_overrides=["model.readout.readout_name=PropagateSignalDown"],
    )
    pipe = drv.build_pipeline(cfg)
    print("training native-readout CWN ...", flush=True)
    trainer, ckpt_cb, seconds, epochs = drv.fit_continuation(cfg, pipe)
    print(f"trained {epochs} epochs in {seconds:.0f}s; "
          f"best: {ckpt_cb.best_model_path}", flush=True)
    state = torch.load(ckpt_cb.best_model_path, map_location="cpu",
                       weights_only=False)["state_dict"]
    pipe.model.load_state_dict(state)
    model = pipe.model.to(device).eval()

    # test AUROC for the panel label
    from sklearn.metrics import roc_auc_score
    loader = pipe.datamodule.test_dataloader()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch["model_state"] = "test"
            o = model.forward(batch)
            o = model.process_outputs(model_out=o, batch=batch)
            ps += torch.softmax(o["logits"].float(), -1)[:, 0].cpu().tolist()
            ys += o["labels"].cpu().tolist()
    auroc = float(roc_auc_score([int(y == 0) for y in ys], ps))
    print(f"native CWN test AUROC (non-orientable) = {auroc:.4f}", flush=True)

    model_fn = pool_faithful_margin(model, 0, 1)

    out = {"seed": SEED, "mask": MASK, "readout": "PropagateSignalDown",
           "epochs": epochs, "test_auroc": auroc, "passes": PASSES,
           "ckpt": ckpt_cb.best_model_path,
           "note": "margin = logit[non-orientable] - logit[orientable]"}
    for name, idx in (("torus", 243), ("klein", 73)):
        batch = single_batch(loader, idx).to(device)
        batch["model_state"] = "test"
        players = [CellPlayer(rank=r, index=i) for r in (0, 1, 2)
                   for i in range(int(getattr(batch, f"x_{r}").shape[0]))]
        print(f"explaining {name} ...", flush=True)
        raw = CellParticipationGame(model_fn, batch, players,
                                    encoder=model.feature_encoder)
        game = CachedGame(n_players=len(players), evaluate=raw)
        att = sampled_shapley(game, len(players), passes=PASSES, seed=0)
        full = (1 << len(players)) - 1
        out[name] = {"phi": [float(v) for v in att.phi],
                     "player_ranks": [p.rank for p in players],
                     "v_full": raw(full), "v_empty": raw(0),
                     "evaluations": game.calls, "test_index": idx}
    path = R / "mantra_attr_fig" / (
        "cwn_native_pair.json" if SEED == 43
        else f"cwn_native_pair_s{SEED}.json")
    path.write_text(json.dumps(out))
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
