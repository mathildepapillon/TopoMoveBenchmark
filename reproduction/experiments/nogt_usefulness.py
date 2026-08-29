"""TopoSHAP without explanation ground truth: TopoBench datasets.

The GraphXAI benchmarks come with annotated motifs; most topological
benchmarks do not. This experiment shows what TopoSHAP delivers there,
with ground-truth-free evaluation only:

  faithfulness  deletion-AUC under the declared removal operator
                (participation), against occlusion / gradient x input /
                random on the SAME graphs — plus the insertion curve
                v(S*_b) that falls out of the prune schedule for free
  diagnostics   per-rank Shapley credit profile (does the model use its
                rings?), higher-rank null-player share

Subjects: structural-lift full-vocabulary GCCNs
(train_structural_nogt.py). Explanations target the PREDICTED class
(no labels consulted). Sample: pre-declared rng(0) draw of N test
graphs, identical across seeds.

Usage: python nogt_usefulness.py DATASET SEED [N=50]
Writes results/nogt/{ds}_s{seed}.json.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

import topobench_driver as drv  # noqa: E402
from mantra_attribution_figure import single_batch  # noqa: E402
from mantra_balanced_eval import R, find_ckpt  # noqa: E402
from toposhap.baselines import (  # noqa: E402
    gradient_x_input,
    occlusion_scores,
    random_scores,
)
from toposhap.cells import CellPlayer  # noqa: E402
from toposhap.cells.participation import (  # noqa: E402
    CellParticipationGame,
    pool_faithful_margin,
)
from toposhap.cells.select import greedy_prune_schedule  # noqa: E402
from toposhap.metrics.deletion import deletion_curve  # noqa: E402
from toposhap.shapley.games import CachedGame  # noqa: E402
from toposhap.shapley.sampled import sampled_shapley  # noqa: E402
from toposhap.vocabulary import NEIGHBORHOODS  # noqa: E402

PASSES = 128


def main() -> None:
    ds = sys.argv[1]
    seed = int(sys.argv[2])
    n_sample = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    base = R / "nogt_subjects"
    rows = [json.loads(x) for x in
            (base / f"nogt_{ds}.jsonl").read_text().splitlines() if x]
    row = next(r for r in rows if r["seed"] == seed)
    cfg = drv.compose_config(
        dataset=ds, seed=seed, neighborhoods=list(NEIGHBORHOODS),
        output_dir=str(R / "nogt" / f"cfg_{ds}_s{seed}"),
        model=(f"{drv.model_domain(ds)}/"
               f"{drv.RECORD_MODEL.split('/', 1)[1]}"),
        extra_overrides=[
            "transforms.graph2cell_lifting.feature_lifting="
            "StructuralOnes",
            "transforms.graph2cell_lifting.preserve_edge_attr=False",
        ],
    )
    pipe = drv.build_pipeline(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_ckpt(base / "runs" / f"{ds}_s{seed}" / "checkpoints",
                     row.get("selected_epoch"))
    state = torch.load(str(ckpt), map_location="cpu",
                       weights_only=False)["state_dict"]
    model = pipe.model
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    loader = pipe.datamodule.test_dataloader()
    correct = total = 0
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            b["model_state"] = "test"
            o = model.forward(b)
            o = model.process_outputs(model_out=o, batch=b)
            correct += int((o["logits"].argmax(-1) == o["labels"]).sum())
            total += int(o["labels"].numel())
    acc = correct / total
    print(f"{ds} s{seed}: test acc {acc:.4f} "
          f"(row: {row['test_accuracy']:.4f})", flush=True)
    if abs(acc - row["test_accuracy"]) > 0.02:
        raise SystemExit("checkpoint does not reproduce trained accuracy")

    test = pipe.datamodule.dataset_test
    rng = np.random.default_rng(0)  # pre-declared, identical across seeds
    offset = int(os.environ.get("TOPOSHAP_NOGT_OFFSET", "0"))
    order = rng.permutation(len(test.data_lst))[offset:n_sample]
    print(f"explaining {len(order)} sampled test complexes "
          f"(offset {offset})", flush=True)

    out = {"dataset": ds, "seed": seed, "passes": PASSES,
           "test_accuracy": acc, "sample_rule": "rng(0) permutation",
           "records": []}
    out_dir = R / "nogt"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_off{offset}" if offset else ""
    out_file = out_dir / f"{ds}_s{seed}{suffix}.json"

    for done, pos in enumerate(int(i) for i in order):
        batch0 = single_batch(loader, pos).to(device)
        batch0["model_state"] = "test"
        players = [CellPlayer(rank=r, index=i) for r in (0, 1, 2)
                   if hasattr(batch0, f"x_{r}")
                   for i in range(int(getattr(batch0, f"x_{r}").shape[0]))]
        n = len(players)

        # explain the PREDICTED class: no labels consulted
        with torch.no_grad():
            b = copy.deepcopy(batch0)
            o = model.forward(b)
            o = model.process_outputs(model_out=o, batch=b)
            pred = int(o["logits"].argmax(-1).item())
        pos_cls, neg_cls = pred, 1 - pred

        gxi = gradient_x_input(model, batch0, players, pos_cls, neg_cls)
        b = copy.deepcopy(batch0)
        raw = CellParticipationGame(
            pool_faithful_margin(model, pos_cls, neg_cls), b, players,
            encoder=model.feature_encoder)
        game = CachedGame(n_players=n, evaluate=raw)

        att = sampled_shapley(game, n, passes=PASSES, seed=0)
        phi = np.asarray(att.phi)
        occ = occlusion_scores(game, n)
        rnd = random_scores(n, seed=pos)

        v_full = game((1 << n) - 1)
        curves = {}
        for name, scores in (("toposhap", phi), ("occlusion", occ),
                             ("gradxinput", gxi), ("random", rnd)):
            ranking = np.argsort(np.asarray(scores))[::-1]
            curves[name] = deletion_curve(game, ranking, n)

        # insertion (sufficiency) curve from the nested prune schedule
        budgets = sorted({max(1, int(round(f * n)))
                          for f in (0.05, 0.1, 0.2, 0.3, 0.5)})
        sched = greedy_prune_schedule(game, n, budgets)
        insertion = {str(b_): v for b_, (_, v) in sorted(sched.items())}

        ranks = np.array([p.rank for p in players])
        rank_share = {int(r): float(np.abs(phi[ranks == r]).sum() /
                                    max(np.abs(phi).sum(), 1e-12))
                      for r in sorted(set(ranks.tolist()))}

        out["records"].append({
            "test_pos": pos, "pred": pred, "n_players": n,
            "n_per_rank": {int(r): int((ranks == r).sum())
                           for r in sorted(set(ranks.tolist()))},
            "phi_rank_share": rank_share,
            "v_full": float(v_full), "v_empty": float(game(0)),
            "deletion_curves": curves, "insertion": insertion,
            "evaluations": game.calls,
        })
        if (done + 1) % 5 == 0:
            print(f"  {done + 1}/{len(order)}", flush=True)
            out_file.write_text(json.dumps(out))

    out_file.write_text(json.dumps(out))
    print(f"wrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
