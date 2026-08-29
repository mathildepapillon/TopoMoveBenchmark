#!/usr/bin/env python
"""Run the one-run selection recipe on a TopoBench dataset.

    python experiments/run_recipe.py --dataset MUTAG --seed 42 --k 6 \
        --selector anchored

Wires TopoBench's hydra/lightning stack into toposhap's trainer-agnostic
:func:`toposhap.neighborhoods.run_recipe` via :class:`TopoBenchHooks`, records
one CostRow per run into results/cost_table.parquet, and applies the framework
patches before anything touches a model.

The heavy lifting (dataset/model instantiation) reuses TopoBench's own hydra
configs with the topotune model forced to the full 11-neighborhood vocabulary
in canonical bit order. Trainer settings follow the dataset's TopoBench
defaults; the recipe only controls epochs (stem vs continuation) and the
neighborhood set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

os.environ.setdefault("TOPOSHAP_INTERRANK_ORIENTATION", "fixed")

from toposhap.costs import CostLedger, CostRow  # noqa: E402
from toposhap.neighborhoods import run_recipe as _run_recipe  # noqa: E402
from toposhap.patches import apply_all  # noqa: E402
from toposhap.vocabulary import NEIGHBORHOODS  # noqa: E402


class TopoBenchHooks:
    """RecipeHooks implementation on TopoBench's hydra + lightning stack.

    The two formerly-deliberate ``NotImplementedError`` seams (``_fit`` and
    ``_evaluate_split``) are wired through ``experiments/topobench_driver.py``,
    the programmatic mirror of ``topobench/run.py::run`` (loader ->
    PreProcessor -> TBDataloader -> model -> trainer -> fit) validated on the
    auto-k arm (gate A parameter match: benzene mask 132 -> backbone 8704,
    total 10528 with ``cell/topotune_nonlinear_exact``).
    """

    def __init__(self, dataset: str, total_epochs: int, stem_epochs: int,
                 accelerator: str = "auto", run_dir: str | None = None,
                 model: str = "cell/topotune_nonlinear_exact"):
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.stem_epochs = stem_epochs
        self.accelerator = accelerator
        self.model = model
        self.run_dir = Path(
            run_dir or REPO / "results" / "recipe_runs" / dataset
        )
        self.pipeline = None
        self.device = None
        self._wrapper = None
        self._ckpt_cb = None
        self._continue_cfg = None

    # -- internal ---------------------------------------------------------
    def _compose_cfg(self, epochs: int, seed: int, subdir: str):
        import topobench_driver as drv

        extra = []
        if self.accelerator != "auto":
            extra.append(f"trainer.accelerator={self.accelerator}")
        return drv.compose_config(
            dataset=self.dataset,
            seed=seed,
            neighborhoods=list(NEIGHBORHOODS),
            output_dir=str(self.run_dir / subdir),
            model=self.model,
            max_epochs=epochs,
            extra_overrides=extra,
        )

    def _fit(self, cfg, stem: bool):
        """Mirror topobench/run.py::run without the hydra main wrapper.

        Stem: fixed epochs, no early stop, no checkpointing. Continuation:
        the dataset's standard callbacks (val checkpointing, early stopping).
        """
        import torch

        import topobench_driver as drv

        if self.pipeline is None:
            self.pipeline = drv.build_pipeline(cfg)
        if stem:
            drv.fit_stem(cfg, self.pipeline, epochs=cfg.trainer.max_epochs)
        else:
            _, self._ckpt_cb, _, _ = drv.fit_continuation(cfg, self.pipeline)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.pipeline.model.to(self.device)
        return self.pipeline.model

    # -- RecipeHooks ------------------------------------------------------
    def train_stem(self, epochs: int, seed: int):
        cfg = self._compose_cfg(epochs, seed, "stem")
        return self._fit(cfg, stem=True)

    def val_accuracy_under_coalition(self, model, mask: int) -> float:
        from toposhap.neighborhoods import CoalitionMaskedBackbone

        if self._wrapper is None:
            # TBModel.backbone is a TuneWrapper; the real TopoTune module is
            # one level down.
            self._wrapper = CoalitionMaskedBackbone(model.backbone.backbone)
        with self._wrapper.coalition(mask):
            return self._evaluate_split(model, "val")

    def prune_and_continue(self, model, mask: int, seed: int):
        from toposhap.neighborhoods import prune_backbone_

        inner = model.backbone.backbone
        if self._wrapper is not None:  # unhook the game before pruning
            inner.aggregate_inter_nbhd = self._wrapper._original_aggregate
            self._wrapper = None
        prune_backbone_(inner, mask)
        remaining = self.total_epochs - self.stem_epochs
        self._continue_cfg = self._compose_cfg(remaining, seed, "continue")
        return self._fit(self._continue_cfg, stem=False)
        # warm start: pruning mid-training is free

    def test_accuracy(self, model) -> float:
        return self._evaluate_split(model, "test")

    def _evaluate_split(self, model, split: str) -> float:
        import topobench_driver as drv

        if split == "test" and self._ckpt_cb is not None:
            # TopoBench's normal test path: best-val checkpoint, framework
            # metrics.
            report = drv.test_best_checkpoint(
                self._continue_cfg, self.pipeline, self._ckpt_cb, self.device
            )
            return float(report["test_metrics"]["test/accuracy"])
        acc, _ = drv.evaluate_split(
            model, self.pipeline.datamodule, split, self.device
        )
        return acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--selector", default="anchored",
                    choices=["anchored", "plain"])
    ap.add_argument("--stem-epochs", type=int, default=5)
    ap.add_argument("--total-epochs", type=int, default=100)
    ap.add_argument("--passes", type=int, default=128)
    ap.add_argument("--out", default="results/cost_table.parquet")
    args = ap.parse_args()

    patches = apply_all()
    print("patches:", patches)

    hooks = TopoBenchHooks(
        dataset=args.dataset,
        total_epochs=args.total_epochs,
        stem_epochs=args.stem_epochs,
    )
    result = _run_recipe(
        hooks,
        seed=args.seed,
        k=args.k,
        stem_epochs=args.stem_epochs,
        passes=args.passes,
        selector=args.selector,
    )

    ledger = CostLedger(REPO / args.out)
    ledger.append(
        CostRow(
            approach="recipe",
            dataset=args.dataset,
            seed=args.seed,
            selector=args.selector,
            k=args.k,
            mask=result.picked_mask,
            test_accuracy=result.test_accuracy,
            wall_seconds=result.stem_seconds
            + result.game_seconds
            + result.continue_seconds,
            stem_seconds=result.stem_seconds,
            game_seconds=result.game_seconds,
            continue_seconds=result.continue_seconds,
            game_evaluations=result.game_evaluations,
            epochs=args.total_epochs,
        )
    )
    ledger.flush()
    print(
        f"picked mask {result.picked_mask} "
        f"({result.selector}, k={result.k}); "
        f"test acc {result.test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
