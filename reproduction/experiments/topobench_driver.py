"""Programmatic TopoBench driver: compose, build, fit, evaluate — no hydra.main.

Mirrors ``topobench/run.py::run`` (hydra instantiate dataset loader ->
PreProcessor -> TBDataloader -> model -> trainer -> fit) as importable
functions, so experiment drivers (run_autok.py, run_recipe.py) can train a
stem, play masked validation-accuracy games on it, prune, continue training,
and run TopoBench's normal best-checkpoint test path — all in one process.

Model config: ``cell/topotune_nonlinear_exact`` reproduces the #14 record
exactly (gate A: benzene mask 132 -> n_parameters_backbone 8704, total 10528).
The upstream ``cell/topotune`` differs only in readout (PropagateSignalDown,
total 16864) — same backbone, but not the recorded model.

Import order matters: callers must run ``toposhap.patches.apply_all()`` before
building any model; this module imports ``topobench.run`` (which registers
hydra resolvers and sets PROJECT_ROOT) but never builds anything at import.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

import topobench.run  # noqa: F401  (registers resolvers, sets PROJECT_ROOT)
from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import TBDataloader
from topobench.utils import instantiate_callbacks, instantiate_loggers

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO / "external" / "TopoBench" / "configs")

#: The model config that reproduces the #14 parameter record (gate A).
RECORD_MODEL = "cell/topotune_nonlinear_exact"

#: Hydra config groups per dataset: (dataset group, model domain). Datasets
#: not listed are graph-domain with cell-domain models (the lifted default).
#: MANTRA is native simplicial — no lifting; the loader builds the
#: neighborhood matrices itself from the model's vocabulary.
DATASET_DOMAINS: dict[str, tuple[str, str]] = {
    "mantra_orientation": ("simplicial", "simplicial"),
}


def dataset_group(dataset: str) -> str:
    """Hydra dataset config group for a dataset."""
    return DATASET_DOMAINS.get(dataset, ("graph", "cell"))[0]


def model_domain(dataset: str) -> str:
    """Model config domain matching the dataset (cell for lifted graphs)."""
    return DATASET_DOMAINS.get(dataset, ("graph", "cell"))[1]


def compose_config(
    dataset: str,
    seed: int,
    neighborhoods: list[str],
    output_dir: str,
    model: str = RECORD_MODEL,
    max_epochs: int | None = None,
    extra_overrides: list[str] | None = None,
    neighborhood_key: str | None = "model.backbone.neighborhoods",
) -> DictConfig:
    """Compose the TopoBench run config with the full neighborhood vocabulary.

    ``neighborhood_key`` is where the vocabulary lives in the model config:
    topotune uses ``model.backbone.neighborhoods`` (default); HOPSE uses
    ``model.preprocessing_params.neighborhoods``. Graph-domain models
    (graph/gcn, graph/gin) have no such key — pass ``None`` to skip it.
    """
    from hydra import compose, initialize_config_dir

    overrides = [
        f"dataset={dataset_group(dataset)}/{dataset}",
        f"model={model}",
        f"seed={seed}",
        "logger=csv",
        f"paths.output_dir={output_dir}",
    ]
    if neighborhood_key is not None:
        overrides.insert(
            2, neighborhood_key + "=[" + ",".join(neighborhoods) + "]")
    if max_epochs is not None:
        overrides.append(f"trainer.max_epochs={max_epochs}")
    overrides.extend(extra_overrides or [])
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="run.yaml", overrides=overrides)


def seed_everything(seed: int) -> None:
    """Seed exactly as topobench/run.py::run does."""
    L.seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass
class Pipeline:
    """Instantiated dataset + datamodule + model for one run config."""

    cfg: DictConfig
    datamodule: TBDataloader
    model: L.LightningModule
    preprocessing_seconds: float


def build_pipeline(cfg: DictConfig, data_only: bool = False) -> Pipeline:
    """Dataset loader -> PreProcessor -> TBDataloader -> model (run.py order)."""
    seed_everything(cfg.seed)

    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()
    transform_config = (
        hydra.utils.instantiate(cfg.transforms)
        if cfg.get("transforms", None) is not None
        else None
    )
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    dataset_train, dataset_val, dataset_test = (
        preprocessor.load_dataset_splits(cfg.dataset.split_params)
    )
    if cfg.dataset.parameters.task_level not in ["node", "graph"]:
        raise ValueError("Invalid task_level")
    datamodule = TBDataloader(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        dataset_test=dataset_test,
        **cfg.dataset.get("dataloader_params", {}),
    )
    if data_only:
        return Pipeline(cfg, datamodule, None, preprocessor.preprocessing_time)

    model: L.LightningModule = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )
    return Pipeline(cfg, datamodule, model, preprocessor.preprocessing_time)


def make_trainer(
    cfg: DictConfig,
    callbacks: list | None = None,
    max_epochs: int | None = None,
    logger=False,
) -> L.Trainer:
    """Instantiate the trainer from the run config (no wandb, no progress bar)."""
    kwargs = dict(
        callbacks=callbacks if callbacks is not None else [],
        logger=logger,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    if max_epochs is not None:
        kwargs["max_epochs"] = max_epochs
    if not torch.cuda.is_available():
        # keep the dataset's standard (GPU) settings on cluster nodes; fall
        # back to CPU only where no GPU exists (login-node smoke tests)
        kwargs["accelerator"] = "cpu"
        kwargs["devices"] = 1
    return hydra.utils.instantiate(cfg.trainer, **kwargs)


def fit_stem(cfg: DictConfig, pipeline: Pipeline, epochs: int) -> float:
    """Train the all-neighborhood stem: fixed epochs, no early stop, no ckpt."""
    trainer = make_trainer(cfg, callbacks=[], max_epochs=epochs)
    t0 = time.perf_counter()
    trainer.fit(model=pipeline.model, datamodule=pipeline.datamodule)
    return time.perf_counter() - t0


def fit_continuation(
    cfg: DictConfig, pipeline: Pipeline
) -> tuple[L.Trainer, ModelCheckpoint, float, int]:
    """Continue training with the dataset's standard TopoBench settings.

    Default trainer max_epochs, checkpointing on the monitored val metric,
    standard callbacks (incl. early stopping, patience 25). The csv logger
    satisfies the LearningRateMonitor callback (#14 also ran logger=csv).
    Returns the trainer, the checkpoint callback, wall seconds, and epochs
    trained.
    """
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    loggers = instantiate_loggers(cfg.get("logger"))
    trainer = make_trainer(cfg, callbacks=callbacks, logger=loggers)
    t0 = time.perf_counter()
    trainer.fit(model=pipeline.model, datamodule=pipeline.datamodule)
    seconds = time.perf_counter() - t0
    ckpt_cb = next(
        cb for cb in callbacks if isinstance(cb, ModelCheckpoint)
    )
    return trainer, ckpt_cb, seconds, trainer.current_epoch


@torch.no_grad()
def accuracy_over_loader(model, loader, state: str, device) -> tuple[float, int]:
    """Argmax accuracy over a dataloader (node masks applied for node tasks).

    Uses TBModel's own forward + process_outputs so node-level splits are
    restricted to the right mask; accuracy scoring per the game convention
    (experiment #8).
    """
    was_training = model.training
    model.eval()
    model.state_str = state
    correct, total = 0, 0
    for batch in loader:
        batch = batch.to(device)
        batch["model_state"] = state
        model_out = model.forward(batch)
        model_out = model.process_outputs(model_out=model_out, batch=batch)
        preds = model_out["logits"].argmax(dim=-1)
        labels = model_out["labels"]
        correct += int((preds == labels).sum())
        total += int(labels.numel())
    if was_training:
        model.train()
    return correct / total, total


def evaluate_split(model, datamodule, split: str, device) -> tuple[float, int]:
    """Validation/test accuracy of the model as-is (full active coalition)."""
    if split == "val":
        return accuracy_over_loader(
            model, datamodule.val_dataloader(), "Validation", device
        )
    if split == "test":
        return accuracy_over_loader(
            model, datamodule.test_dataloader(), "Test", device
        )
    raise ValueError(f"unknown split {split!r}")


def test_best_checkpoint(
    cfg: DictConfig,
    pipeline: Pipeline,
    ckpt_cb: ModelCheckpoint,
    device,
) -> dict:
    """TopoBench's normal test path: load best-val ckpt, validate + test.

    Mirrors topobench/run.py::rerun_best_model_checkpoint (fresh trainer,
    ``on_validation_epoch_start`` no-op'd for the standalone validate pass).
    """
    model = pipeline.model
    best_path = ckpt_cb.best_model_path
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)

    trainer = make_trainer(cfg, callbacks=[])
    model.on_validation_epoch_start = lambda: None
    val_results = trainer.validate(
        model=model, dataloaders=pipeline.datamodule.val_dataloader()
    )
    test_results = trainer.test(
        model=model, dataloaders=pipeline.datamodule.test_dataloader()
    )
    model.to(device)
    return {
        "best_model_path": best_path,
        "selected_epoch": int(ckpt["epoch"]),
        "val_metrics": dict(val_results[0]) if val_results else {},
        "test_metrics": dict(test_results[0]) if test_results else {},
    }


def count_parameters(module: torch.nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
