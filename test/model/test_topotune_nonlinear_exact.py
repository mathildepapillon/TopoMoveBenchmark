"""Config tests for the ``topotune_nonlinear_exact`` model (cell and simplicial)."""

from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

import topobench  # noqa: F401  (registers the custom OmegaConf resolvers)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize(
    "model,dataset",
    [
        ("cell/topotune_nonlinear_exact", "graph/benzene"),
        ("simplicial/topotune_nonlinear_exact", "graph/benzene"),
        (
            "simplicial/topotune_nonlinear_exact",
            "simplicial/mantra_orientation",
        ),
    ],
)
def test_config_composes(model, dataset):
    """The config composes via hydra and keeps the exactness conditions."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(
            config_name="run.yaml",
            overrides=[f"model={model}", f"dataset={dataset}"],
        )

    assert cfg.model.model_name == "topotune_nonlinear_exact"
    assert cfg.model.model_domain == model.split("/", 1)[0]
    assert cfg.model.tune_gnn == "GCN"
    assert list(cfg.model.backbone.neighborhoods) == [
        "up_adjacency-1",
        "up_incidence-0",
        "down_incidence-2",
        "2-up_adjacency-0",
    ]
    assert cfg.model.backbone.layers == 2
    assert cfg.model.backbone.activation == "relu"
    # exactness needs a linear readout: identity on features + sum pooling
    assert cfg.model.readout.readout_name == "NoReadOut"
    assert cfg.model.readout.pooling_type == "sum"


def test_readout_is_registered():
    """The configured readout resolves in the readout registry."""
    import topobench.nn.readouts as readouts
    from topobench.nn.readouts.base import AbstractZeroCellReadOut

    assert issubclass(readouts.NoReadOut, AbstractZeroCellReadOut)


def test_cell_and_simplicial_variants_agree():
    """Both domain variants define the same model up to ``model_domain``."""
    variants = {
        domain: OmegaConf.to_container(
            OmegaConf.load(
                CONFIGS_DIR / "model" / domain / "topotune_nonlinear_exact.yaml"
            )
        )
        for domain in ("cell", "simplicial")
    }
    for domain, cfg in variants.items():
        assert cfg.pop("model_domain") == domain
    assert variants["cell"] == variants["simplicial"]
