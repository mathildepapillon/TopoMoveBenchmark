"""Tests for ``GraphXAIDataset`` and ``GraphXAIDatasetLoader`` (no network).

The real datasets are built from ``.npz`` archives (Benzene,
AlkaneCarbonyl, FluorideCarbonyl) or from the Mutagenicity TUDataset,
both of which would be downloaded on first use. These tests never touch
the network: the ``.npz`` path is exercised with tiny synthetic archives
written in the GraphXAI format, and the Mutagenicity path with a
monkeypatched ``TUDataset`` returning hand-built molecules.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

import topobench.data.datasets.graphxai_dataset as gxai_module
from topobench.data.datasets.graphxai_dataset import (
    NPZ_DATASETS,
    GraphXAIDataset,
)
from topobench.data.loaders.graph.graphxai_datasets import (
    GraphXAIDatasetLoader,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _toy_graph(n_nodes, edges, candidates):
    """Build one graph dict plus its attribution entry, GraphXAI-style.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    edges : list[tuple[int, int]]
        Undirected edges; both directions are emitted.
    candidates : list[list[int]]
        One node-index list per candidate ground-truth explanation.

    Returns
    -------
    tuple[dict, dict]
        The graph dict (``nodes``/``edges``/``receivers``/``senders``)
        and the attribution dict (``nodes``: ``[n_nodes, K]`` floats).
    """
    senders, receivers = [], []
    for a, b in edges:
        senders += [a, b]
        receivers += [b, a]
    graph = {
        "nodes": np.eye(14, dtype=np.float32)[
            np.arange(n_nodes) % 14
        ],
        "edges": np.ones((len(senders), 4), dtype=np.float32),
        "senders": np.asarray(senders, dtype=np.int64),
        "receivers": np.asarray(receivers, dtype=np.int64),
    }
    node_imp = np.zeros((n_nodes, len(candidates)), dtype=np.float32)
    for j, nodes in enumerate(candidates):
        node_imp[nodes, j] = 1.0
    return graph, {"nodes": node_imp}


def _write_archive(path, graphs, attributions, labels):
    """Write a synthetic GraphXAI ``.npz`` archive.

    Parameters
    ----------
    path : pathlib.Path
        Destination file.
    graphs : list[dict]
        Graph dicts from :func:`_toy_graph`.
    attributions : list[dict]
        Attribution dicts from :func:`_toy_graph`.
    labels : list[int]
        Graph labels.
    """
    x_outer = np.empty(1, dtype=object)
    inner = np.empty(len(graphs), dtype=object)
    for i, g in enumerate(graphs):
        inner[i] = g
    x_outer[0] = inner
    attr = np.empty((len(graphs), 1), dtype=object)
    for i, a in enumerate(attributions):
        attr[i, 0] = a
    y = np.asarray([[int(label)] for label in labels], dtype=object)
    np.savez(path, X=x_outer, attr=attr, y=y)


@pytest.fixture
def benzene_archive(tmp_path):
    """Two triangles and a path graph with known candidate masks."""
    raw_dir = tmp_path / "staged"
    raw_dir.mkdir()
    g0, a0 = _toy_graph(
        4,
        [(0, 1), (1, 2), (0, 2), (2, 3)],
        candidates=[[0, 1, 2], [2, 3]],
    )
    g1, a1 = _toy_graph(3, [(0, 1), (1, 2)], candidates=[[1]])
    _write_archive(
        raw_dir / NPZ_DATASETS["Benzene"], [g0, g1], [a0, a1], [1, 0]
    )
    return raw_dir


def _one_hot(index):
    vec = torch.zeros(14)
    vec[index] = 1.0
    return vec


def _molecule(atoms, bonds, label):
    """Build a Mutagenicity-style PyG graph from atom indices and bonds.

    Parameters
    ----------
    atoms : list[int]
        One-hot index per atom (0=C, 1=O, 4=N, ...).
    bonds : list[tuple[int, int]]
        Undirected bonds; both directions are emitted.
    label : int
        Graph label (1 = mutagenic).

    Returns
    -------
    torch_geometric.data.Data
        The molecule graph.
    """
    senders, receivers = [], []
    for a, b in bonds:
        senders += [a, b]
        receivers += [b, a]
    return Data(
        x=torch.stack([_one_hot(i) for i in atoms]),
        edge_index=torch.tensor([senders, receivers], dtype=torch.long),
        y=torch.tensor([label]),
    )


# ---------------------------------------------------------------------------
# .npz datasets (Benzene / AlkaneCarbonyl / FluorideCarbonyl)
# ---------------------------------------------------------------------------


def test_npz_dataset_builds_masks_and_candidates(tmp_path, benzene_archive):
    dataset = GraphXAIDataset(
        root=str(tmp_path / "root"),
        name="Benzene",
        raw_data_dir=str(benzene_archive),
    )
    assert len(dataset) == 2

    first = dataset[0]
    assert first.x.shape == (4, 14)
    assert int(first.y) == 1
    # union of the two candidates {0,1,2} and {2,3}
    assert first.expl_node_mask.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert int(first.n_gt_candidates) == 2
    assert bool(first.gt_positive)

    second = dataset[1]
    assert second.expl_node_mask.tolist() == [0.0, 1.0, 0.0]
    # no edge has both endpoints inside {1}
    assert second.expl_edge_mask.sum() == 0

    candidates = dataset.gt_candidates
    assert len(candidates) == 2
    assert candidates[0]["node"].shape == (2, 4)
    assert candidates[0]["edge"].shape == (2, 8)
    # candidate 0 = triangle {0,1,2}: its 6 directed edges are inside
    assert int(candidates[0]["edge"][0].sum()) == 6


def test_npz_dataset_edge_mask_needs_both_endpoints(
    tmp_path, benzene_archive
):
    dataset = GraphXAIDataset(
        root=str(tmp_path / "root"),
        name="Benzene",
        raw_data_dir=str(benzene_archive),
    )
    first = dataset[0]
    inside = first.expl_node_mask.bool()
    expected = (
        inside[first.edge_index[0]] & inside[first.edge_index[1]]
    ).float()
    assert torch.equal(first.expl_edge_mask, expected)


def test_unknown_dataset_name_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown GraphXAI dataset"):
        GraphXAIDataset(root=str(tmp_path), name="TotallyMadeUp")


def test_alkane_carbonyl_downsampling_is_deterministic(tmp_path):
    raw_dir = tmp_path / "staged"
    raw_dir.mkdir()
    graphs, attributions, labels = [], [], []
    for label in [0, 0, 0, 0, 0, 1]:
        g, a = _toy_graph(3, [(0, 1), (1, 2)], candidates=[[0]])
        graphs.append(g)
        attributions.append(a)
        labels.append(label)
    _write_archive(
        raw_dir / NPZ_DATASETS["AlkaneCarbonyl"], graphs, attributions, labels
    )

    datasets = [
        GraphXAIDataset(
            root=str(tmp_path / f"root{i}"),
            name="AlkaneCarbonyl",
            raw_data_dir=str(raw_dir),
        )
        for i in range(2)
    ]
    for dataset in datasets:
        # 1 positive + 2:1 subsampled negatives
        assert len(dataset) == 3
        assert int(dataset.y.sum()) == 1
        # orig_index is renumbered to index the balanced dataset
        assert [int(dataset[i].orig_index) for i in range(3)] == [0, 1, 2]
    assert torch.equal(datasets[0].y, datasets[1].y)


# ---------------------------------------------------------------------------
# Mutagenicity (substructure-matching ground truth, TUDataset mocked)
# ---------------------------------------------------------------------------


def test_mutagenicity_matches_and_filters(tmp_path, monkeypatch):
    # C-NO2: N (one-hot 4) of degree 3 bonded to two terminal O (one-hot 1)
    nitro = _molecule(
        atoms=[0, 4, 1, 1], bonds=[(0, 1), (1, 2), (1, 3)], label=1
    )
    plain = _molecule(atoms=[0, 0], bonds=[(0, 1)], label=0)
    mismatched = _molecule(
        atoms=[0, 4, 1, 1], bonds=[(0, 1), (1, 2), (1, 3)], label=0
    )
    monkeypatch.setattr(
        gxai_module,
        "TUDataset",
        lambda **kwargs: [nitro, plain, mismatched],
    )

    dataset = GraphXAIDataset(root=str(tmp_path / "root"), name="Mutagenicity")

    # the label/substructure-mismatched molecule is dropped
    assert len(dataset) == 2
    assert dataset.y.tolist() == [1, 0]

    nitro_graph = dataset[0]
    assert nitro_graph.expl_node_mask.tolist() == [0.0, 1.0, 1.0, 1.0]
    assert bool(nitro_graph.gt_positive)
    plain_graph = dataset[1]
    assert plain_graph.expl_node_mask.sum() == 0
    assert not bool(plain_graph.gt_positive)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _loader_cfg(tmp_path, raw_dir, data_name="Benzene"):
    return OmegaConf.create(
        {
            "data_dir": str(tmp_path / "root"),
            "data_name": data_name,
            "data_type": "GraphXAI",
            "raw_data_dir": str(raw_dir),
        }
    )


def test_loader_is_registered():
    """The discovery ``__init__`` re-exports the loader at both levels."""
    import topobench.data.loaders as loaders
    import topobench.data.loaders.graph as graph_loaders

    for module in (loaders, graph_loaders):
        registered = module.GraphXAIDatasetLoader
        assert registered.__name__ == "GraphXAIDatasetLoader"


def test_loader_load_dataset(tmp_path, benzene_archive):
    loader = GraphXAIDatasetLoader(
        _loader_cfg(tmp_path, benzene_archive)
    )
    assert "GraphXAIDatasetLoader" in repr(loader)
    dataset, data_dir = loader.load()
    assert len(dataset) == 2
    assert str(tmp_path) in str(data_dir)


def test_dataset_configs_compose(tmp_path):
    import hydra

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    for config_name in [
        "benzene",
        "alkane_carbonyl",
        "fluoride_carbonyl",
        "mutagenicity_gxai",
    ]:
        with hydra.initialize(
            version_base="1.3", config_path="../../../../configs"
        ):
            cfg = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    f"dataset=graph/{config_name}",
                    "model=graph/gat",
                ],
            )
        loader_cfg = cfg.dataset.loader
        assert loader_cfg._target_ == (
            "topobench.data.loaders.GraphXAIDatasetLoader"
        )
        assert loader_cfg.parameters.raw_data_dir is None
        assert cfg.dataset.parameters.num_classes == 2
        assert cfg.dataset.parameters.task_level == "graph"
