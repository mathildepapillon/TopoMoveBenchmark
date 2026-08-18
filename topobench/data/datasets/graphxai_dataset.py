"""GraphXAI molecular datasets with ground-truth explanation masks, as PyG datasets.

Four graph-classification benchmarks from GraphXAI (Agarwal et al., 2023): Benzene,
AlkaneCarbonyl, FluorideCarbonyl (packaged ``.npz`` files following Sanchez-Lengeling
et al., 2020) and Mutagenicity (TUDataset plus GraphXAI's substructure-matching rules).

Each graph carries the ground-truth explanation as node and edge masks. GraphXAI provides
*several* admissible explanations per graph (e.g. one per benzene ring, or one per matched
mutagenic substructure), and its accuracy metric maximizes over them. We therefore store
both:

* ``expl_node_mask`` / ``expl_edge_mask``: the union over candidates, one float row per
  node/edge, collatable and carried through TopoBench's preprocessing and lifting;
* a sidecar file ``gt_candidates.pt`` with the per-graph list of candidate masks, so an
  explanation-accuracy metric can take the max over admissible candidates.
"""

import os.path as osp
import random
from typing import ClassVar

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import remove_isolated_nodes

from topobench.data.datasets.graphxai_vendor import (
    edge_mask_from_node_mask,
    match_edge_presence,
    to_networkx_conv,
)
from topobench.data.datasets.graphxai_vendor.substruct_chem_match import (
    MUTAG_NH2,
    MUTAG_NO2,
    match_aliphatic_halide,
    match_azo_type,
    match_nitroso,
    match_substruct_mutagenicity,
)

#: Datasets shipped as a single ``.npz`` archive by GraphXAI.
NPZ_DATASETS = {
    "Benzene": "benzene.npz",
    "AlkaneCarbonyl": "alkane_carbonyl.npz",
    "FluorideCarbonyl": "fluoride_carbonyl.npz",
}

#: Maximum number of ground-truth candidate explanations kept per graph. GraphXAI's
#: Mutagenicity builder enumerates every subset of matched substructures, which is
#: exponential in the number of matches; we keep every individual match plus the full
#: union, and the complete subset lattice only when it stays below this bound.
MAX_GT_CANDIDATES = 32

#: Commit of mims-harvard/GraphXAI that the download URLs and the vendored
#: helpers are pinned to, so the datasets stay reproducible even if the
#: upstream main branch moves.
GRAPHXAI_COMMIT = "a11e65ffbc4df737f35522a8accf2283a8aeaa37"


class GraphXAIDataset(InMemoryDataset):
    r"""GraphXAI molecular dataset with ground-truth explanation masks.

    Parameters
    ----------
    root : str
        Root directory for the processed dataset.
    name : str
        One of ``Benzene``, ``AlkaneCarbonyl``, ``FluorideCarbonyl``, ``Mutagenicity``.
    raw_data_dir : str, optional
        Directory holding pre-staged GraphXAI ``.npz`` archives (ignored for
        Mutagenicity, which is downloaded as a TUDataset). When omitted, missing
        archives are downloaded from the GraphXAI repository into ``root/raw``.
    downsample : bool, optional
        Whether to balance AlkaneCarbonyl by subsampling the majority class, as GraphXAI
        does. Default True (no effect on the other datasets).
    downsample_seed : int, optional
        Seed for that subsampling. GraphXAI leaves it unset, which makes the dataset
        nondeterministic; we fix it. Default 42.
    transform : callable, optional
        PyG transform.
    pre_transform : callable, optional
        PyG pre-transform.
    pre_filter : callable, optional
        PyG pre-filter.

    Attributes
    ----------
    URLS (dict): Download URL per ``.npz``-packaged dataset (GraphXAI repository).
    """

    URLS: ClassVar[dict[str, str]] = {
        name: (
            "https://raw.githubusercontent.com/mims-harvard/GraphXAI/"
            f"{GRAPHXAI_COMMIT}/"
            f"graphxai/datasets/real_world/{filename[: -len('.npz')]}/{filename}"
        )
        for name, filename in NPZ_DATASETS.items()
    }

    def __init__(
        self,
        root: str,
        name: str,
        raw_data_dir: str | None = None,
        downsample: bool = True,
        downsample_seed: int = 42,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        if name not in (*NPZ_DATASETS, "Mutagenicity"):
            raise ValueError(f"Unknown GraphXAI dataset {name!r}")
        self.name = name
        self.raw_data_dir = raw_data_dir or ""
        self.downsample = downsample
        self.downsample_seed = downsample_seed
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        self._gt_candidates = None

    @property
    def raw_dir(self) -> str:
        """Directory holding the source archives.

        Returns
        -------
        str
            The raw directory.
        """
        if self.name == "Mutagenicity" or not self.raw_data_dir:
            return osp.join(self.root, "raw")
        return str(self.raw_data_dir)

    @property
    def raw_file_names(self) -> list[str]:
        """Source files required to build the dataset.

        Returns
        -------
        list of str
            File names inside :attr:`raw_dir`.
        """
        if self.name == "Mutagenicity":
            return []
        return [NPZ_DATASETS[self.name]]

    @property
    def processed_file_names(self) -> list[str]:
        """Files produced by :meth:`process`.

        Returns
        -------
        list of str
            Processed file names.
        """
        return ["data.pt", "gt_candidates.pt"]

    @property
    def gt_candidates(self) -> list[dict]:
        """Per-graph candidate ground-truth explanation masks.

        Returns
        -------
        list of dict
            One dict per graph with keys ``node`` (``[K, num_nodes]`` bool) and ``edge``
            (``[K, num_edges]`` bool), ``K`` being the number of candidates.
        """
        if self._gt_candidates is None:
            self._gt_candidates = torch.load(
                self.processed_paths[1], weights_only=False
            )
        return self._gt_candidates

    def download(self) -> None:
        """Fetch the source data.

        Mutagenicity is downloaded as a TUDataset; the other datasets ship as
        ``.npz`` archives inside the GraphXAI repository and are downloaded from
        there unless already staged in :attr:`raw_dir`.
        """
        if self.name == "Mutagenicity":
            TUDataset(
                root=self.raw_dir, name="Mutagenicity", use_node_attr=False
            )
            return
        path = osp.join(self.raw_dir, NPZ_DATASETS[self.name])
        if not osp.isfile(path):
            download_url(self.URLS[self.name], self.raw_dir)

    def process(self) -> None:
        """Build the graphs and their ground-truth explanation masks."""
        if self.name == "Mutagenicity":
            graphs, candidates = self._build_mutagenicity()
        else:
            graphs, candidates = self._build_from_npz()

        data_list = []
        kept_candidates = []
        for data, cand in zip(graphs, candidates, strict=True):
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
            kept_candidates.append(cand)

        self.save(data_list, self.processed_paths[0])
        torch.save(kept_candidates, self.processed_paths[1])

    # ------------------------------------------------------------------ builders

    def _finalize(
        self, x, edge_index, edge_attr, y, cand_node, cand_edge, index
    ):
        """Assemble one graph plus its candidate masks.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge index.
        edge_attr : torch.Tensor or None
            Edge features.
        y : torch.Tensor
            Graph label.
        cand_node : torch.Tensor
            ``[K, num_nodes]`` bool candidate node masks.
        cand_edge : torch.Tensor
            ``[K, num_edges]`` bool candidate edge masks.
        index : int
            Position of this graph in the source dataset, kept so an explanation can find
            the graph's candidate ground-truth masks after splitting.

        Returns
        -------
        tuple
            ``(Data, candidates_dict)``.
        """
        data = Data(
            x=x.float(),
            edge_index=edge_index.long(),
            y=y.long().view(1),
            num_nodes=int(x.shape[0]),
            expl_node_mask=cand_node.any(0).float(),
            expl_edge_mask=cand_edge.any(0).float(),
            n_gt_candidates=torch.tensor([int(cand_node.shape[0])]),
            orig_index=torch.tensor([int(index)]),
            gt_positive=torch.tensor([bool(cand_node.any())]),
        )
        if edge_attr is not None:
            data.edge_attr = edge_attr.float()
        return data, {"node": cand_node.bool(), "edge": cand_edge.bool()}

    def _build_from_npz(self):
        """Build Benzene / AlkaneCarbonyl / FluorideCarbonyl from the packaged archive.

        Returns
        -------
        tuple
            ``(list of Data, list of candidate dicts)``.
        """
        path = osp.join(self.raw_dir, NPZ_DATASETS[self.name])
        blob = np.load(path, allow_pickle=True)
        att, X, y = blob["attr"], blob["X"], blob["y"]
        X = X[0]
        ylist = [y[i][0] for i in range(y.shape[0])]

        graphs, candidates = [], []
        for i in range(len(X)):
            x = torch.from_numpy(X[i]["nodes"])
            edge_attr = torch.from_numpy(X[i]["edges"])
            e1 = torch.from_numpy(X[i]["receivers"]).long()
            e2 = torch.from_numpy(X[i]["senders"]).long()
            edge_index = torch.stack([e1, e2])
            label = torch.tensor([int(ylist[i])], dtype=torch.long)

            node_imp = torch.from_numpy(att[i][0]["nodes"]).float()
            if node_imp.shape[0] != x.shape[0]:
                raise ValueError(
                    f"{self.name} graph {i}: explanation covers {node_imp.shape[0]} "
                    f"nodes but the graph has {x.shape[0]}"
                )
            cand_node, cand_edge = [], []
            for j in range(node_imp.shape[1]):
                nm = node_imp[:, j].bool()
                cand_node.append(nm)
                cand_edge.append(
                    edge_mask_from_node_mask(nm, edge_index).bool()
                )
            data, cand = self._finalize(
                x,
                edge_index,
                edge_attr,
                label,
                torch.stack(cand_node),
                torch.stack(cand_edge),
                i,
            )
            graphs.append(data)
            candidates.append(cand)

        if self.name == "AlkaneCarbonyl" and self.downsample:
            graphs, candidates = self._balance(graphs, candidates)
        return self._reindex(graphs), candidates

    def _balance(self, graphs, candidates):
        """Subsample the majority class 2:1, as GraphXAI does for AlkaneCarbonyl.

        Parameters
        ----------
        graphs : list of Data
            All graphs.
        candidates : list of dict
            Matching candidate masks.

        Returns
        -------
        tuple
            The balanced ``(graphs, candidates)``.
        """
        zero_bin = [i for i, g in enumerate(graphs) if int(g.y) == 0]
        one_bin = [i for i, g in enumerate(graphs) if int(g.y) == 1]
        rng = random.Random(self.downsample_seed)
        keep = rng.sample(zero_bin, k=min(2 * len(one_bin), len(zero_bin)))
        idx = sorted(keep + one_bin)
        return [graphs[i] for i in idx], [candidates[i] for i in idx]

    def _build_mutagenicity(self):
        """Build Mutagenicity with GraphXAI's substructure-matching ground truth.

        Returns
        -------
        tuple
            ``(list of Data, list of candidate dicts)``.
        """
        tu = TUDataset(
            root=self.raw_dir, name="Mutagenicity", use_node_attr=False
        )
        graphs, candidates = [], []
        for data in tu:
            data = data.clone()
            edge_idx, _, node_mask = remove_isolated_nodes(
                data.edge_index, num_nodes=data.x.shape[0]
            )
            data.x = data.x[node_mask]
            data.edge_index = edge_idx
            data.num_nodes = int(data.x.shape[0])

            mol_g = to_networkx_conv(
                data, node_attrs=["x"], to_undirected=True
            )
            matches = (
                match_substruct_mutagenicity(mol_g, MUTAG_NH2, nh2_no2=0)
                + match_substruct_mutagenicity(mol_g, MUTAG_NO2, nh2_no2=1)
                + match_nitroso(mol_g)
                + match_azo_type(mol_g)
                + match_aliphatic_halide(mol_g)
            )
            n_nodes, n_edges = data.num_nodes, data.edge_index.shape[1]
            has_match = len(matches) > 0

            # GraphXAI keeps only graphs whose label agrees with the presence of a
            # known mutagenic substructure.
            if int(has_match) != int(data.y.item()):
                continue

            if not has_match:
                cand_node = torch.zeros(1, n_nodes, dtype=torch.bool)
                cand_edge = torch.zeros(1, n_edges, dtype=torch.bool)
            else:
                singles_n, singles_e = [], []
                for m in matches:
                    nm = torch.zeros(n_nodes, dtype=torch.bool)
                    nm[m] = True
                    singles_n.append(nm)
                    singles_e.append(match_edge_presence(data.edge_index, m))
                cand_node, cand_edge = self._candidate_lattice(
                    singles_n, singles_e
                )
            payload, cand = self._finalize(
                data.x,
                data.edge_index,
                getattr(data, "edge_attr", None),
                data.y,
                cand_node,
                cand_edge,
                len(graphs),
            )
            graphs.append(payload)
            candidates.append(cand)
        return self._reindex(graphs), candidates

    @staticmethod
    def _reindex(graphs):
        """Renumber ``orig_index`` after any filtering, so it indexes this dataset.

        Parameters
        ----------
        graphs : list of Data
            Graphs in final order.

        Returns
        -------
        list of Data
            The same graphs with contiguous ``orig_index``.
        """
        for position, data in enumerate(graphs):
            data.orig_index = torch.tensor([position])
        return graphs

    @staticmethod
    def _candidate_lattice(singles_n, singles_e):
        """Combine matched substructures into candidate explanations.

        GraphXAI takes every non-empty subset of matches. That is exponential, so the
        full lattice is only built while it stays under :data:`MAX_GT_CANDIDATES`;
        otherwise the individual matches and their full union are kept.

        Parameters
        ----------
        singles_n : list of torch.Tensor
            Node mask per matched substructure.
        singles_e : list of torch.Tensor
            Edge mask per matched substructure.

        Returns
        -------
        tuple of torch.Tensor
            ``([K, num_nodes], [K, num_edges])`` candidate masks.
        """
        m = len(singles_n)
        node_stack = torch.stack(singles_n)
        edge_stack = torch.stack(singles_e)
        if 2**m - 1 <= MAX_GT_CANDIDATES:
            cand_n, cand_e = [], []
            for mask in range(1, 2**m):
                sel = [j for j in range(m) if mask >> j & 1]
                cand_n.append(node_stack[sel].any(0))
                cand_e.append(edge_stack[sel].any(0))
            return torch.stack(cand_n), torch.stack(cand_e)
        return (
            torch.cat([node_stack, node_stack.any(0, keepdim=True)]),
            torch.cat([edge_stack, edge_stack.any(0, keepdim=True)]),
        )
