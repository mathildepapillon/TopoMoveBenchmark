"""Cycle lifting with cell-to-graph ground-truth bookkeeping.

Verbatim port of the validated exp #1 module
(data/frozen/reference/exp1_worktree/topobench_repo/topobench/transforms/
liftings/graph2cell/cycle_lifting_gt.py); registered into TopoBench's
TRANSFORMS registry by ``toposhap.patches.apply_all`` so pipelines can
select it with
``transforms.graph2cell_lifting.transform_name=CellCycleLiftingGT``.
"""

import networkx as nx
import torch
import torch_geometric
from toponetx.classes import CellComplex

from topobench.transforms.liftings.graph2cell.base import Graph2CellLifting


class CellCycleLiftingGT(Graph2CellLifting):
    r"""Cycle lifting that preserves the cell-to-graph correspondence.

    Parameters
    ----------
    max_cell_length : int, optional
        Maximum length of a cycle lifted to a 2-cell. Default None (no limit).
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_cell_length=None, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = 2
        self.max_cell_length = max_cell_length

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift a graph to a cell complex and record the cell-to-graph maps.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The graph to lift.

        Returns
        -------
        dict
            The lifted topology, with the extra ``cell_1_endpoints``,
            ``cell_2_boundary_flat`` and ``cell_2_boundary_ptr`` entries.
        """
        graph = self._generate_graph_from_data(data)
        cycles = nx.cycle_basis(graph)
        cell_complex = CellComplex(graph)

        cycles = [cycle for cycle in cycles if len(cycle) != 1]
        if self.max_cell_length is not None:
            cycles = [c for c in cycles if len(c) <= self.max_cell_length]
        # Deterministic 2-cell ordering (the base lifting sorts identically).
        cycles = sorted(cycles, key=lambda c: (len(c), tuple(sorted(c))))
        if len(cycles) != 0:
            cell_complex.add_cells_from(cycles, rank=self.complex_dim)

        lifted = self._get_lifted_topology(cell_complex, graph)

        # 0-cell order must be the original node order for a node mask to transfer.
        x_0 = lifted["x_0"]
        if x_0.shape[0] != data.x.shape[0] or not torch.allclose(
            x_0.float(), data.x.float()
        ):
            raise RuntimeError(
                "0-cell ordering does not match the input node ordering; the "
                "ground-truth mask cannot be lifted safely."
            )

        lifted.update(self._cell_maps(cell_complex))
        return lifted

    @staticmethod
    def _cell_maps(cell_complex: CellComplex) -> dict:
        """Read the boundary maps off the complex's unsigned incidence matrices.

        Parameters
        ----------
        cell_complex : toponetx.classes.CellComplex
            The lifted complex.

        Returns
        -------
        dict
            ``cell_1_endpoints``, ``cell_2_boundary_flat``, ``cell_2_boundary_ptr``.
        """
        b1 = cell_complex.incidence_matrix(rank=1, signed=False).tocsc()
        n_1 = b1.shape[1]
        endpoints = torch.full((n_1, 2), -1, dtype=torch.long)
        for j in range(n_1):
            rows = b1.indices[b1.indptr[j] : b1.indptr[j + 1]]
            if len(rows) != 2:
                raise RuntimeError(
                    f"1-cell {j} has {len(rows)} boundary 0-cells, expected 2"
                )
            endpoints[j, 0], endpoints[j, 1] = int(rows[0]), int(rows[1])

        try:
            b2 = cell_complex.incidence_matrix(rank=2, signed=False).tocsc()
            n_2 = b2.shape[1]
        except Exception:  # noqa: BLE001 - complex with no 2-cells
            n_2 = 0
        flat, ptr = [], [0]
        for j in range(n_2):
            rows = b2.indices[b2.indptr[j] : b2.indptr[j + 1]]
            flat.extend(int(r) for r in rows)
            ptr.append(len(flat))

        return {
            "cell_1_endpoints": endpoints,
            "cell_2_boundary_flat": torch.tensor(flat, dtype=torch.long),
            "cell_2_boundary_ptr": torch.tensor(ptr, dtype=torch.long),
        }
