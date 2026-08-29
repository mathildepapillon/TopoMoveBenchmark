"""Patch the vendored GraphXAI clone (external/GraphXAI) for modern PyG.

GraphXAI is pinned at a11e65ffbc4df737f35522a8accf2283a8aeaa37 — the
same commit the frozen baseline suite used (data/frozen/baselines2/
GRAPHXAI_COMMIT.txt). We use ONLY its SubgraphX implementation;
GNNExplainer and PGExplainer come from PyG's maintained
torch_geometric.explain module instead, which makes most of the frozen
suite's patch list (data/frozen/baselines2/patches.json) unnecessary.
The re-derived patches below correspond to P2 (DataLoader import) and
P4 (trim package imports); P3/P7 (batch vector, device) are applied
where they bite if runtime surfaces them.

Idempotent: safe to run repeatedly. Run after a fresh clone:
    git clone https://github.com/mims-harvard/GraphXAI external/GraphXAI
    git -C external/GraphXAI checkout a11e65ffbc4df737f35522a8accf2283a8aeaa37
    python scripts/apply_graphxai_patches.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "external" / "GraphXAI"


def patch(path: str, old: str, new: str, name: str) -> None:
    p = ROOT / path
    s = p.read_text()
    if new in s:
        print(f"{name}: already applied")
        return
    if old not in s:
        raise SystemExit(f"{name}: anchor not found in {path}")
    p.write_text(s.replace(old, new))
    print(f"{name}: applied")


# P2: modern PyG moved DataLoader to torch_geometric.loader
patch(
    "graphxai/explainers/subgraphx_utils/shapley.py",
    "from torch_geometric.data import Data, Batch, Dataset, DataLoader",
    "from torch_geometric.data import Data, Batch, Dataset\n"
    "from torch_geometric.loader import DataLoader",
    "P2-dataloader-import",
)

# P3: the marginal-contribution loop batches up to 256 masked subgraphs
# per forward, but wrap_value_func drops the collated batch vector — a
# graph-pooling model would read the whole batch as ONE graph. Forward
# the batch vector (zeros for a single uncollated graph).
patch(
    "graphxai/explainers/subgraphx.py",
    "        def wrap_value_func(data):\n"
    "            return value_func(x=data.x, edge_index=data.edge_index, "
    "forward_kwargs=forward_kwargs)\n"
    "\n"
    "        payoff_func = self.get_reward_func(wrap_value_func, "
    "explain_graph = True)",
    "        def wrap_value_func(data):\n"
    "            fk = dict(forward_kwargs)\n"
    "            fk['batch'] = (data.batch if data.batch is not None else\n"
    "                           torch.zeros(data.x.shape[0], "
    "dtype=torch.long, device=data.x.device))\n"
    "            return value_func(x=data.x, edge_index=data.edge_index, "
    "forward_kwargs=fk)\n"
    "\n"
    "        payoff_func = self.get_reward_func(wrap_value_func, "
    "explain_graph = True)",
    "P3-batch-vector",
)

# P3b: drop the stray debug prints on the batched value path.
for old in (
    "            print(forward_kwargs)\n            print(batch.batch)\n",
    "            print(probs, probs.shape)\n",
):
    p = ROOT / "graphxai" / "explainers" / "subgraphx_utils" / "shapley.py"
    s = p.read_text()
    if old in s:
        p.write_text(s.replace(old, ""))
        print("P3b-remove-print: applied")

# P7: __parse_results builds node/edge masks on CPU while edge_index may
# be on GPU — indexing throws on CUDA runs (CPU smoke passes).
patch(
    "graphxai/explainers/subgraphx.py",
    "        subgraph_nodes = torch.tensor([map[c] for c in "
    "best_subgraph.coalition], dtype=torch.long) if map is not None \\\n"
    "            else torch.tensor(best_subgraph.coalition, "
    "dtype=torch.long)\n"
    "\n"
    "        # Create node mask:\n"
    "        node_mask = torch.zeros(all_nodes.shape, dtype=torch.bool)\n"
    "        node_mask[subgraph_nodes] = 1\n"
    "\n"
    "        # Create edge_index mask\n"
    "        num_nodes = maybe_num_nodes(edge_index)\n"
    "        n_mask = torch.zeros(num_nodes, dtype = torch.bool)\n",
    "        subgraph_nodes = torch.tensor([map[c] for c in "
    "best_subgraph.coalition], dtype=torch.long, "
    "device=edge_index.device) if map is not None \\\n"
    "            else torch.tensor(best_subgraph.coalition, "
    "dtype=torch.long, device=edge_index.device)\n"
    "\n"
    "        # Create node mask:\n"
    "        node_mask = torch.zeros(all_nodes.shape, dtype=torch.bool, "
    "device=edge_index.device)\n"
    "        node_mask[subgraph_nodes] = 1\n"
    "\n"
    "        # Create edge_index mask\n"
    "        num_nodes = maybe_num_nodes(edge_index)\n"
    "        n_mask = torch.zeros(num_nodes, dtype=torch.bool, "
    "device=edge_index.device)\n",
    "P7-mask-device",
)

# P4 (extended): the explainers package eagerly imports every explainer;
# several need dead deps (ipdb, pgmpy). Keep only what we use.
init = ROOT / "graphxai" / "explainers" / "__init__.py"
init_new = (
    "# Trimmed by toposhap scripts/apply_graphxai_patches.py: only\n"
    "# SubgraphX is used; the eager imports pulled dead dependencies\n"
    "# (ipdb, pgmpy) on modern environments.\n"
    "from .subgraphx import SubgraphX\n"
    "from ._base import _BaseExplainer\n"
)
if init.read_text() != init_new:
    init.write_text(init_new)
    print("P4-trim-init: applied")
else:
    print("P4-trim-init: already applied")

print("done")
