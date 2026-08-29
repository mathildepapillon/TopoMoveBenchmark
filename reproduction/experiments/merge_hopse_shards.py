"""Merge sharded HOPSE participation runs into one attribution JSON.

Each shard (experiments/mantra_hopse_participation_pair.py under
TOPOSHAP_HOPSE_{TARGET,SEED,PASSES,OUT}) is an unbiased permutation-
sampled Shapley estimate; the equal-passes average over distinct shard
seeds is the same estimator as one run with the summed pass budget.
v_full/v_empty are coalition values, deterministic across shards — the
merge asserts they agree, then averages phi and sums passes/evaluations.

Usage: merge_hopse_shards.py OUT.json SHARD.json [SHARD.json ...]
"""

from __future__ import annotations

import json
import sys

import numpy as np


def main() -> None:
    out_path, shard_paths = sys.argv[1], sys.argv[2:]
    shards = [json.loads(open(p).read()) for p in shard_paths]

    meta = {k: shards[0][k] for k in ("seed", "tag", "ckpt", "game", "note")}
    merged = dict(meta)
    merged["shards"] = [
        {"path": p, "shard_seed": s["shard_seed"], "passes": s["passes"]}
        for p, s in zip(shard_paths, shards)]

    for name in ("torus", "klein"):
        parts = [s for s in shards if name in s]
        if not parts:
            raise SystemExit(f"no shard covers {name}")
        first = parts[0][name]
        passes = {s["passes"] for s in shards if name in s}
        if len(passes) != 1:
            raise SystemExit(f"{name}: unequal pass budgets {passes} — "
                             "equal-weight averaging would be biased")
        for s in parts[1:]:
            e = s[name]
            if e["player_ranks"] != first["player_ranks"]:
                raise SystemExit(f"{name}: player order differs across shards")
            for k in ("v_full", "v_empty"):
                if abs(e[k] - first[k]) > 1e-5:
                    raise SystemExit(
                        f"{name}: {k} differs across shards "
                        f"({e[k]!r} vs {first[k]!r}) — games disagree")
        phi = np.mean([np.asarray(s[name]["phi"]) for s in parts], axis=0)
        merged[name] = {
            "phi": [float(v) for v in phi],
            "player_ranks": first["player_ranks"],
            "v_full": first["v_full"],
            "v_empty": first["v_empty"],
            "evaluations": int(sum(s[name]["evaluations"] for s in parts)),
            "test_index": first["test_index"],
        }
        eff = phi.sum() - (first["v_full"] - first["v_empty"])
        print(f"{name}: {len(parts)} shards, efficiency residual {eff:+.2e}")
    merged["passes"] = int(next(iter(passes)) * len(parts))

    with open(out_path, "w") as f:
        json.dump(merged, f)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
