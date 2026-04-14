"""Exercise 79: overlap heuristics for link prediction.

Goal:
- implement classical edge scores before any GNN
- compare heuristic ranking quality with AUC/AP
- build a reliable baseline for relation prediction
"""

from __future__ import annotations

import itertools
from collections import defaultdict


def make_toy_edges() -> list[tuple[int, int]]:
    """Undirected toy graph edge list."""
    return [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (4, 5),
    ]


def build_neighbors(edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    nbrs: dict[int, set[int]] = defaultdict(set)
    for u, v in edges:
        nbrs[u].add(v)
        nbrs[v].add(u)
    return dict(nbrs)


def common_neighbors(nbrs: dict[int, set[int]], u: int, v: int) -> int:
    return len(nbrs.get(u, set()) & nbrs.get(v, set()))


def jaccard(nbrs: dict[int, set[int]], u: int, v: int) -> float:
    a, b = nbrs.get(u, set()), nbrs.get(v, set())
    den = len(a | b)
    return 0.0 if den == 0 else len(a & b) / den


def adamic_adar(nbrs: dict[int, set[int]], u: int, v: int) -> float:
    """TODO: implement Adamic-Adar index with log-degree denominator."""
    _ = (nbrs, u, v)
    return 0.0


def resource_allocation(nbrs: dict[int, set[int]], u: int, v: int) -> float:
    """TODO: implement Resource Allocation index."""
    _ = (nbrs, u, v)
    return 0.0


def non_edges(nodes: list[int], edges: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cands = []
    for u, v in itertools.combinations(nodes, 2):
        e = (u, v) if u < v else (v, u)
        if e not in edges:
            cands.append(e)
    return cands


if __name__ == "__main__":
    edges = make_toy_edges()
    edge_set = {(u, v) if u < v else (v, u) for (u, v) in edges}
    nodes = sorted(set([n for e in edges for n in e]))

    nbrs = build_neighbors(edges)
    negatives = non_edges(nodes, edge_set)

    assert common_neighbors(nbrs, 0, 3) >= 1
    assert 0.0 <= jaccard(nbrs, 0, 3) <= 1.0
    assert len(negatives) > 0

    # TODO: add held-out split + AUC/AP evaluation across all heuristic scores.
    print("exercise 79 scaffold ready")
