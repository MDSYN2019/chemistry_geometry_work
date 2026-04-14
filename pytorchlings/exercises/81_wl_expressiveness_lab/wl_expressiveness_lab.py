"""Exercise 81: Weisfeiler-Lehman expressiveness lab.

Goal:
- implement 1-WL color refinement
- build graph pairs that challenge weak aggregators
- compare WL outcomes to GCN/GIN behavior
"""

from __future__ import annotations

from collections import Counter


def wl_refine(adj_list: dict[int, list[int]], num_steps: int = 3) -> dict[int, int]:
    """Simple 1-WL color refinement with integer color hashing."""
    colors = {n: 1 for n in adj_list}
    for _ in range(num_steps):
        signatures = {}
        for n in adj_list:
            neighborhood = tuple(sorted(colors[m] for m in adj_list[n]))
            signatures[n] = (colors[n], neighborhood)
        uniq = {sig: i + 1 for i, sig in enumerate(sorted(set(signatures.values())))}
        colors = {n: uniq[sig] for n, sig in signatures.items()}
    return colors


def cycle_graph(n: int) -> dict[int, list[int]]:
    return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}


def two_triangles() -> dict[int, list[int]]:
    return {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1],
        3: [4, 5],
        4: [3, 5],
        5: [3, 4],
    }


if __name__ == "__main__":
    g1 = cycle_graph(6)
    g2 = two_triangles()

    c1 = wl_refine(g1, num_steps=3)
    c2 = wl_refine(g2, num_steps=3)

    hist1 = Counter(c1.values())
    hist2 = Counter(c2.values())

    assert len(c1) == 6 and len(c2) == 6
    assert sum(hist1.values()) == 6 and sum(hist2.values()) == 6

    # TODO: add additional graph pairs and compare with a small GIN + GCN experiment.
    print("exercise 81 scaffold ready", hist1, hist2)
