"""Exercise 95: Limited expressiveness in message-passing GNNs.

Goals:
- show collisions from weak neighborhood aggregators (mean/max)
- implement a multiset-sensitive aggregator demo
- connect behavior to 1-WL limitations on regular graph pairs
"""

from __future__ import annotations

from collections import Counter


def aggregate_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate_sum(values: list[float]) -> float:
    return sum(values)


def aggregate_max(values: list[float]) -> float:
    return max(values)


def multiset_signature(values: list[int]) -> tuple[tuple[int, int], ...]:
    """Return a count-preserving signature for a multiset.

    TODO:
    - replace this baseline with a learnable MLP-over-sum style sketch, or
      another injective multiset map inspired by GIN.
    """
    counts = Counter(values)
    return tuple(sorted(counts.items()))


def demo_neighbor_collision() -> None:
    neigh_a = [1, 3]
    neigh_b = [2, 2]

    assert aggregate_mean(neigh_a) == aggregate_mean(neigh_b)
    assert aggregate_sum(neigh_a) == aggregate_sum(neigh_b)
    assert aggregate_max(neigh_a) != aggregate_max(neigh_b)

    sig_a = multiset_signature(neigh_a)
    sig_b = multiset_signature(neigh_b)
    assert sig_a != sig_b

    print("mean collision:", aggregate_mean(neigh_a), aggregate_mean(neigh_b))
    print("sum collision:", aggregate_sum(neigh_a), aggregate_sum(neigh_b))
    print("multiset signatures:", sig_a, sig_b)


if __name__ == "__main__":
    demo_neighbor_collision()
    # TODO: add a tiny PyG experiment comparing GCNConv vs GINConv on a pair of
    # non-isomorphic graphs that are hard for weak aggregators.
    print("exercise 95 scaffold ready")
