"""Solution 31: weighted graph shortest path with NetworkX."""

import networkx as nx


def build_weighted_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_weighted_edges_from(
        [
            ("A", "B", 2.0),
            ("B", "C", 1.0),
            ("A", "C", 5.0),
            ("C", "D", 1.5),
        ],
        weight="weight",
    )
    return g


def weighted_shortest_path(g: nx.Graph, src: str, dst: str) -> list[str]:
    return nx.shortest_path(g, source=src, target=dst, weight="weight")


def weighted_distance(g: nx.Graph, src: str, dst: str) -> float:
    return nx.shortest_path_length(g, source=src, target=dst, weight="weight")
