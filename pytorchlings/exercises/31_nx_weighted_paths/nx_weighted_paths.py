"""Exercise 31: weighted graph shortest path with NetworkX."""

import networkx as nx


def build_weighted_graph() -> nx.Graph:
    g = nx.Graph()
    # TODO: add weighted edges:
    # ("A", "B", 2.0), ("B", "C", 1.0), ("A", "C", 5.0), ("C", "D", 1.5)
    # Use edge attribute name "weight".
    return g


def weighted_shortest_path(g: nx.Graph, src: str, dst: str) -> list[str]:
    # TODO: return the minimum-total-weight path from src to dst
    return []


def weighted_distance(g: nx.Graph, src: str, dst: str) -> float:
    # TODO: return shortest path distance using "weight"
    return 0.0
