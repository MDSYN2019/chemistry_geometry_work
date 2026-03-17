"""Exercise 29: networkx path helpers and neighborhood queries."""

import networkx as nx


def build_chain_graph(n: int = 5) -> nx.Graph:
    """Build 0-1-2-...-(n-1)."""
    g = nx.Graph()
    # TODO: add chain edges using range
    return g


def shortest_path_nodes(g: nx.Graph, src: int, dst: int) -> list[int]:
    # TODO: return shortest path node sequence from src to dst
    return []


def immediate_neighbors(g: nx.Graph, node: int) -> list[int]:
    # TODO: return sorted neighbors of node
    return []
