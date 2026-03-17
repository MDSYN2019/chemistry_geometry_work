"""Solution 29: networkx path helpers and neighborhood queries."""

import networkx as nx


def build_chain_graph(n: int = 5) -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from((i, i + 1) for i in range(n - 1))
    return g


def shortest_path_nodes(g: nx.Graph, src: int, dst: int) -> list[int]:
    return nx.shortest_path(g, source=src, target=dst)


def immediate_neighbors(g: nx.Graph, node: int) -> list[int]:
    return sorted(g.neighbors(node))
