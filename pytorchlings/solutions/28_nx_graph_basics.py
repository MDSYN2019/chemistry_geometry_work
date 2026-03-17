"""Solution 28: networkx Graph construction and basic queries."""

import networkx as nx


def build_triangle_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(["H", "O", "C"])
    g.add_edges_from([("H", "O"), ("O", "C"), ("C", "H")])
    return g


def node_count(g: nx.Graph) -> int:
    return g.number_of_nodes()


def edge_count(g: nx.Graph) -> int:
    return g.number_of_edges()


def has_edge(g: nx.Graph, u: str, v: str) -> bool:
    return g.has_edge(u, v)
