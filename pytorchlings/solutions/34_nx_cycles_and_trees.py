"""Solution 34: cycle checks and tree checks in undirected graphs."""

import networkx as nx


def build_cyclic_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])
    return g


def has_any_cycle(g: nx.Graph) -> bool:
    return len(nx.cycle_basis(g)) > 0


def is_tree_graph(g: nx.Graph) -> bool:
    return nx.is_tree(g)
