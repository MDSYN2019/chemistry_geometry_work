"""Exercise 34: cycle checks and tree checks in undirected graphs."""

import networkx as nx


def build_cyclic_graph() -> nx.Graph:
    g = nx.Graph()
    # TODO: create graph with a cycle: 0-1-2-0 and a tail edge 2-3
    return g


def has_any_cycle(g: nx.Graph) -> bool:
    # TODO: return True if graph has at least one cycle
    return False


def is_tree_graph(g: nx.Graph) -> bool:
    # TODO: return True only if graph is a tree
    return False
