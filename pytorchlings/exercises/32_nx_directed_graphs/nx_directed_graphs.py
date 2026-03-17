"""Exercise 32: directed graph (DiGraph) predecessors/successors."""

import networkx as nx


def build_dependency_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    # TODO: add edges for workflow:
    # load_data -> preprocess -> train -> evaluate
    # preprocess -> feature_select
    # feature_select -> train
    return g


def direct_parents(g: nx.DiGraph, node: str) -> list[str]:
    # TODO: return sorted immediate predecessors of node
    return []


def direct_children(g: nx.DiGraph, node: str) -> list[str]:
    # TODO: return sorted immediate successors of node
    return []


def is_reachable(g: nx.DiGraph, src: str, dst: str) -> bool:
    # TODO: return whether a directed path exists from src to dst
    return False
