"""Solution 32: directed graph (DiGraph) predecessors/successors."""

import networkx as nx


def build_dependency_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("load_data", "preprocess"),
            ("preprocess", "train"),
            ("train", "evaluate"),
            ("preprocess", "feature_select"),
            ("feature_select", "train"),
        ]
    )
    return g


def direct_parents(g: nx.DiGraph, node: str) -> list[str]:
    return sorted(g.predecessors(node))


def direct_children(g: nx.DiGraph, node: str) -> list[str]:
    return sorted(g.successors(node))


def is_reachable(g: nx.DiGraph, src: str, dst: str) -> bool:
    return nx.has_path(g, source=src, target=dst)
