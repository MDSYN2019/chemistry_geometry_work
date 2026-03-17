"""Solution 30: networkx connectivity and centrality basics."""

import networkx as nx


def build_two_component_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (10, 11)])
    return g


def component_sizes(g: nx.Graph) -> list[int]:
    return sorted(len(c) for c in nx.connected_components(g))


def top_degree_node(g: nx.Graph) -> int:
    return min(g.nodes, key=lambda n: (-g.degree[n], n))
