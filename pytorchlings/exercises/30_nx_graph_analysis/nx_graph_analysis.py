"""Exercise 30: networkx connectivity and centrality basics."""

import networkx as nx


def build_two_component_graph() -> nx.Graph:
    g = nx.Graph()
    # TODO: create two components: (0-1-2) and (10-11)
    return g


def component_sizes(g: nx.Graph) -> list[int]:
    # TODO: return sorted sizes of connected components
    return []


def top_degree_node(g: nx.Graph) -> int:
    # TODO: return node id with highest degree (break ties by smallest id)
    return -1
