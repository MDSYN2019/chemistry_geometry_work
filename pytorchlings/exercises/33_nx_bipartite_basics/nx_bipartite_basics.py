"""Exercise 33: bipartite graph setup and projections."""

import networkx as nx
from networkx.algorithms import bipartite


def build_user_item_graph() -> nx.Graph:
    g = nx.Graph()
    # TODO: add user nodes {"u1", "u2", "u3"} with bipartite=0
    # TODO: add item nodes {"i1", "i2", "i3"} with bipartite=1
    # TODO: add edges: u1-i1, u1-i2, u2-i2, u2-i3, u3-i3
    return g


def user_projection(g: nx.Graph) -> nx.Graph:
    # TODO: return the weighted projection onto user nodes only
    users = set()
    return bipartite.weighted_projected_graph(g, users)


def item_projection(g: nx.Graph) -> nx.Graph:
    # TODO: return the weighted projection onto item nodes only
    items = set()
    return bipartite.weighted_projected_graph(g, items)
