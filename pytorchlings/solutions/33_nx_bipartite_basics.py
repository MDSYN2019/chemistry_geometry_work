"""Solution 33: bipartite graph setup and projections."""

import networkx as nx
from networkx.algorithms import bipartite


def build_user_item_graph() -> nx.Graph:
    g = nx.Graph()
    users = {"u1", "u2", "u3"}
    items = {"i1", "i2", "i3"}
    g.add_nodes_from(users, bipartite=0)
    g.add_nodes_from(items, bipartite=1)
    g.add_edges_from(
        [("u1", "i1"), ("u1", "i2"), ("u2", "i2"), ("u2", "i3"), ("u3", "i3")]
    )
    return g


def user_projection(g: nx.Graph) -> nx.Graph:
    users = {n for n, d in g.nodes(data=True) if d.get("bipartite") == 0}
    return bipartite.weighted_projected_graph(g, users)


def item_projection(g: nx.Graph) -> nx.Graph:
    items = {n for n, d in g.nodes(data=True) if d.get("bipartite") == 1}
    return bipartite.weighted_projected_graph(g, items)
