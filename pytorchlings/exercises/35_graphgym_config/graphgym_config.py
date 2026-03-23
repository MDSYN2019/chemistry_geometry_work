"""Exercise 35: build a minimal GraphGym config with PyG defaults."""
from torch_geometric.graphgym.config import cfg, set_cfg


def build_graphgym_cfg() -> object:
    """Return a GraphGym cfg object configured for a tiny GNN experiment."""
    set_cfg(cfg)

    # TODO: set cfg.dataset.name to "Cora"
    # TODO: set cfg.model.type to "gcn"
    # TODO: set cfg.gnn.layers_mp to 2
    # TODO: set cfg.optim.max_epoch to 50

    return cfg
