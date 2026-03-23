from torch_geometric.graphgym.config import cfg, set_cfg


def build_graphgym_cfg() -> object:
    """Return a GraphGym cfg object configured for a tiny GNN experiment."""
    set_cfg(cfg)
    cfg.dataset.name = "Cora"
    cfg.model.type = "gcn"
    cfg.gnn.layers_mp = 2
    cfg.optim.max_epoch = 50
    return cfg
