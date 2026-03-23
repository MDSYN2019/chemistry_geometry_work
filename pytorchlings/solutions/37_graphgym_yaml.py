def graphgym_yaml() -> str:
    """Return a YAML string for a quick GraphGym run on Cora."""
    return "\n".join(
        [
            "dataset:",
            "  name: Cora",
            "model:",
            "  type: gcn",
            "gnn:",
            "  layers_mp: 2",
            "optim:",
            "  max_epoch: 50",
        ]
    )
