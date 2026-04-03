import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphConv, global_mean_pool


class GIN(torch.nn.Module):
    """Simple graph classification model using GraphConv layers.

    Notes:
    - Despite the class name, this currently uses ``GraphConv`` blocks.
    - Inputs follow the standard PyG ``Data`` batch signature.
    """

    def __init__(self, hidden_channels: int, dataset) -> None:
        super().__init__()
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes

        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        """Run a forward pass.

        Args:
            x: Node feature matrix with shape [num_nodes, num_node_features].
            edge_index: Graph connectivity in COO format.
            batch: Batch vector that maps each node to a graph id.
        """
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = x.relu()

        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        return self.classifier(x)
