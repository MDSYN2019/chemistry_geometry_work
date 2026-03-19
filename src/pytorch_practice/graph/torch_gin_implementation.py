import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool, GraphConv
from torch_geometric.data import Data, DataLoader
from torch.nn import Linear


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super(GIN, self).__init__()
        self.conv_gin1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index) # 1-hop neighbours 
        x = x.relu()
        x = self.conv2(x, edge_index) # 2-hop neighbours 
        x = x.relu()
        x = self.conv3(x, edge_index) # larger structural context 
        x = global_mean_pool(x, batch) # converts the node features -> graph embeddin
        
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.lin(x)
        return x
