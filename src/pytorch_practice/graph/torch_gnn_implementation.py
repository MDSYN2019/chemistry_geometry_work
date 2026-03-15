import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch.nn import Linear

#from rdkit import Chem
#from rdkit.Chem import rdmolfiles, rdmolops
#import networkx as nx
#import numpy as np


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset): 
        """
        """
        super(GCN, self).__init__() # call the parent class method
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        """
        """
        x = self.conv1(x, edge_index) # 1-hop neighbours 
        x = x.relu()
        x = self.conv2(x, edge_index) # 2-hop neighbours 
        x = x.relu()
        x = self.conv3(x, edge_index) # larger structural context 
        x = global_mean_pool(x, batch) # converts the node features -> graph embedding

        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.lin(x) # Linear classifier 
        return x
    
