import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

# Getting the datasets
from torch_geometric.datasets import KarateClub
from graph_tools import visualize_embedding, visualize_graph
from torch_geometric.utils import to_networkx

dataset = KarateClub()
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4,4)
        self.conv3 = GCNConv(4,2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()

        # apply a final (linear) classifier
        out = self.classifier(h)
        return out, h # 

    
        
        
if __name__ == "__main__":
    #model = GCN()
    #print(model)

    G = to_networkx(data, to_undirected = True)
    visualize_graph(G, color = data.y)
    
