"""Exercise 10: build a minimal PyG Data object."""

# pip install torch-geometric
import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter

# torch geometric imports 
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

def build_graph() -> Data:
    # 5 nodes, 2 features each
    x = torch.tensor([
        [1.0, 0.0],  # node 0
        [0.0, 1.0],  # node 1
        [1.0, 1.0],  # node 2
        [0.5, 0.2],  # node 3
        [0.2, 0.8],  # node 4
    ], dtype=torch.float)

    # Undirected chain-like graph: 0-1-2-3-4
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ], dtype=torch.long)

    # Node labels: 2 classes
    y = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)

    # Train on some nodes, test on others
    train_mask = torch.tensor([True, True, True, False, False])
    test_mask = torch.tensor([False, False, False, True, True])

    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, test_mask=test_mask)
#
#def build_graph() -> Data:
#    # 3 nodes, 2 input features each
#    x = torch.tensor([
#        [1.0, 0.0],  # node 0
#        [0.0, 1.0],  # node 1
#        [1.0, 1.0],  # node 2
#    ], dtype=torch.float)
#
#    # Undirected chain: 0 <-> 1 <-> 2
#    # Shape must be [2, E]
#    edge_index = torch.tensor([
#        [0, 1, 1, 2],
#        [1, 0, 2, 1],
#    ], dtype=torch.long)
#
#    return Data(x=x, edge_index=edge_index)

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = 'add') # 

        self.lin = Linear(in_channels, out_channels, bias = False)
        self.bias = Parameter(torch.empty(out_channels))


    def reset_parameters(self):
        self.lin.reset_parameters() # reset the lin and bias to zero 
        self.bias.data.zero_() # set bias to zero 

        
         
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge index has shape, [2, E]

        # Add the self-loops to the adjacency matrix
        print(x.shape)
        print(edge_index.shape)
        
        edge_index, _ = add_self_loops(edge_index, num_nodes= x.size(0)) # we have to ensure that this is of the shape [2,E]

        # transform the node matrix 
        x = self.lin(x) 
        row, col = edge_index
        # Get the degree of each node
        deg = degree(col, x.size(0), dtype = x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x =x , norm = norm) # propate internally calls the message, aggregate and update

        # we pass the node embeddings x and the normalization coefficients norm as additional arguments for
        # message propagation
 
        out += self.bias
        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

# Packaging this up into an actual multlayer GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

        
def accuracy(logits, y):
    """
    Computes the classification accuracy 
    """
    pred = logits.argmax(dim = 1)
    return (pred == y).float().mean().item()
    
if __name__ == "__main__":
    data = build_graph()
    #conv = GCNConv(2, 4)
    #out = conv(data.x, data.edge_index)
    model = GCN(in_channels = 2, hidden_channels = 4, out_channels = 2)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.05, weight_decay = 5e-4)

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index) # give the model the node information and the edge_index ] compute the prediction 
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])  # compute the loss 
#
        loss.backward() 
        optimizer.step()

        if epoch % 20  == 0:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                train_acc = accuracy(logits[data.train_mask], data.y[data.train_mask])
                test_acc = accuracy(logits[data.test_mask], data.y[data.test_mask])

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Acc: {train_acc:.3f} | "
                f"Test Acc: {test_acc:.3f}"
            )
