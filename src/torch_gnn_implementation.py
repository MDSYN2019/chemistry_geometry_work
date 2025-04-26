import networkx as nx
import numpy as np
import seaborn as sns


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

# rdkit for molecular modelling
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops


"""
GNNs are specific layers that input a graph and output a graph. 

For each atom(node), we need to define a feature vector.

For example:

Atom type.
Formal charge.
Hybridization State.
Aromaticity.
Number of connected hydrogens.

"""

def return_molecular_graph(adjacency, node_labels) -> nx.Graph():
    """
    visualize a molecule as a graph
    """
    G = nx.graph()

    for i, label in enumerate(node_labels):
        G.add_node(i, label = label)
        
    rows, cols = np.where(adjacency = 1)
    edges = zip(rows.tolist(), cols.tolist())

    for i, j in edges:
        if i < j:
            G.add_edge(i,j)

    return G

def smiles_to_graph(smiles_mol, elements = ["C", "O", "N", "H", "Other"]):
    if smiles_mol is None:
        raise ValueError(f"Invalid Smiles String: {smiles_mol}")

        # Add hydrogens to make the representation complete
        mol = Chem.AddHs(smiles_mol)

        n_atoms = mol.GetNumAtoms()
        node_features = np.zeros(n_atoms, len(elements))

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            if symbol in elements:
                node_features[idx, elements.index(symbol)] = 1
            else:
                node_features[idx, -1] = 1#
                
        # Create the adjacency matrix and edge features
        adjacency, edge_features, edge_indices = create_adjacency_matrix(n_atoms, mol)

        return node_features, adjacency, edge_features, edge_indices
    
def create_adjacency_matrix(n_atoms, mol):
    """
    Function to create the adjacency matrix for the graph representation of the graph
    """
    # Map bond types to indices
    bond_type_to_idx = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,    
    }

    adjacency = np.zeros((n_atoms, n_atoms))
    edge_features = []
    edge_indices = []
    
    for bond in mol.GetBonds():
        # get the bond types

        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        # update the adjacency index
        adjacency[begin_idx, end_idx] = 1
        adjacency[end_idx, begin_idx] = 1

        # Get the bond type
        bond_type = bond.GetBondType()
        bond_feature = np.zeros(len(bond_type_to_idx))

        if bond_type in bond_type_to_idx:
            bond_feature[bond_type_to_idx[bond_type]] = 1
        else:
            print(f"Warning: Unknown bond type: {bond_type}")

        edge_features.append(bond_feature)
        edge_indices.append((begin_idx, end_idx))

        edge_features.append(bond_feature)
        edge_indices.append((end_idx, begin_idx))

        if edge_features:
            edge_features = np.array(edge_features)
        else:
            edge_features = np.empty((0, len(bond_type_to_idx)))

    return adjacency, edge_features, edge_indices

