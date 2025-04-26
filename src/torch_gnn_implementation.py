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

import logging
logging.basicConfig(level=logging.INFO)

"""
GNNs are specific layers that input a graph and output a graph. 

For each atom(node), we need to define a feature vector.

For example:

Atom type.
Formal charge.
Hybridization State.
Aromaticity.
Number of connected hydrogens.


What does each row in the node feature matrix represent?

=> Each row represents one atom in the molecule, with it's feature encoded


"""
bond_type_to_idx = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

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
        """
        Node features identifies the actual atom type, sample edge features show what type of bond we have,
        and the adjacenecy matrix shows which atoms are bonded to which.
        """
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


def get_molecular_properties(mol):
    """
    return dictionary for properties for the following properties:

    - Atom type
    - Atomic number
    - Formal charge
    - Hybridization
    - Whether it is aromatic
    - Whether it is in a ring
    
    """
    property_dictionary = {}
    properties = ["atom_type", "atomic_num", "formal_charge", "hybridization", "is_aromatic", "is_in_ring"]
    property_values = []
    
    # Get the properties
    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        hybridization = atom.GetHybridization()
        is_aromatic = int(atom.GetIsAromatic())
        is_in_ring = int(atom.IsInRing())
        property_values = [atom_type, atomic_num, formal_charge, hybridization, is_aromatic, is_in_ring]
        
    for property, value in zip(properties, property_values):
        property_dictionary[property] = value
        
    return property_dictionary

    
def advanced_smiles_to_graph(smiles, bond_type_to_idx = bond_type_to_idx):
    """
    The simple representation above use only atom type and bond types, but real-world applications often need
    more sophisticated features.
    
    Convert a SMILES string to graph representation with advanced features.
    """
    if smiles is None:
        raise ValueError(f"Invalid SMILES string")

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()

    node_features = []
    #molecular_properties = get_molecular_properties(mol)
    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        hybridization = atom.GetHybridization()
        is_aromatic = int(atom.GetIsAromatic())
        is_in_ring = int(atom.IsInRing())
        logging.info(f"{atom_type} {atomic_num} {formal_charge} {hybridization} {is_aromatic} {is_in_ring}")
        # Create one-hot encoding for atom type (C, O, N, H, F, S, ..)
        atom_types = ['C', 'O', 'N', 'H', 'F', 'P', 'S', 'Cl', 'Br', 'I']
        atom_type_onehot = [1 if atom_type == t else 0 for t in atom_types]

        # If we don't have this atom type, then we need to account for this new atomic entry
        if atom_type not in atom_types:
            atom_type_onehot.append(1)
        else:
            atom_type_onehot.append(0)

        # One-hot for hybridization

        hybridization_types = [
            Chem.rdchem.HybridizationType.SP, # Which type of hybridization is this?
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3
        ]
        hybridization_onehot = [1 if hybridization == h else 0 for h in hybridization_types]
        logging.info(f"For the molecule we get the following hydbridizations {hybridization_onehot}")
        
        if hybridization not in hybridization_types:
            hybridization_onehot.append(1)  # "Other" hybridization
        else:
            hybridization_onehot.append(0)
            logging.info(f"one hot {atom_type_onehot}, hybridization {hybridization_onehot}")


        # Combine all features

        features = atom_type_onehot + [
            formal_charge,
            is_aromatic,
            is_in_ring,
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons()
            ] + hybridization_onehot

        node_features.append(features)
        logging.info(f"The final feature we get is {features}")
        

    # Convert to numpy array
    node_features = np.array(node_features)
    logging.info(f"The final node features we have combined is: {node_features}")


    # Creating the adjacency matrix and edge features

    adjacency = np.zeros((n_atoms, n_atoms))
    edge_features = []
    edge_indices = []

    for bond in mol.GetBonds():
        # Get the atoms in the bond
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        # Update adjacency matrix (symmetric)
        adjacency[begin_idx, end_idx] = 1
        adjacency[end_idx, begin_idx] = 1

        # Advanced bond features
        bond_type = bond.GetBondType()
        bond_type_onehot = np.zeros(len(bond_type_to_idx))
        if bond_type in bond_type_to_idx:
            bond_type_onehot[bond_type_to_idx[bond_type]] = 1

        is_conjugated = int(bond.GetIsConjugated())
        is_in_ring = int(bond.IsInRing())

        # Combine all bond features
        features = np.concatenate([bond_type_onehot, [is_conjugated, is_in_ring]])

        # Add edge in both directions (undirected graph)
        edge_features.append(features)
        edge_indices.append((begin_idx, end_idx))

        edge_features.append(features)  # Same feature for the reverse direction
        edge_indices.append((end_idx, begin_idx))

    # Convert edge features to numpy array
    if edge_features:
        edge_features = np.array(edge_features)
    else:
        edge_features = np.empty((0, len(bond_type_to_idx) + 2))  # +2 for conjugation and ring

    return node_features, adjacency, edge_features, edge_indices
    
    
def smiles_to_pytorch_graph(smiles):
    """
    Pytorch geometric for molecular graphs

    Now that we understand the fundamentals of graph representation for molecules, let's implement this using Pytorch Geometric (PyG),
    a library specifically designed for graph neural networks
    """
    # Get the graph representation
    node_features, adjacency, edge_features, edge_indices = advanced_smiles_to_graph(smiles)
    

if __name__ == "__main__":
    # Dictionary of molecules for molecular strings
    molecules = {
        "Methanol": "CO",
        "Ethanol": "CCO",
        "Benzene": "c1ccccc1",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O"
    }
    sample_molecule = "Aspirin"    
    aspirin_features, aspirin_adj, aspirin_edge_features, aspirin_edge_indices = advanced_smiles_to_graph(molecules[sample_molecule])
    print(f"features, adjacency matrix, edge_features, edge_indices are as follows: {aspirin_features} {aspirin_adj} {aspirin_edge_features} {aspirin_edge_indices}")


    # Writing code to create and visualize the graph representation for ethanol (CCO)

    
