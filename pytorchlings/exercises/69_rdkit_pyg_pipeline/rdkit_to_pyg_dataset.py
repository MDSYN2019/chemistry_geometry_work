"""Exercise 69: molecule files -> RDKit featurization -> PyG Data objects.

This exercise is intentionally dependency-heavy and mirrors realistic chemistry ML preprocessing.
"""

from typing import Iterable

import torch

try:
    from rdkit import Chem
    from torch_geometric.data import Data
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Install rdkit and torch-geometric to run this exercise."
    ) from exc


ATOM_SET = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]


def atom_features(atom: Chem.Atom) -> list[float]:
    """Simple handcrafted atom features for baseline GNNs."""
    one_hot = [float(atom.GetSymbol() == sym) for sym in ATOM_SET]
    numeric = [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
    ]
    return one_hot + numeric


def smiles_to_data(smiles: str, y_value: float) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)

    edge_pairs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_pairs.extend([(i, j), (j, i)])

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    # TODO: add richer edge features (bond type, conjugation, stereo)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([y_value], dtype=torch.float32))


def build_dataset(rows: Iterable[tuple[str, float]]) -> list[Data]:
    return [smiles_to_data(smiles, y) for smiles, y in rows]


if __name__ == "__main__":
    rows = [("CCO", 0.1), ("c1ccccc1", 0.5), ("CC(=O)O", -0.2)]
    dataset = build_dataset(rows)
    assert len(dataset) == 3
    assert dataset[0].x.ndim == 2
    print("exercise 69 smoke check passed")
