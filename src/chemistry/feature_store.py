"""Reusable RDKit features for a small-molecule feature store.

The functions in this module deliberately keep *identity policy* separate from
parsing.  A canonical SMILES is unique for one RDKit molecular graph, but salts,
tautomers, protonation states, and unspecified stereochemistry may still encode
different graphs.  Callers must therefore choose the normalization appropriate
for their assay before using ``identity_smiles`` as a compound key.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Cluster import Butina
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker


class InvalidMoleculeError(ValueError):
    """Raised when RDKit cannot parse or sanitize a molecule."""


@dataclass(frozen=True)
class StandardizationOptions:
    """Controls which chemical distinctions are removed for identity matching."""

    largest_fragment: bool = True
    neutralize: bool = False
    canonical_tautomer: bool = False
    include_stereochemistry: bool = True


@dataclass(frozen=True)
class MoleculeFeatures:
    """Serializable scalar features and identifiers stored for one molecule."""

    input_smiles: str
    canonical_smiles: str
    identity_smiles: str
    scaffold_smiles: str
    molecular_weight: float
    logp: float
    tpsa: float
    h_bond_donors: int
    h_bond_acceptors: int
    rotatable_bonds: int
    ring_count: int
    heavy_atom_count: int
    formal_charge: int
    fraction_csp3: float


def parse_smiles(smiles: str, *, sanitize: bool = True) -> Chem.Mol:
    """Parse SMILES and raise a useful error instead of returning ``None``.

    Sanitisation checks valence, aromaticity, conjugation, and related chemical
    invariants.  ``sanitize=False`` is primarily useful for diagnosing bad input;
    most descriptor and fingerprint functions require a sanitised molecule.
    """

    if not isinstance(smiles, str) or not smiles.strip():
        raise InvalidMoleculeError("SMILES must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise InvalidMoleculeError(f"RDKit could not parse/sanitize SMILES: {smiles!r}")
    return mol


def canonical_smiles(smiles_or_mol: str | Chem.Mol, *, isomeric: bool = True) -> str:
    """Return RDKit canonical SMILES, optionally retaining stereochemistry."""

    mol = parse_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def standardize_molecule(
    smiles_or_mol: str | Chem.Mol,
    options: StandardizationOptions = StandardizationOptions(),
) -> Chem.Mol:
    """Clean a molecule and optionally remove salts, charge, and tautomer choice."""

    mol = parse_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else Chem.Mol(smiles_or_mol)
    try:
        mol = rdMolStandardize.Cleanup(mol)
        if options.largest_fragment:
            mol = rdMolStandardize.FragmentParent(mol)
        if options.neutralize:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        if options.canonical_tautomer:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise InvalidMoleculeError("RDKit could not standardize the molecule") from exc
    return mol


def identity_smiles(
    smiles_or_mol: str | Chem.Mol,
    options: StandardizationOptions = StandardizationOptions(),
) -> str:
    """Return the normalized canonical SMILES used as a feature-store key."""

    mol = standardize_molecule(smiles_or_mol, options)
    return canonical_smiles(mol, isomeric=options.include_stereochemistry)


def bemis_murcko_scaffold(smiles_or_mol: str | Chem.Mol) -> str:
    """Return the ring/linker Bemis–Murcko scaffold (empty for acyclic molecules)."""

    mol = parse_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return canonical_smiles(scaffold)


def molecular_descriptors(smiles_or_mol: str | Chem.Mol) -> dict[str, float | int]:
    """Calculate a compact, interpretable 2D descriptor set."""

    mol = parse_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "ring_count": Lipinski.RingCount(mol),
        "heavy_atom_count": Lipinski.HeavyAtomCount(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "fraction_csp3": Lipinski.FractionCSP3(mol),
    }


def morgan_fingerprint(
    smiles_or_mol: str | Chem.Mol, *, radius: int = 2, n_bits: int = 2048
) -> DataStructs.ExplicitBitVect:
    """Return an ECFP-like Morgan bit vector (radius 2 corresponds to ECFP4)."""

    if radius < 0 or n_bits <= 0:
        raise ValueError("radius must be non-negative and n_bits must be positive")
    mol = parse_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits).GetFingerprint(mol)


def tanimoto_similarity(left: str | Chem.Mol, right: str | Chem.Mol, **fp_kwargs: int) -> float:
    """Return the intersection-over-union similarity of two Morgan fingerprints."""

    return float(DataStructs.TanimotoSimilarity(
        morgan_fingerprint(left, **fp_kwargs), morgan_fingerprint(right, **fp_kwargs)
    ))


def cluster_molecules(
    molecules: Sequence[str | Chem.Mol], *, cutoff: float = 0.65, radius: int = 2, n_bits: int = 2048
) -> list[list[int]]:
    """Butina-cluster molecule indexes; members have similarity near ``cutoff``."""

    if not 0 <= cutoff <= 1:
        raise ValueError("cutoff must be between 0 and 1")
    fps = [morgan_fingerprint(mol, radius=radius, n_bits=n_bits) for mol in molecules]
    distances = [
        1.0 - similarity
        for i, fp in enumerate(fps)
        for similarity in DataStructs.BulkTanimotoSimilarity(fp, fps[:i])
    ]
    clusters = Butina.ClusterData(distances, len(fps), 1.0 - cutoff, isDistData=True)
    return [list(cluster) for cluster in clusters]


def select_diverse(
    molecules: Sequence[str | Chem.Mol], count: int, *, seed: int = 0, radius: int = 2, n_bits: int = 2048
) -> list[int]:
    """Select molecule indexes with greedy MaxMin fingerprint diversity."""

    if not 0 <= count <= len(molecules):
        raise ValueError("count must be between zero and the number of molecules")
    if count == 0:
        return []
    fps = [morgan_fingerprint(mol, radius=radius, n_bits=n_bits) for mol in molecules]
    picker = MaxMinPicker()
    return list(picker.LazyPick(
        lambda i, j: 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]),
        len(fps), count, seed=seed,
    ))


def featurize_smiles(
    smiles: str, options: StandardizationOptions = StandardizationOptions()
) -> MoleculeFeatures:
    """Build one feature-store record while preserving both raw and normalized IDs."""

    input_mol = parse_smiles(smiles)
    normalized = standardize_molecule(input_mol, options)
    values = molecular_descriptors(normalized)
    return MoleculeFeatures(
        input_smiles=smiles,
        canonical_smiles=canonical_smiles(input_mol),
        identity_smiles=canonical_smiles(normalized, isomeric=options.include_stereochemistry),
        scaffold_smiles=bemis_murcko_scaffold(normalized),
        **values,
    )


def featurize_many(
    smiles_values: Iterable[str], options: StandardizationOptions = StandardizationOptions(), *, skip_invalid: bool = False
) -> list[dict[str, object]]:
    """Create JSON/table-ready records, optionally dropping invalid rows."""

    records = []
    for smiles in smiles_values:
        try:
            records.append(asdict(featurize_smiles(smiles, options)))
        except InvalidMoleculeError:
            if not skip_invalid:
                raise
    return records
