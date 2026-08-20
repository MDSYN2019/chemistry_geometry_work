import pytest

rdkit = pytest.importorskip("rdkit")

from chemistry.feature_store import (  # noqa: E402
    InvalidMoleculeError,
    StandardizationOptions,
    bemis_murcko_scaffold,
    canonical_smiles,
    cluster_molecules,
    featurize_many,
    featurize_smiles,
    identity_smiles,
    select_diverse,
    tanimoto_similarity,
)


def test_equivalent_smiles_have_same_canonical_identity():
    assert canonical_smiles("CCO") == canonical_smiles("OCC")


def test_identity_policy_handles_salts_charge_tautomers_and_stereo():
    assert identity_smiles("CC(=O)[O-].[Na+]") == "CC(=O)[O-]"
    assert identity_smiles("CC(=O)[O-]", StandardizationOptions(neutralize=True)) == "CC(=O)O"
    tautomer_options = StandardizationOptions(canonical_tautomer=True)
    assert identity_smiles("O=c1cccc[nH]1", tautomer_options) == identity_smiles("Oc1ccccn1", tautomer_options)
    assert identity_smiles("C[C@H](O)F") != identity_smiles("C[C@@H](O)F")
    no_stereo = StandardizationOptions(include_stereochemistry=False)
    assert identity_smiles("C[C@H](O)F", no_stereo) == identity_smiles("C[C@@H](O)F", no_stereo)


def test_invalid_smiles_is_explicit_and_batch_can_skip_it():
    with pytest.raises(InvalidMoleculeError):
        featurize_smiles("C1CC")
    assert len(featurize_many(["CCO", "C1CC"], skip_invalid=True)) == 1


def test_features_scaffold_and_similarity():
    aspirin = featurize_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert aspirin.formal_charge == 0
    assert aspirin.molecular_weight == pytest.approx(180.159, abs=0.01)
    assert bemis_murcko_scaffold("CCOc1ccccc1") == "c1ccccc1"
    assert tanimoto_similarity("CCO", "OCC") == 1.0
    assert tanimoto_similarity("CCO", "c1ccccc1") < 0.2


def test_clustering_and_diversity_return_source_indexes():
    molecules = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccncc1"]
    clusters = cluster_molecules(molecules, cutoff=0.4)
    assert sorted(index for cluster in clusters for index in cluster) == list(range(5))
    selected = select_diverse(molecules, 3, seed=7)
    assert len(selected) == len(set(selected)) == 3
