"""Feast definitions for version 1 of the RDKit scalar feature set."""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource
from feast.types import Float32, Float64, Int64, String


molecule = Entity(
    name="molecule",
    join_keys=["identity_smiles"],
    description="Molecule identity after the versioned RDKit normalization policy",
)

molecule_source = PostgreSQLSource(
    name="molecule_features_v1_source",
    table="public.molecule_features_v1",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

molecule_features_v1 = FeatureView(
    name="molecule_features_v1",
    entities=[molecule],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="canonical_smiles", dtype=String),
        Field(name="scaffold_smiles", dtype=String),
        Field(name="molecular_weight", dtype=Float64),
        Field(name="logp", dtype=Float64),
        Field(name="tpsa", dtype=Float64),
        Field(name="h_bond_donors", dtype=Int64),
        Field(name="h_bond_acceptors", dtype=Int64),
        Field(name="rotatable_bonds", dtype=Int64),
        Field(name="ring_count", dtype=Int64),
        Field(name="heavy_atom_count", dtype=Int64),
        Field(name="formal_charge", dtype=Int64),
        Field(name="fraction_csp3", dtype=Float32),
    ],
    source=molecule_source,
    online=True,
    tags={"feature_set_version": "v1", "generator": "rdkit"},
)

molecule_model_v1 = FeatureService(
    name="molecule_model_v1",
    features=[molecule_features_v1],
    tags={"normalization_policy": "largest-fragment,charged,canonical-tautomer,stereo"},
)
