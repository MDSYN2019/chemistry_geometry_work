CREATE SCHEMA IF NOT EXISTS feast_online;

CREATE TABLE IF NOT EXISTS public.molecule_features_v1 (
    identity_smiles TEXT NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    created_timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    input_smiles TEXT NOT NULL,
    canonical_smiles TEXT NOT NULL,
    scaffold_smiles TEXT NOT NULL,
    molecular_weight DOUBLE PRECISION NOT NULL,
    logp DOUBLE PRECISION NOT NULL,
    tpsa DOUBLE PRECISION NOT NULL,
    h_bond_donors BIGINT NOT NULL,
    h_bond_acceptors BIGINT NOT NULL,
    rotatable_bonds BIGINT NOT NULL,
    ring_count BIGINT NOT NULL,
    heavy_atom_count BIGINT NOT NULL,
    formal_charge BIGINT NOT NULL,
    fraction_csp3 REAL NOT NULL,
    PRIMARY KEY (identity_smiles, event_timestamp)
);

CREATE INDEX IF NOT EXISTS molecule_features_v1_event_time_idx
    ON public.molecule_features_v1 (event_timestamp);
