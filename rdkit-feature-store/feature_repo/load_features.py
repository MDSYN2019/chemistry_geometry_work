"""Compute RDKit features and idempotently load the Feast offline table."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from datetime import datetime, timezone

import psycopg

from chemistry.feature_store import StandardizationOptions, featurize_smiles


DEFAULT_SMILES = ["CCO", "OCC", "CC(=O)[O-].[Na+]", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
COLUMNS = [
    "identity_smiles", "event_timestamp", "created_timestamp", "input_smiles",
    "canonical_smiles", "scaffold_smiles", "molecular_weight", "logp", "tpsa",
    "h_bond_donors", "h_bond_acceptors", "rotatable_bonds", "ring_count",
    "heavy_atom_count", "formal_charge", "fraction_csp3",
]


def read_smiles(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_SMILES
    with open(path, newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if "smiles" not in (rows.fieldnames or []):
            raise ValueError("CSV must contain a 'smiles' column")
        return [row["smiles"] for row in rows if row["smiles"].strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="CSV input with a smiles column; defaults to demo molecules")
    args = parser.parse_args()
    observed_at = datetime.now(timezone.utc)
    policy = StandardizationOptions(largest_fragment=True, canonical_tautomer=True)
    records = []
    for smiles in read_smiles(args.csv):
        record = asdict(featurize_smiles(smiles, policy))
        record.update(event_timestamp=observed_at, created_timestamp=observed_at)
        records.append(tuple(record[column] for column in COLUMNS))

    placeholders = ", ".join(["%s"] * len(COLUMNS))
    updates = ", ".join(f"{column} = EXCLUDED.{column}" for column in COLUMNS[2:])
    sql = f"""INSERT INTO public.molecule_features_v1 ({', '.join(COLUMNS)})
              VALUES ({placeholders})
              ON CONFLICT (identity_smiles, event_timestamp) DO UPDATE SET {updates}"""
    connection = psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "feast"),
        user=os.getenv("POSTGRES_USER", "feast"),
        password=os.getenv("POSTGRES_PASSWORD", "feast"),
    )
    with connection, connection.cursor() as cursor:
        cursor.executemany(sql, records)
    print(f"Loaded {len(records)} records with event time {observed_at.isoformat()}")


if __name__ == "__main__":
    main()
