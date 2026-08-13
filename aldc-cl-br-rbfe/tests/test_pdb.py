"""Regression and unit tests for the dependency-free M0 PDB inspector."""
from pathlib import Path
import csv
import json
import subprocess
import sys

import pytest
from structure.inspect_pdb import inspect, parse_pdb

ROOT = Path(__file__).parents[1]
PDB = ROOT / "data/raw/start2_refmac1.pdb"
EXPECTED = {
    ("HIS222", "NE2"): 2.328,
    ("HIS224", "NE2"): 2.190,
    ("HIS235", "ND1"): 2.364,
    ("GLU281", "OE1"): 2.558,
    ("EDO401", "O1"): 2.305,
    ("EDO401", "O2"): 1.920,
}

def test_uploaded_structure_inventory_and_numbering() -> None:
    report = inspect(parse_pdb(PDB), PDB)
    assert set(report["chains"]) == {"A", "B", "C", "S"}
    assert report["chains"]["A"] == {"protein_residue_min": 47,
                                      "protein_residue_max": 285,
                                      "protein_residue_count": 239}
    assert report["chains"]["S"] == {"protein_residue_min": None,
                                      "protein_residue_max": None,
                                      "protein_residue_count": 0}
    assert {(x["chain"], x["residue_number"]) for x in report["zn_atoms"]} == {
        ("A", 301), ("B", 301), ("C", 301)}
    assert len(report["cd_atoms"]) == 4
    assert report["edo_molecules"] == [{"chain": c, "residue": 401} for c in "ABC"]
    assert report["crystallographic_waters"]["count"] == 782
    observed = {(x["residue"], x["atom"]): x["distance_A"] for x in report["coordination"]}
    assert observed == EXPECTED

def test_cli_writes_machine_readable_and_pymol_reports(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, str(ROOT / "structure/inspect_pdb.py"),
                             str(PDB), "--output-dir", str(tmp_path)],
                            check=True, capture_output=True, text=True)
    payload = json.loads((tmp_path / "active_site.json").read_text())
    with (tmp_path / "active_site.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    pymol = (tmp_path / "active_site.pml").read_text()
    assert payload["source_sha256"] and len(payload["source_sha256"]) == 64
    assert len(rows) == 6
    assert pymol.count("distance zn_") == 6
    assert "Found 3 Zn, 4 Cd, 3 EDO, and 782 waters" in result.stderr

def test_empty_pdb_has_readable_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdb"
    empty.write_text("HEADER EMPTY\n")
    with pytest.raises(ValueError, match="No ATOM or HETATM"):
        parse_pdb(empty)
