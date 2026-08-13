"""Tests for the dependency-free validation side of the OpenMM runner."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "md/run_openmm.py"


def test_dry_run_writes_explicit_handoff_manifest(tmp_path: Path) -> None:
    prmtop, inpcrd = tmp_path / "site.prmtop", tmp_path / "site.inpcrd"
    prmtop.write_text("%VERSION placeholder-for-validation\n")
    inpcrd.write_text("placeholder-for-validation\n")
    output = tmp_path / "output"
    subprocess.run([sys.executable, str(RUNNER), "--prmtop", str(prmtop),
                    "--inpcrd", str(inpcrd), "--output-dir", str(output),
                    "--steps", "100", "--report-interval", "10", "--dry-run"], check=True)
    payload = json.loads((output / "run_manifest.json").read_text())
    assert payload["protocol"]["ensemble"] == "NPT"
    assert payload["protocol"]["seed"] == 20240813
    assert "no Zn model" in payload["scientific_scope"]
    assert not (output / "trajectory.dcd").exists()


def test_missing_topology_fails_before_creating_output(tmp_path: Path) -> None:
    inpcrd = tmp_path / "site.inpcrd"
    inpcrd.write_text("coordinates\n")
    output = tmp_path / "output"
    result = subprocess.run([sys.executable, str(RUNNER), "--prmtop",
                             str(tmp_path / "missing.prmtop"), "--inpcrd", str(inpcrd),
                             "--output-dir", str(output), "--dry-run"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "must be a non-empty file" in result.stderr
    assert not output.exists()
