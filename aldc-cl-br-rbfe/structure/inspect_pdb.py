#!/usr/bin/env python3
"""Inspect an ALDC PDB and reproducibly report its crystallographic metal site.

Coordinates, rather than LINK records, are the source for all reported distances.
This deliberately dependency-free M0 tool parses fixed-width ATOM/HETATM records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

LOGGER = logging.getLogger("inspect_pdb")
AMINO_ACIDS = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})
TARGETS = (
    ("HIS", 222, "NE2"),
    ("HIS", 224, "NE2"),
    ("HIS", 235, "ND1"),
    ("GLU", 281, "OE1"),
    ("EDO", 401, "O1"),
    ("EDO", 401, "O2"),
)

@dataclass(frozen=True)
class Atom:
    """The subset of a PDB atom record required for M0."""
    record: str
    serial: int
    name: str
    residue_name: str
    chain: str
    residue_number: int
    insertion_code: str
    x: float
    y: float
    z: float
    element: str


def parse_pdb(path: Path) -> list[Atom]:
    """Parse fixed-column coordinate records from *path*.

    Raises a useful error for absent files or coordinate records; raw input is
    opened read-only and is never modified.
    """
    atoms: list[Atom] = []
    with path.open(encoding="ascii", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            record = line[0:6].strip()
            if record not in {"ATOM", "HETATM"}:
                continue
            try:
                atoms.append(Atom(record, int(line[6:11]), line[12:16].strip(),
                    line[17:20].strip(), line[21:22].strip(), int(line[22:26]),
                    line[26:27].strip(), float(line[30:38]), float(line[38:46]),
                    float(line[46:54]), line[76:78].strip()))
            except ValueError as exc:
                raise ValueError(f"Malformed coordinate record at {path}:{line_number}") from exc
    if not atoms:
        raise ValueError(f"No ATOM or HETATM records found in {path}")
    return atoms


def distance(first: Atom, second: Atom) -> float:
    """Return the Euclidean atom distance in ångströms."""
    return math.dist((first.x, first.y, first.z), (second.x, second.y, second.z))


def _one(atoms: Iterable[Atom], *, chain: str, residue_name: str,
         residue_number: int, atom_name: str | None = None) -> Atom:
    matches = [a for a in atoms if a.chain == chain and a.residue_name == residue_name
               and a.residue_number == residue_number
               and (atom_name is None or a.name == atom_name)]
    if len(matches) != 1:
        label = f"{chain}:{residue_name}{residue_number}:{atom_name or '*'}"
        raise ValueError(f"Expected exactly one {label} atom; found {len(matches)}")
    return matches[0]


def inspect(atoms: Sequence[Atom], source: Path) -> dict[str, object]:
    """Validate the chain-A reference site and construct a report dictionary."""
    zn = _one(atoms, chain="A", residue_name="ZN", residue_number=301)
    coordination = []
    for residue_name, residue_number, atom_name in TARGETS:
        atom = _one(atoms, chain="A", residue_name=residue_name,
                    residue_number=residue_number, atom_name=atom_name)
        coordination.append({"residue": f"{residue_name}{residue_number}",
                             "atom": atom_name, "distance_A": round(distance(zn, atom), 3)})
    chains: dict[str, dict[str, int | None]] = {}
    for chain in sorted({a.chain for a in atoms}):
        # This legacy file labels protein, metals, EDO, and HOH alike as ATOM.
        # Residue chemistry, rather than record type, therefore distinguishes
        # the protein range from solvent and heterogens.
        protein_numbers = sorted({a.residue_number for a in atoms
                                  if a.chain == chain and a.residue_name in AMINO_ACIDS})
        chains[chain or "(blank)"] = {
            "protein_residue_min": protein_numbers[0] if protein_numbers else None,
            "protein_residue_max": protein_numbers[-1] if protein_numbers else None,
            "protein_residue_count": len(protein_numbers),
        }
    sites = lambda name: [asdict(a) for a in atoms if a.residue_name == name]
    residues = lambda name: sorted({(a.chain, a.residue_number) for a in atoms
                                    if a.residue_name == name})
    return {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "chains": chains,
        "zn_atoms": sites("ZN"),
        "cd_atoms": sites("CD"),
        "edo_molecules": [{"chain": c, "residue": n} for c, n in residues("EDO")],
        "crystallographic_waters": {
            "residue_name": "HOH", "count": len(residues("HOH")),
            "residues": [{"chain": c, "residue": n} for c, n in residues("HOH")],
        },
        "metal": "ZN", "chain": "A", "residue": 301,
        "coordination": coordination,
        "notes": ["Distances were calculated directly from Cartesian coordinates, not LINK records.",
                  "Coordination entries are the specification-defined chain-A reference contacts; no bonding assignment is inferred."],
    }


def write_reports(report: dict[str, object], output_dir: Path, pdb_path: Path) -> None:
    """Write deterministic JSON, CSV, and PyMOL reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "active_site.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "active_site.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metal", "chain", "metal_residue",
                                                     "residue", "atom", "distance_A"))
        writer.writeheader()
        for contact in report["coordination"]:  # type: ignore[union-attr]
            writer.writerow({"metal": "ZN", "chain": "A", "metal_residue": 301, **contact})
    relative_pdb = Path("../../data/raw") / pdb_path.name
    lines = [f"load {relative_pdb.as_posix()}, aldc", "hide everything", "show cartoon, polymer",
             "select active_site, chain A and (resi 222+224+235+281+301+401)",
             "show sticks, chain A and (resi 222+224+235+281+401)",
             "show spheres, chain A and resn ZN and resi 301", "color gray70, elem C",
             "color marine, elem N", "color red, elem O", "color slate, resn ZN",
             "set sphere_scale, 0.45, resn ZN"]
    for contact in report["coordination"]:  # type: ignore[union-attr]
        resname = contact["residue"][:3]
        resnum = contact["residue"][3:]
        label = f"zn_{resname.lower()}{resnum}_{contact['atom'].lower()}"
        lines.append(f"distance {label}, chain A and resn ZN and resi 301, "
                     f"chain A and resn {resname} and resi {resnum} and name {contact['atom']}")
    lines += ["set dash_color, yellow", "set dash_width, 2.5", "zoom active_site, 8"]
    (output_dir / "active_site.pml").write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the M0 command-line workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", type=Path, help="raw PDB coordinate file")
    parser.add_argument("--output-dir", type=Path, default=Path("structure/reports"))
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    report = inspect(parse_pdb(args.pdb), args.pdb)
    write_reports(report, args.output_dir, args.pdb)
    LOGGER.info("Found chains: %s", ", ".join(report["chains"]))
    LOGGER.info("Found %d Zn, %d Cd, %d EDO, and %d waters", len(report["zn_atoms"]),
                len(report["cd_atoms"]), len(report["edo_molecules"]),
                report["crystallographic_waters"]["count"])
    for item in report["coordination"]:
        LOGGER.info("Zn301–%s %s: %.3f Å", item["residue"], item["atom"], item["distance_A"])
    LOGGER.info("Wrote M0 reports to %s", args.output_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
