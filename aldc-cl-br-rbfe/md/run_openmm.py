#!/usr/bin/env python3
"""Run OpenMM from an MCPB.py Amber handoff without guessing chemistry."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence

LOG = logging.getLogger("aldc.openmm")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Run OpenMM using an MCPB.py-derived Amber pair.")
    command.add_argument("--prmtop", type=Path, required=True)
    command.add_argument("--inpcrd", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--steps", type=positive_int, default=500_000)
    command.add_argument("--report-interval", type=positive_int, default=5_000)
    command.add_argument("--temperature-k", type=float, default=300.0)
    command.add_argument("--pressure-bar", type=float, default=1.0)
    command.add_argument("--timestep-fs", type=float, default=2.0)
    command.add_argument("--platform", help="OpenMM platform name, for example CPU or CUDA")
    command.add_argument("--seed", type=int, default=20240813)
    command.add_argument("--dry-run", action="store_true",
                         help="validate and write the manifest without importing OpenMM")
    return command


def validate(args: argparse.Namespace) -> None:
    for label in ("prmtop", "inpcrd"):
        path = getattr(args, label)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"--{label} must be a non-empty file: {path}")
    if args.temperature_k <= 0 or args.pressure_bar <= 0 or args.timestep_fs <= 0:
        raise ValueError("temperature, pressure, and timestep must be positive")
    if args.report_interval > args.steps:
        raise ValueError("--report-interval cannot exceed --steps")


def manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "inputs": {"prmtop": str(args.prmtop.resolve()), "inpcrd": str(args.inpcrd.resolve())},
        "protocol": {
            "ensemble": "NPT", "steps": args.steps, "report_interval": args.report_interval,
            "temperature_K": args.temperature_k, "pressure_bar": args.pressure_bar,
            "timestep_fs": args.timestep_fs, "friction_per_ps": 1.0,
            "nonbonded_method": "PME", "cutoff_nm": 1.0, "constraints": "HBonds",
            "seed": args.seed, "platform": args.platform,
        },
        "scientific_scope": "Uses supplied topology chemistry; no Zn model or protonation is inferred.",
    }


def summarize_state_csv(path: Path) -> dict[str, Any]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    return {"samples": len(rows), "columns": list(rows[0]) if rows else [],
            "complete": bool(rows),
            "note": "Thermodynamic trace only; inspect Zn geometry and stability separately."}


def run(args: argparse.Namespace) -> None:
    # Lazy loading keeps validation useful on preparation hosts without OpenMM.
    app = importlib.import_module("openmm.app")
    mm = importlib.import_module("openmm")
    unit = importlib.import_module("openmm.unit")
    topology = app.AmberPrmtopFile(str(args.prmtop))
    coordinates = app.AmberInpcrdFile(str(args.inpcrd))
    system = topology.createSystem(nonbondedMethod=app.PME,
                                   nonbondedCutoff=1.0 * unit.nanometer,
                                   constraints=app.HBonds)
    system.addForce(mm.MonteCarloBarostat(args.pressure_bar * unit.bar,
                                          args.temperature_k * unit.kelvin))
    integrator = mm.LangevinMiddleIntegrator(args.temperature_k * unit.kelvin,
                                             1.0 / unit.picosecond,
                                             args.timestep_fs * unit.femtoseconds)
    integrator.setRandomNumberSeed(args.seed)
    platform = mm.Platform.getPlatformByName(args.platform) if args.platform else None
    properties = {"Precision": "mixed"} if args.platform and args.platform.upper() in {"CUDA", "OPENCL"} else {}
    simulation = app.Simulation(topology.topology, system, integrator, platform, properties)
    simulation.context.setPositions(coordinates.positions)
    if coordinates.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*coordinates.boxVectors)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(args.temperature_k * unit.kelvin, args.seed)
    state_path = args.output_dir / "state.csv"
    simulation.reporters.append(app.DCDReporter(str(args.output_dir / "trajectory.dcd"),
                                                 args.report_interval))
    simulation.reporters.append(app.StateDataReporter(
        str(state_path), args.report_interval, step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, temperature=True, density=True, volume=True, speed=True))
    LOG.info("Running %d NPT steps", args.steps)
    simulation.step(args.steps)
    simulation.saveCheckpoint(str(args.output_dir / "final.chk"))
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    with (args.output_dir / "final.pdb").open("w") as handle:
        app.PDBFile.writeFile(topology.topology, state.getPositions(), handle)
    (args.output_dir / "qc.json").write_text(json.dumps(summarize_state_csv(state_path), indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        validate(args)
    except ValueError as exc:
        parser().error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest(args), indent=2) + "\n")
    if args.dry_run:
        LOG.info("Handoff valid; dry run wrote run_manifest.json")
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
