# ALDC Cl → Br relative-binding study

> **The purpose of this project is not to manufacture an exact reconstruction of missing historical calculations. It is to reconstruct the historical ALDC Cl/Br experiment as a transparent modern physics-based computational study.**

## Scope and current status

This repository is a staged reconstruction of a proposed relative binding free-energy
(RBFE) study involving acetolactate decarboxylase (ALDC). **Only milestone M0 is
implemented.** It inventories the supplied crystallographic model and measures its
chain-A Zn/EDO geometry. No Zn parameters, ligand poses, molecular dynamics, quantum
chemistry, or free-energy values have been generated. M1 must not start until this
structural baseline is reviewed.

The primary scientific specification is the separately supplied
`ALDC_Cl_Br_RBFE_Reproduction_Guide.pdf`. It is intentionally not committed because
this source repository does not track binary files. Place a local copy under `docs/`
when consulting the specification; the path is ignored by Git. The preserved source coordinates are
[`data/raw/start2_refmac1.pdb`](data/raw/start2_refmac1.pdb); generated reports never
overwrite them.

## Historical context, evidence, and reconstruction

ALDC is a Zn-containing enzyme, and the supplied refined structure is the concrete
experimental starting point for this project. The PDB contains three protein chains,
Zn and Cd atoms, crystallographic waters, and EDO (ethylene glycol) at the reference
active sites. In chain A the specification-designated site comprises His222, His224,
His235, Glu281, Zn301, and EDO401.

Important evidence boundaries are maintained throughout:

- The structure contains **EDO**, not either proposed halogenated ligand.
- The historical chlorine compound's exact identity is reconstructed, not conclusively
  established. The working Cl and Br SMILES belong to a later ligand-building milestone.
- Any future Cl/Br pose will be a hypothesis, not a crystallographically observed pose.
- M0 contact distances are geometric observations. Listing them does not itself assert
  a coordination bond, protonation state, or parameterization.
- No missing historical observable or computational result is inferred.

## Scientific question and thermodynamic cycle

The eventual, explicitly gated question is the relative binding of reconstructed methyl
3-chloro-3-methyl-2-oxobutanoate and methyl 3-bromo-3-methyl-2-oxobutanoate. A classical
cycle would use the same mapped Cl → Br transformation in protein complex and solvent:

```text
ΔΔG_bind = ΔG_complex^(Cl→Br) − ΔG_solvent^(Cl→Br)
```

That quantity must not be computed or reported until validated Cl/Br endpoints and the
Zn site pass their milestone criteria. Net charge should remain unchanged by the
halogen transformation.

## Why Zn²⁺ is the central modelling problem

A metalloprotein first shell is not reliably represented by blindly applying ordinary
protein preparation or treating Zn²⁺ as an unvalidated naked +2 point charge. Metal
coordination, ligand protonation, oligomeric state, waters, and force-field choices are
coupled scientific decisions. The intended primary route therefore validates a
Zn/EDO Amber/MCPB.py reference before endpoint or alchemical work. An OpenMM/OpenFE
route is secondary and cannot substitute default preparation for that validation.
M0 supplies only the coordinate-derived reference target.

## M0 measured chain-A geometry

Distances are calculated from Cartesian coordinates, not copied from `LINK` records.
Values below are rounded to three decimal places, as in the machine-readable reports.

| Zn atom | Reference atom | Distance (Å) |
|---|---|---:|
| A:ZN301 | A:HIS222 NE2 | 2.328 |
| A:ZN301 | A:HIS224 NE2 | 2.190 |
| A:ZN301 | A:HIS235 ND1 | 2.364 |
| A:ZN301 | A:GLU281 OE1 | 2.558 |
| A:ZN301 | A:EDO401 O1 | 2.305 |
| A:ZN301 | A:EDO401 O2 | 1.920 |

The inventory recovers chains A, B, C, and the water chain S; 3 Zn atoms; 4 Cd atoms;
3 EDO residues; and 782 HOH residues. Each protein chain spans residues 47–285 (239
observed residue numbers). Protein residue ranges/counts are included per chain in JSON;
the solvent-only S chain is not misrepresented as protein.

## Installation

The inspector uses only Python's standard library. Tests require pytest.

```bash
conda env create -f environment.yml
conda activate aldc-cl-br-rbfe
# or, in an existing Python >=3.10 environment:
python -m pip install -e '.[test]'
```

## Reproduction commands

Run from this directory:

```bash
make inspect
make test
```

Equivalently, the exact acceptance command is:

```bash
python structure/inspect_pdb.py data/raw/start2_refmac1.pdb
```

It deterministically rewrites:

- `structure/reports/active_site.json` — full inventory, source SHA-256, and contacts;
- `structure/reports/active_site.csv` — tabular chain-A contact distances;
- `structure/reports/active_site.pml` — PyMOL active-site display and distance objects.

Open the visualization from the repository root with, for example,
`pymol structure/reports/active_site.pml`. Future Make targets exist but deliberately
fail with a milestone-gating message.

## Repository layout

- `data/raw/`: immutable uploaded coordinates.
- `docs/`: location for the separately supplied, Git-ignored scientific specification.
- `structure/`: M0 parser and generated structural reports.
- `tests/`: fast structural regression and CLI tests.
- `ligands/`, `metal_site/`, `systems/`, `md/`, `rbfe/`, `analysis/`: scaffolded,
  intentionally empty future-milestone areas.

## Validation criteria and milestone sequence

- **M0 (complete here):** reproducibly recover and report the raw Zn/EDO site.
- **M1:** stable, chemically defensible Zn/EDO reference without arbitrary heavy restraints.
- **M2:** stable, plausible matched Cl/Br endpoint pair across replicas.
- **M3:** converged solvent transformation with overlap and repeat diagnostics.
- **M4:** converged complex transformation with defensible Zn geometry at every lambda.
- **M5:** only then calculate the cycle with uncertainty and convergence analysis.
- **M6:** optional, separate QM/MM catalytic-barrier design.

## Assumptions, limitations, and unresolved decisions

M0 assumes fixed-width PDB coordinate records and uses the exact chain/residue/atom
identifiers required by the specification. It does not infer alternate locations,
chemical bonds, biological assembly, protonation, or oxidation state from distances.
The supplied coordinate header is minimally identifying; historical identity should not
be embellished beyond the supplied evidence.

Before M1, review is required to decide and document the biologically appropriate
oligomer/assembly, protonation and EDO treatment, Zn model and QM method/software, and
whether the specification-listed first-shell interpretation is chemically adequate.
No QM output may be fabricated when software or calculations are unavailable.

## Binding affinity is not enzyme activity

Classical RBFE estimates a relative equilibrium **binding free energy**. With suitable
experimental definitions and assumptions, it may be compared to relative `Kd` or `Ki`.
It does **not** predict `kcat`, catalytic turnover, or a reaction barrier. If the
historical observation concerns turnover, comparing activation free energies for Cl and
Br would require a separately scoped, validated QM/MM project after the binding work;
it is not part of M0.

## Exact next command

After scientific review and explicit approval to begin M1, the first reproducibility
check remains:

```bash
make inspect && make test
```

Do not begin Zn parameterization before that review.
