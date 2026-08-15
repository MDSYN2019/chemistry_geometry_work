# Boltz-2 practice lab

This folder is a small, safe place to learn the Boltz-2 prediction workflow.
The examples progress from one protein chain to a protein complex and then to a
protein--ligand affinity request. They are deliberately small enough for
practice; they are **not** validated biological hypotheses.

## 1. Set up Boltz

Use a fresh environment so that model dependencies do not alter the rest of
this repository:

```bash
python -m venv .venv-boltz2
source .venv-boltz2/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade boltz
boltz --help
```

Model inference is substantially faster on a supported GPU. The first run also
downloads model weights. If installation or command-line flags differ in the
version you installed, treat `boltz predict --help` as authoritative.

## 2. Run the examples

Start with the monomer, then work through the other inputs:

```bash
boltz predict inputs/01_monomer.yaml --use_msa_server --out_dir results/01_monomer
boltz predict inputs/02_protein_complex.yaml --use_msa_server --out_dir results/02_complex
boltz predict inputs/03_protein_ligand_affinity.yaml --use_msa_server --out_dir results/03_affinity
```

`--use_msa_server` sends protein sequences to a remote MSA service. Omit it if
the sequence must remain private and instead follow the installed Boltz
documentation to provide local MSAs. Never submit confidential sequences to a
third-party service without approval.

Boltz may create an extra input-named directory below `--out_dir`. File names
and JSON fields can change between releases. Locate and summarize the useful
artifacts without assuming an exact version-specific layout:

```bash
find results -type f | sort
python inspect_results.py results/03_affinity
```

## 3. What each example teaches

| Input | Concept | Inspect first |
| --- | --- | --- |
| `01_monomer.yaml` | Basic sequence-to-structure prediction | Chain continuity and confidence |
| `02_protein_complex.yaml` | Multiple chains and a predicted interface | Relative chain placement and interface confidence |
| `03_protein_ligand_affinity.yaml` | A SMILES ligand plus an affinity request | Ligand pose, clashes, confidence, and affinity JSON |

The amino-acid sequences and ligand in this lab are toy fixtures. A plausible
picture does not establish binding, affinity, or biological function.

## 4. Read the result correctly

Read [`EXPECTED_RESULTS.md`](EXPECTED_RESULTS.md) before interpreting a run.
In brief, expect a predicted coordinate file (usually mmCIF), confidence data,
and, for the last example, affinity-related output. Confidence is not an
experimental measurement. Compare predictions with known structures or assays
before drawing scientific conclusions.

## 5. Practice tasks

1. Run the monomer and color the structure by per-residue confidence in a
   molecular viewer.
2. Run the two-chain input. Identify residues within 5 Å across the predicted
   interface and decide whether the interface confidence supports the pose.
3. Run the ligand example twice with different random seeds (see your installed
   CLI help). Compare the ligand poses rather than trusting only the top rank.
4. Replace the toy ligand with a ligand represented as a valid SMILES string.
   Check protonation, stereochemistry, and tautomer choice before prediction.
5. Replace a protein sequence with one of your own, record the Boltz version and
   command, and keep the unedited input beside the results for reproducibility.

## Suggested folder hygiene

Generated predictions can be large and are ignored by this lab's `.gitignore`.
Keep only small inputs, notes, and selected figures in version control; keep
weights and bulk prediction outputs elsewhere.
