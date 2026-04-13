# Chemistry Neural-Network Exercises (PyTorch)

This exercise set follows your preferred progression:

1. MLP on fingerprints/descriptors (baseline)
2. Multitask MLP on related endpoints
3. GCN/MPNN on molecular graphs
4. LSTM on SMILES strings
5. 3D CNN on protein-ligand voxel grids

---

## Before you start: setup expectations

- Use reproducible seeds (`torch`, `numpy`, Python `random`).
- Report at least one baseline metric and one uncertainty/error analysis view.
- Always compare against a simple non-deep baseline (e.g., random forest) for chemistry tasks.

---

## Exercise 1 — Fingerprint MLP (single-task regression)

### Goal
Predict a scalar molecular property from fixed-size molecular fingerprints.

### Recommended dataset
- **Local option (already in this repo):** `src/data/esol/raw/delaney-processed.csv`
  - Typical target: aqueous solubility (`logS` style endpoint).

### What to implement
1. Build a dataset class from CSV.
2. Convert SMILES to fingerprints (e.g., ECFP/Morgan) or use precomputed descriptors.
3. Train an MLP with dropout + weight decay.
4. Evaluate with RMSE + MAE + calibration plot (predicted vs true).

### Starter model shape
- Input: 1024/2048 fingerprint bits
- Hidden: `[512, 256]`
- Output: `1`

### Deliverables
- Training script
- `results.json` with metrics
- Short note: where model underfits/overfits

---

## Exercise 2 — Multitask MLP (missing labels)

### Goal
Train a shared trunk with multiple task heads (or one multi-output head) and masked loss.

### Recommended datasets
- **Public benchmark option:** MoleculeNet multitask datasets (e.g., Tox21, SIDER).
- **Local project-style option:** create synthetic multitask labels from ESOL + auxiliary tasks (e.g., binned solubility class, scaffold flag).

### What to implement
1. `y` as shape `[batch, n_tasks]` and `mask` for missing labels.
2. Shared encoder MLP + output head(s).
3. Masked loss (`BCEWithLogitsLoss` or `MSELoss` per task with mask).
4. Per-task metrics + macro average.

### Deliverables
- Multitask training loop
- Per-task dashboard table
- Analysis: when multitask helps vs hurts single-task

---

## Exercise 3 — GCN/MPNN on molecular graphs

### Goal
Predict molecular properties directly from graph structure instead of fixed descriptors.

### Recommended datasets
- **Local option:** ESOL from `src/data/esol/raw/delaney-processed.csv`
- **Public option:** QM9 (for quantum properties), FreeSolv, Lipophilicity

### What to implement
1. Convert molecules into graph objects (`x`, `edge_index`, optional `edge_attr`).
2. Build 2-4 message-passing layers + global pooling.
3. Compare against Exercise 1 MLP baseline on the same split.
4. Add scaffold split to test generalization.

### Deliverables
- Graph featurization pipeline
- GCN/MPNN model
- Fair comparison report (same split and metric)

---

## Exercise 4 — SMILES LSTM baseline

### Goal
Treat molecules as token sequences and learn sequence-based representations.

### Recommended datasets
- ESOL (local) or any small property dataset with SMILES + scalar target

### What to implement
1. SMILES tokenizer (char-level is enough to start).
2. Embedding + LSTM encoder + regression/classification head.
3. Padding + packed sequence handling.
4. Error analysis by sequence length and token rarity.

### Deliverables
- Tokenizer/vocabulary stats
- LSTM training script
- Comparison vs MLP and GCN

---

## Exercise 5 — 3D CNN for structure-based scoring (advanced)

### Goal
Predict binding affinity or binder/non-binder from voxelized protein-ligand neighborhoods.

### Recommended datasets
- **Public options:** PDBbind-derived curated sets (small cleaned subsets for starter runs).
- **Practical note:** start with a tiny subset first due to preprocessing cost.

### What to implement
1. Voxelizer that converts atom environments to a 3D tensor.
2. Small 3D CNN + MLP head.
3. Data augmentation (random rotations/translations).
4. Benchmark against simple docking-score-only baseline.

### Deliverables
- Voxelization script
- 3D CNN training run on a toy subset
- Failure cases (pose sensitivity, noisy labels)

---

## Proposed capstone projects

## Project A — Baseline-to-Graph comparison on one endpoint

### Plan
1. Train fingerprint MLP (Exercise 1).
2. Train GCN/MPNN (Exercise 3) on identical train/val/test split.
3. Compare performance + inference speed + robustness to scaffold split.

### Why this is strong
- Gives a realistic chemistry ML workflow: simple baseline first, then graph model.
- Produces a clean story for interviews or research proposals.

---

## Project B — Multitask property panel

### Plan
1. Build multitask MLP with masking (Exercise 2).
2. Add uncertainty estimation (MC dropout or deep ensembles).
3. Produce a ranked candidate list with confidence filtering.

### Why this is strong
- Mirrors real medicinal chemistry settings with sparse labels across endpoints.

---

## Project C — Polymer extension track

### Plan
1. Start from local polymer archive: `src/data/neurips-open-polymer-prediction-2025.zip`.
2. Build a property predictor pipeline (tabular baseline -> GNN if graph data available).
3. Add domain split by chemistry family to evaluate out-of-domain behavior.

### Why this is strong
- Directly aligned with modern materials/polymer property prediction workflows.

---

## Sample dataset menu (what to use right now)

## Immediately available in this repo

1. `src/data/esol/raw/delaney-processed.csv`
   - Best for Exercises 1, 3, and 4.
2. `src/data/neurips-open-polymer-prediction-2025.zip`
   - Best for Project C and advanced custom tasks.

## Good external datasets to add next

1. **MoleculeNet**: ESOL, FreeSolv, Lipophilicity, Tox21, SIDER, ClinTox, HIV
2. **QM9**: quantum property regression
3. **PDBbind (curated subset)**: 3D protein-ligand scoring

---

## Suggested 4-week execution plan

- **Week 1:** Exercise 1 (MLP baseline) + robust evaluation template
- **Week 2:** Exercise 3 (GCN/MPNN) + apples-to-apples comparison
- **Week 3:** Exercise 2 (multitask) + missing-label masking
- **Week 4:** Exercise 4 (SMILES LSTM) or small Exercise 5 prototype

---

## Minimal success criteria per exercise

- Reproducible run (fixed seeds)
- Clear split strategy (random and scaffold if possible)
- At least one ablation (depth, hidden size, dropout, or featurization)
- Save model checkpoints and prediction CSVs
- 5-10 bullet error analysis
