# What to expect from Boltz-2

## Typical artifacts

A successful prediction normally produces:

- one or more coordinate models, commonly in mmCIF (`.cif`) format;
- confidence JSON files containing global and/or per-chain confidence values;
- predicted alignment-error or related confidence arrays/visualizations,
  depending on the installed release and options; and
- affinity output for an input that contains an `affinity` property.

Exact names and JSON schemas are version-dependent. Use
`python inspect_results.py <result-directory>` to inventory a run, then inspect
the original JSON rather than coding against the utility's human-readable text.

## Expected outcome by exercise

### 01: monomer

Expect one protein chain and no ligand or affinity result. The compact example
often yields an apparently folded model, but the useful lesson is to inspect
local confidence: termini and flexible loops may be less certain than the core.

### 02: protein complex

Expect two protein chains in the same coordinate model. A model can have
reasonable individual chains while their relative orientation remains
uncertain. Inspect interface-focused confidence and predicted alignment error;
do not infer an interaction solely because the chains touch.

### 03: protein--ligand affinity

Expect a protein chain, a ligand chain generated from SMILES, and affinity
output in addition to structural confidence. This is a deliberately artificial
pair, so there is no expected experimental affinity or correct binding pose.
The exercise succeeds when the pipeline runs and you can distinguish:

1. **pose confidence** -- whether the model is confident about the geometry;
2. **affinity prediction** -- the model's estimate for binding; and
3. **experimental evidence** -- which these examples do not provide.

## Sanity checklist

- Confirm all requested chain IDs exist in the coordinate file.
- Check for broken chains, severe steric clashes, and implausible ligand bonds.
- Review per-chain and interface confidence, not only a global rank.
- Compare multiple samples/seeds when the decision matters.
- Record software version, input YAML, command, and model/ranking used.
- Treat affinity and confidence as model outputs with uncertainty, never as
  substitutes for assay data.
