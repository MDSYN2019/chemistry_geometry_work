# RDKit small-molecule feature store

The implementation lives in [`src/chemistry/feature_store.py`](../src/chemistry/feature_store.py). It returns ordinary dataclasses, dictionaries, and RDKit bit vectors so a caller can write the results to a dataframe, database, or online feature store without coupling chemistry policy to a storage vendor.

## What was added

| Capability | API | Example | What it does |
|---|---|---|---|
| Parse and sanitise | `parse_smiles("CCO")` | invalid `C1CC` raises `InvalidMoleculeError` | Parses the graph and checks valence, aromaticity, and other invariants rather than allowing RDKit's `None` to fail later. |
| Canonical SMILES | `canonical_smiles("OCC") -> "CCO"` | `OCC` and `CCO` | Produces one deterministic serialization of the *same molecular graph*. It is not, by itself, a universal compound identifier. |
| Stereochemistry | `canonical_smiles(..., isomeric=True)` | `C[C@H](O)F` differs from `C[C@@H](O)F` | Retains defined tetrahedral stereochemistry; disabling it intentionally collapses stereoisomers. |
| Salt/mixture handling | `StandardizationOptions(largest_fragment=True)` | `CC(=O)[O-].[Na+] -> CC(=O)[O-]` | Selects RDKit's parent fragment. Do not use it when every mixture component is scientifically meaningful. |
| Charge/protonation policy | `StandardizationOptions(neutralize=True)` | acetate becomes acetic acid | Optionally removes neutralisable formal charges. The default preserves charge because protonation may affect activity and descriptors. |
| Tautomers | `StandardizationOptions(canonical_tautomer=True)` | 2-pyridone and 2-hydroxypyridine converge | Optionally maps tautomeric graphs to RDKit's canonical tautomer. |
| Identity key | `identity_smiles(smiles, options)` | use as a deduplication key | Applies the selected salt/charge/tautomer/stereo policy and then canonicalises. Store the input too, so the transformation remains auditable. |
| Descriptors | `molecular_descriptors("CCO")` | MW, LogP, TPSA, HBD/HBA, rings, charge, etc. | Computes interpretable 2D properties suitable for filtering and classical models. Values depend on the standardized graph. |
| Morgan/ECFP | `morgan_fingerprint("CCO", radius=2, n_bits=2048)` | radius 2 is commonly called ECFP4 | Hashes circular atom environments into a fixed-size bit vector. Collisions are possible, so it is a similarity/model feature, not an identity key. |
| Tanimoto | `tanimoto_similarity("CCO", "OCC") -> 1.0` | ethanol vs benzene is low | Measures bit-set intersection divided by union for two Morgan fingerprints. |
| Bemis–Murcko scaffold | `bemis_murcko_scaffold("CCOc1ccccc1") -> "c1ccccc1"` | side chains are removed | Extracts ring systems and their linkers for scaffold splits or series analysis; acyclic molecules return an empty scaffold. |
| Clustering | `cluster_molecules(smiles, cutoff=0.65)` | returns lists of input indexes | Uses Butina clustering over Morgan/Tanimoto distance. The cutoff is a similarity threshold, not a universal chemical boundary. |
| Diversity selection | `select_diverse(smiles, count=10, seed=7)` | returns selected input indexes | Uses greedy MaxMin selection to spread a subset through fingerprint space reproducibly. |
| Store record | `featurize_smiles(smiles)` / `featurize_many(...)` | returns a dataclass / list of dictionaries | Keeps raw, canonical, and policy-normalized SMILES next to scaffold and descriptors; batches may fail fast or skip invalid rows. |

## Why “the same” compound can have different SMILES

SMILES is a traversal of a molecular graph, not a database identity. `CCO` and `OCC` are different traversals of ethanol and canonicalize to the same string. Other differences need an explicit business rule: a sodium salt has an extra fragment, a conjugate acid has a different formal charge and hydrogen count, tautomers have different bond/hydrogen placement, and stereoisomers can have the same non-isomeric SMILES while being different compounds. Even canonical SMILES is toolkit/version and policy dependent. For durable interchange, retain the submitted representation and consider an InChIKey alongside the versioned normalization policy.

## End-to-end example

```python
from src.chemistry.feature_store import (
    StandardizationOptions, featurize_many, morgan_fingerprint,
    cluster_molecules, select_diverse,
)

raw = ["CCO", "OCC", "CC(=O)[O-].[Na+]", "c1ccccc1", "not-smiles"]
policy = StandardizationOptions(
    largest_fragment=True,
    neutralize=False,
    canonical_tautomer=True,
    include_stereochemistry=True,
)
rows = featurize_many(raw, policy, skip_invalid=True)
# rows[0]["canonical_smiles"] == rows[1]["canonical_smiles"] == "CCO"
# rows[2]["identity_smiles"] == "CC(=O)[O-]" (sodium removed, charge kept)

fingerprint = morgan_fingerprint(rows[0]["identity_smiles"])
clusters = cluster_molecules([row["identity_smiles"] for row in rows])
diverse_indexes = select_diverse([row["identity_smiles"] for row in rows], count=2, seed=7)
```

Persist the normalization options and RDKit version with generated features. Changing either can change identifiers, descriptor values, fingerprints, clusters, and selected compounds; feature-store keys should therefore include a feature-set version.
