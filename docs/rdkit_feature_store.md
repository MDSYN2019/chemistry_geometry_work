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

## Use it with Feast and PostgreSQL

The RDKit module is the **feature producer** and Feast is the **feature delivery layer**. PostgreSQL holds the timestamped, computed scalar values as Feast's offline store and also holds the materialized low-latency values in a separate `feast_online` schema. Feast does not run RDKit transformations during retrieval: recompute and reload the table whenever the chemistry policy or RDKit version changes.

The runnable example is in [`feature_repo/`](../feature_repo):

| File | Purpose |
|---|---|
| `features.py` | Declares the molecule entity, PostgreSQL source, versioned feature view, and feature service. |
| `feature_store.yaml` | Connects Feast's registry, offline store, and online store. Credentials are read from environment variables. |
| `load_features.py` | Computes RDKit records and upserts them into the offline table. It accepts a CSV with a `smiles` header. |
| `read_features.py` | Demonstrates a point-in-time historical join and an online lookup. |
| `sql/init.sql` | Creates the source table, timestamp index, and isolated online-store schema. |

### One-command Docker quick start

Docker Compose starts PostgreSQL, waits until it is healthy, loads demo molecules, applies the Feast definitions, materializes the latest features, and runs both retrieval examples:

```bash
docker compose -f docker-compose.feast.yml up --build --abort-on-container-exit feast
```

On success, the `feast` container prints historical and online values for ethanol and benzene. The database and registry use named volumes, so subsequent runs retain state. Reset the complete demo—including the source data, online values, and registry—when changing schemas:

```bash
docker compose -f docker-compose.feast.yml down --volumes
```

The credentials in Compose are deliberately local-development defaults. Do not expose this database or reuse those credentials in a deployed environment; supply secrets through the platform's secret manager instead.

### Work interactively

Start only PostgreSQL, then use the Feast image as an ephemeral CLI. Compose supplies the database environment variables and mounts a persistent registry volume:

```bash
docker compose -f docker-compose.feast.yml up -d postgres
docker compose -f docker-compose.feast.yml run --rm feast python feature_repo/load_features.py
docker compose -f docker-compose.feast.yml run --rm --workdir /workspace/feature_repo feast feast apply
docker compose -f docker-compose.feast.yml run --rm --workdir /workspace/feature_repo feast \
  feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)"
docker compose -f docker-compose.feast.yml run --rm feast python feature_repo/read_features.py
```

To ingest your own molecules, provide a CSV whose header includes `smiles`:

```csv
smiles
CCO
CC(=O)Oc1ccccc1C(=O)O
```

Mount it into the container and pass `--csv`:

```bash
docker compose -f docker-compose.feast.yml run --rm \
  -v "$PWD/molecules.csv:/input/molecules.csv:ro" \
  feast python feature_repo/load_features.py --csv /input/molecules.csv
```

Run `feast materialize-incremental` again after every load before expecting new values from `get_online_features`.

### Run without Docker

Install the Feast integration dependencies with `python -m pip install -r feature_repo/requirements.txt`, start any reachable PostgreSQL instance, execute [`feature_repo/sql/init.sql`](../feature_repo/sql/init.sql), and export these variables:

```bash
export POSTGRES_HOST=localhost POSTGRES_PORT=5432 POSTGRES_DB=feast
export POSTGRES_USER=feast POSTGRES_PASSWORD=feast
python feature_repo/load_features.py --csv molecules.csv
(cd feature_repo && feast apply)
(cd feature_repo && feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)")
python feature_repo/read_features.py
```

`get_historical_features` performs a point-in-time join: each entity row needs an `event_timestamp`, and Feast returns the newest source record no later than that time (within the feature view TTL). Use it to construct training sets without future leakage. `get_online_features` takes only entity keys and returns the values last materialized into the online store; use it on inference paths.

### Versioning and production notes

* The current definitions are explicitly named `molecule_features_v1` and `molecule_model_v1`. Create `v2` objects and a new table—not an in-place semantic change—when normalization, descriptor definitions, or RDKit versions change.
* `identity_smiles` is the entity join key under the documented standardization policy. The raw submitted string stays in PostgreSQL for audit but is not an entity key.
* Fingerprints are intentionally absent from this initial Feast view. Store a stable byte/array encoding with explicit radius, bit length, and generator version before adding them; never treat a collision-prone fingerprint as identity.
* The loader uses one event time per batch. Production ingestion should use the source observation time, retain `created_timestamp` for deduplication, validate rejected molecules, and monitor freshness.
* The bundled PostgreSQL online store makes the example easy to operate. For stricter latency or scale requirements, retain PostgreSQL as the offline source and configure a supported dedicated online store without changing feature names or client calls.
