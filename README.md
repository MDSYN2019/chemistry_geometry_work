<div align="center">

# Chemistry, Geometry & ML Lab

**A hands-on portfolio of computational chemistry, molecular ML, scientific software, and learn-by-fixing exercise tracks.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![C++](https://img.shields.io/badge/C%2B%2B-11-00599C?logo=cplusplus&logoColor=white)](cxx/README.md)
[![Rust](https://img.shields.io/badge/Rust-practice-000000?logo=rust&logoColor=white)](practical-rustlings/README.md)
[![JAX](https://img.shields.io/badge/JAX-exercises-8A2BE2)](jaxlings/README.md)
[![PyTorch](https://img.shields.io/badge/PyTorch%20%2B%20PyG-exercises-EE4C2C?logo=pytorch&logoColor=white)](pytorchlings/README.md)

<sub>Explore a product below, open its README, and follow the local quick-start.</sub>

</div>

---

## Portfolio map

```mermaid
flowchart LR
    repo["🧪 Chemistry, Geometry<br/>& ML Lab"]

    repo --> science["⚗️ Scientific products"]
    repo --> learning["🎓 Interactive learning"]
    repo --> systems["🧰 Engineering products"]
    repo --> guides["📚 Practice guides"]

    science --> aldc["ALDC Cl → Br RBFE"]
    science --> boltz["Boltz-2 practice lab"]
    science --> feature["RDKit feature store"]
    science --> forecast["GB demand forecasting"]

    learning --> torch["pytorchlings"]
    learning --> jax["jaxlings"]
    learning --> psi["psi4lings"]
    learning --> py["py03lings"]
    learning --> algo["algolings"]
    learning --> funcs["functoolslings"]
    learning --> rust["practical-rustlings"]

    systems --> llm["LLM implementation template"]
    systems --> cpp["C++ geometry core"]
    systems --> workspace["Python research workspace"]
```

## Products at a glance

| Product | What it is | Start here |
| --- | --- | --- |
| **ALDC Cl → Br RBFE** | Milestone-gated reconstruction of a Zn-containing enzyme relative-binding free-energy study; M0 currently inventories and measures the supplied structure. | [`aldc-cl-br-rbfe/README.md`](aldc-cl-br-rbfe/README.md) |
| **Boltz-2 practice lab** | Small monomer, protein-complex, and protein–ligand fixtures for learning prediction and responsible result interpretation. | [`boltz2-practice/README.md`](boltz2-practice/README.md) |
| **RDKit small-molecule feature store** | Vendor-neutral molecule parsing, normalization, descriptors, fingerprints, clustering, and diversity selection. | [`docs/rdkit_feature_store.md`](docs/rdkit_feature_store.md) |
| **GB demand forecasting lab** | Leakage-aware, point-in-time-correct forecasting exercises with generated half-hourly demand and standard-library baselines. | [`time_series_forecasting/README.md`](time_series_forecasting/README.md) |
| **pytorchlings** | Fix-the-code curriculum spanning PyTorch, PyG, chemistry ML, graph learning, Lightning, and chemistry LLMs. | [`pytorchlings/README.md`](pytorchlings/README.md) |
| **jaxlings** | Progressive exercises in arrays, transformations, autodiff, pytrees, training, device mapping, and checkpointing. | [`jaxlings/README.md`](jaxlings/README.md) |
| **psi4lings** | Quantum-chemistry exercises from molecular setup and SCF through optimization, vibrations, properties, and scans. | [`psi4lings/README.md`](psi4lings/README.md) |
| **py03lings** | Python fundamentals through async, data engineering, systems concepts, and object-oriented design. | [`py03lings/README.md`](py03lings/README.md) |
| **algolings** | Core algorithms track covering search, hashing, graphs, shortest paths, disjoint sets, and dynamic programming. | [`algolings/README.md`](algolings/README.md) |
| **functoolslings** | Focused practice for `reduce`, `partial`, decorators, caching, dispatch, ordering, and integration. | [`functoolslings/README.md`](functoolslings/README.md) |
| **practical-rustlings** | Broad Rust practice from ownership fundamentals to Tokio, systems programming, and numerical FFI. | [`practical-rustlings/README.md`](practical-rustlings/README.md) |
| **LLM implementation template** | A staged blueprint for a reliable LLM application: fundamentals, evaluation, serving, deeper ML, and specialization. | [`llm_implementation_template/README.md`](llm_implementation_template/README.md) |
| **C++ geometry core** | The original reusable geometry implementation, examples, and application entry point. | [`cxx/README.md`](cxx/README.md) |
| **Python research workspace** | Exploratory PyTorch, chemistry, simulation, and platform-engineering source organized under `src/`. | [`src/README.md`](src/README.md) |

## Scientific systems

### Molecular modelling and discovery

The scientific projects deliberately separate **inputs**, **computation**, **quality checks**, and **claims**. The ALDC project is milestone-gated: its current M0 geometry report is a structural baseline, not an RBFE result.

```mermaid
flowchart LR
    pdb["Experimental structure<br/>PDB"] --> inspect["ALDC M0<br/>inventory + Zn geometry"]
    inspect --> gate{"Scientific<br/>review gate"}
    gate -. "future milestone" .-> prep["Metal-site and<br/>ligand preparation"]
    prep -.-> rbfe["Cl → Br<br/>RBFE workflow"]

    yaml["Protein / ligand<br/>YAML fixtures"] --> boltz["Boltz-2<br/>prediction"]
    boltz --> outputs["Coordinates +<br/>confidence outputs"]
    outputs --> validation["Structure / assay<br/>validation"]

    smiles["Raw SMILES"] --> policy["Versioned chemistry<br/>standardization policy"]
    policy --> rdkit["RDKit feature store"]
    rdkit --> features["Descriptors • Morgan FP<br/>scaffolds • clusters"]
    features --> model["Classical or ML<br/>downstream model"]
```

> [!IMPORTANT]
> Predicted structures, confidence values, geometric contacts, and computed molecular features are not experimental evidence of binding or biological function. Read each product's evidence boundaries before interpreting output.

### Forecasting system

```mermaid
flowchart LR
    raw["Raw demand data"] --> ingest["Idempotent ingestion<br/>+ schema checks"]
    ingest --> curated["Canonical observations<br/>event_time + available_at"]
    curated --> features["Point-in-time-safe<br/>feature generation"]
    features --> backtest["Chronological<br/>backtesting"]
    backtest --> metrics["MAE • RMSE • bias<br/>slice diagnostics"]
    metrics --> decision["Reproducible forecast<br/>decision record"]
```

The included worked example uses generated data and only the Python standard library, so the core mechanics can be explored before connecting a production data source.

## Learn-by-fixing tracks

Every `*lings` Python track follows the same loop: open an exercise, replace its `TODO`s, run the checker, then compare with the corresponding solution. Dependencies vary by track.

```mermaid
flowchart LR
    choose["1 · Choose a track"] --> edit["2 · Complete TODOs"]
    edit --> check["3 · Run check.py<br/>or the track command"]
    check --> pass{"Checks pass?"}
    pass -- No --> compare["Inspect feedback<br/>and solution"]
    compare --> edit
    pass -- Yes --> next["4 · Advance to the<br/>next exercise"]

    choose --- foundations["Python • functools<br/>algorithms • Rust"]
    choose --- compute["JAX • PyTorch • PyG"]
    choose --- chemistry["Psi4 • chemistry ML"]
```

### Suggested learning routes

| Goal | Route |
| --- | --- |
| **Scientific Python foundations** | [`py03lings`](py03lings/README.md) → [`functoolslings`](functoolslings/README.md) → [`algolings`](algolings/README.md) |
| **Modern differentiable ML** | [`jaxlings`](jaxlings/README.md) → [`pytorchlings`](pytorchlings/README.md) → [`src/pytorch_practice`](src/README.md) |
| **Computational chemistry + ML** | [`psi4lings`](psi4lings/README.md) → [RDKit feature store](docs/rdkit_feature_store.md) → [`pytorchlings`](pytorchlings/README.md) → [ALDC RBFE](aldc-cl-br-rbfe/README.md) |
| **Systems breadth** | [`practical-rustlings`](practical-rustlings/README.md) → [C++ geometry](cxx/README.md) → [LLM implementation template](llm_implementation_template/README.md) |

## Engineering blueprints

### Reliable LLM application lifecycle

```mermaid
flowchart LR
    app["Fundamentals<br/>retrieval + tools"] --> eval["Evaluation<br/>goldens + regressions"]
    eval --> ops["Serving & ops<br/>APIs + observability"]
    ops --> depth["Applied ML<br/>reranking + tuning"]
    depth --> special["Domain<br/>specialization"]
    special -. "new failures" .-> eval
```

### Repository runtime layers

```mermaid
flowchart TB
    products["Product READMEs & exercises"]
    products --> python["Python packages and scripts<br/>src/ • time_series_forecasting/"]
    products --> native["Native code<br/>cxx/ • practical-rustlings/"]
    python --> tests["pytest • unittest<br/>track checkers"]
    native --> build["Make • Cargo"]
    tests --> feedback["Fast learning and<br/>research feedback"]
    build --> feedback
```

## Practice packs and roadmaps

These standalone guides complement the runnable products:

- [Modern deep-learning research exercises](modern_deep_learning_research_exercises.md) — GNN, Transformer, and diffusion research practice.
- [Force-field coarse-graining exercises](force_field_coarse_graining_exercises.md) — multiscale molecular-modelling practice.
- [PyG exercise roadmap](pyg_exercise_roadmap.md) — a structured graph-learning progression.
- [Data-engineering interview pack](data_engineering_interview_practice_pack.md) — practical data-system interview preparation.
- [SQL performance exercises](sql_performance_complex_query_exercises.md) — complex-query and performance practice.

## Quick start

```bash
# Clone and enter the repository, then install the root Python project
python -m pip install -e .
python -m pip install pytest

# Run the root test suite
pytest

# Or start with a dependency-light learning track
python algolings/check.py
python py03lings/check.py

# Run the dependency-free forecasting example
python time_series_forecasting/example.py
```

Most scripts are exploratory and runnable independently. For JAX, PyTorch/PyG, Psi4, Boltz-2, OpenMM, or other specialist tools, follow the environment notes in that product's README instead of installing everything into one environment.

## Legacy C++ build

The top-level `Makefile` builds the original geometry executable:

```bash
make
./main
```

The compiler requires the configured Eigen headers and `spdlog` library. See the [C++ layout](cxx/README.md) before extending the native code.

## Repository conventions

- Treat each product README as the source of truth for its setup, scope, and scientific limitations.
- Keep generated model weights, trajectories, predictions, and other large artifacts out of Git unless a product explicitly documents otherwise.
- Prefer reproducible commands, seeded examples, versioned inputs, and tests over notebook-only results.
- Do not infer scientific conclusions beyond what a product's current milestone and validation evidence support.
