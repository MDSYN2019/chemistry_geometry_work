# pytorchlings

A rustlings-style practice track for **PyTorch** and **PyTorch Geometric (PyG)**.

## How to use

1. Open an exercise in `pytorchlings/exercises/...`.
2. Fill every `TODO` section.
3. Run the file with Python.
4. Compare with the matching answer in `pytorchlings/solutions/...`.

## Suggested flow

- Start with 00-09 for core PyTorch foundations.
- Continue with 10-14 for PyG fundamentals.
- Extend with 15-19 for more advanced PyTorch practice.
- Finish 20-24 for advanced PyG workflows.
- Add 25-27 for PyTorch Lightning abstractions.
- Finish with 28-30 for NetworkX `nx.Graph` practice.
- Continue 31-34 for weighted, directed, bipartite, and cycle/tree graph workflows.
- Add 35-37 for PyG GraphGym config, registration, and experiment YAML practice.
- Complete 38-47 for comprehensive PyTorch engineering workflows (losses, hooks, initialization, accumulation, AMP, compile, TorchScript, packed sequences).
- Continue 48-59 for advanced graph architecture coverage (GCN, GAT, GraphSAGE, GIN, spectral, pooling, hyperbolic, dynamic, R-GCN, graph transformer, graph autoencoder, diffusion).
- Add 60-61 for Optuna-based hyperparameter optimization (PyTorch MLP + scikit-learn baseline tuning).
- Add 62-65 for chemistry-oriented generative modeling, hill-climbing evals, and slice-based scientific error analysis.

## Quick check command

```bash
python -m compileall pytorchlings/exercises pytorchlings/solutions
```

> If you do not have PyTorch Geometric installed, complete exercises 00-09 first and treat 10-14 as reading/practice templates.


## Lightning note

Exercises 25-27 expect `lightning` to be installed (`pip install lightning`).


## Comprehensive PyTorch coverage extension

Exercises 38-47 were added to cover production-focused PyTorch functionality that often gets skipped in beginner tracks:

- Correct loss/logit pairings (`CrossEntropyLoss`, `BCEWithLogitsLoss`)
- Device + dtype hygiene for stable training loops
- Safe evaluation mode with `torch.no_grad()`
- Forward hooks for feature extraction and debugging
- Explicit parameter initialization patterns
- Gradient accumulation over micro-batches
- AMP (`autocast`, `GradScaler`)
- `torch.compile` runtime optimization API
- TorchScript export (`script`/`trace`)
- Variable-length sequence modeling with packed RNN inputs


## Advanced graph architecture extension

Exercises 48-59 extend PyG graph learning coverage across major model families:

- GCN (explicit normalized adjacency)
- GAT (attention-weight inspection)
- GraphSAGE
- Graph Isomorphism Network (GIN)
- Spectral GNN (Chebyshev/Laplacian filtering)
- Graph pooling (hierarchical + global)
- Hyperbolic-inspired message passing
- Dynamic temporal GNN (snapshot sequence processing)
- Relational GCN (heterogeneous edge types)
- Graph Transformer
- Graph Autoencoder
- Diffusion-style propagation (APPNP)

## Hyperparameter optimization extension

Exercises 60-61 introduce Optuna workflows so learners can practice search-space design and reproducible studies:

- Optuna trial definitions for PyTorch model architecture/training hyperparameters
- Intermediate metric reporting + pruning hooks
- TPE and random samplers with fixed seeds
- Non-neural baseline tuning with scikit-learn cross-validation
## Chemistry + materials interview-focused extension

Exercises 62-65 target role-relevant fundamentals for generative science workflows:

- Masked-token transformer pretraining on toy chemistry strings
- Property-conditioned sequence generation
- Hill-climb eval loops for iterative model improvement
- Slice-wise scientific error analysis + markdown reporting for collaboration

