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
