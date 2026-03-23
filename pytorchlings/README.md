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

## Quick check command

```bash
python -m compileall pytorchlings/exercises pytorchlings/solutions
```

> If you do not have PyTorch Geometric installed, complete exercises 00-09 first and treat 10-14 as reading/practice templates.


## Lightning note

Exercises 25-27 expect `lightning` to be installed (`pip install lightning`).
