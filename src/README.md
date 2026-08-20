# Python source layout

This `src/` tree is now organized as a **PyTorch practice workspace** plus related chemistry/simulation material.

## Structure

- `pytorch_practice/`
  - `basics/`: tensor basics, synthetic data, linear regression, and gradient descent scripts.
  - `nn/`: neural-network-focused experiments and hyperparameter utilities.
  - `vision/`: torchvision and Optuna + MNIST experiments.
  - `graph/`: graph neural network experimentation.
  - `utils/`: shared helper utilities for model experimentation.
- `chemistry/`: chemistry and molecular modeling scripts (`molecule/`, `stat_mech.py`),
  including the vendor-neutral [RDKit small-molecule feature store](../docs/rdkit_feature_store.md).
- `simulations/`: simulation-focused code snippets.
- `data/`: local data artifacts used by scripts.
- `legacy/`: logs or historical one-off files retained for reference.
- `practice/platform_engineering/`: C#/Python distributed systems and platform engineering practice pack.

## Notes

Most scripts are exploratory and can be run independently.
