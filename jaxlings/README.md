# jaxlings

A rustlings-style practice track for **JAX** fundamentals and model-building workflows.

## How to use

1. Open an exercise in `jaxlings/exercises/...`.
2. Fill every `TODO` section.
3. Run the file with Python.
4. Compare with the matching answer in `jaxlings/solutions/...`.

## Suggested flow

- 00-04: `jax.numpy` arrays, reductions, and PRNG keys.
- 05-10: transforms (`vmap`, `grad`, `jit`) and `lax` control flow.
- 11-16: pytrees, dense layers, training basics, and losses.
- 17-21: convolution and higher-order autodiff.
- 22-24: device mapping, checkpointing, and MLP capstone.

## Quick check commands

```bash
python jaxlings/check.py
python -m compileall jaxlings/exercises jaxlings/solutions
```

> Install JAX first (CPU build): `pip install -U "jax[cpu]"`.
