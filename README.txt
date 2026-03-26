# chemistry_geometry_work

## Repository status

This repository is now a **multi-language learning workspace** rather than a single C++ geometry executable.

- The original C++ geometry prototype is still available in `cxx/`.
- Active experimentation has moved to the Python project layout under `src/`.
- The `src/` tree is organized around PyTorch practice, chemistry scripts, and simulation snippets.

## Current structure

- `src/pytorch_practice/` — PyTorch experiments (basics, NN, vision, graph)
- `src/chemistry/` — chemistry and molecular modeling scripts
- `src/simulations/` — simulation-focused code
- `src/data/` — data files used by scripts
- `cxx/` — legacy C++ geometry code
- `jaxlings/` — rustlings-style JAX exercises and solutions

## Build / run notes

### Python workspace (current)

This repo includes `pyproject.toml` for Python tooling. Most scripts are exploratory and can be run directly from the project root.

### Legacy C++ build command

If you need to build the original geometry binary, use:

```bash
g++ -I lib/eigen-3.4.0/ -I lib/ cxx/geometry_functions.cxx main.cxx -o ./main
```
