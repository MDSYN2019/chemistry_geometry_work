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
- `algolings/` — rustlings-style algorithms exercises from simple to complex
- `modern_deep_learning_research_exercises.md` — research-level GNN,
  Transformer, and diffusion model exercise track

## Build / run notes

### Python workspace (current)

This repo includes `pyproject.toml` for Python tooling. Most scripts are exploratory and can be run directly from the project root.

### Legacy C++ build command

If you need to build the original geometry binary, use:

```bash
g++ -I lib/eigen-3.4.0/ -I lib/ cxx/geometry_functions.cxx main.cxx -o ./main
```

## Research Engineer Interview Exercise Pack (BioAI)

The following exercises are designed for hiring/recruiting loops for a Research Engineer role focused on AI for drug discovery.

### 1) Neural Network Fundamentals (take-home, 60–90 min)

- Build a small neural network to predict a binary property from sequence-like inputs.
- Compare at least two architectures (for example: MLP vs. 1D CNN or a tiny Transformer).
- Report train/validation metrics, overfitting behavior, and one optimization you attempted.

### 2) Distributed Training Systems (practical, 90 min)

- Start from a single-GPU training script and describe/adapt it for multi-GPU training.
- Identify bottlenecks (data loading, communication overhead, memory) and mitigation strategies.
- Discuss tradeoffs between data parallel and sharded approaches.

### 3) Profiling and Optimization (debug session, 60 min)

- Profile a deliberately inefficient training script.
- Identify the top bottlenecks and implement (or propose) fixes.
- Show before/after runtime or memory measurements.

### 4) BioAI Problem Framing (whiteboard, 45 min)

Prompt: design an ML workflow to prioritize antibody candidates for a hard-to-drug target.

Candidate should cover:
- data sources (sequence, structure, assay)
- objective definition (affinity, specificity, developability)
- evaluation plan (offline metrics + wet-lab handoff)
- uncertainty and failure-mode handling

### 5) Foundation Model Design Case (system design, 60 min)

Prompt: design a biology/disease foundation model to improve target elucidation and patient stratification.

Candidate should address:
- multimodal inputs and pretraining objectives
- fine-tuning strategy for downstream tasks
- interpretability, bias, and deployment considerations

### 6) Code Quality + Collaboration (pairing, 45 min)

- Refactor a small research code sample for clarity.
- Add basic tests and concise documentation.
- Explain decisions while pairing with the interviewer.

### 7) Research Communication (presentation, 20–30 min)

- Present one prior project/public repo contribution.
- Explain approach, tradeoffs, failures, and next steps.

### Suggested scoring rubric (1–4 each)

- ML fundamentals
- Scaling/systems engineering
- Optimization/debugging
- BioAI reasoning
- Code quality
- Collaboration and communication

A strong overall signal is typically 3+ in most categories with at least one area of clear excellence.

## Implemented Python exercises (PyTorch)

Concrete exercise scripts are available at:

- `src/pytorch_practice/interview_exercises/ex1_sequence_binary_classifier.py`
- `src/pytorch_practice/interview_exercises/ex2_distributed_training_stub.py`
- `src/pytorch_practice/interview_exercises/ex3_profile_and_optimize.py`
- `src/pytorch_practice/interview_exercises/ex4_bioai_multitask_stub.py`
- `src/pytorch_practice/interview_exercises/README.md`

These are executable Python stubs that can be used directly in take-home or live interview settings.
