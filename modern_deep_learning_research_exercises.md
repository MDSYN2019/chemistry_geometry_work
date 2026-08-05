# Modern Deep Learning Research Exercises

This practice track is aimed at researchers who already know the basics of
PyTorch and want to build deeper, interview- and publication-level knowledge
of graph neural networks (GNNs), Transformers, and diffusion models. The goal
is not just to make models run: every exercise asks for a hypothesis, a fair
baseline, an ablation, and an explanation of the result.

## How to use this track

- Spend **6–10 hours** on each core exercise and **2–4 weeks** on the capstone.
- Keep one repository per project with a reproducible environment, fixed random
  seeds, configuration files, tests for important tensor operations, and a
  one-command training script.
- Before each experiment, write down the expected result and the reason for it.
- Run at least three seeds for comparisons. Report the mean and standard
  deviation, parameter count, peak memory, and training time—not only the best
  score.
- End each exercise with a two-page research memo: question, method, results,
  failure analysis, and next experiment.

For limited hardware, use synthetic data or small public datasets, reduce model
width, and validate an idea on a small subset before running a full experiment.

## Shared research checklist

Every submission should answer these questions:

1. **Claim:** What precise, falsifiable claim is being tested?
2. **Protocol:** Are data splits, preprocessing, and compute budgets identical
   across comparisons?
3. **Baselines:** Is there a simple non-neural baseline and a competitive neural
   baseline?
4. **Metrics:** Do the metrics reflect the real task, including calibration or
   efficiency where appropriate?
5. **Uncertainty:** How variable is the outcome across seeds or data splits?
6. **Ablation:** Which component actually causes the improvement?
7. **Failure modes:** On which examples or distribution shifts does the method
   fail?
8. **Reproducibility:** Can another person regenerate the main table or figure
   with one documented command?

---

## Track A — Graph Neural Networks

### A1. Message passing from first principles

**Question:** What assumptions are hidden inside a standard message-passing
layer?

Implement a GCN-style layer using only tensor operations—do not use a packaged
graph convolution. Add self-loops, symmetric degree normalization, neighborhood
aggregation, and a learnable update. Test it on small hand-constructed graphs.

**Required investigations**

- Verify permutation equivariance numerically after randomly relabeling nodes.
- Compare sum, mean, and max aggregation while holding the parameter budget
  fixed.
- Test an isolated node, a duplicated edge, and a graph with very unequal node
  degrees.
- Compare against an MLP that ignores connectivity.

**Deliverables:** unit tests for equivariance and edge cases; a derivation of the
normalized update; accuracy/runtime results; and a paragraph explaining when
each aggregator loses information.

**Success criterion:** the custom layer matches a library implementation on a
tiny graph within a stated numerical tolerance and beats the MLP on a task where
edges carry useful signal.

### A2. Expressivity, 1-WL, and oversmoothing

**Question:** Which graph distinctions can message-passing GNNs learn, and what
happens as depth increases?

Implement 1-dimensional Weisfeiler–Leman color refinement and create graph pairs
that it can and cannot distinguish. Train GCN and GIN classifiers on a synthetic
dataset containing these cases.

**Required investigations**

- Compare 2, 4, 8, and 16 layers with the same hidden width.
- At every layer, measure pairwise embedding distance, feature variance, and a
  simple class-separation statistic.
- Ablate residual connections, normalization, and Jumping Knowledge.
- Separate an **expressivity failure** from an **optimization failure** with a
  controlled example.

**Deliverables:** a table of graph pairs and predictions; depth curves; an
oversmoothing visualization; and a short proof sketch relating message passing
to 1-WL.

**Success criterion:** identify at least one failure caused by representation
limits and one caused by depth or training dynamics.

### A3. Molecular graphs with geometry

**Question:** When do 3D coordinates improve molecular property prediction?

Build a graph-level predictor with atom and bond features. Compare a 2D
message-passing model, a model using raw pairwise distances, and an
E(3)-invariant or equivariant geometric model on a small molecular dataset.

**Required investigations**

- Use scaffold-based rather than only random splits.
- Numerically test invariance of scalar predictions under translation,
  rotation, and atom reindexing.
- Ablate coordinates, bond features, and radial basis encodings.
- Stratify error by molecule size and by structural novelty.

**Deliverables:** symmetry tests; split-generation code; learning curves; an
ablation table; and a gallery of the largest-error molecules.

**Success criterion:** explain with evidence whether geometry helps in-domain,
out-of-domain, both, or neither—without assuming that a higher-capacity model is
automatically the better geometric model.

### A4. Scaling and graph generalization

**Question:** What accuracy–efficiency trade-offs arise when full-batch graph
training no longer fits in memory?

Train the same node classifier using full-batch training, neighbor sampling, and
subgraph mini-batching.

**Required investigations**

- Measure examples or edges per second, peak memory, convergence time, and final
  quality.
- Vary fan-out and sampling depth and quantify sampling variance.
- Test inductive generalization to held-out nodes or graphs.
- Discuss information leakage when constructing graph splits.

**Deliverables:** a Pareto plot, profiler traces for one bottleneck, and a
recommendation for three different memory budgets.

**Success criterion:** produce a defensible sampling choice based on both model
quality and resource measurements.

---

## Track B — Transformer Architectures

### B1. Build a Transformer encoder block

**Question:** How do attention, residual paths, normalization, and masking
interact?

Implement scaled dot-product attention, multi-head attention, a feed-forward
network, residual connections, and layer normalization using basic tensor
operations.

**Required investigations**

- Write tests for tensor shapes, causal and padding masks, and equivalence to a
  trusted implementation after copying weights.
- Compare pre-norm and post-norm models on a small sequence task.
- Visualize attention weights, but also show one example where attention weight
  is not a reliable feature-importance explanation.
- Derive the time and memory complexity in sequence length, width, and number of
  heads.

**Deliverables:** tested implementation; gradient-norm plots; complexity
derivation; and a diagnosis of any unstable run.

**Success criterion:** match the trusted implementation numerically and explain
the observed optimization difference between pre-norm and post-norm.

### B2. Positional information and length extrapolation

**Question:** Which positional scheme generalizes beyond training lengths?

Train small Transformers on an algorithmic task such as sequence reversal,
modular addition, or associative recall. Compare learned absolute embeddings,
sinusoidal encodings, and one relative or rotary scheme.

**Required investigations**

- Train only on short sequences and evaluate on both interpolated and longer
  sequences.
- Control for parameter count and training tokens.
- Report exact-match accuracy by length, not only an aggregate mean.
- Inspect whether failures occur near the context boundary or throughout the
  sequence.

**Deliverables:** accuracy-versus-length curves, an ablation table, and a
mechanistic hypothesis for each scheme's extrapolation behavior.

**Success criterion:** demonstrate a reproducible generalization gap and propose
an experiment that could falsify your explanation.

### B3. Efficient attention benchmark

**Question:** When does an approximate or local attention mechanism become
worthwhile?

Compare full attention with a local-window, block-sparse, or linear-attention
variant on a long-sequence task.

**Required investigations**

- Benchmark several sequence lengths after warm-up and synchronized device
  execution.
- Track wall-clock time, peak memory, and task quality.
- Construct one task requiring local context and another requiring a distant
  dependency.
- Distinguish theoretical complexity from realized hardware performance.

**Deliverables:** log-log scaling plots, quality–latency Pareto curves, profiler
evidence, and a discussion of approximation failures.

**Success criterion:** identify the empirical crossover point and show how it
changes with batch size or hardware.

### B4. Representation analysis and fine-tuning

**Question:** What changes inside a pretrained Transformer during adaptation?

Fine-tune a small pretrained encoder on a classification task. Compare full
fine-tuning, a frozen linear probe, and a parameter-efficient method.

**Required investigations**

- Match optimization budgets as fairly as possible and report trainable as well
  as total parameters.
- Measure calibration and performance under a simple distribution shift.
- Compare layer representations before and after adaptation with a similarity
  measure such as centered kernel alignment.
- Test whether the conclusion holds in a low-data regime.

**Deliverables:** performance/calibration table, representation-similarity
heatmap, compute accounting, and error clusters.

**Success criterion:** connect a measurable representation change to a concrete
behavioral change while clearly labeling correlation versus causation.

---

## Track C — Diffusion Models

### C1. Derive and implement a 2D diffusion model

**Question:** How does denoising score matching recover a multimodal data
distribution?

Use a mixture of 2D Gaussians or a two-moons dataset. Implement the forward
noising process, noise-prediction objective, reverse sampler, and timestep
embedding without a diffusion library.

**Required investigations**

- Derive the closed-form marginal \(q(x_t\mid x_0)\).
- Compare linear and cosine noise schedules.
- Plot learned and analytic scores where the analytic density is available.
- Vary the number of sampling steps and measure sample quality and runtime.

**Deliverables:** derivation; tests for the forward distribution; vector-field
plots; generated samples; and a quantitative distribution distance such as
maximum mean discrepancy.

**Success criterion:** recover all major modes without obvious collapse and
explain the bias–speed trade-off in the sampler.

### C2. DDPM on a small image dataset

**Question:** Which implementation and training choices most affect sample
quality?

Build a compact U-Net diffusion model for MNIST, Fashion-MNIST, or another small
image dataset. Maintain an exponential moving average of parameters and save a
fixed sampling grid during training.

**Required investigations**

- Compare noise prediction with one alternative parameterization.
- Ablate the EMA, schedule, and timestep sampling strategy.
- Track denoising loss by timestep rather than only its global mean.
- Evaluate samples with at least one feature-space metric and a nearest-neighbor
  check for memorization; state the limitations of both.

**Deliverables:** architecture diagram, learning and per-timestep loss curves,
sample grids, evaluation code, and an ablation table.

**Success criterion:** make a causal, ablation-supported claim about one choice
that improves quality or stability.

### C3. Conditional guidance and controllability

**Question:** How does guidance trade diversity for condition adherence?

Add class conditioning and classifier-free guidance to the model from C2.

**Required investigations**

- Train with conditioning dropout and sweep guidance strength at inference.
- Measure conditional accuracy, diversity, and sample quality at each strength.
- Test invalid, ambiguous, or underrepresented conditions.
- Explain the relationship between conditional and unconditional predictions in
  the guidance equation.

**Deliverables:** guidance-sweep grids, a three-metric trade-off plot, derivation
of the update, and failure examples.

**Success criterion:** choose a guidance setting from quantitative evidence and
describe why no single setting optimizes all objectives.

### C4. Fast sampling and likelihood trade-offs

**Question:** How much quality is lost when reverse diffusion is accelerated?

Compare a stochastic DDPM sampler with DDIM or another deterministic/fewer-step
sampler using the same trained denoiser.

**Required investigations**

- Sweep sampling steps over at least one order of magnitude.
- Use fixed initial noise to compare trajectories across samplers.
- Report throughput, memory, quality, and diversity.
- Explain why denoising loss, perceptual quality, and likelihood need not rank
  models identically.

**Deliverables:** trajectory visualizations, quality–latency curves, and a
deployment recommendation under a stated latency budget.

**Success criterion:** locate a practical knee in the quality–latency curve and
support it with repeated measurements.

---

## Track D — Synthesis and research maturity

### D1. Unified architecture comparison

Choose a structured prediction problem that admits at least two representations
(for example, a molecule as a graph and as a token sequence). Compare a GNN and
a Transformer under matched data, tuning budget, and approximate parameter
count.

**Research prompts**

- Which inductive bias is more data-efficient?
- Which model is more robust to a meaningful distribution shift?
- Does ensembling help because the errors are complementary?
- How sensitive is the conclusion to the split and evaluation metric?

**Deliverables:** preregistered hypotheses, matched-budget table, learning
curves, paired error analysis, and a statement of what the experiment cannot
conclude.

**Success criterion:** reach a conclusion that remains consistent across
multiple seeds and at least two sensible evaluation views, or clearly explain
why the evidence is inconclusive.

### D2. Geometry-aware diffusion capstone

Build a diffusion model that generates a small structured object, such as 2D
molecular graphs, 3D point clouds, or coarse-grained conformations. Use a GNN or
geometric Transformer as the denoiser.

The project must include:

- an explicit treatment of permutation and geometric symmetries;
- validity, uniqueness, novelty, and task-specific quality metrics;
- simple empirical-distribution and autoregressive baselines;
- a train/validation/test split that prevents near-duplicate leakage;
- at least three component ablations;
- compute accounting and multi-seed uncertainty;
- qualitative successes, failure cases, and an ethics or misuse note where
  appropriate.

**Success criterion:** produce a workshop-style paper (6–8 pages) whose central
claim is supported by a controlled table, an informative figure, and a serious
limitations section.

### D3. Paper reproduction and extension

Select one influential or recent paper in one of the three areas. Before reading
the official code, reproduce the core method from the paper's equations and
pseudocode on a reduced dataset.

1. Write a one-page claim map: each major claim, its supporting experiment, and
   possible confounders.
2. Reproduce one central table or figure with a realistic compute budget.
3. Compare your protocol line-by-line with the paper and document deviations.
4. Add one baseline or ablation missing from the paper.
5. Propose an extension, preregister the predicted outcome, and run it.
6. Write a reproducibility report that separates exact reproduction, partial
   reproduction, and unresolved discrepancies.

**Success criterion:** another researcher can run your reproduction, trace each
reported number to an artifact, and understand why discrepancies occurred.

---

## Suggested 12-week schedule

| Week | Focus | Output |
|---:|---|---|
| 1 | A1: message passing | Tested implementation and baseline |
| 2 | A2: expressivity | Depth/oversmoothing memo |
| 3 | B1: Transformer block | Numerical parity and stability study |
| 4 | B2: positions | Length-generalization report |
| 5 | C1: 2D diffusion | Derivation and score visualization |
| 6 | C2: image diffusion | Samples and first ablation |
| 7 | Choose A3/A4 | Domain or scaling study |
| 8 | Choose B3/B4 | Efficiency or adaptation study |
| 9 | Choose C3/C4 | Guidance or fast-sampling study |
| 10 | D3: reproduction | Reproduction result and discrepancy log |
| 11 | D2: capstone | Baselines and main experiments |
| 12 | D2: capstone | Paper, code release, and presentation |

## Self-assessment rubric

Score each category from **0 (missing)** to **4 (research-ready)**:

| Category | 2 — competent | 4 — research-ready |
|---|---|---|
| Theory | Can state and use core equations | Derives assumptions and predicts failure regimes |
| Implementation | Model trains correctly | Critical operations are tested, profiled, and reproducible |
| Experimental design | Includes a baseline | Controls confounders, uses multiple seeds, and tests hypotheses |
| Evaluation | Reports standard task metric | Adds calibration, robustness, efficiency, and uncertainty where relevant |
| Analysis | Describes aggregate results | Uses ablations and error analysis to support a bounded claim |
| Communication | Provides a readable README | Produces a concise paper-quality narrative with limitations |

A strong portfolio should contain at least one project scoring **3 or 4 in every
category**, not many projects that only demonstrate model training. Revisit any
exercise where the conclusion rests on a single seed, an unmatched baseline, or
the best checkpoint selected using test data.
