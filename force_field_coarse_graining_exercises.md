# Exercises: Understanding Force-Field Coarse-Graining

These exercises are designed to move from intuition to practical model building. They assume basic familiarity with molecular simulation concepts (coordinates, trajectories, thermodynamics).

---

## 1) Conceptual warm-up: Why coarse-grain?

### Goal
Build intuition about what is gained and lost when moving from atomistic to coarse-grained (CG) models.

### Tasks
1. In 5–7 bullet points, explain why atomistic simulations can become impractical for large systems or long timescales.
2. List **three quantities** that are often preserved reasonably well in CG models (e.g., structural observables), and **three** that can degrade (e.g., fine-grained kinetics).
3. Explain, in your own words, what is meant by:
   - representability error
   - transferability
   - state-point dependence

### Check yourself
- Did you explicitly mention both computational speedups and information loss?
- Can you distinguish between “fitting at one condition” and “generalizing across conditions”?

---

## 2) Mapping exercise: Atomistic to bead representation

### Goal
Practice defining a CG mapping operator.

### Setup
Take a short molecule (e.g., butane, ethanol, or a simple lipid tail segment).

### Tasks
1. Propose a mapping from atoms to beads (e.g., 2–4 atoms per bead).
2. Write the mapping mathematically:
   \[
   \mathbf{R}_I = \sum_{i \in I} w_i \mathbf{r}_i
   \]
   and define your weights \(w_i\) (e.g., mass-weighted).
3. Compare two mapping choices and predict which one better preserves:
   - radius of gyration
   - end-to-end distance distribution
4. Briefly justify your choice in terms of chemistry and symmetry.

### Stretch
Discuss whether your mapping should preserve charge distribution explicitly or implicitly.

---

## 3) Potential of mean force from a toy distribution

### Goal
Connect probability distributions to effective potentials.

### Data (toy)
You are given a bond-length histogram for a CG bond coordinate \(r\):

| r (nm) | P(r) |
|---|---:|
| 0.30 | 0.08 |
| 0.35 | 0.20 |
| 0.40 | 0.34 |
| 0.45 | 0.24 |
| 0.50 | 0.10 |
| 0.55 | 0.04 |

### Tasks
1. Normalize \(P(r)\) (if needed).
2. Compute an unshifted PMF:
   \[
   U(r) = -k_B T \ln P(r)
   \]
   at \(T=300\,\text{K}\).
3. Shift \(U(r)\) so that \(\min U = 0\).
4. Fit the region near the minimum to a harmonic form:
   \[
   U(r) \approx \tfrac12 k (r-r_0)^2
   \]
   and estimate \(r_0\), \(k\).

### Check yourself
- Is the minimum at the most probable \(r\)?
- Does fitted \(k\) look physically reasonable for a soft CG bond?

---

## 4) Iterative Boltzmann Inversion (IBI) by hand (mini)

### Goal
Understand one update step in IBI.

### Given
Target RDF \(g_\text{target}(r)\) and current model RDF \(g_n(r)\) at selected points:

| r (nm) | g_target | g_n |
|---|---:|---:|
| 0.45 | 0.20 | 0.30 |
| 0.50 | 0.80 | 0.60 |
| 0.55 | 1.30 | 1.10 |
| 0.60 | 1.10 | 1.20 |
| 0.65 | 0.90 | 1.00 |

Use:
\[
U_{n+1}(r)=U_n(r)+\alpha k_B T\ln\frac{g_n(r)}{g_{\text{target}}(r)}
\]
with \(\alpha=0.2\), \(T=300\,\text{K}\).

### Tasks
1. For each \(r\), compute the correction term \(\Delta U(r)\).
2. Identify where interaction should become more repulsive vs more attractive.
3. Explain why a damping factor \(\alpha<1\) is often needed.

### Stretch
What problems can happen if \(g(r)\) is noisy near small \(r\)? Propose a stabilization strategy.

---

## 5) Force matching objective derivation

### Goal
Write and interpret the force-matching loss.

### Tasks
1. Starting from atomistic mapped forces \(\mathbf{F}^{\text{ref}}\) and model forces \(\mathbf{F}^{\text{CG}}(\theta)\), write a least-squares loss over snapshots.
2. Expand how this loss scales with:
   - number of frames
   - number of CG sites
   - dimensionality
3. Explain two reasons why minimizing force error does **not** automatically guarantee correct dynamics.

### Expected form
\[
\mathcal{L}(\theta)=\frac{1}{N}\sum_{n=1}^{N}\sum_{I=1}^{N_{\text{CG}}}
\left\|\mathbf{F}^{\text{CG}}_{I}(\mathbf{R}^{(n)};\theta)-\mathbf{F}^{\text{ref}}_{I}(\mathbf{R}^{(n)})\right\|^2
\]

---

## 6) Relative entropy perspective

### Goal
Interpret CG parameter optimization as distribution matching.

### Tasks
1. Write the relative entropy objective:
   \[
   S_{\text{rel}} = \int d\mathbf{R}\; p_{\text{AA}}(\mathbf{R})\ln\frac{p_{\text{AA}}(\mathbf{R})}{p_{\text{CG}}(\mathbf{R};\theta)}
   \]
2. In words, explain what minimizing this objective does.
3. Compare relative entropy matching to force matching in terms of what each “cares about.”
4. Give one scenario where you would prefer each method.

---

## 7) Practical diagnostics checklist

### Goal
Learn what to validate after building a CG model.

### Tasks
Create a validation table with columns:
- observable
- target (AA/experiment)
- CG result
- error metric
- pass/fail criterion

Include at least 8 observables spanning:
- structure (RDFs, bond/angle distributions)
- thermodynamics (density, compressibility or pressure trend)
- dynamics (diffusion; with caveat about time rescaling)
- transferability (second thermodynamic state)

### Reflection
Which failed observable is most critical for your intended application, and why?

---

## 8) Capstone: Design a complete CG workflow

### Goal
Integrate the pieces into one coherent plan.

### Scenario
You need a CG model for a solvated polymer system to study self-assembly over microseconds.

### Tasks
1. Define your mapping strategy and bead types.
2. Choose one fitting method as primary (IBI, force matching, relative entropy, or hybrid) and justify.
3. Specify training data requirements:
   - number of AA trajectories
   - state points
   - sampling quality checks
4. Define your validation suite (minimum 10 metrics).
5. Provide a risk register with mitigation plans for at least 5 risks.
6. State what “good enough” means for deployment.

### Deliverable format
- 1-page workflow diagram (boxes + arrows)
- 1-page assumptions/limitations note
- 1-page validation report template

---

## Optional coding mini-projects

1. **IBI notebook:** implement one IBI loop for a 1D pair potential.
2. **Force-matching toy model:** fit a pairwise potential to synthetic force labels.
3. **Transferability test:** train at one temperature and evaluate structural mismatch at another.

---

## Suggested answer discipline

For each exercise, write:
1. assumptions,
2. equations used,
3. computation steps,
4. physical interpretation,
5. limitations.

That habit is the fastest way to become strong at force-field coarse-graining.
