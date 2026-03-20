# Practical Rustlings Exercises (20)

These exercises are grouped by the skill gaps identified in your assessment.

---

## Track A — Ownership / Container / Type-Shape

### 1) `ok-or-not-ok`
**Focus:** `Result<T, E>` shape discipline.
- Write 10 small transform functions that return `Result`.
- Ensure only fallible branches produce `Err`.
- Avoid redundant `Ok(...)` wrapping when not needed.

**Done when:** tests show expected success/failure paths and no unnecessary wrappers in review.

### 2) `option-result-flip`
**Focus:** `Option<Result<T, E>>` <-> `Result<Option<T>, E>`.
- Implement two conversion helpers and compare ergonomics.
- Use a config parsing example with optional fields.

**Done when:** property-style tests cover `None`, `Some(Ok)`, `Some(Err)`.

### 3) `borrow-vs-own-api`
**Focus:** `&str`/`String`, `&[T]`/`Vec<T>` API choices.
- Provide two APIs for the same operation.
- Benchmark ergonomics with call sites.

**Done when:** final API accepts borrowed data where ownership is unnecessary.

### 4) `clone-budget`
**Focus:** reducing accidental allocation/copies.
- Start with code using many `.clone()` calls.
- Refactor with references, iterators, and lifetimes.

**Done when:** clone count is explicitly minimized and justified.

---

## Track B — Conceptual Layers / Architecture

### 5) `pure-model-vs-runner`
**Focus:** separating equations from simulation state.
- Module A: pure, deterministic physics functions.
- Module B: mutable runner, time integration, history.

**Done when:** model layer has no global mutable state.

### 6) `state-machine-sim`
**Focus:** explicit lifecycle modeling.
- Define states (`Initialized`, `Configured`, `Running`, `Finished`).
- Enforce transitions via typed API.

**Done when:** invalid transitions are unrepresentable or return strong errors.

### 7) `adapter-two-representations`
**Focus:** anti-corruption layer between abstractions.
- Build particle-centric and grid-centric representations.
- Add adapters with clear invariants.

**Done when:** conversion code is isolated and test-covered.

### 8) `dependency-direction-check`
**Focus:** architectural discipline.
- Organize modules so high-level policy does not depend on low-level details.
- Use traits at boundaries.

**Done when:** compile graph reflects intended architecture.

---

## Track C — API Design / Interface Quality

### 9) `one-system-or-many`
**Focus:** domain-driven return types.
- Implement both designs: `Vec<System>` and `System`.
- Evaluate with misuse examples.

**Done when:** you document which interface better matches domain truth and why.

### 10) `typestate-builder`
**Focus:** compile-time required-field enforcement.
- Build `SimulationBuilder` with typed phases.

**Done when:** impossible to build invalid simulation config without unsafe escapes.

### 11) `error-taxonomy`
**Focus:** rich domain errors over ad-hoc strings.
- Define `enum SimError` variants (e.g., `UnitMismatch`, `NonConvergent`).
- Add context fields.

**Done when:** caller can pattern-match and act differently per error kind.

### 12) `iterator-friendly-api`
**Focus:** zero-cost flexible input handling.
- Accept `IntoIterator` where appropriate.
- Compare with concrete container APIs.

**Done when:** API is more reusable without losing readability.

---

## Track D — Physics / Unit Consistency (Scientific Correctness)

### 13) `newtype-units`
**Focus:** units as types.
- Create newtypes: `Meters`, `Seconds`, `Kilograms`, `Joules`.
- Implement valid operations only.

**Done when:** impossible (or hard) to accidentally add incompatible units.

### 14) `dimensional-analysis-tests`
**Focus:** protecting formulas.
- Encode dimension expectations in tests for core equations.

**Done when:** tests fail when unit dimensions are perturbed.

### 15) `conservation-laws`
**Focus:** invariant-based testing.
- Add checks for mass/energy conservation in toy simulations.

**Done when:** invariant regressions fail fast with clear diagnostics.

### 16) `fault-injection-units`
**Focus:** resilience to realistic mistakes.
- Deliberately pass cm as m, ms as s, etc.
- Ensure typed or runtime guards catch errors.

**Done when:** each injected bug has a failing test and a fix.

---

## Track E — Integration / Design Maturity

### 17) `mini-md-engine`
**Focus:** coherent architecture under load.
- Build a tiny molecular dynamics core with pluggable integrators.

**Done when:** integrator swap needs no changes in force model code.

### 18) `io-boundary-clean`
**Focus:** serialization and domain separation.
- Keep parsing/format concerns separate from simulation domain types.

**Done when:** domain logic tests run without file I/O.

### 19) `refactor-for-clarity`
**Focus:** deliberate redesign.
- Take one previous exercise and refactor API after writing usage pain points.

**Done when:** call sites become shorter/clearer with unchanged behavior.

### 20) `capstone-review`
**Focus:** synthesis + communication.
- Choose one integrated simulation exercise.
- Produce short design doc:
  - ownership strategy,
  - abstraction boundaries,
  - error model,
  - unit/invariant strategy.

**Done when:** another developer can understand and extend your design quickly.

---

## Starter + reference code in this repo

- `src/exercises.rs` provides starter APIs for every exercise number in this document.
- `src/references.rs` provides compact, working references for selected exercises (transpose, typestate builder, units).

You can either:

1. work directly in `src/exercises.rs`, or
2. create per-exercise crates inside `exercises/<slug>/` as you iterate.

If you create per-exercise crates, include:

- `Cargo.toml`
- `src/lib.rs`
- `tests/integration.rs`
- `NOTES.md` (what hurt, what improved, what to refactor)

## Optional scoring rubric

Score each exercise 1–5 on:

1. Type clarity
2. Ownership ergonomics
3. API ergonomics
4. Architectural separation
5. Scientific correctness guarantees

Track the trend weekly. Improvement in consistency is more important than speed.
