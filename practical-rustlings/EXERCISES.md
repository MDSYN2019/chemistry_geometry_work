# Practical Rustlings Exercises (40)

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

## Track F — PyO3 / Python Interop from Rust

### 21) `pyo3-function-signatures`
**Focus:** ergonomic Python-callable signatures.
- Design a `#[pyfunction]` API using `&str`, `Vec<f64>`, and optional keyword arguments.
- Compare one strict signature and one permissive signature.

**Done when:** call sites in Python are clear and Rust-side error cases are explicit.

### 22) `python-index-semantics`
**Focus:** Python negative-index behavior in Rust.
- Implement index normalization (`-1` means last element, etc.).
- Return clear errors for out-of-bounds access.

**Done when:** tests cover positive, negative, and invalid indexes.

### 23) `overflow-to-pyerr`
**Focus:** safe arithmetic surfaced as Python exceptions.
- Build a Rust helper using checked arithmetic.
- Map overflow to a Python-facing error shape.

**Done when:** overflow branches are test-covered and produce useful messages.

### 24) `kwargs-validation`
**Focus:** validating dynamic Python input.
- Accept a dict-like structure (`kwargs`) with required/optional keys.
- Convert and validate numbers (for example positive-only).

**Done when:** missing keys, wrong types, and invalid ranges are distinguishable.

### 25) `vectorized-bridge`
**Focus:** batch operations over Python lists/tuples.
- Implement a Rust batch transform suitable for `#[pyfunction]`.
- Avoid unnecessary cloning and support empty inputs.

**Done when:** function handles large inputs efficiently and tests include edge cases.

---

## Track G — Advanced Rust Patterns (More Complex)

### 26) `interior-mutability-bus`
**Focus:** `Rc<RefCell<T>>` tradeoffs and runtime borrow safety.
- Build a tiny event bus where handlers can enqueue follow-up events.
- Explore how shared mutable state can deadlock at runtime borrow boundaries.

**Done when:** tests demonstrate safe handler sequencing and avoid nested mutable borrows.

### 27) `concurrent-pipeline`
**Focus:** channels + worker coordination.
- Build a 3-stage pipeline (`parse -> transform -> aggregate`) using threads and channels.
- Ensure all sender handles are dropped so receivers terminate cleanly.

**Done when:** no hangs, deterministic output ordering strategy is documented, and shutdown is explicit.

### 28) `pin-and-self-reference`
**Focus:** why pinning exists and when self-referential structs are dangerous.
- Implement a safe API around pinned buffers or futures.
- Add notes explaining what cannot be expressed safely without pinning.

**Done when:** compile-time constraints prevent moves after pinning and tests verify address stability.

### 29) `async-timeouts-retries`
**Focus:** robust async orchestration.
- Build an async operation with timeout + retry policy and backoff.
- Classify retryable vs terminal errors.

**Done when:** tests cover timeout, transient failure recovery, and terminal failure behavior.

### 30) `ffi-boundary-safety`
**Focus:** safe wrappers around `unsafe` boundary.
- Create a tiny C-ABI style interface and a safe Rust wrapper.
- Validate UTF-8 / null pointer / ownership transfer assumptions.

**Done when:** all `unsafe` blocks are documented with invariants and wrapper API is panic-safe.

---

## Track H — Extra Advanced Folder Drills (`24_advanced_patterns`)

### 31) `advanced_patterns1_arc_mutex`
**Focus:** thread-safe shared mutation with `Arc<Mutex<T>>`.
- Spawn multiple threads incrementing one shared counter.

**Done when:** final count matches `threads * increments_per_thread`.

### 32) `advanced_patterns2_rwlock_cache`
**Focus:** read-heavy locking with `RwLock`.
- Build a small cache API with read and write methods.

**Done when:** repeated reads avoid write lock contention in design.

### 33) `advanced_patterns3_mpsc_fan_in`
**Focus:** producer fan-in over channels.
- Send values from N producers into one consumer sum.

**Done when:** all sender handles are dropped and receiver terminates naturally.

### 34) `advanced_patterns4_scoped_threads`
**Focus:** borrowing into threads with `std::thread::scope`.
- Split a slice and compute partial sums concurrently.

**Done when:** no cloning for borrowed halves and total is correct.

### 35) `advanced_patterns5_condvar_queue`
**Focus:** blocking synchronization primitives.
- Implement put/take semantics for a one-slot queue.

**Done when:** producer/consumer coordination works without busy-waiting.

### 36) `advanced_patterns6_pin_basics`
**Focus:** basic pinning semantics.
- Pin a heap allocation and validate stable pointer identity.

**Done when:** tests prove address stability while pinned.

### 37) `advanced_patterns7_retry_backoff`
**Focus:** robust retry loops.
- Retry a fallible closure for a fixed attempt budget.

**Done when:** success and final-error paths are both tested.

### 38) `advanced_patterns8_cow_normalize`
**Focus:** allocation-aware string APIs with `Cow<'_, str>`.
- Borrow unchanged input, allocate only for normalized output.

**Done when:** tests verify borrowed vs owned branches.

### 39) `advanced_patterns9_ffi_guard`
**Focus:** nullable pointer validation at FFI boundary.
- Convert `*const c_char` into safe `&str`-backed behavior.

**Done when:** null and invalid UTF-8 are handled as explicit errors.

### 40) `advanced_patterns10_state_machine`
**Focus:** compile-time state transitions.
- Build `Initialized -> Running -> Finished` typed flow.

**Done when:** impossible transitions are rejected by the type system.

---

## Starter + reference code in this repo

- `src/exercises.rs` provides starter APIs for exercises 1–30 in this document.
- `24_advanced_patterns/` provides an additional 10 file-based drills (31–40).
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
