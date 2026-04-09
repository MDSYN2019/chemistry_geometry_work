# Practical Rustlings Exercises (90)

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



## Track I — Collections / Data Processing

### 41) `const-generics-window`
**Focus:** fixed-size windows and predictable aggregation.
- Build moving-average helpers over contiguous windows.
- Compare ergonomics of specialized (`N=3`) vs generic signatures.

**Done when:** short and long slices are both handled cleanly.

### 42) `phantom-type-phase`
**Focus:** marker types to prevent phase/unit confusion.
- Model at least two temperature phases with phantom types.
- Permit only explicit conversion steps.

**Done when:** mixed-phase usage fails at compile-time.

### 43) `iterator-chunking`
**Focus:** chunk-wise transformations with `chunks` and iterators.
- Sum or reduce each chunk independently.
- Define behavior when final chunk is shorter.

**Done when:** edge cases (`chunk_size = 0`, uneven input) are tested.

### 44) `serde-boundary-plan`
**Focus:** parsing boundaries before domain logic.
- Parse `key=value` records into typed domain fields.
- Separate parse errors from semantic validation errors.

**Done when:** parser helpers can be reused without pulling in business logic.

### 45) `deterministic-rng-injection`
**Focus:** testability through dependency injection.
- Define a tiny RNG trait and inject it into domain code.
- Use deterministic fake RNG in tests.

**Done when:** behavior is reproducible and independent from global randomness.

## Track J — Algorithmic Fluency / Stdlib Mastery

### 46) `btreemap-vs-hashmap`
**Focus:** choosing map semantics intentionally.
- Implement counting with both `HashMap` and `BTreeMap`.
- Compare deterministic ordering vs average-case speed.

**Done when:** API docs explain the chosen map tradeoff.

### 47) `saturating-vs-checked`
**Focus:** overflow strategy as an API contract.
- Implement one saturating and one checked accumulator.
- Document where each behavior is appropriate.

**Done when:** callers can clearly choose desired overflow policy.

### 48) `slice-pattern-matching`
**Focus:** expressive branching with slice patterns.
- Classify fixed-length input patterns with guarded matches.
- Keep fallback behavior explicit.

**Done when:** pattern branches are exhaustive and readable.

### 49) `binary-search-contract`
**Focus:** lower-bound style APIs and sorted preconditions.
- Wrap `binary_search` into a stable insertion-point helper.
- Clarify behavior for hit and miss paths.

**Done when:** contract is documented and unit-tested on duplicates.

### 50) `small-dsl-evaluator`
**Focus:** lightweight parsing + safe arithmetic.
- Evaluate a tiny arithmetic DSL (for example `+` and `*`).
- Use checked math to surface overflow errors.

**Done when:** parser failures and overflow paths map to explicit error variants.



## Track K — Production Rust Workflow Topics

### 51) `trait-object-dispatch`
**Focus:** dynamic dispatch and trait-object boundaries.
- Pass behavior as `&dyn Trait` and evaluate call-site ergonomics.
- Compare with generic dispatch for hot loops.

**Done when:** tradeoffs are documented and both forms compile cleanly.

### 52) `enum-driven-dispatch`
**Focus:** explicit strategy selection via enums.
- Model transform behavior with an enum and `match`.
- Keep exhaustiveness checks working for future variants.

**Done when:** adding a new strategy yields clear compiler guidance.

### 53) `builder-default-overrides`
**Focus:** safe defaults with explicit overrides.
- Add a config type with useful defaults.
- Validate user overrides before building runtime state.

**Done when:** invalid overrides fail early with structured errors.

### 54) `derive-more-manual-impl`
**Focus:** mixing derives with hand-written trait impls.
- Use `derive` where possible and implement remaining traits manually.
- Explain correctness constraints for manual impls.

**Done when:** semantics are clear and trait impl set is coherent.

### 55) `parsing-state-machine`
**Focus:** incremental parsing with stable contracts.
- Parse a small structured string (`a:b`) with robust error handling.
- Isolate parse helpers from domain logic.

**Done when:** malformed inputs are rejected deterministically.

### 56) `result-collect-partition`
**Focus:** gathering successes while counting failures.
- Partition parsed values into accepted/rejected groups.
- Keep error accounting cheap and explicit.

**Done when:** caller can inspect both output values and failure count.

### 57) `lifetime-carrying-view`
**Focus:** returning borrowed views without allocations.
- Expose borrowed token slices tied to input lifetime.
- Avoid unnecessary `String` allocations.

**Done when:** API conveys borrowing constraints directly in signature.

### 58) `path-dependent-errors`
**Focus:** domain-specific error mapping.
- Map distinct failure paths to distinct error variants.
- Keep error branches easy to pattern-match by caller.

**Done when:** each branch has a test and clear semantic meaning.

### 59) `map-entry-api`
**Focus:** `entry`-based in-place mutation.
- Update counters atomically with `entry`.
- Avoid extra lookups and temporary allocations.

**Done when:** implementation uses one map lookup path.

### 60) `mini-benchmark-harness`
**Focus:** repeatable local performance probes.
- Build a tiny repeat-run harness for closures.
- Keep interface generic and zero-cost in release mode.

**Done when:** harness is reusable across prior exercises.

## Track L — `?` Operator Mastery

### 61) `question-mark-boundary`
**Focus:** understanding where `?` is legal.
- Write one function that fails to compile with `?` in a `fn -> i32`.
- Refactor it into `fn -> Result<i32, E>` and explain the change.

**Done when:** you can explain, in your own words, why return type compatibility is required.

### 62) `option-to-result-bridge`
**Focus:** using `ok_or`/`ok_or_else` before `?`.
- Parse config from a map where keys are optional.
- Convert missing values (`Option`) into domain errors (`Result`) and propagate with `?`.

**Done when:** missing key and parse failures produce distinct error variants.

### 63) `result-to-option-bridge`
**Focus:** using `?` with `Option`.
- Implement a function returning `Option<T>` that calls helpers returning `Option`.
- Compare with an equivalent `Result` version and note tradeoffs.

**Done when:** both versions pass tests and the loss of error detail is explicit.

### 64) `error-conversion-with-from`
**Focus:** implicit conversion during `?` propagation.
- Define two error enums and implement `From<InnerErr> for OuterErr`.
- Use `?` across helper calls without manual `map_err`.

**Done when:** conversion happens automatically and tests cover multiple failure sources.

### 65) `main-return-result`
**Focus:** using `?` in executable entry points.
- Change a small CLI-style `main` from panicking calls to `main() -> Result<(), E>`.
- Propagate parse + I/O style errors with `?`.

**Done when:** no `unwrap`/`expect` is needed in the happy path and error output remains useful.


## Track M — Async Rust with Tokio

### 66) `tokio-spawn-join`
**Focus:** task spawning and structured result collection.
- Spawn several async tasks with `tokio::spawn`.
- Await all `JoinHandle`s and collect outputs.

**Done when:** all task results are gathered deterministically and no handles are leaked.

### 67) `tokio-shared-state`
**Focus:** async-aware synchronization primitives.
- Share mutable state using `Arc<tokio::sync::Mutex<T>>`.
- Update state from multiple concurrently running tasks.

**Done when:** final state is correct and locking points are minimal and explicit.

### 68) `tokio-channel-pipeline`
**Focus:** producer/consumer coordination with async channels.
- Build a mini pipeline with `tokio::sync::mpsc`.
- Ensure proper shutdown by dropping senders and draining receiver loops.

**Done when:** pipeline terminates cleanly and aggregates expected values.


### 69) `tokio-select-race`
**Focus:** cancellation behavior with `tokio::select!`.
- Race two operations and return the earliest completion.
- Verify the losing branch is cancelled cleanly.

**Done when:** earliest result is deterministic under controlled timing.

### 70) `tokio-timeout-retry`
**Focus:** resilient async orchestration.
- Wrap operations in `tokio::time::timeout`.
- Implement bounded retries for transient failures.

**Done when:** timeout and retry behavior are both test-covered.

### 71) `tokio-broadcast-shutdown`
**Focus:** cooperative shutdown signaling.
- Run multiple workers listening on `tokio::sync::broadcast`.
- Send one shutdown signal and join workers predictably.

**Done when:** all workers terminate without hangs and report completion.



## Track N — Systems Programming (Low-level + OS-facing)

### 72) `ring-buffer-bytes`
**Focus:** fixed-capacity FIFO buffers without reallocations.
- Implement byte-oriented push/pop semantics over a circular buffer.
- Handle full/empty cases explicitly.

**Done when:** wrap-around behavior is correct and no allocations occur after initialization.

### 73) `endian-encoding`
**Focus:** explicit endianness at protocol boundaries.
- Encode/decode integer headers in little-endian format.
- Add cross-check tests against known byte sequences.

**Done when:** round-trip tests pass and byte order is documented.

### 74) `packed-header-layout`
**Focus:** layout predictability for wire formats.
- Define a compact header struct with explicit field serialization.
- Keep layout assumptions isolated from business logic.

**Done when:** serialized bytes match spec and field extraction is deterministic.

### 75) `bounded-io-slice`
**Focus:** safe slicing for parser boundaries.
- Build helpers that return subslices with overflow-safe index math.
- Reject out-of-range windows with structured errors.

**Done when:** fuzz-like edge cases (`start+len` overflow, short buffers) are covered.

### 76) `atomic-progress-counter`
**Focus:** lock-free counters with explicit memory ordering.
- Increment a shared counter from multiple workers.
- Compare relaxed vs stronger ordering choices.

**Done when:** correctness tests pass and ordering rationale is explained.

### 77) `resource-guard-drop`
**Focus:** RAII for deterministic cleanup.
- Model a guard that toggles or closes a resource in `Drop`.
- Ensure cleanup happens on early returns and panic paths.

**Done when:** tests prove cleanup without manual close calls.

### 78) `polling-state-machine`
**Focus:** event-loop style state transitions.
- Implement a minimal poll state machine (`Idle/Readable/Writable/Closed`).
- Reject impossible event transitions.

**Done when:** transition table is explicit and invalid paths return meaningful errors.

### 79) `bounded-queue`
**Focus:** backpressure-aware queue APIs.
- Build a bounded queue push API that fails when full.
- Define behavior for capacity 0 and consumer lag.

**Done when:** producer behavior under pressure is deterministic and test-covered.

### 80) `arena-indices`
**Focus:** index-based ownership instead of references.
- Store values in an arena and return stable numeric handles.
- Avoid borrow checker contention in graph-like data.

**Done when:** insert/get semantics are O(1) and invalid handles are safe.

### 81) `bitmask-permissions`
**Focus:** compact capability representation.
- Represent read/write/execute permissions with bitflags.
- Provide helper APIs for checking combined permissions.

**Done when:** permission checks are branch-light and easy to compose.

### 82) `rolling-checksum`
**Focus:** cheap integrity checks.
- Implement a small checksum primitive for byte streams.
- Discuss tradeoffs vs stronger cryptographic hashes.

**Done when:** incremental updates and known-vector tests pass.

### 83) `free-list-reuse`
**Focus:** slot reuse and stable IDs.
- Build a tiny slot map/free-list style container.
- Reuse freed slots to avoid unbounded growth.

**Done when:** remove/insert cycles reuse indices predictably.

### 84) `zero-copy-field-parse`
**Focus:** borrowing from byte buffers.
- Parse headers/payloads into borrowed subslices.
- Avoid allocating intermediate `Vec`/`String` values.

**Done when:** API lifetimes prevent use-after-free and allocations are minimized.

### 85) `io-error-mapping`
**Focus:** translating OS/I/O failures to domain errors.
- Map `std::io::ErrorKind` into your own error taxonomy.
- Preserve meaningful distinctions for callers.

**Done when:** retryable vs terminal failures are distinguishable.

### 86) `alignment-check`
**Focus:** pointer alignment and low-level safety assumptions.
- Write helpers that validate alignment requirements for typed access.
- Document why misalignment is dangerous.

**Done when:** aligned/misaligned test cases are explicit and easy to reason about.

### 87) `stack-vs-heap-buffer`
**Focus:** buffer placement tradeoffs.
- Compare fixed-size stack scratch space with heap allocation fallback.
- Benchmark for small/large sizes.

**Done when:** threshold choice is justified with measured data.

### 88) `cooperative-tick-scheduler`
**Focus:** deterministic task scheduling basics.
- Implement round-robin tick distribution over task counters.
- Keep scheduler state explicit and serializable.

**Done when:** fairness and edge cases (`0 tasks`, `0 ticks`) are test-covered.

### 89) `length-prefixed-protocol`
**Focus:** robust framing for binary protocols.
- Encode payloads with explicit length prefix.
- Validate frame size before exposing payload slices.

**Done when:** malformed and truncated frames produce clear errors.

### 90) `bump-allocator-drill`
**Focus:** monotonic allocation patterns.
- Implement a tiny bump allocator over a preallocated byte buffer.
- Return mutable slices for each allocation request.

**Done when:** overflow/capacity paths are safe and allocation order is deterministic.

## Starter + reference code in this repo

- `src/exercises.rs` provides starter APIs for exercises 1–30 and 41–90 in this document.
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
