# Practical Rustlings: Design-Focused Practice Track

This track is meant for the stage **after basic syntax fluency**.

## Honest assessment (where you are now)

You have moved from beginner toward **solid early-intermediate Rust**:

- You are building nontrivial programs (not just reading examples).
- Remaining mistakes are mostly in:
  - ownership/container/type-shape mismatches (for example, unnecessary wrapping in `Ok(...)`),
  - conceptual-layer mixing (code compiles toward one abstraction while simulation logic expects another),
  - API design awkwardness (`Vec<System>` vs one global `System`),
  - physics/unit consistency bugs (scientific correctness beyond compiler correctness).

A simple summary:

> You are past “learning Rust keywords” and into “learning how to design software in Rust.”

That is real progress.

## Goal of this track

Shift practice from syntax to:

1. Type design
2. Ownership fluency
3. Interface/API clarity
4. Architectural discipline
5. Scientific invariants and unit consistency

## Recommended weekly cadence (6 weeks)

- **Mon (45 min):** ownership/type-shape drills
- **Wed (60 min):** architecture/API drills
- **Fri (45 min):** units/invariant drills
- **Sat (30 min):** retrospective notes (what design mistake repeated?)

## How to use

1. Start with `EXERCISES.md` and complete exercises in order.
2. For each exercise:
   - make a small crate in `exercises/<slug>/`,
   - add tests,
   - write 3–5 lines of reflection in `NOTES.md`.
3. Keep each exercise < 90 minutes.
4. Prioritize correctness and API design over cleverness.

## Success criteria

After one cycle, you should notice:

- fewer accidental clones,
- cleaner function signatures,
- fewer “shape conversion” mistakes,
- clearer separation between model/state/execution,
- better scientific trust (unit-safe code + invariants tested).

## Included Rust files

This folder now includes a small Rust crate you can use immediately:

- `Cargo.toml`
- `src/lib.rs`
- `src/exercises.rs` (starter surfaces for all 20 exercises)
- `src/references.rs` (compact working references for selected patterns)

Run locally:

```bash
cd practical-rustlings
cargo test
```

