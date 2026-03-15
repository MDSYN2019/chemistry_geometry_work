# psi4lings

A rustlings-style practice track for **Psi4** quantum chemistry workflows.

## How to use

1. Open an exercise in `psi4lings/exercises/...`.
2. Fill every `TODO` section.
3. Run the file with Python.
4. Compare your implementation with the matching answer in `psi4lings/solutions/...`.

## Suggested flow

- Start with 00-01 for molecule setup and first SCF energies.
- Continue with 02-03 for basis effects and open-shell references.
- Complete 04-05 to run optimization and vibrational analysis.
- Finish with 06-07 for molecular properties and a potential-energy scan.

## Quick check command

```bash
python psi4lings/check.py
python -m compileall psi4lings/exercises psi4lings/solutions
```

## Environment note

These exercises assume Psi4 is installed in your Python environment.
