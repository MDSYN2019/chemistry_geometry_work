# C++ layout

The C++ code has been reorganized by intent:

- `apps/`: entry-point executables (currently `main.cxx`).
- `include/`: project headers.
- `src/core/`: reusable domain/core implementation files.
- `src/examples/`: smaller experiments and language/library demos.

Use the top-level `Makefile` to build the main application target.
