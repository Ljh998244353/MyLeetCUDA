# Repository Guidelines

## Project Structure & Module Organization

- `kernels/`: primary implementations. Each subfolder (`kernels/<kernel-name>/`) is typically self-contained with CUDA/C++ (`*.cu`, `*.cuh`), optional PyTorch bindings (`pybind/`), and a small Python runner/benchmark (`*.py`).
- `third-party/`: git submodules (notably `third-party/cutlass`).
- `HGEMM/`, `ffpa-attn/`: additional submodules used by some kernels.
- `docs/`, `slides/`, `others/`: supporting notes, references, and experiments.

When adding a new kernel, use `kernels/elementwise/` as the reference layout (code + runnable Python entry + short README).

## Build, Test, and Development Commands

- Submodules (required for some kernels):
  - `git submodule update --init --recursive --force`
- Pre-commit (recommended before PRs):
  - `pip3 install pre-commit && pre-commit install`
  - `pre-commit run --all-files`
- Example: build/install a PyTorch CUDA extension (HGEMM):
  - `cd kernels/hgemm && python3 setup.py bdist_wheel`
  - `cd kernels/hgemm/dist && python3 -m pip install *.whl`
- Example: run a JIT-built kernel benchmark (common pattern):
  - `cd kernels/elementwise && python3 elementwise.py`
  - `cd kernels/flash-attn && python3 flash_attn_mma.py --D 64`
- Example: native C++ benchmark (when a kernel provides a `makefile`):
  - `cd kernels/hgemm && make && ./hgemm_mma_stage.bin`
- Optional arch control for builds:
  - `export TORCH_CUDA_ARCH_LIST=Ada` (or `Ampere`, etc.)

## Coding Style & Naming Conventions

- Python: format with Black (line length 80) and sort imports with isort (`profile=black`).
- C++/CUDA: follow existing 2-space indentation and keep diffs minimal; run the `clang-format` pre-commit hook when it works locally (avoid sweeping reformat-only diffs).
- Naming: prefer descriptive kernel/function names that encode dtype/vectorization (e.g., `*_f16x8_*`, `*_kernel`) and keep new files within the owning `kernels/<name>/` directory.

## Testing Guidelines

- There is no single monolithic test suite; tests live next to the relevant kernel.
- If you add tests, use `pytest` and name files `test_*.py` (example):
  - `cd kernels/openai-triton/merge-attn-states && pytest -s test_merge_attn_states.py`
- Include correctness checks (tolerances + dtype coverage) and gate GPU-specific tests with clear skips when CUDA isn’t available.

## Commit & Pull Request Guidelines

- Commit messages follow a Conventional Commits style seen in history: `feat: ...`, `fix(scope): ...`, `chore: ...`, `misc: ...`.
- PRs should include:
  - What changed + where (`kernels/<name>/...`), and how to run it (one command).
  - Any required submodules/dependencies.
  - If performance-related, include a brief benchmark note (GPU, CUDA, shapes).
