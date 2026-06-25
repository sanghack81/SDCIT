# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SDCIT (Self-Discrepancy Conditional Independence Test) is a Python package implementing kernel-based conditional independence tests from Lee and Honavar (2017). It tests whether X is conditionally independent of Y given Z, using kernel matrices and conditional permutation via minimum-weight perfect matching (Blossom-V).

## Build & Install

The package has a Cython/C++ extension that links against Blossom-V (external dependency):

```bash
# Download and extract Blossom-V into blossom5/ directory
./setup.sh
# Build extensions and install in editable mode
pip install -e .
# Or build extensions in-place only
python setup.py build_ext --inplace
```

The `blossom5/` directory must exist before building. It is downloaded by `setup.sh` from the Blossom-V website.

## Running Tests

```bash
pytest sdcit/tests/
# Run a single test
pytest sdcit/tests/test_reproducibility.py::test_hsics
```

Coverage is configured in `setup.cfg` (`--cov=sdcit --cov-report=term-missing`).

## Architecture

All statistical tests take **kernel matrices** (N×N numpy arrays) as primary inputs, not raw data. Use `rbf_kernel_median()` from `sdcit.utils` to compute kernel matrices with median heuristic.

### Core test functions

- **`sdcit.sdcit_mod.SDCIT`** — Pure Python SDCIT. Returns `(test_statistic, p_value)`.
- **`sdcit.sdcit_mod.c_SDCIT`** — C++-accelerated SDCIT via Cython. Same interface, supports `n_jobs` for threading.
- **`sdcit.kcipt.c_KCIPT`** — C++-accelerated KCIPT (Doran et al. 2014). Permutation-based kernel CI test.
- **`sdcit.kcit.python_kcit`** — KCIT (Zhang et al. 2011). Takes raw data (X, Y, Z), not kernel matrices. Requires `gpflow` (<2.0) and `tensorflow` (<2.0).
- **`sdcit.kcit.python_kcit_K`** — KCIT variant that takes kernel matrices instead of raw data.
- **`sdcit.hsic.HSIC` / `c_HSIC`** — Hilbert-Schmidt Independence Criterion (unconditional independence test).
- **`sdcit.flaxman.FCIT`** — Flaxman et al. (2016) residualization-based CI test. Requires `gpflow`.

### Cython/C++ layer

`sdcit/cython_impl/cy_sdcit.pyx` wraps C++ implementations:
- `SDCIT.cpp` — SDCIT core (MMSD computation, null distribution)
- `KCIPT.cpp` — KCIPT core (MMD computation, bootstrap)
- `HSIC.cpp` — HSIC bootstrap
- `permutation.cpp` — Minimum-weight perfect matching permutation via Blossom-V (`split_permutation` and `dense_2n_permutation`)

### Key utilities (`sdcit.utils`)

- `rbf_kernel_median(*arrays)` — RBF kernel matrices with median heuristic bandwidth
- `K2D(K)` — Convert kernel matrix to RKHS distance matrix
- `centering(M)` — Center a kernel matrix (H @ M @ H)
- `residual_kernel(K_Y, K_X)` — Kernel matrix of residual Y|X (requires `gpflow`)
- `cythonize(*matrices)` — Cast to contiguous float64 for C++ layer

### Synthetic data (`sdcit.synthetic_data`)

- `henon(seed, n, gamma, independence)` — Chaotic time series (Hénon map)
- `zhang2012(seed, N, dimensions, the_case, independent)` — Post-nonlinear noise model

## Important Notes

- `gpflow` and `tensorflow` are optional dependencies, only needed for KCIT, FCIT, and GP-based residualization. The core SDCIT/KCIPT/HSIC tests work without them.
- Blossom-V is free for research but requires a commercial license for commercial use.
- On macOS, the extension compiles with `-std=c++17`; on Linux with `-std=c++11`.
- Test tolerances in `test_reproducibility.py` may differ between macOS and Linux due to platform-specific numerical behavior in Blossom-V.
