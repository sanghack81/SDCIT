# Changelog

All notable changes to SDCIT are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-06-26

This is a **major release** that modernizes SDCIT for current Python, NumPy, SciPy,
Cython, and (most significantly) **gpflow 2.x**. The gpflow GP code path was rewritten
against an API that is not backward compatible with gpflow 1.x, hence the major version
bump.

### Breaking changes

- **gpflow 2.x is now required for GP-based tests** (KCIT, FCIT, GP residualization).
  The previous code targeted the gpflow **1.x** API, which depended on `tensorflow<2`
  — a combination that can no longer be installed on modern Python (TensorFlow 1.x
  ships wheels only up to CPython 3.7). All gpflow 1.x calls were ported:
  - `gpflow.train.ScipyOptimizer().minimize(m)` → `gpflow.optimizers.Scipy().minimize(m.training_loss, m.trainable_variables)`
  - `GPR(X, Y, kernel)` → `GPR((X, Y), kernel=kernel)`
  - `RBF(dim, ARD=True)` / `Linear(dim, ARD=True)` → `RBF(lengthscales=...)` / `Linear(variance=...)`
  - `kernel.compute_K_symm(X)` → `kernel(X)`
  - `param.value` → `param.numpy()`
- **`python_requires` is now `>=3.9`** (was effectively 3.8 in docs).
- The GP optional dependencies moved to an extra: install with `pip install '.[gp]'`
  (previously implied `gpflow<2.0` / `tensorflow<2.0`).

### Added

- `gp` install extra (`pip install '.[gp]'`) pulling `gpflow>=2.0` and `tensorflow>=2.0`.
- Internal gpflow 2.x helpers in `sdcit.utils`: `_require_gpflow()` (version-guards
  gpflow >= 2.0), `_optimize_gpr()`, `_kernel_matrix()`.
- Reproducibility: a `seed` is threaded through the minimum-weight perfect-matching
  permutation path (`permutation.cpp` / `cy_sdcit.pyx` / `sdcit_mod.py`). The seed is
  optional and defaults to a fresh random seed for backward compatibility.
- New tests: `test_utils.py`, `test_synthetic_data.py`, and a permutation-seed
  reproducibility test.
- Packaging: `pyproject.toml` (PEP 517 build backend), `MANIFEST.in`.

### Changed

- `setup.py` migrated from `distutils` to `setuptools`, with proper
  `install_requires` (`numpy`, `scipy>=1.5`, `scikit-learn`) instead of the
  deprecated/ignored `requires=['numpy']`.
- NumPy/SciPy API updates: dropped `numpy.matlib.repmat` (now a small `np.tile`
  wrapper); `scipy.linalg.eigh(eigvals=...)` → `subset_by_index=...`.
- Cython wrappers use typed memoryviews.
- `gpflow` is imported lazily and **version-guarded**: a gpflow 1.x install is now
  treated as "GP unavailable" rather than crashing at call time.
- Expanded docstrings across the public API.

### Fixed

- Clean-environment build: `requirements.txt` now lists build-time dependencies
  (`setuptools`, `cython`, `numpy`), so `python setup.py build_ext --inplace` no longer
  fails with `ModuleNotFoundError: No module named 'setuptools'`. `pip install -e .`
  remains the recommended path (build deps resolved via `pyproject.toml`).
- `requirements.txt` gained `pytest-cov` to match the `--cov` option in `setup.cfg`.

### Verified

- Core C++/Cython tests pass in a clean Python 3.12 environment (58/58).
- GP path runs end-to-end on gpflow 2.9.2 / TensorFlow 2.16: `residualize`,
  `residual_kernel`, `python_kcit`, `python_kcit_K`, and `FCIT`. The reproducibility
  test matches the historical gpflow-1.x reference p-values, confirming a
  numerically faithful port.
- No growing memory in the C++ extension across repeated `c_SDCIT`/`c_KCIPT`/`c_HSIC`
  calls (flat max-RSS).

## [1.3.0]

Superseded by 2.0.0 before release (the gpflow 1.x→2.x port is a breaking change, so
the version was promoted to 2.0.0). See the 2.0.0 entry for the full set of changes.

## [1.2.0]

Last release on the gpflow 1.x / TensorFlow 1.x stack.
