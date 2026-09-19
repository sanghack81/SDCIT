"""Reject unsupported sample sizes before entering native matching code."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sdcit.sdcit_mod import SDCIT, c_SDCIT


@pytest.mark.parametrize("implementation", ["SDCIT", "c_SDCIT"])
@pytest.mark.parametrize("size", [0, 1, 4, 6, 10])
def test_unsupported_sample_size_raises_without_exiting(implementation, size):
    # Older native matching calls exit(1) for odd half-sample sizes. Isolate
    # each call so a regression fails this test without terminating pytest.
    script = """
import sys
import numpy as np
from sdcit import sdcit_mod

implementation, size = sys.argv[1], int(sys.argv[2])
points = np.arange(size, dtype=float)
kernel = np.exp(-0.1 * (points[:, None] - points[None, :]) ** 2)
try:
    getattr(sdcit_mod, implementation)(
        kernel, kernel, kernel, seed=123, size_of_null_sample=4
    )
except ValueError as error:
    assert "at least 8" in str(error) and "divisible by 4" in str(error), str(error)
    print("caught expected ValueError")
else:
    raise AssertionError("Unsupported sample size was accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, implementation, str(size)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "caught expected ValueError" in result.stdout


@pytest.mark.parametrize("implementation", [SDCIT, c_SDCIT])
def test_minimum_supported_sample_size(implementation):
    kernel = np.eye(8)
    statistic, pvalue, null = implementation(
        kernel, kernel, kernel, seed=123, size_of_null_sample=4, with_null=True
    )
    assert np.isfinite(statistic)
    assert 0 < pvalue <= 1
    assert np.isfinite(null).all()


@pytest.mark.parametrize("implementation", [SDCIT, c_SDCIT])
@pytest.mark.parametrize("matrix_name", ["Kx", "Ky", "Kz", "Dz"])
def test_kernel_and_distance_shapes_must_match(implementation, matrix_name):
    matrices = {name: np.eye(8) for name in ("Kx", "Ky", "Kz", "Dz")}
    matrices[matrix_name] = np.ones((8, 1))
    with pytest.raises(ValueError, match="square matrices of the same shape"):
        implementation(**matrices, seed=123, size_of_null_sample=4)


@pytest.mark.parametrize("implementation", ["SDCIT", "c_SDCIT"])
@pytest.mark.parametrize(
    "case, message",
    [
        ("Kx_nan", "kernel matrices must contain only finite"),
        ("Ky_inf", "kernel matrices must contain only finite"),
        ("Kz_nan", "kernel matrices must contain only finite"),
        ("distance_nan", "distances must be finite and nonnegative"),
        ("distance_inf", "distances must be finite and nonnegative"),
        ("distance_negative", "distances must be finite and nonnegative"),
        ("distance_too_large", "distances are too large"),
        ("derived_overflow", "distances must be finite and nonnegative"),
    ],
)
def test_invalid_values_raise_without_terminating(implementation, case, message):
    script = """
import sys
import numpy as np
from sdcit import sdcit_mod

implementation, case, expected = sys.argv[1:]
matrices = {name: np.eye(8) for name in ('Kx', 'Ky', 'Kz')}
if case.startswith('distance_'):
    distance = np.ones((8, 8)) - np.eye(8)
    value = {
        'distance_nan': np.nan,
        'distance_inf': np.inf,
        'distance_negative': -1,
        'distance_too_large': np.finfo(float).max,
    }[case]
    distance[0, 1] = distance[1, 0] = value
    matrices['Dz'] = distance
elif case == 'derived_overflow':
    matrices['Kz'] *= 1e308
else:
    name, kind = case.split('_')
    matrices[name][0, 0] = np.nan if kind == 'nan' else np.inf
try:
    getattr(sdcit_mod, implementation)(
        **matrices, seed=123, size_of_null_sample=4
    )
except ValueError as error:
    assert expected in str(error), str(error)
    print('caught expected ValueError')
else:
    raise AssertionError('Invalid matrix values were accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, implementation, case, message],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "caught expected ValueError" in result.stdout


@pytest.mark.parametrize(
    "implementation", ["cy_sdcit", "cy_kcipt", "cy_dense_permutation", "cy_split_permutation"]
)
@pytest.mark.parametrize("invalid", ["negative", "nan"])
def test_native_wrappers_reject_invalid_distances(implementation, invalid):
    script = """
import sys
import numpy as np
from sdcit.cython_impl import cy_sdcit as native

implementation, invalid = sys.argv[1:]
kernel = np.eye(8)
distance = np.ones((8, 8)) - np.eye(8)
distance[0, 1] = distance[1, 0] = -1 if invalid == 'negative' else np.nan
try:
    if implementation == 'cy_sdcit':
        native.cy_sdcit(kernel, kernel, kernel, distance, 4, 123, 1,
                        np.zeros(1), np.zeros(1), np.zeros(4), np.zeros(4))
    elif implementation == 'cy_kcipt':
        native.cy_kcipt(kernel, kernel, kernel, distance, 2, 2,
                        np.zeros((2, 2)), np.zeros(2), 123, 1, np.zeros(4), 4)
    else:
        getattr(native, implementation)(distance, np.zeros(8, dtype='int32'), 123)
except ValueError as error:
    assert 'nonnegative' in str(error) and 'NaN' in str(error), str(error)
    print('caught expected ValueError')
else:
    raise AssertionError('Invalid native distance values were accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, implementation, invalid],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "caught expected ValueError" in result.stdout


@pytest.mark.parametrize("invalid", ["infinite", "huge"])
def test_native_sdcit_wrapper_rejects_unsafe_distance_ranges(invalid):
    script = """
import sys
import numpy as np
from sdcit.cython_impl.cy_sdcit import cy_sdcit

kernel = np.eye(8)
distance = np.ones((8, 8)) - np.eye(8)
distance[0, 1] = distance[1, 0] = (
    np.inf if sys.argv[1] == 'infinite' else np.finfo(float).max
)
try:
    cy_sdcit(kernel, kernel, kernel, distance, 4, 123, 1,
             np.zeros(1), np.zeros(1), np.zeros(4), np.zeros(4))
except ValueError as error:
    assert 'SDCIT distances' in str(error), str(error)
    print('caught expected ValueError')
else:
    raise AssertionError('Unsafe native SDCIT distances were accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, invalid],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "caught expected ValueError" in result.stdout


@pytest.mark.parametrize("implementation", ["cy_split_permutation", "cy_kcipt"])
def test_split_matching_preserves_infinite_missing_edges(implementation):
    script = """
import sys
import numpy as np
from sdcit.cython_impl import cy_sdcit as native

distance = np.full((8, 8), np.inf)
distance[:4, :4] = distance[4:, 4:] = 1
np.fill_diagonal(distance, 0)
if sys.argv[1] == 'cy_split_permutation':
    permutation = np.zeros(8, dtype='int32')
    native.cy_split_permutation(distance, permutation, 123)
    np.testing.assert_array_equal(np.sort(permutation), np.arange(8))
    assert np.all(permutation != np.arange(8))
    assert np.isfinite(distance[np.arange(8), permutation]).all()
else:
    kernel = np.ones((8, 8))
    inner, mmds, outer = np.zeros((2, 2)), np.zeros(2), np.zeros(4)
    native.cy_kcipt(kernel, kernel, kernel, distance, 2, 2,
                    inner, mmds, 123, 1, outer, 4)
    assert np.isfinite(inner).all() and np.isfinite(mmds).all()
    assert np.isfinite(outer).all()
print('infinite missing edges preserved')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, implementation],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "infinite missing edges preserved" in result.stdout
