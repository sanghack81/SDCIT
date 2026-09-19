"""Regression against the native source, independent of the Python extension.

Set SDCIT_NATIVE_SANITIZERS=1 to run AddressSanitizer and UndefinedBehaviorSanitizer
as well. The C++ driver stubs the matching symbol, which the shuffle never calls.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    "sanitized", [False, True] if os.environ.get("SDCIT_NATIVE_SANITIZERS") == "1" else [False]
)
def test_native_shuffle_matrix(tmp_path, sanitized):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the native regression")
    root = Path(__file__).resolve().parents[1]
    source = root / "cython_impl" / "SDCIT.cpp"
    driver = Path(__file__).with_name("native") / "shuffle_matrix_regression.cpp"
    binary = tmp_path / "shuffle_matrix_regression"
    flags = ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if sanitized else ["-O2"]
    command = [
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
        # Existing MMSD return-by-move is unrelated to this helper regression.
        "-Wno-pessimizing-move", "-pthread",
        *flags, "-I", str(source.parent), str(source), str(driver), "-o", str(binary),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "760 cases passed" in result.stdout
