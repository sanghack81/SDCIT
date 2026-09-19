"""Exercise integer matching costs using actual native sources and an oracle."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    "sanitized", [False, True] if os.environ.get("SDCIT_NATIVE_SANITIZERS") == "1" else [False]
)
def test_native_matching_costs(tmp_path, sanitized):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("a C++ compiler is required for the native regression")
    root = Path(__file__).resolve().parents[2]
    if not (root / "blossom5/PerfectMatching.h").is_file():
        pytest.skip("the bundled Blossom-V sources are required")
    sources = [
        "sdcit/cython_impl/SDCIT.cpp", "sdcit/cython_impl/permutation.cpp",
        "blossom5/MinCost/MinCost.cpp", "blossom5/misc.cpp",
        "blossom5/PMduals.cpp", "blossom5/PMexpand.cpp", "blossom5/PMinit.cpp",
        "blossom5/PMinterface.cpp", "blossom5/PMmain.cpp",
        "blossom5/PMrepair.cpp", "blossom5/PMshrink.cpp",
    ]
    driver = Path(__file__).with_name("native") / "matching_cost_regression.cpp"
    binary = tmp_path / "matching_cost_regression"
    flags = ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if sanitized else ["-O2"]
    command = [
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-pthread",
        *flags, "-I", str(root / "sdcit/cython_impl"),
        *(str(root / source) for source in sources), str(driver), "-o", str(binary),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    environment = dict(os.environ, UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    result = subprocess.run([str(binary)], capture_output=True, text=True,
                            env=environment, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "328 oracle cases" in result.stdout
