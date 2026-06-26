[![SDCIT Tests](https://github.com/sanghack81/SDCIT/actions/workflows/python-app.yml/badge.svg)](https://github.com/sanghack81/SDCIT/actions)

SDCIT: Self-Discrepancy Conditional Independence Test
==

Overview
-------
`sdcit` is a package for testing conditional independence in python implementing **SDCIT** by Lee and Honavar (2017). The algorithm utilizes the notion of closeness among observations, defined by a kernel function, and conditional permutation, which allows us to yield a pseudo-null sample.

This algorithm depends on [`Blossom-V`](http://pub.ist.ac.at/~vnk/software/blossom5-v2.05.src.tar.gz) (Kolmogorov 2009), which is freely available for the research purpose. To be used commercially, users must buy [commercial license](http://pub.ist.ac.at/~vnk/software.html) for `Blossom-V`.



Installation
-----
This package requires `python>=3.9`. Other required packages are described in [requirements.txt](https://github.com/sanghack81/SDCIT/blob/master/requirements.txt). The following script will clone the SDCIT code, download `Blossom-V`, and install the package using `pip`.

```bash
git clone https://github.com/sanghack81/SDCIT
cd SDCIT
# Prepare C++ external dependencies (Blossom-V)
./setup.sh
# Build and install the package (build deps are resolved via pyproject.toml)
pip install -e .
```

The GP-based tests (KCIT, FCIT, GP residualization) are optional and require the
modern `gpflow` 2.x API. Install them with the `gp` extra:

```bash
pip install -e '.[gp]'   # adds gpflow>=2.0 and tensorflow>=2.0
```

> **Note:** `pip install -e .` is the recommended install — it resolves the
> build-time dependencies (`setuptools`, `cython`, `numpy`) automatically.
> Running `python setup.py build_ext --inplace` directly requires those three
> to be installed first.

### Running Tests
To verify your installation and run unit tests:
```bash
pip install pytest pytest-cov
pytest --cov=sdcit sdcit/tests/
```



Examples
-----
We provide three simple examples, where kernel matrices are computed based on median heuristic.

```python
import numpy as np
from sdcit.sdcit_mod import SDCIT
from sdcit.utils import rbf_kernel_median

np.random.seed(0)

N = 200
# Three independent random variables
X = np.random.randn(N, 2)
Y = np.random.randn(N, 2)
Z = np.random.randn(N, 2)
Kx, Ky, Kz = rbf_kernel_median(X, Y, Z)  # median heuristic
test_statistic, p_value = SDCIT(Kx, Ky, Kz)
print('p_value: {:.4f}'.format(p_value))

# (conditionally dependent)
# X --> Z <-- Y
Z = X + Y + np.random.randn(N, 2)
Kx, Ky, Kz = rbf_kernel_median(X, Y, Z)  # median heuristic
test_statistic, p_value = SDCIT(Kx, Ky, Kz)
print('p_value: {:.4f}'.format(p_value))

# (conditionally independent)
# X <-- Z --> Y
Z = np.random.randn(N, 2)
X = Z + np.random.randn(N, 2)
Y = Z + np.random.randn(N, 2)
Kx, Ky, Kz = rbf_kernel_median(X, Y, Z)  # median heuristic
test_statistic, p_value = SDCIT(Kx, Ky, Kz)
print('p_value: {:.4f}'.format(p_value))

```

References
-------


> Sanghack Lee, Vasant Honavar **Self-Discrepancy Conditional Independence Test**
> _Proceedings of the 33rd Conference on Uncertainty in Artificial Intelligence._ 2017. (to appear)


> Gary Doran, Krikamol Muandet, Kun Zhang, and Bernhard Schölkopf. **A Permutation-Based Kernel Conditional Independence Test**
> _Proceedings of the 30th Conference on Uncertainty in Artificial Intelligence._ 2014.


> Vladimir Kolmogorov. **Blossom V: A new implementation of a minimum cost perfect matching algorithm.**
>        In Mathematical Programming Computation (MPC), July 2009, 1(1):43-67.


