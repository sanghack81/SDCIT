[![Build Status](https://travis-ci.org/sanghack81/SDCIT.svg?branch=master)](https://travis-ci.org/sanghack81/SDCIT)

SDCIT: Self-Discrepancy Conditional Independence Test
==

Overview
-------
`sdcit` is a package for testing conditional independence in python implementing **SDCIT** by Lee and Honavar (2017). The algorithm utilizes the notion of closeness among observations, defined by a kernel function, and conditional permutation, which allows us to yield a pseudo-null sample.

This algorithm depends on [`Blossom-V`](http://pub.ist.ac.at/~vnk/software/blossom5-v2.05.src.tar.gz) (Kolmogorov 2009), which is freely available for the research purpose. To be used commercially, users must buy [commercial license](http://pub.ist.ac.at/~vnk/software.html) for `Blossom-V`. 

 
 
Installation
-----
You are required to use `python>=3.6`. Other required packages are described in [requirements.txt](https://github.com/sanghack81/SDCIT/blob/master/requirements.txt). Following scripts will download `sdcit` code together with `Blossom-V` and install it using `python3` and `pip` in your path.


```
cd ~/Downloads
git clone https://github.com/sanghack81/SDCIT
cd SDCIT
pip install -r requirements.txt
pip install 'gpflow<2.0' 'tensorflow<2.0'
./setup.sh
python3 setup.py build_ext --inplace
pip install -e .
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


