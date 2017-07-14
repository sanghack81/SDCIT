import numpy as np

from sdcit.kcit import python_kcit, python_kcit_K
from sdcit.sdcit import SDCIT, c_SDCIT, shuffling
from sdcit.synthetic_data import henon
from sdcit.utils import rbf_kernel_median


def test_reproducible():
    np.random.seed(0)

    X, Y, Z = henon(49, 200, 0.25, True)
    KX, KY, KZ = rbf_kernel_median(X, Y, Z)
    _, p1 = SDCIT(KX, KY, KZ, seed=55)
    _, p2 = c_SDCIT(KX, KY, KZ, seed=55)
    _, _, p3, *_ = python_kcit(X, Y, Z, seed=99)
    _, _, p4, *_ = python_kcit_K(KX, KY, KZ, seed=99)

    print(p1, p2, p3, p4)
    assert np.allclose([p1, p2, p3, p4], [0.345, 0.337, 0.095, 0.0606])


def test_shuffling():
    X = np.arange(100).reshape((10, 10))
    Y = np.arange(100).reshape((10, 10))

    X, Y = shuffling(77, X, Y)
    print(np.allclose(X, Y))
    print(X)
