import numpy as np

from sdcit.hsic import c_HSIC, HSIC
from sdcit.kcit import python_kcit, python_kcit_K
from sdcit.sdcit_mod import SDCIT, c_SDCIT, shuffling
from sdcit.synthetic_data import henon
from sdcit.utils import rbf_kernel_median


def test_hsics():
    np.random.seed(0)

    X = np.random.randn(600, 3)
    Y = np.random.randn(600, 3) + 0.01 * X
    KX, KY = rbf_kernel_median(X, Y)
    t0, p0 = c_HSIC(KX, KY, n_jobs=1, size_of_null_sample=5000)
    p2 = HSIC(KX, KY, num_boot=5000)

    assert np.allclose([p0, p2], [0.0338, 0.0316], atol=0.005)


def test_reproducible():
    np.random.seed(0)

    X, Y, Z = henon(49, 200, 0.25, True)
    KX, KY, KZ = rbf_kernel_median(X, Y, Z)
    _, p1 = SDCIT(KX, KY, KZ, seed=55)
    _, p2 = c_SDCIT(KX, KY, KZ, seed=55)  # macOS and Linux may have different result.
    _, _, p3, *_ = python_kcit(X, Y, Z, seed=99)
    _, _, p4, *_ = python_kcit_K(KX, KY, KZ, seed=99)

    # [0.345, 0.347, 0.095, 0.0606]
    assert np.allclose([p1, p2, p3, p4], [0.345, 0.348, 0.095, 0.0606], atol=0.005, rtol=0)


def test_shuffling():
    X = np.arange(100).reshape((10, 10))
    Y = np.arange(100).reshape((10, 10))

    X, Y = shuffling(77, X, Y)
    print(np.allclose(X, Y))
