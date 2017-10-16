import numpy as np

from sdcit.hsic import c_HSIC, HSIC
from sdcit.kcit import python_kcit, python_kcit_K
from sdcit.sdcit import SDCIT, c_SDCIT, shuffling
from sdcit.synthetic_data import henon
from sdcit.utils import rbf_kernel_median


def test_hsics():
    np.random.seed(0)

    X = np.random.randn(600, 3)
    Y = np.random.randn(600, 3) + 0.01 * X
    KX, KY = rbf_kernel_median(X, Y)
    import time
    at0 = time.time()
    t0, p0 = c_HSIC(KX, KY, n_jobs=1, size_of_null_sample=5000)
    at1 = time.time()
    t1, p1 = c_HSIC(KX, KY, n_jobs=4, size_of_null_sample=5000)
    at2 = time.time()
    p2 = HSIC(KX, KY, num_boot=5000)
    at3 = time.time()
    print(f'Python:{at3-at2:.3f} vs C4:{at2-at1:.3f} vs C1:{at1-at0:.3f}')

    print(t0, t1)
    print(p0, p1, p2)


def test_reproducible():
    np.random.seed(0)

    X, Y, Z = henon(49, 200, 0.25, True)
    KX, KY, KZ = rbf_kernel_median(X, Y, Z)
    _, p1 = SDCIT(KX, KY, KZ, seed=55)
    _, p2 = c_SDCIT(KX, KY, KZ, seed=55)  # macOS and Linux may have different result.
    _, _, p3, *_ = python_kcit(X, Y, Z, seed=99)
    _, _, p4, *_ = python_kcit_K(KX, KY, KZ, seed=99)

    print(p1, p2, p3, p4)
    assert np.allclose([p1, p3, p4], [0.345, 0.095, 0.0606])


def test_shuffling():
    X = np.arange(100).reshape((10, 10))
    Y = np.arange(100).reshape((10, 10))

    X, Y = shuffling(77, X, Y)
    print(np.allclose(X, Y))
    print(X)
