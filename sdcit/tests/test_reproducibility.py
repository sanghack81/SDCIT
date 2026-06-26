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

    import sdcit.kcit as kcit
    if kcit.gpflow is not None:  # gpflow>=2.0 available (1.x is treated as unavailable)
        _, _, p3, *_ = python_kcit(X, Y, Z, seed=99)
        _, _, p4, *_ = python_kcit_K(KX, KY, KZ, seed=99)
        assert np.allclose([p1, p2, p3, p4], [0.345, 0.348, 0.095, 0.0606], atol=0.005, rtol=0)
    else:
        assert np.allclose([p1, p2], [0.345, 0.348], atol=0.005, rtol=0)


def test_shuffling():
    X = np.arange(100).reshape((10, 10))
    Y = np.arange(100).reshape((10, 10))

    X, Y = shuffling(77, X, Y)
    print(np.allclose(X, Y))


def test_permutation_seed():
    from sdcit.sdcit_mod import permuted

    # D is all zeros, so Blossom-V gets 0 distance and relies entirely on random tie-breaking / fallback
    D = np.zeros((10, 10))
    p1 = permuted(D, seed=42, dense=True)
    p2 = permuted(D, seed=42, dense=True)
    p3 = permuted(D, seed=99, dense=True)

    assert np.array_equal(p1, p2)
    assert not np.array_equal(p1, p3)

    p4 = permuted(D, seed=42, dense=False)
    p5 = permuted(D, seed=42, dense=False)
    p6 = permuted(D, seed=99, dense=False)

    assert np.array_equal(p4, p5)
    assert not np.array_equal(p4, p6)
