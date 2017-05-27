from sdcit.cython_impl.cy_kcipt import cy_kcipt, cy_adj_kcipt

from sdcit.utils import *


def c_KCIPT(K_X, K_Y, K_Z, D_Z, B, b, M, n_jobs=1, seed=None):
    K_X = np.ascontiguousarray(K_X, 'float64')
    K_Y = np.ascontiguousarray(K_Y, 'float64')
    K_Z = np.ascontiguousarray(K_Z, 'float64')
    D_Z = np.ascontiguousarray(D_Z, 'float64')

    inner_null = np.zeros((B, b), dtype='float64')
    outer_null = np.zeros((M,), dtype='float64')
    mmds = np.zeros((B,), dtype='float64')

    cy_kcipt(K_X, K_Y, K_Z, D_Z, B, b, inner_null, mmds, random_seeds() if seed is None else seed, n_jobs, outer_null, M)

    if M > 0:
        outer_null -= outer_null.mean()

    test_statistic = mmds.mean()

    return p_value_of(test_statistic, outer_null) if M > 0 else float('nan'), mmds, inner_null, outer_null


def c_adj_KCIPT(K_X, K_Y, K_Z, D_Z, B, b, M, n_jobs=1, seed=None):
    K_X = np.ascontiguousarray(K_X, 'float64')
    K_Y = np.ascontiguousarray(K_Y, 'float64')
    K_Z = np.ascontiguousarray(K_Z, 'float64')
    K_XYZ = np.ascontiguousarray(K_X * K_Y * K_Z, 'float64')
    D_Z = np.ascontiguousarray(D_Z, 'float64')

    inner_null = np.zeros((B, b), dtype='float64')
    outer_null = np.zeros((M,), dtype='float64')
    mmds = np.zeros((B,), dtype='float64')

    cy_adj_kcipt(K_X, K_Y, K_Z, K_XYZ, D_Z, B, b, inner_null, mmds, random_seeds() if seed is None else seed, n_jobs, outer_null, M, 1)

    if M > 0:
        outer_null -= outer_null.mean()

    test_statistic = mmds.mean()

    return p_value_of(test_statistic, outer_null) if M > 0 else float('nan'), mmds, inner_null, outer_null
