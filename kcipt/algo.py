from kcipt.utils import *


def c_KCIPT(K_X, K_Y, K_Z, D_Z, B, b, M, n_jobs=1):
    from kcipt.cython_impl.cy_kcipt import cy_kcipt

    K_X = np.ascontiguousarray(K_X, 'float64')
    K_Y = np.ascontiguousarray(K_Y, 'float64')
    K_Z = np.ascontiguousarray(K_Z, 'float64')
    D_Z = np.ascontiguousarray(D_Z, 'float64')

    inner_null = np.zeros((B, b), dtype='float64')
    outer_null = np.zeros((M,), dtype='float64')
    mmds = np.zeros((B,), dtype='float64')

    cy_kcipt(K_X, K_Y, K_Z, D_Z, B, b, inner_null, mmds, random_seeds(), n_jobs, outer_null, M)
    if b > 0:
        inner_null -= inner_null.mean()
    if M > 0:
        outer_null -= outer_null.mean()
    test_statistic = mmds.mean()
    return p_value_of(test_statistic, outer_null, approxmation=True) if M > 0 else float('nan'), mmds, inner_null, outer_null
