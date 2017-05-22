import numpy as np
import numpy.ma as ma
from sklearn.linear_model import LinearRegression

from kcipt.cython_impl.cy_kcipt import cy_sdcit2
from kcipt.sdcit import penalized_distance, perm_and_mask
from kcipt.utils import K2D, p_value_of, random_seeds


def MMSD(ky: np.ndarray, kz: np.ndarray, kxz: np.ndarray, Dz: np.ndarray):
    """Maximum Mean Self-Discrepancy"""
    n = len(kxz)
    full_idx = np.arange(0, n)

    mask, perm = perm_and_mask(Dz)
    ky_fp = ky[np.ix_(full_idx, perm)]
    ky_pp = ky[np.ix_(perm, perm)]

    kk = (ky + ky_pp - 2 * ky_fp)
    statistic = ma.array(kxz * kk, mask=mask).mean()
    error_statistic = ma.array(kz * kk, mask=mask).mean()

    return statistic, error_statistic, mask, perm


def emp_MMSD(kxz: np.ndarray, ky: np.ndarray, kz: np.ndarray, Dz: np.ndarray, b: int):
    """Empirical distribution of MMSD"""
    n = len(kxz)
    empirical_distr = np.zeros((b,))
    empirical_error_distr = np.zeros((b,))

    for b_i in range(b):
        idx1 = np.random.choice(n, n // 2, replace=False)
        _11 = np.ix_(idx1, idx1)
        empirical_distr[b_i], empirical_error_distr[b_i], *_ = MMSD(ky[_11], kz[_11], kxz[_11], Dz[_11])

    return 0.5 * (empirical_distr - empirical_distr.mean()) + empirical_distr.mean(), \
           0.5 * (empirical_error_distr - empirical_error_distr.mean()) + empirical_error_distr.mean()


def fix_you(null_errors, null, error, test_statistic):
    model = LinearRegression().fit(null_errors[:, None], null[:, None])
    beta = model.coef_[0, 0]
    beta = max(0, beta)
    return null - null_errors * beta, test_statistic - error * beta


def bias_reduced_SDCIT(kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, Dz=None, size_of_null_sample=1000, with_null=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if Dz is None:
        Dz = K2D(kz)

    kxz = kx * kz

    test_statistic, error_statistic, mask, Pidx = MMSD(ky, kz, kxz, Dz)
    mask, Pidx = perm_and_mask(penalized_distance(Dz, mask))

    # avoid permutation between already permuted pairs.
    raw_null, raw_null_error = emp_MMSD(kxz,
                                        ky[np.ix_(Pidx, Pidx)],
                                        kz,
                                        penalized_distance(Dz, mask),
                                        size_of_null_sample)

    fix_null, fix_test_statistic = fix_you(raw_null_error, raw_null, error_statistic, test_statistic)
    fix_null = fix_null - fix_null.mean()  # why? why not?
    if with_null:
        return fix_test_statistic, p_value_of(fix_test_statistic, fix_null), fix_null
    else:
        return fix_test_statistic, p_value_of(fix_test_statistic, fix_null)


def SDCIT2(kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, Dz=None, size_of_null_sample=1000, with_null=False, seed=None):
    return bias_reduced_SDCIT(kx, ky, kz, Dz, size_of_null_sample, with_null, seed)


def c_SDCIT2(kx, ky, kz, Dz=None, size_of_null_sample=1000, with_null=False, seed=None, n_jobs=1):
    if seed is None:
        seed = random_seeds()
    if Dz is None:
        Dz = K2D(kz)

    kxz = kx * kz

    K_XZ = np.ascontiguousarray(kxz, dtype=np.float64)
    K_Y = np.ascontiguousarray(ky, dtype=np.float64)
    K_Z = np.ascontiguousarray(kz, dtype=np.float64)
    D_Z = np.ascontiguousarray(Dz, dtype=np.float64)
    raw_null = np.zeros((size_of_null_sample,), dtype='float64')
    error_raw_null = np.zeros((size_of_null_sample,), dtype='float64')
    mmsd = np.zeros((1,), dtype='float64')
    error_mmsd = np.zeros((1,), dtype='float64')

    cy_sdcit2(K_XZ, K_Y, K_Z, D_Z, size_of_null_sample, seed, n_jobs, mmsd, error_mmsd, raw_null, error_raw_null)
    raw_null = 0.5 * (raw_null - raw_null.mean()) + raw_null.mean()
    error_raw_null = 0.5 * (error_raw_null - error_raw_null.mean()) + error_raw_null.mean()

    test_statistic = mmsd[0]
    error_statistic = error_mmsd[0]

    fix_null, fix_test_statistic = fix_you(error_raw_null, raw_null, error_statistic, test_statistic)
    fix_null = fix_null - fix_null.mean()  # why? why not?
    if with_null:
        return fix_test_statistic, p_value_of(fix_test_statistic, fix_null), fix_null
    else:
        return fix_test_statistic, p_value_of(fix_test_statistic, fix_null)

# if __name__ == '__main__':
#     n = 200
#     for i in range(10):
#         print(i)
#         for gamma in [0.0]:
#             x, y, z = henon(i, n, gamma, 0)
#             kx, ky, kz = median_heuristic(x, y, z)
#             dz = K2D(kz)
#             t, p1 = c_SDCIT(kx, ky, kz, dz, seed=i, n_jobs=1)
#             t, p2 = c_SDCIT2(kx, ky, kz, dz, seed=i, n_jobs=1)
#             t, p3 = SDCIT(kx, ky, kz, dz)
#             t, p4 = bias_reduced_SDCIT(kx, ky, kz, dz)

# import pandas as pd
#
#     df = pd.read_csv('allfour.csv', names=['gamma', 'c_sdcit', 'c_sdcit2', 'py_sdcit', 'py_sdcit2'])
#     for key, gdf in df.groupby('gamma'):
#         print(key, aupc(gdf['c_sdcit']), aupc(gdf['c_sdcit2']), aupc(gdf['py_sdcit']), aupc(gdf['py_sdcit2']))
#
#
#     # 0.0 0.489301 0.49024 0.489126 0.490071
#     # 0.25 0.936781 0.947112 0.936875 0.947134
#     # 0.4 0.981879 0.985635 0.98187 0.985481
