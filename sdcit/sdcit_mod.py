from typing import Tuple

import numpy as np
import numpy.ma as ma
from sdcit.cython_impl.cy_sdcit import cy_sdcit, cy_split_permutation, cy_dense_permutation
from sklearn.linear_model import LinearRegression

from sdcit.utils import K2D, p_value_of, random_seeds, cythonize


def permuted(D, dense=True):
    out = np.zeros((len(D),), 'int32')
    if dense:
        cy_dense_permutation(D, out)
    else:
        cy_split_permutation(D, out)
    return out


def mask_and_perm(Dz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(Dz)
    full_idx = np.arange(0, n)
    perm = permuted(Dz)

    # 1 for masked (=excluded)
    mask = np.zeros((n, n))
    mask[full_idx, full_idx] = 1  # i==j
    mask[full_idx, perm] = 1  # pi_i = j
    mask[perm, full_idx] = 1  # i = pi_j

    return mask, perm


def penalized_distance(Dz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # add some big value to "masked" values except diagonal.
    return Dz + (mask - np.diag(np.diag(mask))) * Dz.max()  # soft penalty


def MMSD(Ky: np.ndarray, Kz: np.ndarray, Kxz: np.ndarray, Dz: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Maximum Mean Self-Discrepancy"""
    n = len(Kxz)
    full_idx = np.arange(0, n)

    mask, perm = mask_and_perm(Dz)
    Ky_fp = Ky[np.ix_(full_idx, perm)]
    Ky_pp = Ky[np.ix_(perm, perm)]

    kk = (Ky + Ky_pp - 2 * Ky_fp)
    statistic = ma.array(Kxz * kk, mask=mask).mean()
    error_statistic = ma.array(Kz * kk, mask=mask).mean()

    return statistic, error_statistic, mask, perm


def emp_MMSD(Kxz: np.ndarray, Ky: np.ndarray, Kz: np.ndarray, Dz: np.ndarray, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical distribution of MMSD"""
    n = len(Kxz)
    mmsd_distr = np.zeros((num_samples,))
    error_distr = np.zeros((num_samples,))

    for i_th in range(num_samples):
        selected = np.random.choice(n, n // 2, replace=False)
        grid = np.ix_(selected, selected)
        mmsd_distr[i_th], error_distr[i_th], *_ = MMSD(Ky[grid], Kz[grid], Kxz[grid], Dz[grid])

    return (0.5 * (mmsd_distr - mmsd_distr.mean()) + mmsd_distr.mean(),
            0.5 * (error_distr - error_distr.mean()) + error_distr.mean())


def adjust_errors(null_errors, null, error=None, test_statistic=None):
    if error is not None:
        assert test_statistic is not None

    model = LinearRegression().fit(null_errors[:, None], null[:, None])
    beta = max(0, model.coef_[0, 0])

    if error is not None:
        return null - null_errors * beta, test_statistic - error * beta
    else:
        return null - null_errors * beta


def SDCIT(Kx: np.ndarray, Ky: np.ndarray, Kz: np.ndarray, Dz=None, size_of_null_sample=1000, with_null=False, seed=None, adjust=True, to_shuffle=True):
    """SDCIT (Lee and Honavar, 2017)

    Parameters
    ----------
    Kx : np.ndarray
        N by N kernel matrix of X
    Ky : np.ndarray
        N by N kernel matrix of Y
    Kz : np.ndarray
        N by N kernel matrix of Z
    Dz : np.ndarray
        N by N pairwise distance matrix of Z
    size_of_null_sample : int
        The number of samples in a null distribution
    with_null : bool
        If true, resulting null distribution is also returned
    seed : int
        Random seed
    adjust : bool
        whether to adjust null distribution and test statistics based on 'permutation error' information
    to_shuffle : bool
        shuffle the order of given data at the beginning, which minimize possible issues with getting a bad permutation

    References
    ----------
        Lee, S., Honavar, V. (2017). Self-Discrepancy Conditional Independence Test.
        In Proceedings of the Thirty-third Conference on Uncertainty in Artificial Intelligence. Corvallis, Oregon: AUAI Press.
    """
    if seed is not None:
        np.random.seed(seed)

    if Dz is None:
        Dz = K2D(Kz)

    if to_shuffle:
        Kx, Ky, Kz, Dz = shuffling(seed, Kx, Ky, Kz, Dz)  # categorical Z may yield an ordered 'block' matrix and it may harm permutation.

    Kxz = Kx * Kz

    test_statistic, error_statistic, mask, _ = MMSD(Ky, Kz, Kxz, Dz)
    mask, Pidx = mask_and_perm(penalized_distance(Dz, mask))

    # avoid permutation between already permuted pairs.
    mmsd_distr_under_null, error_distr_under_null = emp_MMSD(Kxz, Ky[np.ix_(Pidx, Pidx)], Kz, penalized_distance(Dz, mask), size_of_null_sample)

    if adjust:
        fix_null, fix_test_statistic = adjust_errors(error_distr_under_null, mmsd_distr_under_null, error_statistic, test_statistic)
        fix_null = fix_null - fix_null.mean()
    else:
        fix_null = mmsd_distr_under_null - mmsd_distr_under_null.mean()
        fix_test_statistic = test_statistic

    if with_null:
        return fix_test_statistic, p_value_of(fix_test_statistic, fix_null), fix_null
    else:
        return fix_test_statistic, p_value_of(fix_test_statistic, fix_null)


def c_SDCIT(Kx, Ky, Kz, Dz=None, size_of_null_sample=1000, with_null=False, seed=None, n_jobs=1, adjust=True, to_shuffle=True):
    """C-based SDCIT (Lee and Honavar, 2017)

    Parameters
    ----------
    Kx : np.ndarray
        N by N kernel matrix of X
    Ky : np.ndarray
        N by N kernel matrix of Y
    Kz : np.ndarray
        N by N kernel matrix of Z
    Dz : np.ndarray
        N by N pairwise distance matrix of Z
    size_of_null_sample : int
        The number of samples in a null distribution
    with_null : bool
        If true, a resulting null distribution is also returned
    seed : int
        Random seed
    n_jobs: int
        number of threads to be used
    adjust : bool
        whether to adjust null distribution and test statistics based on 'permutation error' information
    to_shuffle : bool
        shuffle the order of given data at the beginning, which minimize possible issues with getting a bad permutation


    References
    ----------
        Lee, S., Honavar, V. (2017). Self-Discrepancy Conditional Independence Test.
        In Proceedings of the Thirty-third Conference on Uncertainty in Artificial Intelligence. Corvallis, Oregon: AUAI Press.
    """
    if seed is not None:
        np.random.seed(seed)

    if Dz is None:
        Dz = K2D(Kz)

    if to_shuffle:
        Kx, Ky, Kz, Dz = shuffling(seed, Kx, Ky, Kz, Dz)  # categorical Z may yield an ordered 'block' matrix and it may harm permutation.

    Kxz = Kx * Kz

    # prepare parameters & output variables
    Kxz, Ky, Kz, Dz = cythonize(Kxz, Ky, Kz, Dz)
    raw_null = np.zeros((size_of_null_sample,), dtype='float64')
    error_raw_null = np.zeros((size_of_null_sample,), dtype='float64')
    mmsd = np.zeros((1,), dtype='float64')
    error_mmsd = np.zeros((1,), dtype='float64')

    # run SDCIT
    cy_sdcit(Kxz, Ky, Kz, Dz, size_of_null_sample, random_seeds(), n_jobs, mmsd, error_mmsd, raw_null, error_raw_null)

    # post-process outputs
    test_statistic = mmsd[0]
    error_statistic = error_mmsd[0]
    raw_null = 0.5 * (raw_null - raw_null.mean()) + raw_null.mean()
    error_raw_null = 0.5 * (error_raw_null - error_raw_null.mean()) + error_raw_null.mean()

    if adjust:
        fix_null, fix_test_statistic = adjust_errors(error_raw_null, raw_null, error_statistic, test_statistic)
        fix_null = fix_null - fix_null.mean()
    else:
        fix_null = raw_null - raw_null.mean()
        fix_test_statistic = test_statistic

    if with_null:
        return fix_test_statistic, p_value_of(fix_test_statistic, fix_null), fix_null
    else:
        return fix_test_statistic, p_value_of(fix_test_statistic, fix_null)


def shuffling(seed, *matrices):
    if seed is not None:
        np.random.seed(seed)

    n = -1
    for matrix in matrices:
        if n < 0:
            n = len(matrix)
            idxs = np.arange(n)
            np.random.shuffle(idxs)

        yield matrix[np.ix_(idxs, idxs)]
