import os
import time

import scipy.io

from kcipt.permutation import slim_permuted
from kcipt.utils import *


def MMSD(kxz, ky, Dz, with_post=True, Pidx=None):
    """Maximum Mean Self-Discrepancy

    :param kxz: a Gram matrix for (x, z) values.
    :param ky: a Gram matrix for y values
    :param Dz: a pairwise distance matrix for z values
    :param with_post: whether to perform local improvement heuristics for the perfect matching result.
    :param Pidx: provided permutation.
    :return:
    """
    n = len(kxz)
    full_idx = np.arange(0, n)

    mask, Pidx = perm_and_mask(Dz, with_post, Pidx)

    K11 = kxz * ky
    K12 = kxz * ky[np.ix_(full_idx, Pidx)]
    K22 = kxz * ky[np.ix_(Pidx, Pidx)]

    mmd = ma.array(K11 + K22 - K12 - K12.T, mask=mask).mean()

    return mmd, mask, Pidx


def perm_and_mask(Dz, with_post=True, Pidx=None):
    n = len(Dz)
    full_idx = np.arange(0, n)
    if Pidx is None:
        Pidx = slim_permuted(full_idx, D=Dz, with_post=with_post)

    # 1 for masked (=excluded)
    mask = np.zeros((n, n))
    mask[full_idx, full_idx] = 1  # i==j
    mask[full_idx, Pidx] = 1  # pi_i = j
    mask[Pidx, full_idx] = 1  # i = pi_j

    return mask, Pidx


def jackknife_MMSD(kxz, ky, Dz, with_post=True):
    """Jackknife-based estiamte of Maximum Mean Self-Discrepancy"""
    n = len(kxz)
    jack = np.zeros((n // 2,))

    test_statistic, mask, Pidx = MMSD(kxz, ky, Dz, with_post=with_post)
    for i, offset in enumerate(range(0, n, 2)):
        idx1 = list(set(range(n)) - {offset, offset + 1})
        _11 = np.ix_(idx1, idx1)
        jack[i], _, _ = MMSD(kxz[_11], ky[_11], Dz[_11], with_post=with_post)

    return jack.mean(), mask, Pidx


def emp_MMSD(kxz, ky, Dz, B, with_post=True):
    """Empirical distribution of Maximum Mean Self-Discrepancy"""
    n = len(kxz)
    empirical_distr = np.zeros((B,))

    for B_i in range(B):
        idx1 = np.random.choice(n, n // 2, replace=False)
        _11 = np.ix_(idx1, idx1)
        empirical_distr[B_i], _, _ = MMSD(kxz[_11], ky[_11], Dz[_11], with_post=with_post)

    return 0.5 * (empirical_distr - empirical_distr.mean()) + empirical_distr.mean()


def SDCIT(kx, ky, kz, Dz=None, size_of_null_sample=500, reserve_perm=False, with_null=False, with_post=True, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if Dz is None:
        Dz = K2D(kz)

    kxz = kx * kz

    test_statistic, mask, Pidx = MMSD(kxz, ky, Dz, with_post=with_post)

    # second-class permuted sample
    if reserve_perm:
        mask, Pidx = perm_and_mask(penaltied_distance(Dz, mask), with_post=with_post)

    # avoid permutation between already permuted pairs.
    raw_null = emp_MMSD(kxz,
                        ky[np.ix_(Pidx, Pidx)],
                        penaltied_distance(Dz, mask),
                        size_of_null_sample, with_post=with_post)

    null = raw_null - raw_null.mean()

    if with_null:
        return test_statistic, p_value_of(test_statistic, null), null
    else:
        return test_statistic, p_value_of(test_statistic, null)


def jackknife_SDCIT(kx, ky, kz, Dz=None, size_of_null_sample=500, reserve_perm=False, with_null=False, with_post=True, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if Dz is None:
        Dz = K2D(kz)

    n = len(kx)
    shuf = np.arange(n)
    np.random.shuffle(shuf)
    shufidx = np.ix_(shuf, shuf)
    kx, ky, kz, Dz = kx[shufidx], ky[shufidx], kz[shufidx], Dz[shufidx]

    kxz = kx * kz

    test_statistic, mask, Pidx = jackknife_MMSD(kxz, ky, Dz, with_post=with_post)

    # second-class permuted sample
    if reserve_perm:
        mask, Pidx = perm_and_mask(penaltied_distance(Dz, mask), with_post=with_post)

    # avoid permutation between already permuted pairs.
    raw_null = emp_MMSD(kxz,
                        ky[np.ix_(Pidx, Pidx)],
                        penaltied_distance(Dz, mask),
                        size_of_null_sample, with_post=with_post)

    null = raw_null - raw_null.mean()

    if with_null:
        return test_statistic, p_value_of(test_statistic, null), null
    else:
        return test_statistic, p_value_of(test_statistic, null)


def penaltied_distance(Dz, mask):
    return Dz + (mask - np.diag(np.diag(mask))) * Dz.max()


