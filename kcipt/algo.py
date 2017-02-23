import numpy.ma as ma
from kcipt.cython_impl.cy_kcipt import cy_kcipt, cy_adj_kcipt, cy_null_distribution, cy_bootstrap_single_null
from scipy.stats import norm

from kcipt.permutation import blossom_permutation
from kcipt.utils import *


def permuted(idx, K):
    Pidx = idx.copy()
    D = K2D(K)
    rows, cols = np.nonzero(blossom_permutation(D))
    Pidx[rows] = idx[cols]
    return Pidx


def c_KCIPT(K_X, K_Y, K_Z, D_Z, B, b, M, n_jobs=1):
    K_X = np.ascontiguousarray(K_X, 'float64')
    K_Y = np.ascontiguousarray(K_Y, 'float64')
    K_Z = np.ascontiguousarray(K_Z, 'float64')
    D_Z = np.ascontiguousarray(D_Z, 'float64')

    inner_null = np.zeros((B, b), dtype='float64')
    outer_null = np.zeros((M,), dtype='float64')
    mmds = np.zeros((B,), dtype='float64')

    cy_kcipt(K_X, K_Y, K_Z, D_Z, B, b, inner_null, mmds, random_seeds(), n_jobs, outer_null, M)
    if b > 0:
        inner_null -= np.mean(inner_null)
    if M > 0:
        outer_null -= np.mean(outer_null)
    # test_statistic, p, z = compute_p_values(mmds, inner_null, options.get('p_value_method', None))
    test_statistic = np.mean(mmds)
    return p_value_of(test_statistic, outer_null, approxmation=True) if M > 0 else float('nan'), mmds, inner_null, outer_null


# MMD-based
def c_adj_KCIPT(K_X, K_Y, K_Z, D_Z, B, b, M, variance_reduced=True, n_jobs=1):
    # print('something wrong ... working on...')
    K_X = np.ascontiguousarray(K_X, 'float64')
    K_Y = np.ascontiguousarray(K_Y, 'float64')
    K_Z = np.ascontiguousarray(K_Z, 'float64')
    K_XYZ = np.ascontiguousarray(K_X * K_Y * K_Z, 'float64')
    D_Z = np.ascontiguousarray(D_Z, 'float64')

    inner_null = np.zeros((b,), dtype='float64')
    outer_null = np.zeros((M,), dtype='float64')
    mmds = np.zeros((1,), dtype='float64')

    cy_adj_kcipt(K_X, K_Y, K_Z, K_XYZ, D_Z, B, b, inner_null, mmds, random_seeds(), n_jobs, outer_null, M, 1 if variance_reduced else 0)
    if b > 0:
        inner_null -= np.mean(inner_null)
    if M > 0:
        outer_null -= np.mean(outer_null)

    test_statistic = mmds[0]
    p = p_value_of(test_statistic, outer_null, approxmation=True) if M > 0 else float('nan')
    return p, mmds[0], inner_null, outer_null


def one_shot_KCIPT(kx, ky, kz, num_null=10 ** 4, with_null=True):
    kxyz = kx * ky * kz
    kxz = kx * kz

    n = len(kx)
    full_idx = np.arange(0, n)
    Pidx = permuted(full_idx, kz)  #
    inversedP = np.zeros((n,))
    for i, pi_i in enumerate(Pidx):
        inversedP[pi_i] = i

    mask = np.zeros((n, n))
    mask[full_idx, full_idx] = 1  # i==j
    mask[full_idx, Pidx] = 1  # pi_i = j
    mask[Pidx, full_idx] = 1  # i = pi_j

    K11 = kxyz
    K12 = kxz * ky[np.ix_(full_idx, Pidx)]
    K22 = kxz * ky[np.ix_(Pidx, Pidx)]

    mmd = ma.array(K11, mask=mask).mean() + \
          ma.array(K22, mask=mask).mean() - \
          2 * ma.array(K12, mask=mask).mean()

    if with_null:
        UL, UR, BR = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
        # (xi,yi,zi) ~ (xj, yj, zj)
        UL[full_idx, full_idx] = 1  # i==j
        # (xi,yi,zi) ~ (xj, y pi_j, zj)
        UR[full_idx, full_idx] = 1  # i==j
        UR[Pidx, full_idx] = 1  # i==pi_j
        # (xi,y pi_i,zi) ~ (xj, y pi_j, zj)
        BR[full_idx, full_idx] = 1
        BR[Pidx, full_idx] = 1  # i==pi_j
        BR[full_idx, Pidx] = 1  # pi_i == j
        big_mask = np.bmat([[UL, UR], [UR.T, BR]])

        traditional_null = np.zeros((num_null,))
        new_k = np.bmat([[K11, K12], [K12.T, K22]])
        for b_i in range(num_null):
            idx1, idx2 = split_1_to_r(2 * n, 1)
            kk = new_k[np.ix_(idx1, idx1)] + new_k[np.ix_(idx2, idx2)] - new_k[np.ix_(idx1, idx2)] - new_k[np.ix_(idx2, idx1)]
            mask = big_mask[np.ix_(idx1, idx1)] + big_mask[np.ix_(idx2, idx2)] + big_mask[np.ix_(idx1, idx2)] + big_mask[np.ix_(idx2, idx1)]
            traditional_null[b_i] = ma.array(kk, mask=mask).mean()

        return mmd, traditional_null
    else:
        return mmd


def bootstrap_null(nulls, B, b, M=10000, floc=None):
    rows = np.arange(B)
    out = np.array([np.mean(nulls[rows, np.random.choice(b, B)]) for _ in range(M)])
    if floc is not None:
        out = out - np.mean(out) + floc
    return out


def bootstrap_single_null(null, B, b=None, M=10000, floc=None, no_cython=False):
    if M <= 0:
        return np.zeros((0,))
    if b is None:
        b = len(null)
    if not no_cython:
        out = np.zeros((M,), dtype='float64')
        cy_bootstrap_single_null(random_seeds(), B, b, M, null, out)
        return out

    out = np.array([np.mean(null[np.random.choice(b, B)]) for _ in range(M)])
    if floc is not None:
        out = out - np.mean(out) + floc
    return out


# Perform split-MMD multiple times.
def null_distribution(kxyz, sample_size=5000, split_ratio=1, with_cross_nulls=False, no_cython=False, floc=None):
    if sample_size <= 0:
        return np.zeros((0,))
    if floc is not None and with_cross_nulls:
        import warnings
        warnings.warn('floc is incompatible with cross-nulls.')

    if split_ratio == 1 and not with_cross_nulls and not no_cython:
        inner_null = np.zeros((sample_size,), dtype='float64')
        cy_null_distribution(random_seeds(), kxyz, len(kxyz), inner_null, len(inner_null))
        if floc is not None:
            inner_null = inner_null - np.mean(inner_null) + floc
        return inner_null

    sample_size = int(sample_size)
    unbiased_nulls = np.zeros((sample_size,))
    between_nulls = np.zeros((sample_size,)) if with_cross_nulls else None

    for b_i in range(sample_size):
        idx1, idx2 = split_1_to_r(len(kxyz), split_ratio)

        u11 = mean_without_diag(kxyz[np.ix_(idx1, idx1)])
        u22 = mean_without_diag(kxyz[np.ix_(idx2, idx2)])
        k12 = np.mean(kxyz[np.ix_(idx1, idx2)])
        unbiased_nulls[b_i] = u11 + u22 - 2 * k12
        if with_cross_nulls:
            between_nulls[b_i] = - k12

    if with_cross_nulls:
        return unbiased_nulls, between_nulls
    else:
        if floc is not None:
            unbiased_nulls = unbiased_nulls - np.mean(unbiased_nulls) + floc
        return unbiased_nulls


def combined_test(kx, ky, kz, ratio=1, effect_size=0.025, confidence=0.9999):
    n = len(kx)
    ci_factor = norm.ppf(confidence)  # 0.9999 ~ 3.7, 0.999 ~ 3.1, 0.99 ~ 2.3
    # base
    kxz = kx * kz
    kxy = kx * ky
    kxyz = kx * ky * kz

    unbiased_nulls = null_distribution(kxyz, 5000, floc=0)
    null_std = np.std(unbiased_nulls)

    upperbound_mmd = 2 * one_shot_KCIPT(kx, ky, kz, with_null=False)
    if upperbound_mmd / null_std <= effect_size:
        return [upperbound_mmd / null_std, 0, 0, 0, upperbound_mmd]

    effective_mmd = effect_size * null_std

    #################### test statistic ##########################
    MMD00, MMD0Y, MMD0Z, MMDYY, MMDZZ = 0, 0, 0, 0, 0
    last_50 = np.zeros((50,))
    for B_i in range(1, 10001):
        idx1, idx2 = split_1_to_r(n, ratio)
        _11, _22 = np.ix_(idx1, idx1), np.ix_(idx2, idx2)

        Pidx1, Pidx2 = permuted(idx1, kz[_11]), permuted(idx2, kz[_22])
        #
        _P1P1, _P2P2 = np.ix_(Pidx1, Pidx1), np.ix_(Pidx2, Pidx2)
        #
        within_k_00_1 = mean_without_diag(kxyz[_11])
        within_k_00_2 = mean_without_diag(kxyz[_22])
        within_k_yy_1 = mean_without_diag(kxz[_11] * ky[_P1P1])
        within_k_zz_1 = mean_without_diag(kxy[_11] * kz[_P1P1])
        within_k_yy_2 = mean_without_diag(kxz[_22] * ky[_P2P2])
        within_k_zz_2 = mean_without_diag(kxy[_22] * kz[_P2P2])
        #
        _12, _1P2, _P1P2 = np.ix_(idx1, idx2), np.ix_(idx1, Pidx2), np.ix_(Pidx1, Pidx2)

        bw_k_00 = np.mean(kxyz[_12])
        bw_k_0y = np.mean(kxz[_12] * ky[_1P2])
        bw_k_0z = np.mean(kxy[_12] * kz[_1P2])
        bw_k_yy = np.mean(kxz[_12] * ky[_P1P2])
        bw_k_zz = np.mean(kxy[_12] * kz[_P1P2])

        MMD00 += (within_k_00_1 + within_k_00_2 - 2 * bw_k_00 - MMD00) / B_i
        MMD0Y += (within_k_00_1 + within_k_yy_2 - 2 * bw_k_0y - MMD0Y) / B_i
        MMD0Z += (within_k_00_1 + within_k_zz_2 - 2 * bw_k_0z - MMD0Z) / B_i
        MMDYY += (within_k_yy_1 + within_k_yy_2 - 2 * bw_k_yy - MMDYY) / B_i
        MMDZZ += (within_k_zz_1 + within_k_zz_2 - 2 * bw_k_zz - MMDZZ) / B_i

        y_to_z_factor = ((MMDYY - MMD00) / max(MMD0Z - MMD00, MMDZZ - MMD00)) if (MMDZZ - MMD00) != 0 else 1
        corrected = (MMD0Y - MMD00) - (MMD0Z - MMD00) * y_to_z_factor
        last_50 = np.roll(last_50, 1)
        last_50[0] = corrected

        if B_i >= len(last_50):
            current_stdev = np.std(last_50)
            if effective_mmd < (corrected - current_stdev * ci_factor) / null_std or (corrected + current_stdev * ci_factor) / null_std < effective_mmd:
                return [corrected / null_std, current_stdev, B_i, corrected, upperbound_mmd]

    return [corrected / null_std, current_stdev, B_i, corrected, upperbound_mmd]


def adj_KCIPT(kx, ky, kz, B, b=10000, M=10000, variance_reduced=True):
    n = len(kx)
    # base
    kxz = kx * kz
    kxy = kx * ky
    kxyz = kx * ky * kz

    unbiased_nulls = null_distribution(kxyz, b, floc=0)
    outer_null = bootstrap_single_null(unbiased_nulls, B, b, M, floc=0)

    #################### test statistic ##########################
    MMD00, MMD0Y, MMD0Z, MMDYY, MMDZZ = 0, 0, 0, 0, 0
    for B_i in range(1, B + 1):
        idx1, idx2 = split_1_to_r(n, 1)
        _11, _22 = np.ix_(idx1, idx1), np.ix_(idx2, idx2)

        Pidx1, Pidx2 = permuted(idx1, kz[_11]), permuted(idx2, kz[_22])
        #
        _P1P1, _P2P2 = np.ix_(Pidx1, Pidx1), np.ix_(Pidx2, Pidx2)
        #
        within_k_00_1 = mean_without_diag(kxyz[_11])
        within_k_00_2 = mean_without_diag(kxyz[_22])
        within_k_yy_1 = mean_without_diag(kxz[_11] * ky[_P1P1])
        within_k_zz_1 = mean_without_diag(kxy[_11] * kz[_P1P1])
        within_k_yy_2 = mean_without_diag(kxz[_22] * ky[_P2P2])
        within_k_zz_2 = mean_without_diag(kxy[_22] * kz[_P2P2])
        #
        _12, _1P2, _P1P2 = np.ix_(idx1, idx2), np.ix_(idx1, Pidx2), np.ix_(Pidx1, Pidx2)

        bw_k_00 = np.mean(kxyz[_12])
        bw_k_0y = np.mean(kxz[_12] * ky[_1P2])
        bw_k_0z = np.mean(kxy[_12] * kz[_1P2])
        bw_k_yy = np.mean(kxz[_12] * ky[_P1P2])
        bw_k_zz = np.mean(kxy[_12] * kz[_P1P2])

        MMD00 += (within_k_00_1 + within_k_00_2 - 2 * bw_k_00 - MMD00) / B_i
        MMD0Y += (within_k_00_1 + within_k_yy_2 - 2 * bw_k_0y - MMD0Y) / B_i
        MMD0Z += (within_k_00_1 + within_k_zz_2 - 2 * bw_k_0z - MMD0Z) / B_i
        MMDYY += (within_k_yy_1 + within_k_yy_2 - 2 * bw_k_yy - MMDYY) / B_i
        MMDZZ += (within_k_zz_1 + within_k_zz_2 - 2 * bw_k_zz - MMDZZ) / B_i

    y_to_z_factor = ((MMDYY - MMD00) / max(MMD0Z - MMD00, MMDZZ - MMD00)) if (MMDZZ - MMD00) != 0 else 1
    if variance_reduced:
        corrected = (MMD0Y - MMD00) - (MMD0Z - MMD00) * y_to_z_factor
    else:
        corrected = MMD0Y - (MMD0Z - MMD00) * y_to_z_factor

    # print(MMD00, MMD0Y, MMD0Z, MMDYY, MMDZZ, sep=',')

    return p_value_of(corrected, outer_null, approxmation=True) if M > 0 else float('nan'), corrected, unbiased_nulls, outer_null
