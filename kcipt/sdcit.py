from kcipt.algo import c_KCIPT
from kcipt.cython_impl.cy_kcipt import cy_sdcit
from kcipt.permutation import permuted
from kcipt.utils import *


def MMSD(kxz, ky, Dz):
    """Maximum Mean Self-Discrepancy"""
    n = len(kxz)
    full_idx = np.arange(0, n)

    mask, perm = perm_and_mask(Dz)

    K11 = kxz * ky
    K12 = kxz * ky[np.ix_(full_idx, perm)]
    K22 = kxz * ky[np.ix_(perm, perm)]

    statistic = ma.array(K11 + K22 - K12 - K12.T, mask=mask).mean()

    return statistic, mask, perm


def perm_and_mask(Dz):
    n = len(Dz)
    full_idx = np.arange(0, n)
    perm = permuted(Dz)

    # 1 for masked (=excluded)
    mask = np.zeros((n, n))
    mask[full_idx, full_idx] = 1  # i==j
    mask[full_idx, perm] = 1  # pi_i = j
    mask[perm, full_idx] = 1  # i = pi_j

    return mask, perm


def jackknife_MMSD(kxz, ky, Dz):
    """Jackknife-based estiamte of MMSD"""
    n = len(kxz)
    jack = np.zeros((n // 2,))

    test_statistic, mask, Pidx = MMSD(kxz, ky, Dz)
    for i, offset in enumerate(range(0, n, 2)):
        idx1 = list(set(range(n)) - {offset, offset + 1})
        _11 = np.ix_(idx1, idx1)
        jack[i], _, _ = MMSD(kxz[_11], ky[_11], Dz[_11])

    return jack.mean(), mask, Pidx


def emp_MMSD(kxz, ky, Dz, b):
    """Empirical distribution of MMSD"""
    n = len(kxz)
    empirical_distr = np.zeros((b,))

    for b_i in range(b):
        idx1 = np.random.choice(n, n // 2, replace=False)
        _11 = np.ix_(idx1, idx1)
        empirical_distr[b_i], _, _ = MMSD(kxz[_11], ky[_11], Dz[_11])

    return 0.5 * (empirical_distr - empirical_distr.mean()) + empirical_distr.mean()


def SDCIT(kx, ky, kz, Dz=None, size_of_null_sample=1000, with_null=False, seed=None, adjust_null=True, adjust_statistic_factor=0):
    if adjust_statistic_factor and not adjust_null:
        warnings.warn('test statistic is only adjusted if null is adjusted (set adjust_null=True)')
    if adjust_statistic_factor < 0:
        warnings.warn('0 <= adjust_statistic_factor <= 1.0, (Recommended: 0.5)')

    if seed is not None:
        np.random.seed(seed)

    if Dz is None:
        Dz = K2D(kz)

    kxz = kx * kz

    test_statistic, mask, Pidx = MMSD(kxz, ky, Dz)
    mask, Pidx = perm_and_mask(penalized_distance(Dz, mask))

    # avoid permutation between already permuted pairs.
    raw_null = emp_MMSD(kxz,
                        ky[np.ix_(Pidx, Pidx)],
                        penalized_distance(Dz, mask),
                        size_of_null_sample)

    null_bias = raw_null.mean()
    if adjust_null:
        null = raw_null - raw_null.mean()
        if adjust_statistic_factor:
            test_statistic -= adjust_statistic_factor * null_bias
    else:
        null = raw_null

    if with_null:
        return test_statistic, p_value_of(test_statistic, null), null
    else:
        return test_statistic, p_value_of(test_statistic, null)


def jackknife_SDCIT(kx, ky, kz, Dz=None, size_of_null_sample=1000, with_null=False, seed=None):
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

    test_statistic, mask, Pidx = jackknife_MMSD(kxz, ky, Dz)
    mask, Pidx = perm_and_mask(penalized_distance(Dz, mask))

    # avoid permutation between already permuted pairs.
    raw_null = emp_MMSD(kxz,
                        ky[np.ix_(Pidx, Pidx)],
                        penalized_distance(Dz, mask),
                        size_of_null_sample)

    null = raw_null - raw_null.mean()

    if with_null:
        return test_statistic, p_value_of(test_statistic, null), null
    else:
        return test_statistic, p_value_of(test_statistic, null)


def penalized_distance(Dz, mask):
    return Dz + (mask - np.diag(np.diag(mask))) * Dz.max()  # soft penalty


def suggest_B_for_KCIPT(kx, ky, kz, Dz):
    _, _, null_sdcit = c_SDCIT(kx, ky, kz, Dz, 250, with_null=True)
    _, _, inner_null, _ = c_KCIPT(kx, ky, kz, Dz, 10, 1000, 0)
    inner_null = inner_null.flatten()
    return int(1 + (np.std(inner_null) / np.std(null_sdcit)) ** 2)


def c_SDCIT(kx, ky, kz, Dz=None, size_of_null_sample=1000, with_null=False, seed=None, n_jobs=1, adjust_null=True, adjust_statistic_factor=0):
    if seed is None:
        seed = random_seeds()
    if Dz is None:
        Dz = K2D(kz)

    kxz = kx * kz

    K_XZ = np.ascontiguousarray(kxz, dtype=np.float64)
    K_Y = np.ascontiguousarray(ky, dtype=np.float64)
    D_Z = np.ascontiguousarray(Dz, dtype=np.float64)
    raw_null = np.zeros((size_of_null_sample,), dtype='float64')
    mmsd = np.zeros((1,), dtype='float64')

    cy_sdcit(K_XZ, K_Y, D_Z, size_of_null_sample, seed, n_jobs, mmsd, raw_null)

    test_statistic = mmsd[0]
    bias = raw_null.mean()
    if adjust_null:
        null = 0.5 * (raw_null - bias)
        if adjust_statistic_factor:
            test_statistic -= adjust_statistic_factor * bias
    else:
        null = 0.5 * (raw_null - bias) + bias

    if with_null:
        return test_statistic, p_value_of(test_statistic, null), null
    else:
        return test_statistic, p_value_of(test_statistic, null)
