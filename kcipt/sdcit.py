from kcipt.cython_impl.cy_kcipt import cy_sdcit

from experiments.synthetic import henon
from kcipt.permutation import slim_permuted, permuted
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
        if (n % 2) == 0:
            Pidx = slim_permuted(full_idx, D=Dz, with_post=with_post)
        else:
            Pidx = permuted(full_idx, D=Dz, with_post=with_post)

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


def emp_MMSD(kxz, ky, Dz, b, with_post=True):
    """Empirical distribution of Maximum Mean Self-Discrepancy"""
    n = len(kxz)
    empirical_distr = np.zeros((b,))

    for b_i in range(b):
        idx1 = np.random.choice(n, n // 2, replace=False)
        _11 = np.ix_(idx1, idx1)
        empirical_distr[b_i], _, _ = MMSD(kxz[_11], ky[_11], Dz[_11], with_post=with_post)

    return 0.5 * (empirical_distr - empirical_distr.mean()) + empirical_distr.mean()


def SDCIT(kx, ky, kz, Dz=None, size_of_null_sample=1000, reserve_perm=True, with_null=False, with_post=True, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if Dz is None:
        Dz = K2D(kz)

    kxz = kx * kz

    test_statistic, mask, Pidx = MMSD(kxz, ky, Dz, with_post=with_post)

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


def jackknife_SDCIT(kx, ky, kz, Dz=None, size_of_null_sample=1000, reserve_perm=True, with_null=False, with_post=True, seed=None):
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
    # Dz2 = Dz.copy()
    # rows, cols = np.nonzero(mask)
    # Dz2[rows, cols] = float('inf')
    # return Dz2
    return Dz + (mask - np.diag(np.diag(mask))) * 2 * Dz.max()
    # return Dz + (mask - np.diag(np.diag(mask))) * float('inf')


def c_SDCIT(kx, ky, kz, Dz=None, size_of_null_sample=1000, with_null=False, seed=None, n_jobs=1):
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
    null = 0.5 * (raw_null - raw_null.mean())
    if with_null:
        return test_statistic, p_value_of(test_statistic, null), null
    else:
        return test_statistic, p_value_of(test_statistic, null)


if __name__ == '__main__':
    np.random.seed(0)
    xs, ys = [], []
    for _ in range(1, 10):
        x, y, z = henon(_, 200, np.random.rand() * 0.5, 0)
        kx, ky, kz = median_heuristic(x, y, z)
        dz = K2D(kz)

        import time

        start = time.time()
        t0, p0, n0 = c_SDCIT(kx, ky, kz, dz, 1, True, _, 1)
        xx = time.time()
        t1, p1, n1 = SDCIT(kx, ky, kz, dz, 1, with_null=True, seed=_)
        yy = time.time()
        print(t0, t1, sep=',')
        # xs.append(t0)
        # ys.append(t1)
        #
        # if (_ % 10) == 0:
        #     import seaborn as sns
        #     import matplotlib.pyplot as plt
        #
        #     sns.set()
        #     plt.scatter(np.array(xs) * 15, np.array(ys) * 15, alpha=0.2)
        #     plt.legend()
        #     plt.savefig('testing_c.pdf')
        #     plt.close()
