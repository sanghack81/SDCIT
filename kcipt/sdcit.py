import os
import time

import scipy.io

from experiments.synthetic import henon
from kcipt.algo import slim_permuted
from kcipt.utils import *


def MMSD(kxz, ky, Dz, with_post=True, Pidx=None):
    n = len(kxz)
    full_idx = np.arange(0, n)
    if Pidx is None:
        Pidx = slim_permuted(full_idx, D=Dz, with_post=with_post)

    mask = np.zeros((n, n))
    mask[full_idx, full_idx] = 1  # i==j
    mask[full_idx, Pidx] = 1  # pi_i = j
    mask[Pidx, full_idx] = 1  # i = pi_j

    K11 = kxz * ky
    K12 = kxz * ky[np.ix_(full_idx, Pidx)]
    K22 = kxz * ky[np.ix_(Pidx, Pidx)]

    unbiased_hs = ma.array(K11 + K22 - K12 - K12.T, mask=mask)
    mmd = unbiased_hs.mean()

    return mmd, mask, Pidx


def jackknife_MMSD(kxz, ky, Dz, with_post=True):
    n = len(kxz)
    jack = np.zeros((n // 2,))

    test_statistic, mask, Pidx = MMSD(kxz, ky, Dz, with_post=with_post)
    for i, offset in enumerate(range(0, n, 2)):
        idx1 = list(set(range(n)) - {offset, offset + 1})
        _11 = np.ix_(idx1, idx1)
        jack[i], _, _ = MMSD(kxz[_11], ky[_11], Dz[_11], with_post=with_post)

    # statistic_jack = (n * test_statistic - (n - 2) * jack.mean()) / 2

    return jack.mean(), mask, Pidx


def emp_MMSD(kxz, ky, Dz, B, with_post=True):
    n = len(kxz)
    empirical_distr = np.zeros((B,))

    for B_i in range(B):
        idx1 = np.random.choice(n, n // 2, replace=False)
        _11 = np.ix_(idx1, idx1)
        empirical_distr[B_i], _, _ = MMSD(kxz[_11], ky[_11], Dz[_11], with_post=with_post)

    return 0.5 * (empirical_distr - empirical_distr.mean()) + empirical_distr.mean()


def emp_MMSD_XY(kxz, ky, kyz, kx, Dz, B, with_post=True):
    n = len(kxz)
    empirical_distr = np.zeros((B,))

    for B_i in range(B):
        idx1 = np.random.choice(n, n // 2, replace=False)
        _11 = np.ix_(idx1, idx1)
        t1, _, Pidx = MMSD(kxz[_11], ky[_11], Dz[_11], with_post=with_post)
        t2, _, _ = MMSD(kyz[_11], kx[_11], Dz[_11], with_post=with_post, Pidx=Pidx)
        empirical_distr[B_i] = t1 + t2

    return 0.5 * (empirical_distr - empirical_distr.mean()) + empirical_distr.mean()


def SDCIT(kx, ky, kz, Dz=None, size_of_null_sample=500, reserve_perm=False, with_null=False, with_post=True):
    if Dz is None:
        Dz = K2D(kz)

    kxz = kx * kz

    test_statistic, mask, Pidx = MMSD(kxz, ky, Dz, with_post=with_post)

    # second-class permuted sample
    if reserve_perm:
        # TODO only perm
        _, mask, Pidx = MMSD(kxz, ky, penaltied_distance(Dz, mask), with_post=with_post)

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


def jack_SDCIT(kx, ky, kz, Dz=None, size_of_null_sample=500, reserve_perm=False, with_null=False, with_post=True):
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
        # TODO only perm
        _, mask, Pidx = MMSD(kxz, ky, penaltied_distance(Dz, mask), with_post=with_post)

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


def SDCIT_XY(kx, ky, kz, Dz=None, size_of_null_sample=500, reserve_perm=False, with_null=False, with_post=True):
    if Dz is None:
        Dz = K2D(kz)

    kxz = kx * kz
    kyz = ky * kz

    test_statistic_Y, mask, Pidx = MMSD(kxz, ky, Dz, with_post=with_post)
    test_statistic_X, _, Pidx = MMSD(kyz, kx, Dz, with_post=with_post, Pidx=Pidx)

    test_statistic = test_statistic_X + test_statistic_Y

    # second-class permuted sample
    if reserve_perm:
        # TODO only perm
        _, mask, Pidx = MMSD(kxz, ky, penaltied_distance(Dz, mask), with_post=with_post)

    # avoid permutation between already permuted pairs.
    raw_null = emp_MMSD_XY(kxz,
                           ky[np.ix_(Pidx, Pidx)],
                           kyz,
                           kx[np.ix_(Pidx, Pidx)],
                           penaltied_distance(Dz, mask),
                           size_of_null_sample, with_post=with_post)

    null = raw_null - raw_null.mean()

    if with_null:
        return test_statistic, p_value_of(test_statistic, null), null
    else:
        return test_statistic, p_value_of(test_statistic, null)


def penaltied_distance(Dz, mask):
    return Dz + (mask - np.diag(np.diag(mask))) * Dz.max()


def testing_ing(N, trial, gamma, reserve_perm, with_post=True):
    mmd, pval, null = SDCIT(*median_heuristic(*henon(trial, N, gamma, 0, noise_dim=2, noise_std=0.5)),
                            size_of_null_sample=500, reserve_perm=reserve_perm, with_null=True, with_post=with_post)

    return [gamma, mmd, pval, null]


def time_sdcit():
    #
    with open('sdcit_time.csv', 'w') as f:
        for N in [200, 400]:
            for b in [250, 500]:
                for trial in range(300):
                    mat_load = scipy.io.loadmat(os.path.expanduser('~/kcipt_data/{}_{}_{}_{}_chaotic.mat'.format('0.0', trial, 0, N)), squeeze_me=True, struct_as_record=False)
                    data = mat_load['data']
                    X = data.Yt1
                    Y = data.Xt
                    Z = data.Yt[:, 0: 2]

                    start = time.time()
                    kkk400 = median_heuristic(X, Y, Z)
                    D400 = K2D(kkk400[-1])
                    SDCIT(*kkk400, Dz=D400, size_of_null_sample=b, reserve_perm=True, with_post=True)
                    endtime = time.time()
                    print(endtime - start, trial, N, b, file=f, sep=',', flush=True)


if __name__ == '__main__':
    time_sdcit()
