from joblib import Parallel
from joblib import delayed

from experiments.synthetic import henon
from kcipt.algo import permuted
from kcipt.utils import *


def mmd_selfperm(kxz, ky, Dz):
    n = len(kxz)
    full_idx = np.arange(0, n)
    Pidx = permuted(full_idx, D=Dz)  #

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


def bootstrap_selfperm(kxz, ky, Dz, B):
    n = len(kxz)
    nulls = np.zeros((B,))

    for B_i in range(B):
        idx1 = np.random.choice(n, n // 2, replace=False)
        _11 = np.ix_(idx1, idx1)
        nulls[B_i], _, _ = mmd_selfperm(kxz[_11], ky[_11], Dz[_11])

    return nulls


def lee_KCIPT(kx, ky, kz, Dz=None, size_of_null_sample=500, reserve_perm=False, with_null=False):
    if Dz is None:
        Dz = K2D(kz)
    kxz = kx * kz

    test_statistic, mask, Pidx = mmd_selfperm(kxz, ky, Dz)

    if reserve_perm:
        _, mask, Pidx = mmd_selfperm(kxz, ky, penaltied_distance(Dz, mask))

    # avoid permutation between already permuted pairs.
    raw_null_distr = bootstrap_selfperm(kxz,
                                        ky[np.ix_(Pidx, Pidx)],
                                        penaltied_distance(Dz, mask),
                                        size_of_null_sample)
    correct_null_distr = 0.5 * (raw_null_distr - raw_null_distr.mean())
    if with_null:
        return test_statistic, p_value_of(test_statistic, correct_null_distr), correct_null_distr
    else:
        return test_statistic, p_value_of(test_statistic, correct_null_distr)


def penaltied_distance(Dz, mask):
    return Dz + (mask - np.diag(np.diag(mask))) * Dz.max()


def testing_ing(trial, gamma, reserve_perm):
    mmd, pval, null = lee_KCIPT(*median_heuristic(*henon(trial, 400, gamma, 0, noise_dim=2, noise_std=0.5)),
                                size_of_null_sample=1000, reserve_perm=reserve_perm, with_null=True)

    return [gamma, mmd, pval, null]


if __name__ == '__main__':

    import seaborn as sns
    import matplotlib.pyplot as plt

    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    outs = Parallel(-1)(delayed(testing_ing)(trial, gamma, False) for trial in range(10) for gamma in gammas)
    T_outs = Parallel(-1)(delayed(testing_ing)(trial, gamma, True) for trial in range(10) for gamma in gammas)

    for gamma in gammas:
        temp = np.where([out[0] == gamma for out in outs])[0]  # list of bool

        sns.set(palette=sns.color_palette('Paired', 2))
        for iii, idx in enumerate(temp):
            _, mmd, pval, null = outs[idx]
            _, T_mmd, T_pval, T_null = T_outs[idx]
            if iii == 0:
                sns.distplot(0.5 * (null - null.mean()), hist=False, label='second-class null')
                sns.distplot(0.5 * (T_null - T_null.mean()), hist=False, label='first-class null')
            else:
                sns.distplot(0.5 * (null - null.mean()), hist=False)
                sns.distplot(0.5 * (T_null - T_null.mean()), hist=False)
        plt.legend()
        plt.savefig('yoyo_{}.pdf'.format(gamma))
        plt.close()
