import multiprocessing

import numpy.ma as ma
from joblib import Parallel
from joblib import delayed
from sklearn.metrics import euclidean_distances

from experiments.synthetic import henon
from kcipt.algo import permuted
from kcipt.utils import *


def learn_apply(kx, ky, kz, learn_with='Z', apply_to='Y'):
    assert apply_to in {'X', 'Y', 'Z'}
    kxyz = kx * ky * kz
    if apply_to == 'X':
        k_perm, k_unperm = kx, ky * kz
    elif apply_to == 'Y':
        k_perm, k_unperm = ky, kx * kz
    else:
        k_perm, k_unperm = kz, kx * ky
    k_learns = {'X': kx, 'Y': ky, 'Z': kz}
    if len(learn_with) == 1:
        k_learn = k_learns[learn_with]
    else:
        k_learn = np.multiply(*[k_learns[v] for v in learn_with])

    n = len(kx)
    full_idx = np.arange(0, n)
    Pidx = permuted(full_idx, k_learn)  #

    mask = np.zeros((n, n))
    mask[full_idx, full_idx] = 1  # i==j
    mask[full_idx, Pidx] = 1  # pi_i = j
    mask[Pidx, full_idx] = 1  # i = pi_j

    K11 = kxyz
    K12 = k_unperm * k_perm[np.ix_(full_idx, Pidx)]
    K22 = k_unperm * k_perm[np.ix_(Pidx, Pidx)]

    mmd = ma.array(K11, mask=mask).mean() + \
          ma.array(K22, mask=mask).mean() - \
          2 * ma.array(K12, mask=mask).mean()

    return mmd


def for_null(kx, ky, kz, B, to_learn='YZ', to_perm='Y'):
    n = len(kx)
    nulls = np.zeros((B,))
    for B_i in range(B):
        idx1 = np.random.choice(n, n // 2, replace=False)
        _11 = np.ix_(idx1, idx1)
        nulls[B_i] = learn_apply(kx[_11], ky[_11], kz[_11], to_learn, to_perm)

    return nulls


def one_and_another_shot_KCIPT(kx, ky, kz, b=500):
    mmd0Y = learn_apply(kx, ky, kz, 'Z', 'Y')
    # mmd0X = learn_apply(kx, ky, kz, 'Z', 'X')
    # mmd_helper_Y = learn_apply(kx, ky, kz, 'YZ', 'Y')
    # mmd_helper_X = learn_apply(kx, ky, kz, 'XZ', 'X')

    nullsY = for_null(kx, ky, kz, b, 'Z', 'Y')
    p_0_org = p_value_of(0, nullsY)
    nullsY -= np.mean(nullsY)
    p_0_centered = p_value_of(0, nullsY)
    to_shrink = p_0_org / p_0_centered
    # nullsX = for_null(kx, ky, kz, b, ratio, 'XZ', 'X')

    return mmd0Y, p_value_of(2 * mmd0Y, nullsY / to_shrink)  # , p_value_of(mmd0X, nullsX), p_value_of(2 * mmd_helper_Y, nullsY), p_value_of(2 * mmd_helper_X, nullsX)


# def para(seed, n, slope, ratio, datatype='henon'):
#     if datatype == 'henon':
#         x, y, z = henon(seed, n, slope, False)
#         # kx = auto_rbf_kernel(x)
#         # ky = auto_rbf_kernel(y)
#         dx = euclidean_distances(x, squared=True)
#         dy = euclidean_distances(y, squared=True)
#         dz = euclidean_distances(z, squared=True)
#         mx = np.median(dx)
#         my = np.median(dy)
#         mz = np.median(dz)  # without diag?
#         kx = exp(-0.5 * dx / mx)
#         ky = exp(-0.5 * dy / my)
#         kz = exp(-0.5 * dz / mz)
#     elif datatype == 'old':
#         kx, ky, kz, _, _, _ = data_gen_old(n, seed, slope)
#     elif datatype == 'one':
#         kx, ky, kz, _, _, _ = data_gen_one(n, seed, slope)
#
#     np.random.seed(seed)
#     return one_and_another_shot_KCIPT(kx, ky, kz, ratio=ratio)


if __name__ == '__main__':
    pass
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    #
    # if multiprocessing.cpu_count() == 32:
    #     cnt = 0
    #     trial = 500
    #     with open('newnull_henon.csv', 'a') as f:
    #         for oi in range(1 + (trial // 32)):
    #             for slope in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    #                 for ratio in [1]:
    #                     ps = Parallel(-1)(delayed(para)(i + oi * 32, 200, slope, ratio) for i in range(32))
    #                     for p in ps:
    #                         print(slope, p, sep=',', file=f)
    #                         # print(slope, *p, sep=',', file=f)
    #                     f.flush()
    #
    #             sns.set()
    #             data = pd.read_csv('newnull_henon.csv', names=['slope', 'py'])
    #             data = data[data['slope'] <= 0.5]
    #             grid = sns.FacetGrid(data, hue='slope', sharey=False)
    #             grid.map(sns.distplot, 'py', bins=20, kde=False)
    #             plt.legend()
    #             plt.savefig('newnull_henon.pdf')
    #             plt.close()
    #
    # # sns.set()
    # data = pd.read_csv('newnull_henon.csv', names=['slope', 'py'])
    # inde_data = data[data['slope'] == 0.0]
    # _, ksp = scipy.stats.kstest(np.array(inde_data['py'])[0:300], 'uniform')
    # print(ksp)
    #
    # print()
    # for s, dd in pd.groupby(data, 'slope'):
    #     ttt = [(uniq_v, np.mean(dd['py'] <= uniq_v)) for uniq_v in np.unique(dd['py'])]
    #     area = 0
    #     prev_x, prev_y = 0, 0
    #     for x, y in ttt:
    #         area += (x - prev_x) * prev_y
    #         prev_x, prev_y = x, y
    #     area += (1 - prev_x) * prev_y
    #     assert prev_y == 1
    #     print(s, area)

    #
    # grid.map(plt.scatter, 'px', 'py', alpha=0.2, s=5)
    # plt.savefig('newnull_henon_power.pdf')
    # plt.close()
