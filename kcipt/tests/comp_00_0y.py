import matplotlib.pyplot as plt
import seaborn as sns

from kcipt.algo import permuted
from kcipt.utils import *


def comp_KCIPT(kx, ky, kz, B):
    n = len(kx)
    # base
    kxz = kx * kz
    kxyz = kx * ky * kz

    #################### test statistic ##########################
    MMD00, MMD0Y, MMD0Z, MMDYY, MMDZZ = 0, 0, 0, 0, 0
    pairs = np.zeros((B, 2))
    for B_i in range(1, B + 1):
        idx1, idx2 = split_1_to_r(n, 1)
        _11, _22 = np.ix_(idx1, idx1), np.ix_(idx2, idx2)

        Pidx1, Pidx2 = permuted(idx1, kz[_11]), permuted(idx2, kz[_22])
        #
        _P1P1, _P2P2 = np.ix_(Pidx1, Pidx1), np.ix_(Pidx2, Pidx2)
        #
        within_k_00_1 = mean_without_diag(kxyz[_11])
        within_k_00_2 = mean_without_diag(kxyz[_22])
        within_k_yy_2 = mean_without_diag(kxz[_22] * ky[_P2P2])
        #
        _12, _1P2, _P1P2 = np.ix_(idx1, idx2), np.ix_(idx1, Pidx2), np.ix_(Pidx1, Pidx2)

        bw_k_00 = np.mean(kxyz[_12])
        bw_k_0y = np.mean(kxz[_12] * ky[_1P2])

        cur_MMD00 = (within_k_00_1 + within_k_00_2 - 2 * bw_k_00 - MMD00)
        curMMD0Y = (within_k_00_1 + within_k_yy_2 - 2 * bw_k_0y - MMD0Y)

        pairs[B_i - 1, :] = [cur_MMD00, curMMD0Y]

    return pairs


if __name__ == '__main__':
    file_names_withinner = list()
    file_names_noinner = list()

    B = 1000
    palette = sns.color_palette("Set1", 3)
    sns.set(palette=palette)
    for i, trial in enumerate([0]):
        kx, ky, kz, _, _, _ = data_gen_one(100, trial, trial)
        pairs = comp_KCIPT(kx, ky, kz, B)
        sns.distplot(pairs[:, 0], hist=False, label='mmd00')
        sns.distplot(pairs[:, 1], hist=False, label='mmd0y')
        sns.distplot(pairs[:, 1] - pairs[:, 0], hist=False, label='mmd0y-mmd00')
        # plt.scatter(pairs[:, 0], pairs[:, 1], c=palette[i], alpha=0.4, label=str(trial))
        plt.plot([np.mean(pairs[:, 0]), np.mean(pairs[:, 0])], [0, 10], label='mean mmd00')
        plt.plot([np.mean(pairs[:, 1]), np.mean(pairs[:, 1])], [0, 10], label='mean mmd0y')
        plt.plot([np.mean(pairs[:, 1]) - np.mean(pairs[:, 0]), np.mean(pairs[:, 1]) - np.mean(pairs[:, 0])], [0, 10], label='mean mmd0y-mmd00')

    plt.legend()
    plt.axes().set_xlabel('MMD00')
    plt.axes().set_ylabel('MMD0Y')
    plt.savefig('mmd00-mmd0y.pdf')
    plt.close()
