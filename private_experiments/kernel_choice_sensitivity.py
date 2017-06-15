import os

import numpy as np
import numpy.ma as ma
import scipy.io
from sklearn.metrics import euclidean_distances

from UAI_2017_SDCIT_experiments.exp_setup import SDCIT_RESULT_DIR, SDCIT_DATA_DIR
from sdcit.kcit import python_kcit_K
from sdcit.sdcit import SDCIT
from sdcit.tests.t_utils import KS_statistic, AUPC
from sdcit.utils import K2D


def medd(D_squared):
    mask = np.triu(np.ones(D_squared.shape), 0)
    return ma.median(ma.array(D_squared, mask=mask))


def para(N, independent, trial):
    outs = []
    mat_load = scipy.io.loadmat(os.path.expanduser(SDCIT_DATA_DIR + '/{}_{}_{}_{}_chaotic.mat'.format('0.3', trial, independent, N)), squeeze_me=True, struct_as_record=False)
    data = mat_load['data']
    if independent:
        X = data.Xt1
        Y = data.Yt
        Z = data.Xt[:, 0:2]
    else:
        X = data.Yt1
        Y = data.Xt
        Z = data.Yt[:, 0: 2]

    DX = euclidean_distances(X, squared=True)
    DY = euclidean_distances(Y, squared=True)
    DZ = euclidean_distances(Z, squared=True)

    DX /= np.max(DX)
    DY /= np.max(DY)
    DZ /= np.max(DZ)

    mX = 0.5 / medd(DX)
    mY = 0.5 / medd(DY)
    mZ = 0.5 / medd(DZ)

    for multiplier in [0.01, 0.02, 50, 100]:
        KX = np.exp(-mX * DX * multiplier)
        KY = np.exp(-mY * DY * multiplier)
        KZ = np.exp(-mZ * DZ * multiplier)
        Dz = K2D(KZ)

        if np.random.rand() < 0.5:
            p_SDCIT = SDCIT(KX, KY, KZ, Dz=Dz, size_of_null_sample=500, seed=trial)[1]
            p_KCIT = python_kcit_K(KX, KY, KZ, seed=trial)[2]
        else:
            p_KCIT = python_kcit_K(KX, KY, KZ, seed=trial)[2]
            p_SDCIT = SDCIT(KX, KY, KZ, Dz=Dz, size_of_null_sample=500, seed=trial)[1]

        outs.append(['SDCIT', N, trial, multiplier, independent, p_SDCIT])
        outs.append(['KCIT', N, trial, multiplier, independent, p_KCIT])
    return outs


if __name__ == '__main__':
    #
    # import random
    #
    # n_cpus = multiprocessing.cpu_count()
    #
    # configs = list(itertools.product([200, 400], [0, 1], list(range(300))))
    # random.shuffle(configs)
    #
    # with Parallel(4 * n_cpus // 5, verbose=100) as parallel:
    #     lll = parallel(delayed(para)(*param) for param in tqdm(configs))
    #     fname = SDCIT_RESULT_DIR + '/sensitivity.csv'
    #     with open(fname, 'a') as f:
    #         for list_of_list in lll:
    #             for a_list in list_of_list:
    #                 print(*a_list, file=f, sep=',')

    fname = SDCIT_RESULT_DIR + '/sensitivity.csv'
    import pandas as pd

    all_multipliers = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    df = pd.read_csv(fname, names=['method', 'N', 'trial', 'multiplier', 'independent', 'p'])

    aupcs = {(N, method, multiplier): AUPC(gdf['p'])
             for (N, method, multiplier), gdf
             in df[df['independent'] == 0].groupby(by=['N', 'method', 'multiplier'])}

    ks = {(N, method, multiplier): KS_statistic(gdf['p'])
          for (N, method, multiplier), gdf
          in df[df['independent'] == 1].groupby(by=['N', 'method', 'multiplier'])}

    import seaborn as sns
    import matplotlib.pyplot as plt

    colors = sns.color_palette('Paired', 4)
    sns.set(style='white', font_scale=1.2)
    paper_rc = {'lines.linewidth': 0.8, 'lines.markersize': 2, 'patch.linewidth': 1}
    sns.set_context("paper", rc=paper_rc)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{cmbright}')

    aupc_sdcit_200 = np.array([aupcs[(200, 'SDCIT', mp)] for mp in all_multipliers])
    aupc_kcit_200 = np.array([aupcs[(200, 'KCIT', mp)] for mp in all_multipliers])
    aupc_sdcit_400 = np.array([aupcs[(400, 'SDCIT', mp)] for mp in all_multipliers])
    aupc_kcit_400 = np.array([aupcs[(400, 'KCIT', mp)] for mp in all_multipliers])
    ks_sdcit_200 = np.array([ks[(200, 'SDCIT', mp)] for mp in all_multipliers])
    ks_kcit_200 = np.array([ks[(200, 'KCIT', mp)] for mp in all_multipliers])
    ks_sdcit_400 = np.array([ks[(400, 'SDCIT', mp)] for mp in all_multipliers])
    ks_kcit_400 = np.array([ks[(400, 'KCIT', mp)] for mp in all_multipliers])

    plt.semilogx(all_multipliers, aupc_sdcit_200, label='_nolegend_', ls=':', c=colors[3])
    plt.semilogx(all_multipliers, aupc_kcit_200, label='_nolegend_', ls=':', c=colors[1])
    plt.semilogx(all_multipliers, aupc_sdcit_400, label='AUPC SDCIT', c=colors[3])
    plt.semilogx(all_multipliers, aupc_kcit_400, label='AUPC KCIT', c=colors[1])

    plt.semilogx(all_multipliers, ks_sdcit_200, label='_nolegend_', ls=':', c=colors[2])
    plt.semilogx(all_multipliers, ks_kcit_200, label='_nolegend_', ls=':', c=colors[0])
    plt.semilogx(all_multipliers, ks_sdcit_400, label='KS SDCIT', c=colors[2])
    plt.semilogx(all_multipliers, ks_kcit_400, label='KS KCIT', c=colors[0])

    plt.legend()
    plt.savefig('figures/sensitivity.pdf')
    plt.close()
