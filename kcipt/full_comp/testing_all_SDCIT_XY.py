import itertools
import os
import time
from os.path import exists

import scipy.io
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm

from kcipt.sdcit import SDCIT_XY
from kcipt.utils import *


def test_chaotic(independent, gamma, trial, N):
    np.random.seed(trial)

    mat_load = scipy.io.loadmat(os.path.expanduser('~/kcipt_data/{}_{}_{}_{}_chaotic.mat'.format(gamma, trial, independent, N)), squeeze_me=True, struct_as_record=False)
    data = mat_load['data']
    if independent:
        X = data.Xt1
        Y = data.Yt
        Z = data.Xt[:, 0:2]
    else:
        X = data.Yt1
        Y = data.Xt
        Z = data.Yt[:, 0: 2]

    kx, ky, kz = median_heuristic(X, Y, Z)

    mmd, pval = SDCIT_XY(kx, ky, kz, size_of_null_sample=1000, reserve_perm=True)
    return (independent, gamma, trial, N, mmd, pval)


def test_chaotic_with_null(independent, gamma, trial, N):
    np.random.seed(trial)

    mat_load = scipy.io.loadmat(os.path.expanduser('~/kcipt_data/{}_{}_{}_{}_chaotic.mat'.format(gamma, trial, independent, N)), squeeze_me=True, struct_as_record=False)
    data = mat_load['data']
    if independent:
        X = data.Xt1
        Y = data.Yt
        Z = data.Xt[:, 0:2]
    else:
        X = data.Yt1
        Y = data.Xt
        Z = data.Yt[:, 0: 2]

    kx, ky, kz = median_heuristic(X, Y, Z)

    mmd, pval, null = SDCIT_XY(kx, ky, kz, size_of_null_sample=1000, reserve_perm=True, with_null=True)
    return mmd, null


def test_postnonlinear(independent, noise, trial, N):
    np.random.seed(trial)

    data_file = os.path.expanduser('~/kcipt_data/{}_{}_{}_{}_postnonlinear.mat'.format(noise, trial, independent, N))
    dist_mat_file = os.path.expanduser('~/kcipt_data/dist_{}_{}_{}_{}_postnonlinear.mat'.format(noise, trial, independent, N))

    if not exists(dist_mat_file):
        while not exists(dist_mat_file):
            time.sleep(5)
        time.sleep(1)

    mat_load = scipy.io.loadmat(data_file, squeeze_me=True, struct_as_record=False)
    data = mat_load['data']
    X = np.array(data.X).reshape((len(data.X), -1))
    Y = np.array(data.Y).reshape((len(data.Y), -1))
    Z = np.array(data.Z).reshape((len(data.Z), -1))

    kx, ky, kz = median_heuristic(X, Y, Z)

    mat_load = scipy.io.loadmat(dist_mat_file, squeeze_me=True, struct_as_record=False)
    Dz = np.array(mat_load['D'])

    mmd, pval = SDCIT_XY(kx, ky, kz, Dz, size_of_null_sample=1000, reserve_perm=True)
    return (independent, noise, trial, N, mmd, pval)


if __name__ == '__main__':
    if True:
        import seaborn as sns
        import matplotlib.pyplot as plt

        for independent, N, gamma in list(itertools.product([1], [400], ['0.0'])):
            outs = Parallel(-1,verbose=100)(delayed(test_chaotic_with_null)(independent, gamma, trial, N) for trial in range(300))
            print('plotting....')
            sns.set()
            mmds = [out[0] for out in outs]
            sns.distplot(mmds, bins=20, norm_hist=True, label='mmds')
            for out in outs:
                mmd, null = out
                sns.distplot(null, hist=False, kde=True, kde_kws={'lw': 0.1, 'alpha': 0.1}, color='k')

            allnull = np.hstack([out[1] for out in outs]).squeeze()
            sns.distplot(allnull, bins=20, norm_hist=True, label='all null')

            plt.savefig('xylee_inspect.pdf')
            plt.close()

    if False:
        for independent, N, gamma in tqdm(list(itertools.product([0, 1], [200, 400], ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']))):
            if independent == 0 and N == 400:
                continue
            outs = Parallel(16)(delayed(test_chaotic)(independent, gamma, trial, N) for trial in range(300))
            with open('xylee_chaotic.csv', 'a') as f:
                for out in outs:
                    print(*out, sep=',', file=f, flush=True)

        for noise, independent, N in tqdm(list(itertools.product(range(5), [0, 1], [200, 400]))):
            outs = Parallel(16)(delayed(test_postnonlinear)(independent, noise, trial, N) for trial in range(300))
            with open('xylee_postnonlinear.csv', 'a') as f:
                for out in outs:
                    print(*out, sep=',', file=f, flush=True)

        for noise, independent, N in tqdm(list(itertools.product([9, 19, 49], [0, 1], [400]))):
            outs = Parallel(16)(delayed(test_postnonlinear)(independent, noise, trial, N) for trial in range(300))
            with open('xylee_postnonlinear.csv', 'a') as f:
                for out in outs:
                    print(*out, sep=',', file=f, flush=True)
