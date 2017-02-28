import itertools
import scipy.io
import multiprocessing

import numpy.ma as ma
from joblib import Parallel
from joblib import delayed
from sklearn.metrics import euclidean_distances
from tqdm import tqdm

from experiments.synthetic import henon
from kcipt.algo import permuted
from kcipt.twoshot import one_and_another_shot_KCIPT
from kcipt.utils import *


def toK(x, y, z):
    dx = euclidean_distances(x, squared=True)
    dy = euclidean_distances(y, squared=True)
    dz = euclidean_distances(z, squared=True)

    mx = ma.median(ma.array(dx, mask=np.triu(np.ones(dx.shape), 0)))
    my = ma.median(ma.array(dy, mask=np.triu(np.ones(dy.shape), 0)))
    mz = ma.median(ma.array(dz, mask=np.triu(np.ones(dz.shape), 0)))

    kx = exp(-0.5 * dx / mx)
    ky = exp(-0.5 * dy / my)
    kz = exp(-0.5 * dz / mz)
    return kx, ky, kz


def tonp(xxx):
    np.array(xxx)


def test_chaotic(independent, gamma, trial, N):
    np.random.seed(trial)

    mat_load = scipy.io.loadmat('/Users/sxl439/kcipt_data/{}_{}_{}_{}_chaotic.mat'.format(gamma, trial, independent, N), squeeze_me=True, struct_as_record=False)
    data = mat_load['data']
    if independent:
        X = data.Xt1
        Y = data.Yt
        Z = data.Xt[:, 0:2]
    else:
        X = data.Yt1
        Y = data.Xt
        Z = data.Yt[:, 0: 2]

    kx, ky, kz = toK(X, Y, Z)

    mmd, pval = one_and_another_shot_KCIPT(kx, ky, kz, 500)
    return (independent, gamma, trial, N, mmd, pval)


def test_postnonlinear(independent, noise, trial, N):
    np.random.seed(trial)

    mat_load = scipy.io.loadmat('/Users/sxl439/kcipt_data/{}_{}_{}_{}_postnonlinear.mat'.format(noise, trial, independent, N), squeeze_me=True, struct_as_record=False)
    data = mat_load['data']
    X = np.array(data.X).reshape((len(data.X), -1))
    Y = np.array(data.Y).reshape((len(data.Y), -1))
    Z = np.array(data.Z).reshape((len(data.Z), -1))

    kx, ky, kz = toK(X, Y, Z)

    mmd, pval = one_and_another_shot_KCIPT(kx, ky, kz, 500)
    return (independent, noise, trial, N, mmd, pval)


if __name__ == '__main__':
    if True:
        # for independent, N, gamma in tqdm(list(itertools.product([0, 1], [200, 400], ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']))):
        #     outs = Parallel(-1)(delayed(test_chaotic)(independent, gamma, trial, N) for trial in range(300))
        #     with open('lee_chaotic.csv', 'a') as f:
        #         for out in outs:
        #             print(*out, sep=',', file=f, flush=True)

        for noise, independent, N in tqdm(list(itertools.product(range(5), [0, 1], [200, 400]))):
            outs = Parallel(-3)(delayed(test_postnonlinear)(independent, noise, trial, N) for trial in range(300))
            with open('lee_postnonlinear.csv', 'a') as f:
                for out in outs:
                    print(*out, sep=',', file=f, flush=True)

        for noise, independent, N in tqdm(list(itertools.product([9, 19, 49], [0, 1], [400]))):
            outs = Parallel(-3)(delayed(test_postnonlinear)(independent, noise, trial, N) for trial in range(300))
            with open('lee_postnonlinear.csv', 'a') as f:
                for out in outs:
                    print(*out, sep=',', file=f, flush=True)
