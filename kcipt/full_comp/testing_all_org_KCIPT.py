import itertools
import multiprocessing
import os
import time
from os.path import exists

import scipy.io
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm
from tqdm import trange

from kcipt.algo import c_KCIPT
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


def test_chaotic(independent, gamma, trial, N, B=25, n_jobs=1):
    np.random.seed(trial)

    # if exists('pykcipt_chaotic.csv'):
    #     for line in open('pykcipt_chaotic.csv', 'r'):
    #         if line.startswith('{},{},{},{},'.format(independent, gamma, trial, N)):
    #             return None

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

    kx, ky, kz = toK(X, Y, Z)
    Dz = K2D(kz)
    pval, mmds, _, _ = c_KCIPT(kx, ky, kz, Dz, B, 200, 10000, n_jobs)
    return (independent, gamma, trial, N, np.mean(mmds), pval, B)


def test_postnonlinear(independent, noise, trial, N, B=25):
    np.random.seed(trial)

    # if exists('pykcipt_postnonlinear.csv'):
    #     for line in open('pykcipt_postnonlinear.csv', 'r'):
    #         if line.startswith('{},{},{},{},'.format(independent, noise, trial, N)):
    #             return None

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

    kx, ky, kz = toK(X, Y, Z)

    mat_load = scipy.io.loadmat(dist_mat_file, squeeze_me=True, struct_as_record=False)
    Dz = np.array(mat_load['D'])

    pval, mmds, _, _ = c_KCIPT(kx, ky, kz, Dz, B, 200, 10000)
    return (independent, noise, trial, N, np.mean(mmds), pval, B)


if __name__ == '__main__':
    if False:
        if multiprocessing.cpu_count() == 32:
            for independent, gamma, N in tqdm(list(itertools.product([0, 1], ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5'], [200, 400]))):
                print(independent, gamma, N)
                outs = Parallel(-5)(delayed(test_chaotic)(independent, gamma, trial, N) for trial in range(300))
                with open('pykcipt_chaotic.csv', 'a') as f:
                    for out in outs:
                        if out is not None:
                            print(*out, sep=',', file=f, flush=True)
        #
        if multiprocessing.cpu_count() == 32:
            for noise, independent, N in tqdm(list(itertools.product(range(5), [0, 1], [200, 400]))):
                print(noise, independent, N)
                outs = Parallel(-5)(delayed(test_postnonlinear)(independent, noise, trial, N) for trial in range(300))
                with open('pykcipt_postnonlinear.csv', 'a') as f:
                    for out in outs:
                        if out is not None:
                            print(*out, sep=',', file=f, flush=True)

        if multiprocessing.cpu_count() == 32:
            for noise, independent, N in tqdm(list(itertools.product([9, 19, 49], [0, 1], [400]))):
                print(noise, independent, N)
                outs = Parallel(-5)(delayed(test_postnonlinear)(independent, noise, trial, N) for trial in range(300))
                with open('pykcipt_postnonlinear.csv', 'a') as f:
                    for out in outs:
                        if out is not None:
                            print(*out, sep=',', file=f, flush=True)
    #
    if True:
        if multiprocessing.cpu_count() == 32:
            independent, gamma, N = 0, '0.0', 400
            print(independent, gamma, N)
            outs = [test_chaotic(independent, gamma, trial, N, 1470, 32) for trial in trange(300)]
            with open('pykcipt_chaotic_1470.csv', 'a') as f:
                for out in outs:
                    if out is not None:
                        print(*out, sep=',', file=f, flush=True)
