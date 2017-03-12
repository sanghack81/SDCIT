import multiprocessing
import time

import scipy.io
import scipy.stats
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm
from tqdm import trange

from experiments.testing_utils import *
from kcipt.algo import c_KCIPT
from kcipt.utils import *


def test_chaotic(independent, gamma, trial, N, B=25, n_jobs=1):
    np.random.seed(trial)

    kx, ky, kz, Dz = read_chaotic(independent, gamma, trial, N)

    if B == 25:
        pval, mmds, inner_null, _ = c_KCIPT(kx, ky, kz, Dz, B, 10000, 10000, n_jobs)
    else:
        _, mmds, inner_null, _ = c_KCIPT(kx, ky, kz, Dz, B, 100, 0, n_jobs)
        inner_null = np.squeeze(inner_null)
        pval = scipy.stats.norm.sf(mmds.mean(), 0, inner_null.std() / np.sqrt(B))

    return independent, gamma, trial, N, np.mean(mmds), pval, B


def test_postnonlinear(independent, noise, trial, N, B=25):
    np.random.seed(trial)

    kx, ky, kz, Dz = read_postnonlinear_noise(independent, noise, trial, N)

    pval, mmds, _, _ = c_KCIPT(kx, ky, kz, Dz, B, 200, 10000)
    return independent, noise, trial, N, np.mean(mmds), pval, B


if __name__ == '__main__':
    # General Empirical Evaluation
    if False:
        if multiprocessing.cpu_count() == 32:
            for independent, N, gamma in tqdm(chaotic_configs()):
                print(independent, gamma, N)
                outs = Parallel(-5)(delayed(test_chaotic)(independent, gamma, trial, N) for trial in range(300))
                with open('../results/kcipt_chaotic.csv', 'a') as f:
                    for out in outs:
                        if out is not None:
                            print(*out, sep=',', file=f, flush=True)
        #
        if multiprocessing.cpu_count() == 32:
            for noise, independent, N in tqdm(postnonlinear_noise_configs()):
                print(noise, independent, N)
                outs = Parallel(-5)(delayed(test_postnonlinear)(independent, noise, trial, N) for trial in range(300))
                with open('../results/kcipt_postnonlinear.csv', 'a') as f:
                    for out in outs:
                        if out is not None:
                            print(*out, sep=',', file=f, flush=True)

    #
    if False:
        if multiprocessing.cpu_count() == 32:
            independent, gamma, N = 0, '0.0', 400
            print(independent, gamma, N)
            outs = [test_chaotic(independent, gamma, trial, N, 1470, 32) for trial in trange(300)]
            with open('../results/kcipt_chaotic_1470.csv', 'a') as f:
                for out in outs:
                    if out is not None:
                        print(*out, sep=',', file=f, flush=True)

    #
    if False:
        if multiprocessing.cpu_count() == 32:
            independent, gamma, N = 0, '0.0', 400
            print(independent, gamma, N)
            outs = [test_chaotic(independent, gamma, trial, N, 5000, 32) for trial in trange(300)]
            with open('../results/kcipt_chaotic_5000.csv', 'a') as f:
                for out in outs:
                    if out is not None:
                        print(*out, sep=',', file=f, flush=True)

    # for what???
    if False:
        if multiprocessing.cpu_count() == 32:
            for independent, gamma, N in tqdm(list(itertools.product([0, 1], ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5'], [200, 400]))):
                print(independent, gamma, N)
                outs = [test_chaotic(independent, gamma, trial, N, 1470, 32) for trial in range(300)]
                with open('../results/kcipt_chaotic_1470_full.csv', 'a') as f:
                    for out in outs:
                        if out is not None:
                            print(*out, sep=',', file=f, flush=True)
                time.sleep(5)
    if True:
        if multiprocessing.cpu_count() == 32:
            independent, gamma, N = 0, '0.0', 400
            print(independent, gamma, N)
            outs = [test_chaotic(independent, gamma, trial, N, 20000, 16) for trial in trange(300)]
            with open('../results/kcipt_chaotic_20000.csv', 'a') as f:
                for out in outs:
                    if out is not None:
                        print(*out, sep=',', file=f, flush=True)
