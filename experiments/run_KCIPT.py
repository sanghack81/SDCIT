import time
from os.path import exists

import scipy.io
import scipy.stats
from joblib import Parallel
from joblib import delayed
from tqdm import trange

from experiments.exp_setup import SDCIT_RESULT_DIR, PARALLEL_JOBS
from experiments.exp_utils import *
from sdcit.kcipt import c_KCIPT
from sdcit.utils import *


def test_chaotic(independent, gamma, trial, N, B=25, n_jobs=1):
    kx, ky, kz, Dz = read_chaotic(independent, gamma, trial, N)

    if B < 100:
        # classic mode
        pval, mmds, _, _ = c_KCIPT(kx, ky, kz, Dz, B, 10000, 10000, n_jobs, trial)
    else:
        # advanced mode
        _, mmds, inner_null, _ = c_KCIPT(kx, ky, kz, Dz, B, 100, 0, n_jobs, trial)
        inner_null = np.squeeze(inner_null)
        pval = scipy.stats.norm.sf(mmds.mean(), 0, inner_null.std() / np.sqrt(B))

    return independent, gamma, trial, N, mmds.mean(), pval, B


def test_postnonlinear(independent, noise, trial, N, B=25, n_jobs=1):
    kx, ky, kz, Dz = read_postnonlinear_noise(independent, noise, trial, N)

    if B < 100:
        # classic mode
        pval, mmds, _, _ = c_KCIPT(kx, ky, kz, Dz, B, 10000, 10000, n_jobs, trial)
    else:
        # advanced mode
        _, mmds, inner_null, _ = c_KCIPT(kx, ky, kz, Dz, B, 100, 0, n_jobs, trial)
        inner_null = np.squeeze(inner_null)
        pval = scipy.stats.norm.sf(mmds.mean(), 0, inner_null.std() / np.sqrt(B))

    return independent, noise, trial, N, mmds.mean(), pval, B


def main():
    if not exists(SDCIT_RESULT_DIR + '/kcipt_chaotic.csv'):
        for independent, N, gamma in chaotic_configs():
            print(independent, N, gamma)
            time.sleep(1)
            outs = Parallel(PARALLEL_JOBS)(delayed(test_chaotic)(independent, gamma, trial, N, 25) for trial in range(300))
            with open(SDCIT_RESULT_DIR + '/kcipt_chaotic.csv', 'a') as f:
                for out in outs:
                    if out is not None:
                        print(*out, sep=',', file=f, flush=True)
    else:
        print('skipping KCIPT on chaotic time series data')

    # #
    if not exists(SDCIT_RESULT_DIR + '/kcipt_postnonlinear.csv'):
        for noise, independent, N in postnonlinear_noise_configs():
            print(noise, independent, N)
            time.sleep(1)
            outs = Parallel(PARALLEL_JOBS)(delayed(test_postnonlinear)(independent, noise, trial, N, 25) for trial in range(300))
            with open(SDCIT_RESULT_DIR + '/kcipt_postnonlinear.csv', 'a') as f:
                for out in outs:
                    print(*out, sep=',', file=f, flush=True)
    else:
        print('skipping KCIPT on post nonlinear noise data')

    independent, gamma, N = 1, '0.0', 400
    for B in [5000, 20000]:
        if not exists(SDCIT_RESULT_DIR + '/kcipt_chaotic_{}.csv'.format(B)):
            print(independent, gamma, N, B)
            time.sleep(1)
            for trial in trange(300):
                out = test_chaotic(independent, gamma, trial, N, B=B, n_jobs=PARALLEL_JOBS)
                with open(SDCIT_RESULT_DIR + '/kcipt_chaotic_{}.csv'.format(B), 'a') as f:
                    print(*out, sep=',', file=f, flush=True)
        else:
            print('skipping KCIPT on chaotic time series data with B={}'.format(B))


if __name__ == '__main__':
    main()
