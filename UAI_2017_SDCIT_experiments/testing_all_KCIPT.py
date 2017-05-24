import time

import scipy.io
import scipy.stats
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm
from tqdm import trange

from UAI_2017_SDCIT_experiments.testing_utils import *
from sdcit.algo import c_KCIPT
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
    # # General Empirical Evaluation
    existing_chaotic = set(tuple(line.split(',')[:4]) for line in open('results/kcipt_chaotic.csv', 'r'))

    for independent, N, gamma in chaotic_configs():
        to_test_trials = set(range(300))
        for trial in range(300):
            if (str(independent), str(gamma), str(trial), str(N)) in existing_chaotic:
                to_test_trials.remove(trial)
        if not to_test_trials:
            print('empty: {},{},{}'.format(independent, N, gamma))
            continue
        to_test_trials = sorted(list(to_test_trials))

        print(independent, N, gamma)
        time.sleep(1)
        outs = Parallel(-1)(delayed(test_chaotic)(independent, gamma, trial, N, 25) for trial in tqdm(to_test_trials))
        with open('results/kcipt_chaotic.csv', 'a') as f:
            for out in outs:
                if out is not None:
                    print(*out, sep=',', file=f, flush=True)

    if False:
        # #
        for noise, independent, N in postnonlinear_noise_configs():
            print(noise, independent, N)
            time.sleep(1)
            outs = Parallel(-1)(delayed(test_postnonlinear)(independent, noise, trial, N, 25) for trial in trange(300))
            with open('results/kcipt_postnonlinear.csv', 'a') as f:
                for out in outs:
                    print(*out, sep=',', file=f, flush=True)

        independent, gamma, N = 1, '0.0', 400
        for B in [5000, 20000]:
            print(independent, gamma, N, B)
            time.sleep(1)
            for trial in trange(300):
                out = test_chaotic(independent, gamma, trial, N, B=B, n_jobs=32)
                with open('results/kcipt_chaotic_{}.csv'.format(B), 'a') as f:
                    print(*out, sep=',', file=f, flush=True)


if __name__ == '__main__':
    main()
