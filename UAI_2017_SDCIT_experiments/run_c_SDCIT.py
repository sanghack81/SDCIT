from os.path import exists

from joblib import Parallel
from joblib import delayed
from tqdm import tqdm, trange

from UAI_2017_SDCIT_experiments.exp_setup import SDCIT_RESULT_DIR, PARALLEL_JOBS
from UAI_2017_SDCIT_experiments.testing_utils import read_chaotic, read_postnonlinear_noise, chaotic_configs, postnonlinear_noise_configs
from sdcit.sdcit2 import c_SDCIT2
from sdcit.utils import *


def test_chaotic(independent, gamma, trial, N):
    np.random.seed(trial)
    mmd, pval = c_SDCIT2(*read_chaotic(independent, gamma, trial, N), seed=trial)
    return independent, gamma, trial, N, mmd, pval


def test_postnonlinear(independent, noise, trial, N):
    np.random.seed(trial)
    mmd, pval = c_SDCIT2(*read_postnonlinear_noise(independent, noise, trial, N), seed=trial)
    return independent, noise, trial, N, mmd, pval


def main():
    if not exists(SDCIT_RESULT_DIR + '/csdcit_chaotic.csv'):
        for independent, N, gamma in tqdm(chaotic_configs()):
            outs = Parallel(PARALLEL_JOBS)(delayed(test_chaotic)(independent, gamma, trial, N) for trial in trange(300))
            with open(SDCIT_RESULT_DIR + '/csdcit_chaotic.csv', 'a') as f:
                for out in outs:
                    print(*out, sep=',', file=f, flush=True)

    if not exists(SDCIT_RESULT_DIR + '/csdcit_postnonlinear.csv'):
        for noise, independent, N in tqdm(postnonlinear_noise_configs()):
            test_postnonlinear(independent, noise, 7, N)
            outs = Parallel(PARALLEL_JOBS)(delayed(test_postnonlinear)(independent, noise, trial, N) for trial in trange(300))
            with open(SDCIT_RESULT_DIR + '/csdcit_postnonlinear.csv', 'a') as f:
                for out in outs:
                    print(*out, sep=',', file=f, flush=True)


if __name__ == '__main__':
    main()
