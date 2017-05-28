from os.path import exists

from joblib import Parallel
from joblib import delayed
from tqdm import tqdm

from UAI_2017_SDCIT_experiments.exp_setup import SDCIT_RESULT_DIR
from UAI_2017_SDCIT_experiments.testing_utils import read_chaotic, read_postnonlinear_noise, chaotic_configs, postnonlinear_noise_configs
from sdcit.sdcit2 import SDCIT2
from sdcit.utils import *


def test_chaotic(independent, gamma, trial, N):
    np.random.seed(trial)
    mmd, pval = SDCIT2(*read_chaotic(independent, gamma, trial, N), seed=trial)
    return independent, gamma, trial, N, mmd, pval


def test_postnonlinear(independent, noise, trial, N):
    np.random.seed(trial)
    mmd, pval = SDCIT2(*read_postnonlinear_noise(independent, noise, trial, N), seed=trial)
    return independent, noise, trial, N, mmd, pval


def main():
    if not exists(SDCIT_RESULT_DIR + '/sdcit_chaotic.csv'):
        for independent, N, gamma in tqdm(chaotic_configs()):
            outs = Parallel(-1)(delayed(test_chaotic)(independent, gamma, trial, N) for trial in range(300))
            with open(SDCIT_RESULT_DIR + '/sdcit_chaotic.csv', 'a') as f:
                for out in outs:
                    print(*out, sep=',', file=f, flush=True)
    else:
        print('skipping SDCIT on chaotic time series data')

    if not exists(SDCIT_RESULT_DIR + '/sdcit_postnonlinear.csv'):
        for noise, independent, N in tqdm(postnonlinear_noise_configs()):
            outs = Parallel(-1)(delayed(test_postnonlinear)(independent, noise, trial, N) for trial in range(300))
            with open(SDCIT_RESULT_DIR + '/sdcit_postnonlinear.csv', 'a') as f:
                for out in outs:
                    print(*out, sep=',', file=f, flush=True)
    else:
        print('skipping SDCIT on post nonlinear noise data')


if __name__ == '__main__':
    main()
