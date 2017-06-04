from os.path import exists

from joblib import Parallel
from joblib import delayed
from tqdm import tqdm

from UAI_2017_SDCIT_experiments.exp_setup import SDCIT_RESULT_DIR, PARALLEL_JOBS
from UAI_2017_SDCIT_experiments.testing_utils import chaotic_configs, postnonlinear_noise_configs, read_postnonlinear_noise_data, read_chaotic_data
from sdcit.flaxman import FCIT
from sdcit.utils import *


def test_chaotic(independent, gamma, trial, N):
    np.random.seed(trial)
    pval = FCIT(*read_chaotic_data(independent, gamma, trial, N), seed=trial)
    return independent, gamma, trial, N, pval


def test_postnonlinear(independent, noise, trial, N):
    np.random.seed(trial)
    pval = FCIT(*read_postnonlinear_noise_data(independent, noise, trial, N), seed=trial)
    return independent, noise, trial, N, pval


def main():
    with Parallel(PARALLEL_JOBS) as parallel:
        if not exists(SDCIT_RESULT_DIR + '/fcit2_chaotic.csv'):
            for independent, N, gamma in tqdm(chaotic_configs()):
                outs = parallel(delayed(test_chaotic)(independent, gamma, trial, N) for trial in range(300))
                with open(SDCIT_RESULT_DIR + '/fcit2_chaotic.csv', 'a') as f:
                    for out in outs:
                        print(*out, sep=',', file=f, flush=True)
        else:
            print('skipping FCIT on chaotic time series data')

        if not exists(SDCIT_RESULT_DIR + '/fcit2_postnonlinear.csv'):
            for noise, independent, N in tqdm(postnonlinear_noise_configs()):
                outs = parallel(delayed(test_postnonlinear)(independent, noise, trial, N) for trial in range(300))
                with open(SDCIT_RESULT_DIR + '/fcit2_postnonlinear.csv', 'a') as f:
                    for out in outs:
                        print(*out, sep=',', file=f, flush=True)
        else:
            print('skipping FCIT on post nonlinear noise data')


if __name__ == '__main__':
    main()
