from os.path import exists

from joblib import Parallel
from joblib import delayed
from tqdm import tqdm, trange

from UAI_2017_SDCIT_experiments.exp_setup import SDCIT_RESULT_DIR
from UAI_2017_SDCIT_experiments.testing_utils import chaotic_configs, read_chaotic_data, read_postnonlinear_noise_data
from sdcit.kcit import python_kcit
from sdcit.utils import *


def test_chaotic(independent, gamma, trial, N, sigma_squared):
    np.random.seed(trial)
    pval = python_kcit(*read_chaotic_data(independent, gamma, trial, N), seed=trial, with_gp=False, noise=sigma_squared)[2]
    return independent, gamma, trial, N, pval


def test_postnonlinear(independent, noise, trial, N, sigma_squared):
    np.random.seed(trial)
    pval = python_kcit(*read_postnonlinear_noise_data(independent, noise, trial, N), seed=trial, with_gp=False, noise=sigma_squared)[2]
    return independent, noise, trial, N, pval


def main():
    with Parallel(16) as parallel:
        if not exists(SDCIT_RESULT_DIR + '/kcit3_chaotic.csv'):
            for independent, N, gamma in tqdm(chaotic_configs()):
                for sigma_squared in [0.001]:
                    outs = parallel(delayed(test_chaotic)(independent, gamma, trial, N, sigma_squared) for trial in trange(300))
                    with open(SDCIT_RESULT_DIR + '/kcit3_chaotic.csv', 'a') as f:
                        for out in outs:
                            print(sigma_squared, *out, sep=',', file=f, flush=True)
        else:
            print('skipping KCIT3 on chaotic time series data')
            #
            # if not exists(SDCIT_RESULT_DIR + '/kcit3_postnonlinear.csv'):
            #     for noise, independent, N in tqdm(postnonlinear_noise_configs()):
            #         for sigma_squared in [1.0, 0.1, 0.01, 0.001]:
            #             outs = parallel(delayed(test_postnonlinear)(independent, noise, trial, N, sigma_squared) for trial in trange(300))
            #             with open(SDCIT_RESULT_DIR + '/kcit3_postnonlinear.csv', 'a') as f:
            #                 for out in outs:
            #                     print(sigma_squared, *out, sep=',', file=f, flush=True)
            # else:
            #     print('skipping KCIT3 on post nonlinear noise data')


if __name__ == '__main__':
    main()
