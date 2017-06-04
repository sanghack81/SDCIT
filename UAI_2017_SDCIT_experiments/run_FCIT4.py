from os.path import exists

from joblib import Parallel
from joblib import delayed
from tqdm import tqdm, trange

from UAI_2017_SDCIT_experiments.exp_setup import SDCIT_RESULT_DIR
from UAI_2017_SDCIT_experiments.testing_utils import chaotic_configs, postnonlinear_noise_configs, read_chaotic, read_postnonlinear_noise
from sdcit.flaxman import FCIT_K
from sdcit.utils import *


def test_chaotic(independent, gamma, trial, N, sigma_squared):
    np.random.seed(trial)
    KX, KY, KZ, _ = read_chaotic(independent, gamma, trial, N)
    pval = FCIT_K(KX, KY, KZ, seed=trial, with_gp=False, eq_17_as_is=False, sigma_squared=sigma_squared)
    return independent, gamma, trial, N, pval


def test_postnonlinear(independent, noise, trial, N, sigma_squared):
    np.random.seed(trial)
    KX, KY, KZ, _ = read_postnonlinear_noise(independent, noise, trial, N)
    pval = FCIT_K(KX, KY, KZ, seed=trial, with_gp=False, eq_17_as_is=False, sigma_squared=sigma_squared)
    return independent, noise, trial, N, pval


def main():
    with Parallel(4) as parallel:
        if not exists(SDCIT_RESULT_DIR + '/fcit4_chaotic.csv'):
            for independent, N, gamma in tqdm(chaotic_configs()):
                for sigma_squared in [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
                    outs = parallel(delayed(test_chaotic)(independent, gamma, trial, N, sigma_squared) for trial in trange(300))
                    with open(SDCIT_RESULT_DIR + '/fcit4_chaotic.csv', 'a') as f:
                        for out in outs:
                            print(sigma_squared, *out, sep=',', file=f, flush=True)
        else:
            print('skipping FCIT on chaotic time series data')

        if not exists(SDCIT_RESULT_DIR + '/fcit4_postnonlinear.csv'):
            for noise, independent, N in tqdm(postnonlinear_noise_configs()):
                for sigma_squared in [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
                    outs = parallel(delayed(test_postnonlinear)(independent, noise, trial, N, sigma_squared) for trial in trange(300))
                    with open(SDCIT_RESULT_DIR + '/fcit4_postnonlinear.csv', 'a') as f:
                        for out in outs:
                            print(sigma_squared, *out, sep=',', file=f, flush=True)
        else:
            print('skipping FCIT on post nonlinear noise data')


if __name__ == '__main__':
    main()
