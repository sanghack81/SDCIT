from os.path import exists

from joblib import Parallel
from joblib import delayed
from tqdm import tqdm, trange

from UAI_2017_SDCIT_experiments.exp_setup import SDCIT_RESULT_DIR
from UAI_2017_SDCIT_experiments.testing_utils import read_chaotic, read_postnonlinear_noise, chaotic_configs, postnonlinear_noise_configs
from sdcit.kcit import python_kcit_K
from sdcit.utils import *


def test_chaotic(independent, gamma, trial, N):
    np.random.seed(trial)
    KX, KY, KZ, _ = read_chaotic(independent, gamma, trial, N)
    pval = python_kcit_K(KX, KY, KZ, seed=trial, with_gp=True)[2]
    return independent, gamma, trial, N, pval


def test_postnonlinear(independent, noise, trial, N):
    np.random.seed(trial)
    KX, KY, KZ, _ = read_postnonlinear_noise(independent, noise, trial, N)
    pval = python_kcit_K(KX, KY, KZ, seed=trial, with_gp=True)[2]
    return independent, noise, trial, N, pval


def main():
    with Parallel(5) as parallel:
        if not exists(SDCIT_RESULT_DIR + '/kcit5_chaotic.csv'):
            for independent, N, gamma in tqdm(chaotic_configs()):
                outs = parallel(delayed(test_chaotic)(independent, gamma, trial, N) for trial in trange(300))
                with open(SDCIT_RESULT_DIR + '/kcit5_chaotic.csv', 'a') as f:
                    for out in outs:
                        print(*out, sep=',', file=f, flush=True)
        else:
            print('skipping KCIT5 on chaotic time series data')

        if not exists(SDCIT_RESULT_DIR + '/kcit5_postnonlinear.csv'):
            for noise, independent, N in tqdm(postnonlinear_noise_configs()):
                outs = parallel(delayed(test_postnonlinear)(independent, noise, trial, N) for trial in trange(300))
                with open(SDCIT_RESULT_DIR + '/kcit5_postnonlinear.csv', 'a') as f:
                    for out in outs:
                        print(*out, sep=',', file=f, flush=True)
        else:
            print('skipping KCIT5 on post nonlinear noise data')


if __name__ == '__main__':
    main()
