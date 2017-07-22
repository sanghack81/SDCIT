from os.path import exists

from joblib import Parallel
from joblib import delayed
from tqdm import tqdm, trange

from experiments.exp_setup import SDCIT_RESULT_DIR
from experiments.exp_utils import chaotic_configs, postnonlinear_noise_configs, read_chaotic_data, read_postnonlinear_noise_data, read_postnonlinear_noise, read_chaotic
from sdcit.hybrid import hybrid
from sdcit.utils import *


def test_chaotic(independent, gamma, trial, N):
    np.random.seed(trial)
    Kx, Ky, Kz, Dz = read_chaotic(independent, gamma, trial, N)
    X, Y, Z = read_chaotic_data(independent, gamma, trial, N)
    pval = hybrid(X, Y, Z, Kx, Ky, Kz, Dz)
    return independent, gamma, trial, N, pval


def test_postnonlinear(independent, noise, trial, N):
    np.random.seed(trial)
    Kx, Ky, Kz, Dz = read_postnonlinear_noise(independent, noise, trial, N)
    X, Y, Z = read_postnonlinear_noise_data(independent, noise, trial, N)
    pval = hybrid(X, Y, Z, Kx, Ky, Kz, Dz)
    return independent, noise, trial, N, pval


def main():
    with Parallel(20) as parallel:
        if not exists(SDCIT_RESULT_DIR + '/hybrid_chaotic.csv'):
            for independent, N, gamma in tqdm(chaotic_configs()):
                outs = parallel(delayed(test_chaotic)(independent, gamma, trial, N) for trial in trange(300))
                with open(SDCIT_RESULT_DIR + '/hybrid_chaotic.csv', 'a') as f:
                    for out in outs:
                        print(*out, sep=',', file=f, flush=True)
        else:
            print('skipping hybrid on chaotic time series data')

        if not exists(SDCIT_RESULT_DIR + '/hybrid_postnonlinear.csv'):
            for noise, independent, N in tqdm(postnonlinear_noise_configs()):
                outs = parallel(delayed(test_postnonlinear)(independent, noise, trial, N) for trial in trange(300))
                with open(SDCIT_RESULT_DIR + '/hybrid_postnonlinear.csv', 'a') as f:
                    for out in outs:
                        print(*out, sep=',', file=f, flush=True)
        else:
            print('skipping hybrid on post nonlinear noise data')


if __name__ == '__main__':
    main()
