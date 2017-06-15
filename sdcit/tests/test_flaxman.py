import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

from sdcit.flaxman import FCIT, FCIT_K
from sdcit.tests.synthetic_data import henon
from sdcit.tests.t_utils import AUPC
from sdcit.utils import rbf_kernel_median


def para(trial, gamma, normalize=False):
    np.random.seed(trial)
    return FCIT(*henon(trial, 200, gamma, 0), normalize=normalize)


def test_flaxman_henon():
    n_jobs = multiprocessing.cpu_count() // 3
    n_trial = 200
    with Parallel(n_jobs) as parallel:
        for gamma in [0.0, 0.1, 0.2]:
            ps = parallel(delayed(para)(trial, gamma) for trial in range(n_trial))
            aupc_gamma = AUPC(ps)
            print('test_flaxman_henon', gamma, aupc_gamma)


def para_K(trial, gamma, use_expectation, sigma_squared):
    np.random.seed(trial)
    return FCIT_K(*rbf_kernel_median(*henon(trial, 200, gamma, 0)), use_expectation=use_expectation, with_gp=False, sigma_squared=sigma_squared)


def test_flaxman_henon_K():
    n_jobs = multiprocessing.cpu_count() // 3
    n_trial = 200
    with Parallel(n_jobs) as parallel:
        for use_expectation in [True, False]:
            for gamma in [0.0, 0.1, 0.2]:
                for sigma_squared in [1.0, 0.1, 0.01, 0.001]:
                    ps = parallel(delayed(para_K)(trial, gamma, use_expectation, sigma_squared) for trial in range(n_trial))
                    aupc_gamma = AUPC(ps)
                    print('test_flaxman_henon_K', use_expectation, gamma, sigma_squared, aupc_gamma)


def para_K2(trial, gamma, use_expectation):
    np.random.seed(trial)
    return FCIT_K(*rbf_kernel_median(*henon(trial, 200, gamma, 0)), use_expectation=use_expectation, with_gp=True)


def test_flaxman_henon_K2():
    n_jobs = multiprocessing.cpu_count() // 2
    n_trial = 200
    with Parallel(n_jobs) as parallel:
        for use_expectation in [True, False]:
            for gamma in [0.0, 0.1, 0.2]:
                ps = parallel(delayed(para_K2)(trial, gamma, use_expectation) for trial in trange(n_trial))
                aupc_gamma = AUPC(ps)
                print('test_flaxman_henon_K2', use_expectation, gamma, aupc_gamma)
