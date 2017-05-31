import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

from sdcit.flaxman import FCIT, FCIT_K
from sdcit.tests.synthetic import henon
from sdcit.tests.t_utils import aupc
from sdcit.utils import rbf_kernel_with_median_heuristic, columnwise_normalizes


def para(trial, gamma, normalize=False):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    return FCIT(X, Y, Z, normalize=normalize)


def test_flaxman_henon():
    """
    True 0.0 0.432265
    True 0.2 0.99849
    True 0.4 0.99712

    False 0.0 0.47327
    False 0.2 0.95744
    False 0.4 0.9972
    """
    n_jobs = multiprocessing.cpu_count() // 2
    n_trial = 200
    with Parallel(n_jobs) as parallel:
        for normalize in [True, False]:
            for gamma in [0.0, 0.2, 0.4]:
                ps = parallel(delayed(para)(trial, gamma, normalize) for trial in trange(n_trial))
                aupc_gamma = aupc(ps)
                print(normalize, gamma, aupc_gamma)


def para_K(trial, gamma, normalize, eq_17_as_is):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    if normalize:
        X, Y, Z = columnwise_normalizes(X, Y, Z)
    return FCIT_K(*rbf_kernel_with_median_heuristic(X, Y, Z), eq_17_as_is=eq_17_as_is)


def test_flaxman_henon_K():
    """
    True True 0.0 1.0
    True True 0.2 1.0
    True True 0.4 1.0

    True False 0.0 1.0
    True False 0.2 1.0
    True False 0.4 1.0

    False True 0.0 0.59571
    False True 0.2 0.729125
    False True 0.4 0.923015

    False False 0.0 0.58316
    False False 0.2 0.86386
    False False 0.4 0.99079
    """
    n_jobs = multiprocessing.cpu_count() // 2
    n_trial = 200
    with Parallel(n_jobs) as parallel:
        for eq_17_as_is in [True, False]:
            for normalize in [True, False]:
                for gamma in [0.0, 0.2, 0.4]:
                    ps = parallel(delayed(para_K)(trial, gamma, normalize, eq_17_as_is) for trial in trange(n_trial))
                    aupc_gamma = aupc(ps)
                    print(eq_17_as_is, normalize, gamma, aupc_gamma)


if __name__ == '__main__':
    test_flaxman_henon()
    print()
    test_flaxman_henon_K()
