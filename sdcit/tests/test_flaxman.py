import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

from sdcit.flaxman import FCIT, FCIT_K
from sdcit.tests.synthetic import henon
from sdcit.tests.t_utils import aupc
from sdcit.utils import rbf_kernel_with_median_heuristic


def para(trial, gamma, normalize=False):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    return FCIT(X, Y, Z, normalize=normalize)


def test_flaxman_henon():
    """
    test_flaxman_henon True 0.0 0.448325
    test_flaxman_henon True 0.2 0.9987
    test_flaxman_henon True 0.4 0.997665

    test_flaxman_henon False 0.0 0.48988
    test_flaxman_henon False 0.2 0.965075
    test_flaxman_henon False 0.4 0.997455
    """
    n_jobs = multiprocessing.cpu_count() // 3
    n_trial = 200
    with Parallel(n_jobs) as parallel:
        for normalize in [True, False]:
            for gamma in [0.0, 0.2, 0.4]:
                ps = parallel(delayed(para)(trial, gamma, normalize) for trial in range(n_trial))
                aupc_gamma = aupc(ps)
                print('test_flaxman_henon', normalize, gamma, aupc_gamma)


def para_K(trial, gamma, eq_17_as_is, sigma_squared):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    return FCIT_K(*rbf_kernel_with_median_heuristic(X, Y, Z), eq_17_as_is=eq_17_as_is, with_gp=False, sigma_squared=sigma_squared)


def test_flaxman_henon_K():
    """
    test_flaxman_henon_K True 0.0 1.0 0.713635
    test_flaxman_henon_K True 0.0 0.1 0.46006
    test_flaxman_henon_K True 0.0 0.01 0.51714
    test_flaxman_henon_K True 0.0 0.001 0.54742

    test_flaxman_henon_K True 0.2 1.0 0.99924
    test_flaxman_henon_K True 0.2 0.1 0.997465
    test_flaxman_henon_K True 0.2 0.01 0.998165
    test_flaxman_henon_K True 0.2 0.001 0.99762

    test_flaxman_henon_K True 0.4 1.0 1.0
    test_flaxman_henon_K True 0.4 0.1 1.0
    test_flaxman_henon_K True 0.4 0.01 1.0
    test_flaxman_henon_K True 0.4 0.001 1.0


    test_flaxman_henon_K False 0.0 1.0 0.43718
    test_flaxman_henon_K False 0.0 0.1 0.493685
    test_flaxman_henon_K False 0.0 0.01 0.523725
    test_flaxman_henon_K False 0.0 0.001 0.548395

    test_flaxman_henon_K False 0.2 1.0 0.99509
    test_flaxman_henon_K False 0.2 0.1 0.99768
    test_flaxman_henon_K False 0.2 0.01 0.99821
    test_flaxman_henon_K False 0.2 0.001 0.99764

    test_flaxman_henon_K False 0.4 1.0 1.0
    test_flaxman_henon_K False 0.4 0.1 1.0
    test_flaxman_henon_K False 0.4 0.01 1.0
    test_flaxman_henon_K False 0.4 0.001 1.0
    """
    n_jobs = multiprocessing.cpu_count() // 3
    n_trial = 200
    with Parallel(n_jobs) as parallel:
        for eq_17_as_is in [True, False]:
            for gamma in [0.0, 0.2, 0.4]:
                for sigma_squared in [1.0, 0.1, 0.01, 0.001]:
                    ps = parallel(delayed(para_K)(trial, gamma, eq_17_as_is, sigma_squared) for trial in range(n_trial))
                    aupc_gamma = aupc(ps)
                    print('test_flaxman_henon_K', eq_17_as_is, gamma, sigma_squared, aupc_gamma)


def para_K2(trial, gamma, eq_17_as_is):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    return FCIT_K(*rbf_kernel_with_median_heuristic(X, Y, Z), eq_17_as_is=eq_17_as_is, with_gp=True)


def test_flaxman_henon_K2():
    """
    test_flaxman_henon_K2 True 0.0 0.4848   / ARD 0.4953
    test_flaxman_henon_K2 True 0.2 0.996865 / ARD 0.996495
    test_flaxman_henon_K2 True 0.4 1.0      / ARD 1.0

    test_flaxman_henon_K2 False 0.0 0.48488 /
    test_flaxman_henon_K2 False 0.2 0.996865/
    test_flaxman_henon_K2 False 0.4 1.0     /
    """
    n_jobs = multiprocessing.cpu_count() // 2
    n_trial = 200
    with Parallel(n_jobs) as parallel:
        for eq_17_as_is in [True, False]:
            for gamma in [0.0, 0.2, 0.4]:
                ps = parallel(delayed(para_K2)(trial, gamma, eq_17_as_is) for trial in trange(n_trial))
                aupc_gamma = aupc(ps)
                print('test_flaxman_henon_K2', eq_17_as_is, gamma, aupc_gamma)


if __name__ == '__main__':
    # test_flaxman_henon_K()
    test_flaxman_henon_K2()
