import multiprocessing
import os
import time

import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

from UAI_2017_SDCIT_experiments.testing_utils import read_chaotic_data
from sdcit.kcit import python_kcit, kcit, python_kcit_K
from sdcit.tests.synthetic import henon
from sdcit.tests.t_utils import aupc
from sdcit.utils import random_seeds, rbf_kernel_with_median_heuristic


def test_kcit_debug():
    X, Y, Z = read_chaotic_data(0, 0.0, 0, 200, os.path.expanduser('~/Dropbox/research/2014 rcm/workspace/python/SDCIT/kcit/'))
    outs = python_kcit(X, Y, Z, normalize=False, with_gp=False, noise=0.001)
    return outs


def para(trial, gamma, normalize):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    Sta, Cri, p_val, Cri_appr, p_appr = python_kcit(X, Y, Z, normalize=normalize)
    return p_val


def test_kcit_henon():
    """
    test_kcit_henon True 0.0 0.485895
    test_kcit_henon True 0.2 0.644962

    test_kcit_henon False 0.0 0.503069
    test_kcit_henon False 0.2 0.967547

    """
    n_trial = 200
    with Parallel(multiprocessing.cpu_count() // 3) as parallel:
        for normalize in [True, False]:
            for gamma in [0.0, 0.2]:
                ps = parallel(delayed(para)(trial, gamma, normalize) for trial in range(n_trial))
                aupc_gamma = aupc(ps)
                print('test_kcit_henon', normalize, gamma, aupc_gamma)


def para_K(trial, gamma, sigma_squared):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    KX, KY, KZ = rbf_kernel_with_median_heuristic(X, Y, Z)
    Sta, Cri, p_val, Cri_appr, p_appr = python_kcit_K(KX, KY, KZ, with_gp=False, noise=sigma_squared)
    return p_val


def test_kcit_henon_K():
    """
    test_kcit_henon_K 0.0 1.0 0.392629
    test_kcit_henon_K 0.0 0.1 0.499575
    test_kcit_henon_K 0.0 0.01 0.549117
    test_kcit_henon_K 0.0 0.001 0.580488

    test_kcit_henon_K 0.2 1.0 0.99918
    test_kcit_henon_K 0.2 0.1 0.999053
    test_kcit_henon_K 0.2 0.01 0.99862
    test_kcit_henon_K 0.2 0.001 0.996396
    """
    n_trial = 200
    with Parallel(multiprocessing.cpu_count() // 3) as parallel:
        for gamma in [0.0, 0.2]:
            for sigma_squared in [1.0, 0.1, 0.01, 0.001]:
                ps = parallel(delayed(para_K)(trial, gamma, sigma_squared) for trial in range(n_trial))
                aupc_gamma = aupc(ps)
                print('test_kcit_henon_K', gamma, sigma_squared, aupc_gamma)


def para_K2(trial, gamma):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    KX, KY, KZ = rbf_kernel_with_median_heuristic(X, Y, Z)
    Sta, Cri, p_val, Cri_appr, p_appr = python_kcit_K(KX, KY, KZ, with_gp=True)
    return p_val


def test_kcit_henon_K2():
    """
    test_kcit_henon_K2 0.0 0.508071 31.19998335838318
    test_kcit_henon_K2 0.2 0.998337 30.68874216079712
    """
    n_trial = 200
    with Parallel(-1) as parallel:
        for gamma in [0.0, 0.2]:
            start = time.time()
            ps = parallel(delayed(para_K2)(trial, gamma) for trial in trange(n_trial))
            aupc_gamma = aupc(ps)
            print('test_kcit_henon_K2', gamma, aupc_gamma, time.time() - start)


def test_matlab_kcit_henon():
    """
    0.0 0.498593
    0.2 0.717397
    0.4 0.9175
    """
    if multiprocessing.cpu_count() != 8:
        return
    import matlab.engine
    mateng = matlab.engine.start_matlab()
    dir_at = os.path.expanduser('~/Dropbox/research/2014 rcm/workspace/python/SDCIT/kcit')
    mateng.addpath(mateng.genpath(dir_at))
    mateng.RandStream.setGlobalStream(mateng.RandStream('mcg16807', 'Seed', random_seeds()))

    n_trial = 200
    for gamma in [0.0, 0.2, 0.4]:
        ps = [kcit(*henon(trial, 200, gamma, 0), seed=trial, mateng=mateng)[2] for trial in range(n_trial)]
        aupc_gamma = aupc(ps)
        print(gamma, aupc_gamma)

    mateng.quit()


if __name__ == '__main__':
    # test_kcit_henon()
    # test_kcit_henon_K()
    test_kcit_henon_K2()
