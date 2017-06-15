import multiprocessing
import os
import time

import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

from sdcit.kcit import python_kcit, matlab_kcit, python_kcit_K
from sdcit.tests.synthetic_data import henon
from sdcit.tests.t_utils import AUPC
from sdcit.utils import random_seeds, rbf_kernel_median


def para(trial, gamma, normalize):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    Sta, Cri, p_val, Cri_appr, p_appr = python_kcit(X, Y, Z, normalize=normalize)
    return p_val


def test_kcit_henon():
    n_trial = 200
    with Parallel(multiprocessing.cpu_count() // 3) as parallel:
        for normalize in [True, False]:
            for gamma in [0.0, 0.2]:
                ps = parallel(delayed(para)(trial, gamma, normalize) for trial in range(n_trial))
                aupc_gamma = AUPC(ps)
                print('test_kcit_henon', normalize, gamma, aupc_gamma)


def para_K(trial, gamma, sigma_squared):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    KX, KY, KZ = rbf_kernel_median(X, Y, Z)
    Sta, Cri, p_val, Cri_appr, p_appr = python_kcit_K(KX, KY, KZ, with_gp=False, sigma_squared=sigma_squared)
    return p_val


def test_kcit_henon_K():
    n_trial = 200
    with Parallel(multiprocessing.cpu_count() // 3) as parallel:
        for gamma in [0.0, 0.2]:
            for sigma_squared in [1.0, 0.1, 0.01, 0.001]:
                ps = parallel(delayed(para_K)(trial, gamma, sigma_squared) for trial in range(n_trial))
                aupc_gamma = AUPC(ps)
                print('test_kcit_henon_K', gamma, sigma_squared, aupc_gamma)


def para_K2(trial, gamma):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    KX, KY, KZ = rbf_kernel_median(X, Y, Z)
    Sta, Cri, p_val, Cri_appr, p_appr = python_kcit_K(KX, KY, KZ, with_gp=True)
    return p_val


def test_kcit_henon_K2():
    n_trial = 200
    with Parallel(-1) as parallel:
        for gamma in [0.0, 0.2]:
            start = time.time()
            ps = parallel(delayed(para_K2)(trial, gamma) for trial in trange(n_trial))
            aupc_gamma = AUPC(ps)
            print('test_kcit_henon_K2', gamma, aupc_gamma, time.time() - start)


def test_matlab_kcit_henon():
    import matlab.engine
    mateng = matlab.engine.start_matlab()
    dir_at = os.path.expanduser('~/Dropbox/research/2014 rcm/workspace/python/SDCIT/kcit')
    mateng.addpath(mateng.genpath(dir_at))
    mateng.RandStream.setGlobalStream(mateng.RandStream('mcg16807', 'Seed', random_seeds()))

    n_trial = 200
    for gamma in [0.0, 0.1, 0.2]:
        ps = [matlab_kcit(*henon(trial, 200, gamma, 0), seed=trial, mateng=mateng)[2] for trial in range(n_trial)]
        aupc_gamma = AUPC(ps)
        print(gamma, aupc_gamma)

    mateng.quit()
