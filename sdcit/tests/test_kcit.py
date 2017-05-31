import multiprocessing
import os

import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

from sdcit.kcit import python_kcit, kcit, python_kcit_K
from sdcit.tests.synthetic import henon
from sdcit.tests.t_utils import aupc
from sdcit.utils import random_seeds, rbf_kernel_with_median_heuristic, columnwise_normalizes


def para(trial, gamma, normalize):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    Sta, Cri, p_val, Cri_appr, p_appr = python_kcit(X, Y, Z, normalize=normalize)
    return p_val


def test_kcit_henon():
    """
    True 0.0 0.488643
    True 0.2 0.604434
    True 0.4 0.999994
    
    False 0.0 0.497717
    False 0.2 0.932756
    False 0.4 0.999934
    """
    n_trial = 200
    with Parallel(multiprocessing.cpu_count() // 2) as parallel:
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
    KX, KY, KZ = rbf_kernel_with_median_heuristic(X, Y, Z)
    Sta, Cri, p_val, Cri_appr, p_appr = python_kcit_K(KX, KY, KZ, eq_17_as_is=eq_17_as_is)
    return p_val


def test_kcit_henon_K():
    """
    True True 0.0 1.0
    True True 0.2 1.0
    True True 0.4 1.0

    True False 0.0 1.0
    True False 0.2 1.0
    True False 0.4 1.0

    False True 0.0 0.59243
    False True 0.2 0.810601
    False True 0.4 0.966841

    False False 0.0 0.565238
    False False 0.2 0.915659
    False False 0.4 0.997314
    """
    n_trial = 200
    with Parallel(multiprocessing.cpu_count() // 2) as parallel:
        for eq_17_as_is in [True, False]:
            for normalize in [True, False]:
                for gamma in [0.0, 0.2, 0.4]:
                    ps = parallel(delayed(para_K)(trial, gamma, normalize, eq_17_as_is) for trial in trange(n_trial))
                    aupc_gamma = aupc(ps)
                    print(eq_17_as_is, normalize, gamma, aupc_gamma)


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
        ps = [kcit(*henon(trial, 200, gamma, 0), seed=trial, mateng=mateng)[2] for trial in trange(n_trial)]
        aupc_gamma = aupc(ps)
        print(gamma, aupc_gamma)

    mateng.quit()


if __name__ == '__main__':
    test_matlab_kcit_henon()
    print()
    test_kcit_henon()
    print()
    test_kcit_henon_K()
