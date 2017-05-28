import os

import numpy as np
from joblib import Parallel, delayed

from sdcit.kcit import python_kcit, kcit
from sdcit.tests.synthetic import henon
from sdcit.tests.t_utils import aupc
from sdcit.utils import random_seeds


def para(trial, gamma):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    Sta, Cri, p_val, Cri_appr, p_appr = python_kcit(X, Y, Z, with_gp=True)
    return p_val


def test_kcit_henon():
    n_trial = 200
    for gamma in [0.0, 0.3, 0.5]:
        ps = Parallel(-1)(delayed(para)(trial, gamma) for trial in range(n_trial))
        aupc_gamma = aupc(ps)
        if gamma == 0.0:
            assert 0.45 <= aupc_gamma <= 0.55  # 0.468147   # linux 0.471639
        if gamma == 0.3:
            assert 0.85 <= aupc_gamma <= 0.925  # 0.924583  # linux 0.913242
        if gamma == 0.5:
            assert 0.85 <= aupc_gamma  # 1.0    # linux 1.0
        print(gamma, aupc_gamma)


def test_matlab_kcit_henon():
    import matlab.engine
    mateng = matlab.engine.start_matlab()
    dir_at = os.path.expanduser('~/Dropbox/research/2014 rcm/workspace/python/SDCIT/kcit')
    mateng.addpath(mateng.genpath(dir_at))
    mateng.RandStream.setGlobalStream(mateng.RandStream('mcg16807', 'Seed', random_seeds()))

    n_trial = 200
    for gamma in [0.0, 0.3, 0.5]:
        ps = [kcit(*henon(trial, 200, gamma, 0), seed=trial, mateng=mateng)[2] for trial in range(n_trial)]
        aupc_gamma = aupc(ps)
        if gamma == 0.0:
            assert 0.45 <= aupc_gamma <= 0.55  # 0.498593
        if gamma == 0.3:
            assert 0.85 <= aupc_gamma <= 0.925  # 0.885768
        if gamma == 0.5:
            assert 0.85 <= aupc_gamma  # 0.884607
        print(gamma, aupc_gamma)

    mateng.quit()


if __name__ == '__main__':
    # test_matlab_kcit_henon()
    # print()
    test_kcit_henon()
