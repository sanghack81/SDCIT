import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

from sdcit.flaxman import FCIT, FCIT_K
from sdcit.tests.synthetic import henon
from sdcit.tests.t_utils import aupc
from sdcit.utils import rbf_kernel_with_median_heuristic


def para(trial, gamma):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    return FCIT(X, Y, Z)


def test_flaxman_henon():
    n_trial = 200
    for gamma in [0.0, 0.3, 0.5]:
        ps = Parallel(-1)(delayed(para)(trial, gamma) for trial in trange(n_trial))
        aupc_gamma = aupc(ps)
        if gamma == 0.0:
            assert 0.4 <= aupc_gamma <= 0.55  # 0.4344
        if gamma == 0.3:
            assert 0.95 <= aupc_gamma  # 0.9991
        if gamma == 0.5:
            assert 0.95 <= aupc_gamma  # 0.9974
        print(gamma, aupc_gamma)


def para_K(trial, gamma):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    return FCIT_K(*rbf_kernel_with_median_heuristic(X, Y, Z))


def test_flaxman_henon_K():
    n_trial = 200
    for gamma in [0.0, 0.3, 0.5]:
        ps = Parallel(-1)(delayed(para_K)(trial, gamma) for trial in trange(n_trial))
        aupc_gamma = aupc(ps)
        if gamma == 0.0:
            assert 0.45 <= aupc_gamma <= 0.6  # 0.584385
        if gamma == 0.3:
            assert 0.9 <= aupc_gamma  # 0.93467
        if gamma == 0.5:
            assert 0.95 <= aupc_gamma  # 0.998145
        print(gamma, aupc_gamma)


if __name__ == '__main__':
    test_flaxman_henon()
    print()
    test_flaxman_henon_K()
