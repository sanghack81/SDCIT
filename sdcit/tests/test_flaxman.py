import numpy as np
from joblib import Parallel, delayed

from sdcit.flaxman import check_cond_indep_real
from sdcit.tests.synthetic import henon
from sdcit.tests.t_utils import aupc


def para(trial, gamma):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    return check_cond_indep_real(X, Y, Z)


def test_flaxman_henon():
    n_trial = 200
    for gamma in [0.0, 0.3, 0.5]:
        ps = Parallel(-1)(delayed(para)(trial, gamma) for trial in range(n_trial))
        aupc_gamma = aupc(ps)
        if gamma == 0.0:
            assert 0.4 <= aupc_gamma <= 0.55  # 0.4344
        if gamma == 0.3:
            assert 0.95 <= aupc_gamma  # 0.9991
        if gamma == 0.5:
            assert 0.95 <= aupc_gamma  # 0.9974
        print(gamma, aupc_gamma)


if __name__ == '__main__':
    test_flaxman_henon()
