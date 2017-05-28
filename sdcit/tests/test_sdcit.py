import numpy as np
from joblib import Parallel, delayed

from sdcit.sdcit import rbf_kernel_with_median_heuristic
from sdcit.sdcit2 import c_SDCIT2
from sdcit.tests.synthetic import henon
from sdcit.tests.t_utils import aupc


def para(trial, gamma):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)

    return c_SDCIT2(*rbf_kernel_with_median_heuristic(X, Y, Z), size_of_null_sample=500)[1]


def test_sdcit_henon():
    n_trial = 200
    for gamma in [0.0, 0.3, 0.5]:
        ps = Parallel(4)(delayed(para)(trial, gamma) for trial in range(n_trial))
        aupc_gamma = aupc(ps)
        if gamma == 0.0:
            assert 0.45 <= aupc_gamma <= 0.55   # 0.4785
        if gamma == 0.3:
            assert 0.925 <= aupc_gamma  # 0.97403
        if gamma == 0.5:
            assert 0.95 <= aupc_gamma   # 0.98488
        print(gamma, aupc_gamma)


if __name__ == '__main__':
    test_sdcit_henon()
