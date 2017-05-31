import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

from sdcit.sdcit import rbf_kernel_with_median_heuristic, columnwise_normalizes
from sdcit.sdcit2 import c_SDCIT2
from sdcit.tests.synthetic import henon
from sdcit.tests.t_utils import aupc


def para(trial, gamma, normalize):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    if normalize:
        X, Y, Z = columnwise_normalizes(X, Y, Z)
    return c_SDCIT2(*rbf_kernel_with_median_heuristic(X, Y, Z), size_of_null_sample=500)[1]


def test_sdcit_henon():
    """
    True 0.0 0.45497
    True 0.2 0.60488
    True 0.4 0.72499

    False 0.0 0.46003
    False 0.2 0.88871
    False 0.4 0.98883
    """
    n_trial = 200
    with Parallel(-1) as parallel:
        for normalize in [True, False]:
            for gamma in [0.0, 0.2, 0.4]:
                ps = parallel(delayed(para)(trial, gamma, normalize) for trial in trange(n_trial))
                aupc_gamma = aupc(ps)
                print(normalize, gamma, aupc_gamma)


if __name__ == '__main__':
    test_sdcit_henon()
