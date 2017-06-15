import numpy as np
from joblib import Parallel, delayed
from tqdm import trange

from sdcit.sdcit import c_SDCIT
from sdcit.tests.synthetic_data import henon
from sdcit.tests.t_utils import AUPC
from sdcit.utils import rbf_kernel_median, columnwise_normalizes


def para(trial, gamma, normalize):
    np.random.seed(trial)
    X, Y, Z = henon(trial, 200, gamma, 0)
    if normalize:
        X, Y, Z = columnwise_normalizes(X, Y, Z)
    return c_SDCIT(*rbf_kernel_median(X, Y, Z), size_of_null_sample=500)[1]


def test_sdcit_henon():
    n_trial = 200
    with Parallel(-1) as parallel:
        for normalize in [True, False]:
            for gamma in [0.0, 0.2, 0.4]:
                ps = parallel(delayed(para)(trial, gamma, normalize) for trial in trange(n_trial))
                aupc_gamma = AUPC(ps)
                print(normalize, gamma, aupc_gamma)


if __name__ == '__main__':
    test_sdcit_henon()
