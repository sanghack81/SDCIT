from sdcit.cython_impl.cy_sdcit import cy_kcipt

from sdcit.utils import *


def c_KCIPT(Kx, Ky, Kz, Dz, B, b, M, n_jobs=1, seed=None):
    """Python implementation of KCIPT by Doran et al. to test X _||_ Y | Z

    Parameters
    ----------
    Kx : np.ndarray
        An N by N kernel matrix of X
    Ky : np.ndarray
        An N by N kernel matrix of Y
    Kz : np.ndarray
        An N by N kernel matrix of Z
    Dz : np.ndarray
        An N by N pairwise distance matrix of Z
    B : int
        The number of outer bootstrap
    b : int
        The number of inner bootstrap
    M : int
        The number of Monte Carlo simulation
    n_jobs : int
        The number of threads to be used.
    seed : int
        Random seed to be used

    References
    ----------
    Doran, G., Muandet, K., Zhang, K., & Schölkopf, B. (2014). A Permutation-Based Kernel Conditional Independence Test.
    In Proceedings of the Thirtieth Conference on Uncertainty in Artificial Intelligence (pp. 132–141). Corvallis, Oregon: AUAI Press.
    """
    if seed is None:
        seed = random_seeds()

    Kx, Ky, Kz, Dz = cythonize(Kx, Ky, Kz, Dz)
    inner_null = np.zeros((B, b), dtype='float64')
    outer_null = np.zeros((M,), dtype='float64')
    mmds = np.zeros((B,), dtype='float64')

    cy_kcipt(Kx, Ky, Kz, Dz, B, b, inner_null, mmds, seed, n_jobs, outer_null, M)

    # null correction for test statistic based on a set of unbiased estimates of squared MMD
    if M > 0:
        outer_null -= outer_null.mean()

    test_statistic = mmds.mean()

    return p_value_of(test_statistic, outer_null) if M > 0 else float('nan'), mmds, inner_null, outer_null
