import numpy as np
import scipy.stats
from typing import List, Tuple

from sdcit.cython_impl.cy_sdcit import cy_hsic
from sdcit.utils import p_value_of, cythonize, random_seeds, centering


def HSIC(K: np.ndarray, L: np.ndarray, p_val_method='bootstrap', num_boot=1000) -> float:
    if p_val_method == 'bootstrap':
        return HSIC_boot(K, L, num_boot)
    elif p_val_method == 'gamma':
        return HSIC_gamma_approx(K, L)
    else:
        raise ValueError('unknown p value computation method: {}'.format(p_val_method))


def sum_except_diag(M: np.ndarray):
    return M.sum() - M.trace()


def HSIC_gamma_approx(K: np.ndarray, L: np.ndarray) -> float:
    """Hilbert-Schmidt Independence Criterion where null distribution is based on approximated Gamma distribution

    References
    ----------
    Gretton, A., Herbrich, R., Smola, A., Bousquet, O., & Schölkopf, B. (2005). Kernel Methods for Measuring Independence. Journal of Machine Learning Research, 6, 2075–2129.
    """
    Kc, Lc = centering(K), centering(L)
    m = len(K)

    test_stat = 1 / m * np.sum(Kc * Lc)
    muX = 1 / m / (m - 1) * sum_except_diag(K)
    muY = 1 / m / (m - 1) * sum_except_diag(L)
    mHSIC = 1 / m * (1 + muX * muY - muX - muY)
    varHSIC = 72 * (m - 4) * (m - 5) / m / (m - 1) / (m - 2) / (m - 3) * (1 / m / (m - 1) * (sum_except_diag((1 / 6 * Kc * Lc) ** 2)))

    al = mHSIC ** 2 / varHSIC
    bet = varHSIC * m / mHSIC
    return scipy.stats.gamma.sf(test_stat, al, scale=bet)


def HSIC_stat(K: np.ndarray, L: np.ndarray) -> float:
    """HSIC statistic assuming given two centered kernel matrices.

    References
    ----------
    Gretton, A., Herbrich, R., Smola, A., Bousquet, O., & Schölkopf, B. (2005). Kernel Methods for Measuring Independence. Journal of Machine Learning Research, 6, 2075–2129.
    """
    m = len(K)
    return float(1 / m * np.sum(K * L))


def HSIC_boot(K: np.ndarray, L: np.ndarray, num_boot=1000, seed=None) -> Tuple[float, List[float]]:
    """A Hilbert-Schmidt Independence Criterion where null distribution is based on bootstrapping

    References
    ----------
    Gretton, A., Herbrich, R., Smola, A., Bousquet, O., & Schölkopf, B. (2005). Kernel Methods for Measuring Independence. Journal of Machine Learning Research, 6, 2075–2129.
    """
    if seed is not None:
        np.random.seed(seed)

    Kc, Lc = centering(K), centering(L)

    test_statistics = HSIC_stat(Kc, Lc)

    def shuffled():
        perm = np.random.permutation(len(K))
        return Lc[np.ix_(perm, perm)]

    null_distribution = [HSIC_stat(Kc, shuffled()) for _ in range(num_boot)]

    return p_value_of(test_statistics, null_distribution)


def c_HSIC(K: np.ndarray, L: np.ndarray, size_of_null_sample=1000, with_null=False, seed=None, n_jobs=1):
    if seed is not None:
        np.random.seed(seed)

    K, L = centering(K), centering(L)
    K, L = cythonize(K, L)
    raw_null = np.zeros((size_of_null_sample,), dtype='float64')
    test_statistic = np.zeros((1,), dtype='float64')

    # run SDCIT
    cy_hsic(K, L, size_of_null_sample, random_seeds(), n_jobs, test_statistic, raw_null)

    # post-process outputs
    test_statistic = test_statistic[0]

    if with_null:
        return test_statistic, p_value_of(test_statistic, raw_null), raw_null
    else:
        return test_statistic, p_value_of(test_statistic, raw_null)
