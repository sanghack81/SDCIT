import numpy as np
import scipy.stats

from sdcit.kcit import centering
from sdcit.utils import p_value_of


def HSIC(K: np.ndarray, L: np.ndarray, p_val_method='bootstrap', num_boot=1000):
    if p_val_method == 'bootstrap':
        return HSIC_boot(K, L, num_boot)
    elif p_val_method == 'gamma':
        return HSIC_gamma_approx(K, L)
    else:
        raise ValueError('unknown p value computation method: {}'.format(p_val_method))


def sum_except_diag(M):
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


def HSIC_stat(K, L):
    """HSIC statistic assuming given two centered kernel matrices.

    References
    ----------
    Gretton, A., Herbrich, R., Smola, A., Bousquet, O., & Schölkopf, B. (2005). Kernel Methods for Measuring Independence. Journal of Machine Learning Research, 6, 2075–2129.
    """
    m = len(K)
    return 1 / m * np.sum(K * L)


def HSIC_boot(K: np.ndarray, L: np.ndarray, num_boot=1000, seed=None) -> float:
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
