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


def HSIC_gamma_approx(K: np.ndarray, L: np.ndarray) -> float:
    """Hilbert-Schmidt Independence Criterion

    based on matlab code by Arthur Gretton (hsicTestGamma.m, 03/06/07):
    http://www.gatsby.ucl.ac.uk/~gretton/indepTestFiles/indep.htm
    """
    K, L = K.copy(), L.copy()
    Kc, Lc = centering(K), centering(L)
    m = len(K)

    testStat = 1 / m * np.sum(Kc * Lc)

    bone = np.ones((m, 1))
    varHSIC = (1 / 6 * Kc * Lc) ** 2

    # varHSIC = 1 / m / (m - 1) * (varHSIC.sum() - np.sum(np.diag(varHSIC)))
    varHSIC = 1 / m / (m - 1) * (varHSIC.sum() - varHSIC.trace())
    varHSIC = 72 * (m - 4) * (m - 5) / m / (m - 1) / (m - 2) / (m - 3) * varHSIC
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)

    muX = 1 / m / (m - 1) * (bone.T @ (K @ bone))
    muY = 1 / m / (m - 1) * (bone.T @ (L @ bone))
    mHSIC = 1 / m * (1 + muX @ muY - muX - muY)[0, 0]

    al = mHSIC ** 2 / varHSIC
    bet = varHSIC * m / mHSIC

    return scipy.stats.gamma.sf(testStat, al, scale=bet)


def HSIC_stat(K, L):
    """HSIC statistic assuming given two centered kernel matrices."""
    m = len(K)
    return 1 / m * np.sum(K * L)


def HSIC_boot(K: np.ndarray, L: np.ndarray, num_boot=1000) -> float:
    """A Hilbert-Schmidt Independence Criterion"""
    Kc, Lc = centering(K), centering(L)

    stat = HSIC_stat(Kc, Lc)

    idx = np.arange(len(K))
    boot = np.zeros((num_boot,))
    for i in range(num_boot):
        np.random.shuffle(idx)
        boot[i] = HSIC_stat(Kc, Lc[np.ix_(idx, idx)])

    return p_value_of(stat, boot)
