import numpy as np
import scipy.stats


def centering(M):
    nr, nc = M.shape
    assert nr == nc
    n = nr
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ M @ H


# # based on matlab code by Arthur Gretton (hsicTestGamma.m, 03/06/07):
# # http://www.gatsby.ucl.ac.uk/~gretton/indepTestFiles/indep.htm
def hsicTestGamma(K: np.ndarray, L: np.ndarray, Kcentered=False):
    print('not tested yet!')
    m = len(K)

    if not Kcentered:
        Kc = centering(K)
    Lc = centering(L)

    testStat = 1 / m * np.sum(Kc.T * Lc)

    bone = np.ones((m, 1))
    varHSIC = (1 / 6 * Kc * Lc) ** 2

    varHSIC = 1 / m / (m - 1) * (np.sum(varHSIC) - np.sum(np.diag(varHSIC)))
    varHSIC = 72 * (m - 4) * (m - 5) / m / (m - 1) / (m - 2) / (m - 3) * varHSIC
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)

    muX = 1 / m / (m - 1) * bone.T @ (K @ bone)
    muY = 1 / m / (m - 1) * bone.T @ (L @ bone)
    mHSIC = 1 / m * (1 + muX @ muY - muX - muY)

    al = mHSIC ** 2 / varHSIC
    bet = varHSIC * m / mHSIC

    return scipy.stats.gamma.pdf(testStat, al, scale=bet)


def compute_residual(X, K, sigma=1.0):
    return (np.linalg.inv(np.eye(len(K)) + 1 / (sigma ** 2) * K) @ X).squeeze()




# def check_cond_indep(X, Y, Z, KX, KY, KZ, sigma=1):
#     result1 = residual(Z, Y, sigma=sigma)
#     result2 = residual(Z, X, sigma=sigma)
#     return hsicTestGamma.wrapper(result1, result2)
