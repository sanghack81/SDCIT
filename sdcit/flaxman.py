import GPflow
import numpy as np
import scipy.stats
from numpy.linalg import inv
from sklearn.metrics import pairwise_distances

from sdcit.kcit import centering


def HSIC(K: np.ndarray, L: np.ndarray) -> float:
    """A Hilbert-Schmidt Independence Criterion
    :type K: np.ndarray
        Kernel matrix
    :type L: np.ndarray
        Kernel matrix
    """
    # based on matlab code by Arthur Gretton (hsicTestGamma.m, 03/06/07):
    # http://www.gatsby.ucl.ac.uk/~gretton/indepTestFiles/indep.htm
    K, L = K.copy(), L.copy()
    Kc, Lc = centering(K), centering(L)
    m = len(K)

    testStat = 1 / m * np.sum(Kc.T * Lc)

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


def compute_residual_kernel_sigma(Y, K, sigma=1.0):
    # (X,Y)
    # Y is real-vector
    # Kij = k(X_i, X_j)
    I = np.eye(len(K))
    return (inv(I + sigma ** (-2) * K) @ Y).squeeze()


def compute_residual(Y, X, k=None):
    _, n_feats = X.shape
    if k is None:
        k = GPflow.kernels.RBF(n_feats) + GPflow.kernels.White(n_feats)
    m = GPflow.gpr.GPR(X, Y, k)
    m.optimize()
    Yhat, _ = m.predict_y(X)
    return Y - Yhat


# TODO wrapper for precomputed kernel matrix

# def compute_residual2(Y: np.ndarray, K: np.ndarray, sigma=1.0):
#     I = np.eye(len(K))
#     return Y - K @ inv(K + sigma ** 2 * I) @ Y


def residual_kernel_matrix(K: np.ndarray, L: np.ndarray, sigma=0.1):
    I = np.eye(len(K))
    IK = inv(I + sigma ** (-2) * K)
    # return K @ IK + IK @ L @ IK
    return K @ IK + IK @ L @ IK
    # return (K + IK @ L) @ IK


def check_cond_indep(KX, KY, KZ, sigma1=0.1, sigma2=0.1):
    R1 = residual_kernel_matrix(KZ, KX, sigma=sigma1)
    R2 = residual_kernel_matrix(KZ, KY, sigma=sigma2)
    return HSIC(R1, R2)


def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def check_cond_indep_real(X, Y, Z=None):
    if Z is not None:
        X = normalize(X)
        Y = normalize(Y)
        Z = normalize(Z)

        e_YZ = compute_residual(Y, Z)
        e_XZ = compute_residual(X, Z)

        e_YZ = normalize(e_YZ)
        e_XZ = normalize(e_XZ)
        if e_YZ.ndim == 1:
            e_YZ = e_YZ[:, None]
        if e_XZ.ndim == 1:
            e_XZ = e_XZ[:, None]

        D1_sq = pairwise_distances(e_YZ, metric='sqeuclidean')
        D2_sq = pairwise_distances(e_XZ, metric='sqeuclidean')

        mask = np.triu(np.ones(D1_sq.shape), 0)
        D1_sq_median = np.ma.median(np.ma.array(D1_sq, mask=mask))
        D2_sq_median = np.ma.median(np.ma.array(D2_sq, mask=mask))

        R1 = np.exp(-D1_sq / 2 / D1_sq_median)
        R2 = np.exp(-D2_sq / 2 / D2_sq_median)

        return HSIC(R1, R2)
    else:
        D1_sq = pairwise_distances(X, metric='sqeuclidean')
        D2_sq = pairwise_distances(Y, metric='sqeuclidean')

        mask = np.triu(np.ones(D1_sq.shape), 0)
        D1_sq_median = np.ma.median(np.ma.array(D1_sq, mask=mask))
        D2_sq_median = np.ma.median(np.ma.array(D2_sq, mask=mask))

        R1 = np.exp(-D1_sq / 2 / D1_sq_median)
        R2 = np.exp(-D2_sq / 2 / D2_sq_median)
        return HSIC(R1, R2)


if __name__ == '__main__':
    n = 500
    for mix in [0.0, 0.1, 0.2, 0.5, 1.0]:
        X = np.random.randn(n, 1)
        Y = mix * X + np.random.randn(n, 1)
        Z = X + np.random.randn(n, 1)

        DX = pairwise_distances(X, metric='sqeuclidean')
        DY = pairwise_distances(Y, metric='sqeuclidean')
        DZ = pairwise_distances(Z, metric='sqeuclidean')
        KX = np.exp(-DX / 2)
        KY = np.exp(-DY / 2)
        KZ = np.exp(-DZ / 2)

        I = np.eye(len(KX))
        sigma = 0.5

        phi = np.random.randn(n, 1)

        # assert np.allclose(KX - KX @ inv(KX + sigma ** 2 * I) @ KX, KX @ inv(I + sigma ** (-2) * KX))  # check Eq 15.
        # assert np.allclose(KX @ inv(KX + sigma ** 2 * I) @ phi - phi, -inv(I + sigma ** (-2) * KX) @ phi)  # check Eq 16
        print('{:.4f} {:.4f}'.format(mix, HSIC(KX, KY)))
        print('{:.4f} {:.4f}'.format(mix, check_cond_indep(KX, KY, KZ, 1.0e-4, 1.0e-4)))
        print('{:.4f} {:.4f}'.format(mix, check_cond_indep_real(X, Y, Z)))
        print()
