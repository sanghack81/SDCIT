import warnings

import GPflow
import numpy as np
import numpy.ma as ma
import scipy.linalg
import scipy.optimize
from GPflow.kernels import White, Linear, RBF
from numpy import diag, exp, sqrt
from numpy.matlib import repmat
from sklearn.metrics import euclidean_distances


def columnwise_normalizes(*xs):
    return [columnwise_normalize(x) for x in xs]


def columnwise_normalize(x: np.ndarray) -> np.ndarray:
    """normalize per column"""
    if x is None:
        return None
    return (x - np.mean(x, 0)) / np.std(x, 0)  # broadcast


def ensure_symmetric(x):
    return (x + x.T) / 2


def truncated_eigen(eig_vals, eig_vecs=None, relative_threshold=1e-5):
    indices = np.where(eig_vals > max(eig_vals) * relative_threshold)[0]
    if eig_vecs is not None:
        return eig_vals[indices], eig_vecs[:, indices]
    else:
        return eig_vals[indices]


def eigdec(x: np.ndarray, n: int = None):
    """Top N descending ordered eigenvalues and corresponding eigenvectors"""
    if n is None:
        n = len(x)

    x = ensure_symmetric(x)
    M = len(x)

    # ascending M-1-N <= <= M-1
    w, v = scipy.linalg.eigh(x, eigvals=(M - 1 - n + 1, M - 1))
    # descending
    return w[::-1], v[:, ::-1]


def centering(M: np.ndarray) -> np.ndarray:
    """Matrix Centering"""
    nr, nc = M.shape
    assert nr == nc
    n = nr
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ M @ H


# inverse of a positive definite matrix
def pdinv(x):
    U = scipy.linalg.cholesky(x)
    Uinv = scipy.linalg.inv(U)
    return Uinv @ Uinv.T


def residualize(Y, X, gp_kernel=None):
    # Y_i - E[Y_i|X_i]
    if gp_kernel is None:
        _, n_feats = X.shape
        gp_kernel = RBF(n_feats) + White(n_feats)

    m = GPflow.gpr.GPR(X, Y, gp_kernel)
    m.optimize()

    Yhat, _ = m.predict_y(X)
    return Y - Yhat


def residual_kernel(K_Y: np.ndarray, K_X: np.ndarray, eq_17_as_is=True, with_gp=False, sigma_squared=1e-3):
    # R_Y|X
    K_Y, K_X = centering(K_Y), centering(K_X)
    T = len(K_Y)
    if with_gp:
        eig_Ky, eiy = truncated_eigen(*eigdec(K_Y, min(100, T // 5)))
        eig_Kx, eix = truncated_eigen(*eigdec(K_X, min(100, T // 5)))

        X = eix @ diag(sqrt(eig_Kx))
        Y = eiy @ diag(sqrt(eig_Ky))
        n_feats = X.shape[1]

        # TODO ARD
        gp_x = GPflow.gpr.GPR(X, Y, Linear(n_feats) + White(n_feats))
        gp_x.optimize()

        K_X = gp_x.kern.linear.compute_K_symm(X)
        # TODO ARD, diag variance
        sigma_squared = gp_x.kern.white.variance.value[0]

    I = np.eye(len(K_X))
    IK = pdinv(I + K_X / sigma_squared)  # I-K @ inv(K+Sigma)
    if eq_17_as_is:
        return (K_X + IK @ K_Y) @ IK
    else:
        return IK @ K_Y @ IK


def rbf_kernel_with_median_heuristic_wo2(data, *args):
    return rbf_kernel_with_median_heuristic(data, *args, without_two=True)


def rbf_kernel_with_median_heuristic(data, *args, without_two=False):
    """A list of RBF kernel matrices for data sets in arguments based on median heuristic"""
    if args is None:
        args = []

    outs = []
    for x in [data, *args]:
        D_squared = euclidean_distances(x, squared=True)
        # masking upper triangle and the diagonal.
        mask = np.triu(np.ones(D_squared.shape), 0)
        median_squared_distance = ma.median(ma.array(D_squared, mask=mask))
        if without_two:
            kx = exp(-D_squared / median_squared_distance)
        else:
            kx = exp(-0.5 * D_squared / median_squared_distance)
        outs.append(kx)

    if len(outs) == 1:
        return outs[0]
    else:
        return outs


def p_value_of(val, data) -> float:
    """The percentile of a value given a data"""

    data = np.sort(data)
    return 1 - np.searchsorted(data, val, side='right') / len(data)


def random_seeds(n=None):
    """Random seeds of given size or a random seed if n is None"""
    if n is None:
        return np.random.randint(np.iinfo(np.int32).max)
    else:
        return [np.random.randint(np.iinfo(np.int32).max) for _ in range(n)]


def K2D(K):
    """An RKHS distance matrix from a kernel matrix

    A distance matrix D of the same size of the given kernel matrix K
     :math:`d^2(i,j)=k(i,i)+k(j,j)-2k(i,j)`.
    """
    if K is None:
        return None

    Kd = repmat(diag(K).reshape((len(K), 1)), 1, len(K))
    temp = Kd + Kd.transpose() - 2 * K
    min_val = np.min(temp)
    if min_val < 0.0:
        if min_val < -1e-15:
            warnings.warn('K2D: negative values will be treated as zero. Observed: {}'.format(min_val))
        temp *= (temp > 0)
    return np.sqrt(temp)
