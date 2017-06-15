import typing
import warnings
from typing import Union

import numpy as np
import numpy.ma as ma
import scipy.linalg
import scipy.optimize
from GPflow.gpr import GPR
from GPflow.kernels import White, Linear, RBF, Kern
from numpy import diag, exp, sqrt
from numpy.matlib import repmat
from sklearn.metrics import euclidean_distances


def columnwise_normalizes(*Xs) -> typing.List[Union[None, np.ndarray]]:
    """normalize per column for multiple data"""
    return [columnwise_normalize(X) for X in Xs]


def columnwise_normalize(X: np.ndarray) -> Union[None, np.ndarray]:
    """normalize per column"""
    if X is None:
        return None
    return (X - np.mean(X, 0)) / np.std(X, 0)  # broadcast


def ensure_symmetric(x: np.ndarray) -> np.ndarray:
    return (x + x.T) / 2


def truncated_eigen(eig_vals, eig_vecs=None, relative_threshold=1e-5):
    """Retain eigenvalues and corresponding eigenvectors where an eigenvalue > max(eigenvalues)*relative_threshold"""
    indices = np.where(eig_vals > max(eig_vals) * relative_threshold)[0]
    if eig_vecs is not None:
        return eig_vals[indices], eig_vecs[:, indices]
    else:
        return eig_vals[indices]


def eigdec(X: np.ndarray, top_N: int = None):
    """Eigendecomposition with top N descending ordered eigenvalues and corresponding eigenvectors"""
    if top_N is None:
        top_N = len(X)

    X = ensure_symmetric(X)
    M = len(X)

    # ascending M-1-N <= <= M-1
    w, v = scipy.linalg.eigh(X, eigvals=(M - 1 - top_N + 1, M - 1))

    # descending
    return w[::-1], v[:, ::-1]


def centering(M: np.ndarray) -> Union[None, np.ndarray]:
    """Matrix Centering"""
    if M is None:
        return None
    n = len(M)
    H = np.eye(n) - 1 / n
    return H @ M @ H


def pdinv(x: np.ndarray) -> np.ndarray:
    """Inverse of a positive definite matrix"""
    U = scipy.linalg.cholesky(x)
    Uinv = scipy.linalg.inv(U)
    return Uinv @ Uinv.T


def default_gp_kernel(X: np.ndarray) -> Kern:
    _, n_feats = X.shape
    return RBF(n_feats, ARD=True) + White(n_feats)


def residualize(Y, X=None, gp_kernel=None):
    """Residual of Y given X. Y_i - E[Y_i|X_i]"""
    if X is None:
        return Y - np.mean(Y)  # nothing is residualized!

    if gp_kernel is None:
        gp_kernel = default_gp_kernel(X)

    m = GPR(X, Y, gp_kernel)
    m.optimize()

    Yhat, _ = m.predict_y(X)
    return Y - Yhat


def residual_kernel(K_Y: np.ndarray, K_X: np.ndarray, use_expectation=True, with_gp=True, sigma_squared=1e-3):
    """Kernel matrix of residual of Y given X based on their kernel matrices"""
    K_Y, K_X = centering(K_Y), centering(K_X)
    T = len(K_Y)

    if with_gp:
        eig_Ky, eiy = truncated_eigen(*eigdec(K_Y, min(100, T // 4)))
        eig_Kx, eix = truncated_eigen(*eigdec(K_X, min(100, T // 4)))

        X = eix @ diag(sqrt(eig_Kx))  # X @ X.T is close to K_X
        Y = eiy @ diag(sqrt(eig_Ky))
        n_feats = X.shape[1]

        gp_model = GPR(X, Y, Linear(n_feats, ARD=True) + White(n_feats))
        gp_model.optimize()

        K_X = gp_model.kern.linear.compute_K_symm(X)
        sigma_squared = gp_model.kern.white.variance.value[0]

    P = pdinv(np.eye(T) + K_X / sigma_squared)  # == I-K @ inv(K+Sigma) in Zhang et al. 2011
    if use_expectation:  # Flaxman et al. 2016 Gaussian Processes for Independence Tests with Non-iid Data in Causal Inference.
        return (K_X + P @ K_Y) @ P
    else:  # Zhang et al. 2011. Kernel-based Conditional Independence Test and Application in Causal Discovery.
        return P @ K_Y @ P


def rbf_kernel_median(data: np.ndarray, *args, without_two=False):
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


def p_value_of(val: float, data: np.ndarray) -> float:
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
    """An RKHS distance matrix given a kernel matrix

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


def cythonize(*matrices):
    return tuple(np.ascontiguousarray(matrix, dtype=np.float64) for matrix in matrices)
