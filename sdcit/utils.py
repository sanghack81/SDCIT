import warnings

import GPflow
import numpy as np
import numpy.ma as ma
import scipy.linalg
import tensorflow as tf
from numpy import diag, exp, sqrt
from numpy.matlib import repmat
from sklearn.metrics import euclidean_distances


def columnwise_normalizes(*xs):
    return [columnwise_normalize(x) for x in xs]


def columnwise_normalize(x: np.ndarray) -> np.ndarray:
    """normalize per column"""
    x = x - np.vstack([np.mean(x, 0)] * len(x))
    x = x / np.std(x, 0)  # broadcast
    return x


def noise_variance(K_Y, K_X):
    T = len(K_Y)
    eig_Kx, eix = reduced_eig(*eigdec(K_Y, min(100, int(T / 5))))
    gp_x = GPflow.gpr.GPR(np.arange(len(K_X))[:, None], 2 * sqrt(T) * eix @ diag(sqrt(eig_Kx)) / sqrt(eig_Kx[0]), PrecomputedKernel(K_X) + GPflow.kernels.White(1))
    gp_x.optimize()
    return gp_x.kern.white.variance.value[0]


def ensure_symmetric(x):
    return (x + x.T) / 2


def reduced_eig(eig_Ky, eiy=None, Thresh=1e-5):
    IIy = np.where(eig_Ky > max(eig_Ky) * Thresh)[0]
    if eiy is not None:
        return eig_Ky[IIy], eiy[:, IIy]
    else:
        return eig_Ky[IIy]


def eigdec(x: np.ndarray, n: int = None):
    """Top N descending ordered eigenvalues and corresponding eigenvectors"""
    if n is None:
        n = len(x)
    x = ensure_symmetric(x)
    M = len(x)
    # ascending M-1-N <= <= M-1
    w, v = scipy.linalg.eigh(x, eigvals=(M - 1 - n + 1, M - 1))
    w = w[::-1]
    v = v[:, ::-1]
    return w, v


def centering(M):
    """Matrix Centering"""
    nr, nc = M.shape
    assert nr == nc
    n = nr
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ M @ H


# inverse of a positive definite matrix
def pdinv(x):
    L = scipy.linalg.cholesky(x)
    Linv = scipy.linalg.inv(L)
    return Linv.T @ Linv


def residualize_real_and_kernel(Y, K_X, sigma=1.0):
    # Y_i - E[Y_i|X_i]
    I = np.eye(len(K_X))
    return (pdinv(I + sigma ** (-2) * K_X) @ Y).squeeze()


def residualize(Y, X, gp_kernel=None):
    # Y_i - E[Y_i|X_i]
    if gp_kernel is None:
        _, n_feats = X.shape
        gp_kernel = GPflow.kernels.RBF(n_feats) + GPflow.kernels.White(n_feats)

    m = GPflow.gpr.GPR(X, Y, gp_kernel)
    m.optimize()

    Yhat, _ = m.predict_y(X)
    return Y - Yhat


def residual_kernel(K_Y: np.ndarray, K_X: np.ndarray, sigma=None):
    K_Y, K_X = centering(K_Y), centering(K_X)
    # R_Y|X
    if sigma is None:
        sigma = noise_variance(K_Y, K_X)
    I = np.eye(len(K_X))
    IK = pdinv(I + sigma ** (-2) * K_X)
    # TODO WHY WHY?
    warnings.warn('original K@IK + IK@L@IK does not work... remove covariance part.')
    # return (K_X + IK @ K_Y) @ IK
    return (IK @ K_Y) @ IK


def rbf_kernel_with_median_heuristic(data, *args):
    """A list of RBF kernel matrices for data sets in arguments based on median heuristic"""
    if args is None:
        args = []

    outs = []
    for x in [data, *args]:
        squared_distances = euclidean_distances(x, squared=True)
        # masking upper triangle and the diagonal.
        mask = np.triu(np.ones(squared_distances.shape), 0)
        median_squared_distance = ma.median(ma.array(squared_distances, mask=mask))
        kx = exp(-0.5 * squared_distances / median_squared_distance)
        outs.append(kx)

    if len(outs) == 1:
        return outs[0]
    else:
        return outs


def p_value_of(val, data) -> float:
    """The percentile of a value given a data

    Parameters
    ----------
    val: float
        value to compute p-value
    data: array_like
        data representing a reference distribution
    """

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


class PrecomputedKernel(GPflow.kernels.Kern):
    def __init__(self, K):
        warnings.warn('experimental. This kernel ignores the actual input, and just return the given precomputed kernel matrix.')
        GPflow.kernels.Kern.__init__(self, input_dim=1)
        self.variance = GPflow.param.Param(1.0, transform=GPflow.transforms.positive)
        self.given_K = K

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return self.variance * tf.stack(self.given_K)

    def Kdiag(self, X):
        return self.variance * tf.stack(np.diag(self.given_K))
