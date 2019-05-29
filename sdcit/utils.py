import numpy as np
import numpy.ma as ma
import scipy.linalg
import scipy.optimize
import scipy.stats
import typing
import warnings
from numpy import diag, exp, sqrt
from numpy.matlib import repmat
from sklearn.metrics import euclidean_distances
from typing import Union, List


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


def default_gp_kernel(X: np.ndarray):
    from gpflow.kernels import White, RBF

    _, n_feats = X.shape
    return RBF(n_feats, ARD=True) + White(n_feats)


def residualize(Y, X=None, gp_kernel=None):
    """Residual of Y given X. Y_i - E[Y_i|X_i]"""
    import gpflow
    from gpflow.models import GPR

    if X is None:
        return Y - np.mean(Y)  # nothing is residualized!

    if gp_kernel is None:
        gp_kernel = default_gp_kernel(X)

    m = GPR(X, Y, gp_kernel)
    gpflow.train.ScipyOptimizer().minimize(m)

    Yhat, _ = m.predict_y(X)
    return Y - Yhat


def residual_kernel(K_Y: np.ndarray, K_X: np.ndarray, use_expectation=True, with_gp=True, sigma_squared=1e-3, return_learned_K_X=False):
    """Kernel matrix of residual of Y given X based on their kernel matrices, Y=f(X)"""
    import gpflow
    from gpflow.kernels import White, Linear
    from gpflow.models import GPR

    K_Y, K_X = centering(K_Y), centering(K_X)
    T = len(K_Y)

    if with_gp:
        eig_Ky, eiy = truncated_eigen(*eigdec(K_Y, min(100, T // 4)))
        eig_Kx, eix = truncated_eigen(*eigdec(K_X, min(100, T // 4)))

        X = eix @ diag(sqrt(eig_Kx))  # X @ X.T is close to K_X
        Y = eiy @ diag(sqrt(eig_Ky))
        n_feats = X.shape[1]

        linear = Linear(n_feats, ARD=True)
        white = White(n_feats)
        gp_model = GPR(X, Y, linear + white)
        gpflow.train.ScipyOptimizer().minimize(gp_model)

        K_X = linear.compute_K_symm(X)
        sigma_squared = white.variance.value

    P = pdinv(np.eye(T) + K_X / sigma_squared)  # == I-K @ inv(K+Sigma) in Zhang et al. 2011
    if use_expectation:  # Flaxman et al. 2016 Gaussian Processes for Independence Tests with Non-iid Data in Causal Inference.
        RK = (K_X + P @ K_Y) @ P
    else:  # Zhang et al. 2011. Kernel-based Conditional Independence Test and Application in Causal Discovery.
        RK = P @ K_Y @ P

    if return_learned_K_X:
        return RK, K_X
    else:
        return RK


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


def p_value_of(val: float, data: typing.Iterable) -> float:
    """The percentile of a value given a data"""

    data = np.sort(data)
    return float(1 - np.searchsorted(data, val, side='right') / len(data))


def random_seeds(n=None):
    """Random seeds of given size or a random seed if n is None"""
    if n is None:
        return np.random.randint(np.iinfo(np.int32).max)
    else:
        return [np.random.randint(np.iinfo(np.int32).max) for _ in range(n)]


def K2D(K: Union[None, np.ndarray]) -> np.ndarray:
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


def AUPC(p_values: Union[List, np.ndarray]) -> float:
    """Area Under Power Curve"""
    p_values = np.array(p_values)

    # CDF of p-values
    xys = [(uniq_v, np.mean(p_values <= uniq_v)) for uniq_v in np.unique(p_values)]

    area, prev_x, prev_y = 0, 0, 0
    for x, y in xys:
        area += (x - prev_x) * prev_y
        prev_x, prev_y = x, y

    area += (1 - prev_x) * prev_y
    return area


def KS_statistic(p_values: np.ndarray) -> float:
    """Kolmogorov-Smirnov test statistics"""
    return scipy.stats.kstest(p_values, 'uniform')[0]


def p_value_curve(p_values):
    p_values = np.array(p_values)
    xys = [(uniq_v, np.mean(p_values <= uniq_v)) for uniq_v in np.unique(p_values)]
    return [(0, 0), *xys, (1, 1)]


def regression_distance(Y: np.ndarray, Z: np.ndarray, ard=True):
    """d(z,z') = |f(z)-f(z')| where Y=f(Z) + noise and f ~ GP"""
    import gpflow
    from gpflow.kernels import White, RBF
    from gpflow.models import GPR

    n, dims = Z.shape

    rbf = RBF(dims, ARD=ard)
    rbf_white = rbf + White(dims)

    gp_model = GPR(Z, Y, rbf_white)
    gpflow.train.ScipyOptimizer().minimize(gp_model)

    Kz_y = rbf.compute_K_symm(Z)
    Ry = pdinv(rbf_white.compute_K_symm(Z))
    Fy = Y.T @ Ry @ Kz_y  # F(z)

    M = Fy.T @ Fy
    O = np.ones((n, 1))
    N = O @ (np.diag(M)[:, None]).T
    D = np.sqrt(N + N.T - 2 * M)

    return D, Kz_y


def regression_distance_k(Kx: np.ndarray, Ky: np.ndarray):
    warnings.warn('not tested yet!')
    import gpflow
    from gpflow.kernels import White, Linear
    from gpflow.models import GPR

    T = len(Kx)

    eig_Ky, eiy = truncated_eigen(*eigdec(Ky, min(100, T // 4)))
    eig_Kx, eix = truncated_eigen(*eigdec(Kx, min(100, T // 4)))

    X = eix @ diag(sqrt(eig_Kx))  # X @ X.T is close to K_X
    Y = eiy @ diag(sqrt(eig_Ky))
    n_feats = X.shape[1]

    linear = Linear(n_feats, ARD=True)
    white = White(n_feats)
    gp_model = GPR(X, Y, linear + white)
    gpflow.train.ScipyOptimizer().minimize(gp_model)

    Kx = linear.compute_K_symm(X)
    sigma_squared = white.variance.value

    P = Kx @ pdinv(Kx + sigma_squared * np.eye(T))

    M = P @ Ky @ P
    O = np.ones((T, 1))
    N = O @ np.diag(M).T
    D = np.sqrt(N + N.T - 2 * M)
    return D
