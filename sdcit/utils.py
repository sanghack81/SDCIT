import numpy as np
import numpy.ma as ma
import scipy.linalg
import scipy.optimize
import scipy.stats
import typing
import warnings
from numpy import diag, exp, sqrt
from sklearn.metrics import euclidean_distances

from typing import Union, List


def repmat(a, m, n):
    """Repeat a matrix a given number of times.

    Parameters
    ----------
    a : array_like
        The array to repeat.
    m : int
        Number of repetitions along the first axis.
    n : int
        Number of repetitions along the second axis.

    Returns
    -------
    np.ndarray
        The tiled array.
    """
    return np.tile(a, (m, n))


def _require_gpflow():
    """Import gpflow, enforcing the 2.x API that SDCIT targets.

    GP-based functions (KCIT, FCIT, GP residualization) require gpflow>=2.0.
    gpflow is an optional dependency; install it with ``pip install 'SDCIT[gp]'``.
    """
    try:
        import gpflow
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "gpflow>=2.0 is required for GP-based functions. "
            "Install with `pip install 'SDCIT[gp]'`."
        ) from e
    if int(gpflow.__version__.split('.')[0]) < 2:
        raise ImportError(
            f"SDCIT requires gpflow>=2.0 (the 1.x API was removed), but found gpflow {gpflow.__version__}."
        )
    return gpflow


def _optimize_gpr(model) -> None:
    """Fit a gpflow 2.x GPR model's hyperparameters with Scipy L-BFGS-B."""
    import gpflow
    gpflow.optimizers.Scipy().minimize(model.training_loss, model.trainable_variables)


def _kernel_matrix(kernel, X: np.ndarray) -> np.ndarray:
    """Gram matrix K(X, X) of a gpflow 2.x kernel as a numpy array.

    Replaces the gpflow 1.x ``kernel.compute_K_symm(X)`` API.
    """
    return np.asarray(kernel(X), dtype=np.float64)


def columnwise_normalizes(*Xs) -> typing.List[Union[None, np.ndarray]]:
    """Normalize multiple arrays per column.

    Parameters
    ----------
    *Xs : tuple of np.ndarray
        Arrays to be normalized.

    Returns
    -------
    List[Union[None, np.ndarray]]
        A list of normalized arrays.
    """
    return [columnwise_normalize(X) for X in Xs]


def columnwise_normalize(X: np.ndarray) -> Union[None, np.ndarray]:
    """Normalize a single array per column.

    Parameters
    ----------
    X : np.ndarray
        Array to be normalized.

    Returns
    -------
    Union[None, np.ndarray]
        An array where each column has mean 0 and standard deviation 1.
    """
    if X is None:
        return None
    return (X - np.mean(X, 0)) / np.std(X, 0)  # broadcast


def ensure_symmetric(x: np.ndarray) -> np.ndarray:
    """Ensure a matrix is perfectly symmetric.

    Parameters
    ----------
    x : np.ndarray
        Square matrix to be symmetrized.

    Returns
    -------
    np.ndarray
        Symmetric matrix computed as (x + x.T) / 2.
    """
    return (x + x.T) / 2


def truncated_eigen(eig_vals, eig_vecs=None, relative_threshold=1e-5):
    """Retain eigenvalues and corresponding eigenvectors where an eigenvalue > max(eigenvalues)*relative_threshold

    Parameters
    ----------
    eig_vals : np.ndarray
        1D array of eigenvalues.
    eig_vecs : np.ndarray, optional
        2D array of eigenvectors.
    relative_threshold : float, optional
        Threshold criteria to filter eigenvalues (default is 1e-5).

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Filtered eigenvalues, and optionally filtered eigenvectors if they were provided.
    """
    indices = np.where(eig_vals > max(eig_vals) * relative_threshold)[0]
    if eig_vecs is not None:
        return eig_vals[indices], eig_vecs[:, indices]
    else:
        return eig_vals[indices]


def eigdec(X: np.ndarray, top_N: int = None):
    """Eigendecomposition with top N descending ordered eigenvalues and corresponding eigenvectors.

    Parameters
    ----------
    X : np.ndarray
        Symmetric matrix to decompose.
    top_N : int, optional
        Number of top eigenvalues to retain. If None, retains all (default is None).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Descending sorted eigenvalues and their corresponding eigenvectors.
    """
    if top_N is None:
        top_N = len(X)

    X = ensure_symmetric(X)
    M = len(X)

    # ascending M-1-N <= <= M-1
    w, v = scipy.linalg.eigh(X, subset_by_index=(M - 1 - top_N + 1, M - 1))

    # descending
    return w[::-1], v[:, ::-1]


def centering(M: np.ndarray) -> Union[None, np.ndarray]:
    """Matrix Centering operation.

    Parameters
    ----------
    M : np.ndarray
        A square matrix.

    Returns
    -------
    Union[None, np.ndarray]
        The centered matrix H @ M @ H.
    """
    if M is None:
        return None
    n = len(M)
    H = np.eye(n) - 1 / n
    return H @ M @ H


def pdinv(x: np.ndarray) -> np.ndarray:
    """Inverse of a positive definite matrix using Cholesky decomposition.

    Parameters
    ----------
    x : np.ndarray
        Positive definite matrix to invert.

    Returns
    -------
    np.ndarray
        Inverse of the matrix.
    """
    U = scipy.linalg.cholesky(x)
    Uinv = scipy.linalg.inv(U)
    return Uinv @ Uinv.T


def default_gp_kernel(X: np.ndarray):
    """Generates the default GP kernel.

    Parameters
    ----------
    X : np.ndarray
        Input features array to define the number of features.

    Returns
    -------
    gpflow.kernels.Kernel
        An additive kernel of RBF and White.
    """
    from gpflow.kernels import White, RBF

    _, n_feats = X.shape
    return RBF(lengthscales=np.ones(n_feats)) + White()  # ARD via vector lengthscales (gpflow 2.x)


def residualize(Y, X=None, gp_kernel=None):
    """Residual of Y given X. Generates conditional predictions Y_i - E[Y_i|X_i].

    Parameters
    ----------
    Y : np.ndarray
        Target labels or values.
    X : np.ndarray, optional
        Features. If None, Y is simply centered.
    gp_kernel : optional
        Custom GP kernel to use. If None, the default kernel is built.

    Returns
    -------
    np.ndarray
        The residual differences.
    """
    _require_gpflow()
    from gpflow.models import GPR

    if X is None:
        return Y - np.mean(Y)  # nothing is residualized!

    if gp_kernel is None:
        gp_kernel = default_gp_kernel(X)

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    m = GPR((X, Y), kernel=gp_kernel)
    _optimize_gpr(m)

    Yhat, _ = m.predict_y(X)
    return Y - np.asarray(Yhat, dtype=np.float64)


def residual_kernel(K_Y: np.ndarray, K_X: np.ndarray, use_expectation=True, with_gp=True, sigma_squared=1e-3, return_learned_K_X=False):
    """Kernel matrix of residual of Y given X based on their kernel matrices, Y=f(X)

    Parameters
    ----------
    K_Y : np.ndarray
        Kernel matrix of Y.
    K_X : np.ndarray
        Kernel matrix of X.
    use_expectation : bool, optional
        Whether to use expectation correction formulation (default is True).
    with_gp : bool, optional
        Whether to learn hyperparameters using GPFlow (default is True).
    sigma_squared : float, optional
        Default variance noise floor (default is 1e-3).
    return_learned_K_X : bool, optional
        If True, returns the learned residual kernel and the original kernel (default is False).

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        The residual kernel matrix.
    """
    _require_gpflow()
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

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        linear = Linear(variance=np.ones(n_feats))  # ARD via per-dimension variance (gpflow 2.x)
        white = White()
        gp_model = GPR((X, Y), kernel=linear + white)
        _optimize_gpr(gp_model)

        K_X = _kernel_matrix(linear, X)
        sigma_squared = float(white.variance.numpy())

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
    """A list of RBF kernel matrices for data sets in arguments based on median heuristic.

    Parameters
    ----------
    data : np.ndarray
        The primary data array.
    *args : np.ndarray
        Additional sequence of data arrays.
    without_two : bool, optional
        If True, the variance logic scales without the (0.5) multiplier reduction (default is False).

    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
        A single kernel matrix or a list of kernel matrices.
    """
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
    """The percentile of a value given an array of empirical distribution data.

    Parameters
    ----------
    val : float
        The test statistic.
    data : typing.Iterable
        The distribution or array of null hypothesis values.

    Returns
    -------
    float
        The calculated p-value representing area beyond the val.
    """
    data = np.sort(data)
    return float(1 - np.searchsorted(data, val, side='right') / len(data))


def random_seeds(n=None):
    """Random seeds of given size or a random seed if n is None.

    Parameters
    ----------
    n : int, optional
        How many random seeds to return (default is None).

    Returns
    -------
    Union[int, List[int]]
        A single integer seed or a list of random integer seeds.
    """
    if n is None:
        return np.random.randint(np.iinfo(np.int32).max)
    else:
        return [np.random.randint(np.iinfo(np.int32).max) for _ in range(n)]


def K2D(K: Union[None, np.ndarray]) -> np.ndarray:
    """An RKHS distance matrix given a kernel matrix.

    A distance matrix D of the same size of the given kernel matrix K
     :math:`d^2(i,j)=k(i,i)+k(j,j)-2k(i,j)`.

    Parameters
    ----------
    K : Union[None, np.ndarray]
        The source Kernel correlation matrix.

    Returns
    -------
    np.ndarray
        The Euclidean translation distance matrix.
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
    """Casts matrices into a contiguous float64 array type for quick C ingestion.

    Parameters
    ----------
    *matrices : tuple of np.ndarray
        Array matrices to convert.

    Returns
    -------
    tuple
        Sequence of np.float64 matrices.
    """
    return tuple(np.ascontiguousarray(matrix, dtype=np.float64) for matrix in matrices)


def AUPC(p_values: Union[List, np.ndarray]) -> float:
    """Area Under Power Curve mapping calculated p-values into uniform spaces.

    Parameters
    ----------
    p_values : Union[List, np.ndarray]
        Test p-values collection.

    Returns
    -------
    float
        Total curve area integration score.
    """
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
    """Kolmogorov-Smirnov test statistics over uniformity checking.

    Parameters
    ----------
    p_values : np.ndarray
        Series of recorded test p-values.

    Returns
    -------
    float
        Deviation ratio from the expected distribution line.
    """
    return scipy.stats.kstest(p_values, 'uniform')[0]


def p_value_curve(p_values):
    """Yields plot anchor coordinates representing power characteristics of p-values.

    Parameters
    ----------
    p_values : typing.Iterable
        Calculated or empirical p-value array.

    Returns
    -------
    list
        List of 2D coordinates representing discrete empirical points.
    """
    p_values = np.array(p_values)
    xys = [(uniq_v, np.mean(p_values <= uniq_v)) for uniq_v in np.unique(p_values)]
    return [(0, 0), *xys, (1, 1)]


def regression_distance(Y: np.ndarray, Z: np.ndarray, ard=True):
    """d(z,z') = |f(z)-f(z')| where Y=f(Z) + noise and f ~ GP.

    Parameters
    ----------
    Y : np.ndarray
        Dependent target dataset matrix.
    Z : np.ndarray
        Conditional feature subset matrix.
    ard : bool, optional
        Use Automatic Relevance Determination (default is True).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Regression translated RKHS distance matrix and computed kernel mapping.
    """
    _require_gpflow()
    from gpflow.kernels import White, RBF
    from gpflow.models import GPR

    n, dims = Z.shape

    rbf = RBF(lengthscales=np.ones(dims) if ard else 1.0)
    rbf_white = rbf + White()

    Z = np.asarray(Z, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gp_model = GPR((Z, Y), kernel=rbf_white)
    _optimize_gpr(gp_model)

    Kz_y = _kernel_matrix(rbf, Z)
    Ry = pdinv(_kernel_matrix(rbf_white, Z))
    Fy = Y.T @ Ry @ Kz_y  # F(z)

    M = Fy.T @ Fy
    ones_mat = np.ones((n, 1))
    N = ones_mat @ (np.diag(M)[:, None]).T
    D = np.sqrt(N + N.T - 2 * M)

    return D, Kz_y


def regression_distance_k(Kx: np.ndarray, Ky: np.ndarray):
    warnings.warn('not tested yet!')
    _require_gpflow()
    from gpflow.kernels import White, Linear
    from gpflow.models import GPR

    T = len(Kx)

    eig_Ky, eiy = truncated_eigen(*eigdec(Ky, min(100, T // 4)))
    eig_Kx, eix = truncated_eigen(*eigdec(Kx, min(100, T // 4)))

    X = eix @ diag(sqrt(eig_Kx))  # X @ X.T is close to K_X
    Y = eiy @ diag(sqrt(eig_Ky))
    n_feats = X.shape[1]

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    linear = Linear(variance=np.ones(n_feats))  # ARD via per-dimension variance (gpflow 2.x)
    white = White()
    gp_model = GPR((X, Y), kernel=linear + white)
    _optimize_gpr(gp_model)

    Kx = _kernel_matrix(linear, X)
    sigma_squared = float(white.variance.numpy())

    P = Kx @ pdinv(Kx + sigma_squared * np.eye(T))

    M = P @ Ky @ P
    ones_mat = np.ones((T, 1))
    N = ones_mat @ np.diag(M).T
    D = np.sqrt(N + N.T - 2 * M)
    return D
