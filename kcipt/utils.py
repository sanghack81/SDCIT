import numpy as np
import scipy
from numpy import diag, log, sqrt, exp
from numpy.matlib import repmat
from numpy.random import choice, randn
from scipy.stats import expon
from sklearn.metrics import pairwise_distances
from sklearn.metrics import euclidean_distances
import numpy.ma as ma

def median_heuristic(x, y, z):
    dx = euclidean_distances(x, squared=True)
    dy = euclidean_distances(y, squared=True)
    dz = euclidean_distances(z, squared=True)

    mx = ma.median(ma.array(dx, mask=np.triu(np.ones(dx.shape), 0)))
    my = ma.median(ma.array(dy, mask=np.triu(np.ones(dy.shape), 0)))
    mz = ma.median(ma.array(dz, mask=np.triu(np.ones(dz.shape), 0)))

    kx = exp(-0.5 * dx / mx)
    ky = exp(-0.5 * dy / my)
    kz = exp(-0.5 * dz / mz)
    return kx, ky, kz



def safe_iter(iterable, sort=False):
    """Iterator (generator) that can skip removed items."""
    copied = list(iterable)
    if sort:
        copied = sorted(copied)
    for y in copied:
        if y in iterable:
            yield y


def auto_rbf_gamma(X, **kwargs):
    if X.ndim == 1:
        X = X[:, None]
    D = pairwise_distances(X)
    return kde_based_rbf_gamma(D)


def auto_rbf_kernel(X, n_jobs=1):
    if X.ndim == 1:
        X = X[:, None]
    D = pairwise_distances(X, n_jobs=n_jobs)
    gamma = kde_based_rbf_gamma(D)
    return exp(-gamma * (D ** 2))


def auto_rbf_kernel_with_gamma(X, n_jobs=1):
    if X.ndim == 1:
        X = X[:, None]
    D = pairwise_distances(X, n_jobs=n_jobs)
    gamma = kde_based_rbf_gamma(D)
    return exp(-gamma * (D ** 2)), gamma


def mle_score_cv_2(gamma, X, D2):
    LOWER_BOUND = 1e-10
    # adding
    train, test = X[:len(X) // 2], X[len(X) // 2:]
    K = exp(-gamma * D2[np.ix_(test, train)])  # kernel matrix
    return -np.sum(log(LOWER_BOUND + np.sum(K, axis=1) / sqrt(0.5 / gamma)))


def kde_based_rbf_gamma(D, nn=5, cutoff=None, init_x=1.0, repeat=5, seed=None, summarizer=np.median):
    """Compute the gamma parameter for a RBF kernel based on cross validation given an array of float numbers"""
    if seed is not None:
        np.random.seed(seed)
    if cutoff is None:  #
        cutoff = len(D) // 2
    idxs = np.arange(len(D))
    squared_D = D ** 2

    results = []
    for _ in range(repeat):
        Xs = [None] * nn
        for i in range(nn):
            np.random.shuffle(idxs)
            Xs[i] = idxs.copy() if len(idxs) <= cutoff else choice(idxs, cutoff, False)

        def func(ga):
            return sum(mle_score_cv_2(max(1e-10, ga[0]), X, squared_D) for X in Xs)

        res = scipy.optimize.minimize(func, np.array([init_x]), method='BFGS')
        results.append(res.x[0])
        init_x = res.x[0]
    # median is okay.
    # but min gives us an acceptable smooth(est) setting.
    if callable(summarizer):
        return summarizer(results)
    elif isinstance(summarizer, float):
        assert 0.0 <= float < 1.0
        return results[min(int(round(repeat * summarizer)), repeat - 1)]
    else:
        raise ValueError('unknown summarizer')


def p_value_of(val, data, approxmation=False, sorted=False) -> float:
    """The percentile of a value given a data

    Parameters
    ----------
    val: float
        value to compute p-value
    data: array_like
        data representing a reference distribution
    approxmation: bool
        if approximated, gamma distribution is used to approximate right-tail distribution.

    """

    if not sorted:
        data = np.sort(data)

    if approxmation:
        assert len(data) >= 1000, 'not enough data (n={}) to confidently approximate'.format(len(data))
        if data[-50] < val:  # when evidence becomes 'sparse'
            last_50_p = p_value_of(data[-50], data, approxmation=False, sorted=True)  # fine tuning...
            return expon.sf(val, *expon.fit(data[-50:], floc=data[-50])) * last_50_p

    low = np.searchsorted(data, val)
    high = np.searchsorted(data, val, side='right')
    if low == high:  # if not found    3 in [1,2,4,5] = 0.5    (2,2)
        return 1.0 - ((low + 0.5) / (len(data) + 1))
    else:  # if found                  3 in [1,2,3,4,5] = 0.5  (2,3)
        return 1.0 - (0.5 * (low + high) / len(data))


def median_except_diag(D):
    """Median value except diagonal values"""
    if D.ndim != 2:
        raise TypeError('not a matrix')
    if D.shape[0] != D.shape[1]:
        raise TypeError('not a square matrix')
    if len(D) <= 1:
        raise ValueError('No non-diagonal element')

    mask = np.ones(D.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    return np.median(D[mask])


def submatrix(mat, rows, cols=None) -> np.ndarray:
    """A submatrix of the given rows and columns."""
    if cols is None:
        cols = rows
    return mat[np.ix_(rows, cols)]


def is_square_matrix(x) -> bool:
    """whether x is a square matrix """
    return isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[0] == x.shape[1]


def random_seeds(n=None):
    """Random seeds of given size or a random seed if n is None"""
    if n is None:
        return np.random.randint(np.iinfo(np.int32).max)
    else:
        return [np.random.randint(np.iinfo(np.int32).max) for _ in range(n)]


def K2D(K: np.ndarray) -> np.ndarray:
    """A distance matrix D of the same size of the given kernel matrix K
     :math:`d^2(i,j)=k(i,i)+k(j,j)-2k(i,j)`.
    """
    if K is None:
        return None

    Kd = repmat(diag(K).reshape((len(K), 1)), 1, len(K))
    temp = Kd + Kd.transpose() - 2 * K
    min_val = np.min(temp)
    if min_val < 0.0:
        import warnings
        warnings.warn('K2D: negative values will be ignored. Observed: {}'.format(min_val))
        temp *= (temp > 0)
    return np.sqrt(temp)


def split(n):
    return split_1_to_r(n, 1)


def split_1_to_r(n, ratio):
    selector = np.random.choice(n, int(n / (1 + ratio)), False)  # choose n from 2n
    other = np.array(list(set(range(n)) - set(selector)))
    np.random.shuffle(other)
    return selector, other


def mean_with_without_diag(D):
    if D.ndim != 2:
        raise TypeError('not a matrix')
    if D.shape[0] != D.shape[1]:
        raise TypeError('not a square matrix')
    if len(D) <= 1:
        raise ValueError('No non-diagonal element')

    n = len(D)
    total_sum = np.sum(D)
    diagonal_sum = np.trace(D)
    return total_sum / (n * n), (total_sum - diagonal_sum) / (n * (n - 1))


def mean_without_diag(D):
    if D.ndim != 2:
        raise TypeError('not a matrix')
    if D.shape[0] != D.shape[1]:
        raise TypeError('not a square matrix')
    if len(D) <= 1:
        raise ValueError('No non-diagonal element')

    n = len(D)
    total_sum = np.sum(D)
    diagonal_sum = np.trace(D)
    return (total_sum - diagonal_sum) / (n * (n - 1))


def mmd_and_k(kx, ky, kz, idx1, idx2, Pidx1, Pidx2, type1=None, type2=None):
    # meshes
    _11 = np.ix_(idx1, idx1)
    _22 = np.ix_(idx2, idx2)
    _P1P1 = np.ix_(Pidx1, Pidx1)
    _P2P2 = np.ix_(Pidx2, Pidx2)

    k11 = kx[_P1P1 if type1 == 'x' else _11] * \
          ky[_P1P1 if type1 == 'y' else _11] * \
          kz[_P1P1 if type1 == 'z' else _11]

    k22 = kx[_P2P2 if type2 == 'x' else _22] * \
          ky[_P2P2 if type2 == 'y' else _22] * \
          kz[_P2P2 if type2 == 'z' else _22]

    k12 = kx[np.ix_(Pidx1 if type1 == 'x' else idx1,
                    Pidx2 if type2 == 'x' else idx2)] * \
          ky[np.ix_(Pidx1 if type1 == 'y' else idx1,
                    Pidx2 if type2 == 'y' else idx2)] * \
          kz[np.ix_(Pidx1 if type1 == 'z' else idx1,
                    Pidx2 if type2 == 'z' else idx2)]

    mmd = np.mean(k11) + np.mean(k22) - 2 * np.mean(k12)
    new_k = np.bmat([[k11, k12], [k12.T, k22]])
    return mmd, new_k


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def data_gen_old(n, seed, slope=0.0, skip_kernel=False):
    np.random.seed(seed)

    z = normalize(randn(n, 1))
    y = normalize(z + 0.3 * randn(n, 1))
    x = normalize(z + 0.3 * randn(n, 1) + slope * y)

    if skip_kernel:
        return x, y, z
    else:
        kx = auto_rbf_kernel(x)
        ky = auto_rbf_kernel(y)
        kz = auto_rbf_kernel(z)
        return kx, ky, kz, x, y, z


def data_gen_other(n, seed, slope=0.0, slope_xz=1.0, slope_yz=1.0, skip_kernel=False):
    # X --> Z --> Y
    # X <-- Z --> Y
    np.random.seed(seed)

    z = randn(n, 1)
    x = slope_xz * z ** 2 + 0.4 * randn(n, 1)
    y = slope_yz * np.cos(z) + slope * np.sqrt(np.abs(x)) + 0.2 * randn(n, 1)

    if skip_kernel:
        return x, y, z
    else:
        kx = auto_rbf_kernel(x)
        ky = auto_rbf_kernel(y)
        kz = auto_rbf_kernel(z)
        return kx, ky, kz, x, y, z


def data_gen_one(n, seed, slope=0.0, slope_xz=1.0, slope_yz=1.0, skip_kernel=False):
    # X --> Z --> Y
    # X <-- Z --> Y
    np.random.seed(seed)

    z = randn(n, 1)
    y = slope_yz * z ** 2 + 0.4 * randn(n, 1)
    x = slope_xz * np.cos(z) + slope * np.sqrt(np.abs(y)) + 0.2 * randn(n, 1)
    if skip_kernel:
        return x, y, z
    else:
        kx = auto_rbf_kernel(x)
        ky = auto_rbf_kernel(y)
        kz = auto_rbf_kernel(z)
        return kx, ky, kz, x, y, z
