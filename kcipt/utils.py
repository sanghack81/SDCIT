import numpy as np
import numpy.ma as ma
from numpy import diag, exp
from numpy.matlib import repmat
from sklearn.metrics import euclidean_distances


def median_heuristic(data, *args):
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

    return outs


def safe_iter(iterable, sort=False):
    """Iterator (generator) that can skip removed items."""
    copied = list(iterable)
    if sort:
        copied = sorted(copied)
    for y in copied:
        if y in iterable:
            yield y


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
        import warnings
        warnings.warn('K2D: negative values will be ignored. Observed: {}'.format(min_val))
        temp *= (temp > 0)
    return np.sqrt(temp)
