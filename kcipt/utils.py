import warnings

import mpmath
import numpy as np
import numpy.ma as ma
import sympy as sp
from numpy import diag, exp
from numpy.matlib import repmat
from scipy.stats import norm
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
        if min_val < -1e-15:
            warnings.warn('K2D: negative values will be treated as zero. Observed: {}'.format(min_val))
        temp *= (temp > 0)
    return np.sqrt(temp)


def chi_cdf(x, k):
    x, k = mpmath.mpf(x), mpmath.mpf(k)
    return mpmath.gammainc(k / 2, 0, x / 2, regularized=True)


def meta_analysis(ps, meta_analysis_method='fisher'):
    """Meta-analysis for combining p-values

    Returns
    -------
    A tuple of test statistic, p-value, and Z-score (or t-score) where Z-score can be None.

    References
    ----------
    http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/presentations/ohbm2013/Winkler_CombInf_OHBM2013.pdf
    """
    if isinstance(ps, float):
        return float('nan'), ps, float('nan')
    assert np.all(1.0 >= ps) and np.all(ps >= 0.0)

    if len(ps) == 1:
        return float('nan'), ps[0], float('nan')

    K = len(ps)
    Z = None
    if meta_analysis_method == 'fisher':  # <-- works okay
        # Fisher. Statistical Methods for Research Workers. Oliver and Boyd, 1932
        for p in ps:
            if p == 0:
                return float('nan'), 0.0, float('nan')
        S = -2 * np.sum(np.log(ps))
        with mpmath.extradps(50):
            p = float(1 - chi_cdf(S, 2 * K))
            # p = 1 - chi2.cdf(S, 2 * K)

    elif meta_analysis_method == 'tippett':
        # Tippett. The Methods of Statistics. Williams and Northgate, 1931
        S = np.min(ps)
        sp_S = sp.Float(S, prec=500)
        p = float(1 - (1 - sp_S) ** K)

    elif meta_analysis_method == 'stouffer':
        # Stouffer et al. Studies in Social Psychology in World War II. Princeton, 1949
        Z = S = np.sum(norm.ppf(1 - ps)) / np.sqrt(K)
        p = norm.sf(S)

    elif meta_analysis_method == 'friston':
        # Friston et al. Neuroimage. 1999;10(4):385-96
        S = np.max(ps)
        p = S ** K

    elif meta_analysis_method == 'shlee':
        minS, maxS = np.min(ps), np.max(ps)
        p1 = float(1 - (1 - sp.Float(minS, prec=500)) ** K)
        p2 = maxS ** K
        p = norm.sf((norm.isf(p1) + norm.isf(p2)) / 2)
        Z = S = norm.isf(p)

    elif meta_analysis_method == 'nichols':
        p = S = np.max(ps)

    elif meta_analysis_method == 'edgington':
        raise ValueError('not implemented due to numerical instability.')
    else:
        raise ValueError('unknown option: {}'.format(meta_analysis_method))

    return S, p, norm.isf(p) if Z is None else Z
