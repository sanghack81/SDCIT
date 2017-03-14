from itertools import chain

import numpy as np
from numpy import zeros, allclose, ix_, diag
from numpy.random import randint

from kcipt.blossom_v.cy_blossom_v import cy_blossom_v, cy_post_2_2_2_to_3_3, cy_post_2_3_to_5
from kcipt.utils import safe_iter, K2D


def permuted(idx, K=None, D=None, with_post=True):
    assert K is not None or D is not None
    Pidx = idx.copy()
    if D is None:
        D = K2D(K)
    rows, cols = np.nonzero(blossom_permutation(D, with_post))
    Pidx[rows] = idx[cols]
    return Pidx


def slim_permuted(idx, K=None, D=None, with_post=True):
    assert K is not None or D is not None
    if D is None:
        D = K2D(K)
    perm_vector = slim_blossom_permutation(D, with_post)
    return idx[perm_vector]


def _sample_integer_except(n: int, exclude: int) -> int:
    """Sample an integer between 0 (inclusive) and n (exclusive) except the given value"""
    if n <= 0:
        raise ValueError('n must be strictly positive integer')
    if n == 1 and exclude == 0:
        raise ValueError('cannot exclude only available integer: 0')

    at = randint(n)
    while at == exclude:
        at = randint(n)
    return at


def split_permutation(D, perm_comp, **options):
    """Delegate acquiring permutation matrix based on a divide-and-conquer approach to ensure perm_comp only computes
    a permutation matrix of a connected components.

    Parameters
    ----------
    D: np.ndarray
        a distance matrix which might contain positive infinity

    perm_comp: callable
        a function to compute permutation matrix given a distance matrix with given options

    options: dict
        options for perm_comp

    Notes
    -----
    This ensures that permutation computation is based on a distance matrix without infinity.
    """

    # assuming x -- inf -- y, y -- not inf -- z, then, z -- inf -- x
    remains = set(range(len(D)))
    components = set()
    for i in safe_iter(remains):
        comp = frozenset(np.where(D[i, :] < float('inf'))[0])
        components.add(comp)
        remains -= comp

    P = zeros((len(D), len(D)), 'int')
    for comp in components:
        comp = list(comp)
        # impossible to permute
        if len(comp) == 1:
            P[comp[0], comp[0]] = 1
        else:
            mesh = ix_(comp, comp)
            temp = D[mesh]
            assert not np.any(temp == float('inf'))
            assert not np.any(np.isnan(temp))
            P[mesh] = perm_comp(temp, **options)

    return P


def random_permutation(n, **unused_options) -> np.ndarray:
    """ A random permutation matrix of n by n"""
    if unused_options:
        print('ignored: {}'.format(unused_options))
    if n == 1:  # relaxed condition for permutation matrix
        return _perm_to_P([0])
    elif n == 2:
        return _perm_to_P([1, 0])
    elif n == 3:
        return _perm_to_P([1, 2, 0])

    perm = np.random.permutation(n)
    for i in filter(lambda j: perm[j] == j, range(n)):
        to_swap = _sample_integer_except(n, i)
        perm[i] = perm[to_swap]
        perm[to_swap] = i

    return _perm_to_P(perm)


def __flip_coin():
    return np.random.rand() < 0.5


def _perm_to_P(perm):
    n = len(perm)
    P = zeros((n, n), dtype=int)
    P[range(n), perm] = 1
    return P


def is_valid_P(P):
    if len(P) == 1:
        return P[0, 0] == 1
    else:
        return P is not None and \
               _all_ones(np.sum(P, 1)) and \
               _all_ones(np.sum(P, 0)) and \
               _all_zeros(diag(P))


def _all_zeros(x):
    return allclose(x, 0, rtol=0, atol=0)


def _all_ones(x):
    return allclose(x, 1, rtol=0, atol=0)


def blossom_permutation(D, with_post=True):
    """A permutation matrix, which first initialized by minimum-cost perfect matching using Blossom V algorithm.

    """
    _validate_distance_matrix(D)

    n = len(D)
    if np.sum(D) == 0:
        return random_permutation(len(D))

    if n == 1:  # relaxed
        return _perm_to_P([0])
    if n == 2:
        return _perm_to_P([1, 0])
    if n == 3:
        return _perm_to_P([1, 2, 0])

    p2s = []
    p3s = []
    p5s = []
    if n % 2:
        # take off the last one.
        m = n - 1
        perm_array = _execute_blossom_v(D[:-1, :-1])
        for i in perm_array:
            if i < perm_array[i]:
                p2s.append((i, perm_array[i]))

        dist_to_m = float('inf')
        near_to_m = -1
        for i in range(m):
            if D[m, i] + D[m, perm_array[i]] < dist_to_m:
                dist_to_m = D[m, i] + D[m, perm_array[i]]
                near_to_m = i

        i, j = near_to_m, perm_array[near_to_m]
        assert i < j
        p2s.remove((i, j))
        p3s.append(tuple(sorted([i, j, m])))
    else:
        perm_array = _execute_blossom_v(D)
        for i in perm_array:
            if i < perm_array[i]:
                p2s.append((i, perm_array[i]))

    if with_post:
        _post_2_2_2_to_3_3(D, p2s, p3s)
        p5s = _post_2_3_to_5(D, p2s, p3s)

    perm = zeros((n, n), dtype='int')
    for a, b in p2s:
        perm[b, a] = perm[a, b] = 1
    for a, b, c in p3s:
        perm[a, b] = perm[b, c] = perm[c, a] = 1
    for v, w, x, y, z in p5s:
        perm[v, w] = perm[w, x] = perm[x, y] = perm[y, z] = perm[z, v] = 1

    assert is_valid_P(perm)
    return perm


def slim_blossom_permutation(D, with_post=True):
    """A permutation vector, which first initialized by minimum-cost perfect matching using Blossom V algorithm.

    """
    n = len(D)
    assert (n % 2) == 0

    if n == 1:  # relaxed
        return _perm_to_P([0])
    if n == 2:
        return _perm_to_P([1, 0])
    if n == 3:
        return _perm_to_P([1, 2, 0])

    p2s = []
    p3s = []
    p5s = []
    perm_array = _execute_blossom_v(D)
    for i in perm_array:
        if i < perm_array[i]:
            p2s.append((i, perm_array[i]))

    if with_post:
        _post_2_2_2_to_3_3(D, p2s, p3s)
        p5s = _post_2_3_to_5(D, p2s, p3s)

    perm_vector = zeros((n,), dtype='int')
    for a, b in p2s:
        perm_vector[a] = b
        perm_vector[b] = a
    for a, b, c in p3s:
        perm_vector[a] = b
        perm_vector[b] = c
        perm_vector[c] = a
    for v, w, x, y, z in p5s:
        perm_vector[v] = w
        perm_vector[w] = x
        perm_vector[x] = y
        perm_vector[y] = z
        perm_vector[z] = v

    return perm_vector


def _validate_distance_matrix(D: np.ndarray):
    # if len(D) < 2:
    #     raise ValueError("There is no permutation for {} data point.".format(len(D)))
    if not _all_zeros(diag(D)):
        raise ValueError('Distance between the same data points must be 0.')
    if not np.all(D >= 0):  # TODO faster?
        raise ValueError('Negative is not allowed as distance .')
    if np.max(D) == float('inf'):
        raise ValueError('Infinity is not allowed as distance.')
    if np.isnan(np.sum(D)):
        raise ValueError('NaN is not allowed.')


def _execute_blossom_v(D):
    assert len(D) >= 2 and not len(D) % 2
    D = np.ascontiguousarray(D, 'float64')
    summed = np.sum(D)
    if summed > 0:
        D = (2 ** 30 / summed) * D
    perm_array = np.zeros((len(D),), dtype='int32')

    cy_blossom_v(D, perm_array)
    return perm_array


# new one
# greedy! greedy!
def _post_2_2_2_to_3_3(D, p2s: list, p3s: list):
    output = np.zeros((len(p2s) // 3, 6), 'int32')
    cy_post_2_2_2_to_3_3(np.ascontiguousarray(D, 'float64'),
                         np.ascontiguousarray(list(chain(*p2s)), 'int32'),
                         output
                         )
    for a, b, c, d, e, f in output:
        if a == 0 and b == 0:
            break
        p2s.remove((a, b) if a < b else (b, a))
        p2s.remove((c, d) if c < d else (d, c))
        p2s.remove((e, f) if e < f else (f, e))

        p3s.append(tuple(sorted([a, c, d])))
        p3s.append(tuple(sorted([b, e, f])))


def _post_2_3_to_5(D, p2s, p3s):
    p5s = []
    output = np.zeros((len(p3s), 11), 'int32')
    cy_post_2_3_to_5(np.ascontiguousarray(D, 'float64'),
                     np.ascontiguousarray(list(chain(*p2s)), 'int32'),
                     np.ascontiguousarray(list(chain(*p3s)), 'int32'),
                     output
                     )
    for a, b, c, d, e, v, w, x, y, z, v in output:
        if a == 0 and b == 0:
            break
        p2s.remove(tuple(sorted([a, b])))
        p3s.remove(tuple(sorted([c, d, e])))
        p5s.append((v, w, x, y, z))

    return p5s
