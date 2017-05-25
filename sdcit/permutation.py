import numpy as np

from sdcit.cython_impl.cy_kcipt import cy_split_permutation, cy_dense_permutation


def permuted(D, dense=True):
    out = np.zeros((len(D),), 'int32')
    if dense:
        cy_dense_permutation(D, out)
    else:
        cy_split_permutation(D, out)
    return out


def pathgrow(D, full=False):
    n = len(D)
    for i in range(n):
        D[i, i] = float('inf')  # TODO do not change original matrix...
    nearest_neighbors = np.argsort(D, axis=1)
    d_nearest_neighbors = np.sort(D, axis=1)
    threshold = float('inf') if full else np.median(D)

    small_graph = dict()
    for i in range(n):
        small_graph[i] = list(nearest_neighbors[i, d_nearest_neighbors[i, :] < threshold])
    # not undirected!!
    m1 = [None] * n
    m2 = [None] * n
    flag = False
    while small_graph:
        x = np.random.choice(list(small_graph.keys()), 1)[0]
        while small_graph[x]:
            y = small_graph[x][0]
            if flag:
                m1[x] = y
                m1[y] = x
                flag = not flag
            else:
                m2[x] = y
                m2[y] = x
                flag = not flag
            for z in small_graph.pop(x):
                small_graph[z].remove(x)
            x = y
        small_graph.pop(x)
    s1 = 0.0
    for i, j in enumerate(m1):
        if j is not None:
            s1 += D[i, j]
    s2 = 0.0
    for i, j in enumerate(m2):
        if j is not None:
            s2 += D[i, j]
    if s1 < s2:
        return m1
    else:
        return m2


def greedy_perfect_matching(D, full=False):
    """Dense Greedy Perfect Matching"""
    n = len(D)
    dists = [None] * ((n * (n - 1)) // 2)
    k = 0
    threshold = float('inf') if full else np.median(D)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < threshold:
                dists[k] = (D[i, j], i, j)
                k += 1
    dists = dists[:k]
    dists = sorted(dists)
    len_dists = len(dists)

    at = 0
    perm = [None] * n
    filled = set()

    while at < len_dists and len(filled) < n:  # what if odd-sized?
        d, i, j = dists[at]
        if perm[i] is None and perm[j] is None:
            perm[i] = j
            perm[j] = i
            filled.add(i)
            filled.add(j)
        at += 1
    # assert len(filled) >= n - 1
    return perm, n - len(filled)


def greedy_perm(D, full=False):
    n = len(D)
    dists = [None] * ((n * (n - 1)) // 2)
    k = 0
    threshold = float('inf') if full else np.median(D)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < threshold:
                dists[k] = (D[i, j], i, j)
                k += 1
    dists = dists[:k]
    dists = sorted(dists)
    len_dists = len(dists)

    at = 0
    perm = [None] * n
    filled = set()

    waiting = dict()
    while at < len_dists and len(filled) < n:  # what if odd-sized?
        d, i, j = dij = dists[at]
        if perm[i] is None and perm[j] is None:
            perm[i] = j
            perm[j] = i
            filled.add(i)
            filled.add(j)
        elif perm[j] is None or perm[i] is None:
            if perm[i] is None:
                j, i = i, j
            if j in waiting:
                _, k, l = waiting.pop(j)
                if {perm[i], j} == {k, l} and perm[perm[i]] == i:
                    perm[i], perm[j], perm[k] = j, k, i
                    filled.add(j)
            else:
                if perm[perm[i]] == i:
                    waiting[j] = dij
        at += 1
    # assert len(filled) >= n - 1
    return perm, n - len(filled)

#
# def sparsify(D: np.ndarray, min_degree=5):
#     n = len(D)
#     if n <= 20:
#         return D
#     dists = [(D[i, j], i, j) for i in range(n) for j in range(i + 1, n)]
#     dists = sorted(dists)
#
#     newD = np.ones(D.shape) * np.inf
#     for i in range(n):
#         newD[i, i] = 0.0
#
#     degrees = np.zeros(n)
#     at = 0
#     while at < len(dists):
#         d, i, j = dists[at]
#         at += 1
#         if degrees[i] < min_degree or degrees[j] < min_degree:
#             newD[j, i] = newD[i, j] = d
#             degrees[i] += 1
#             degrees[j] += 1
#
#     return newD
#
#
# def is_valid_P(P):
#     if len(P) == 1:
#         return P[0, 0] == 1
#     else:
#         return P is not None and \
#                _all_ones(np.sum(P, 1)) and \
#                _all_ones(np.sum(P, 0)) and \
#                _all_zeros(np.diag(P))
#
#
# def _all_zeros(x):
#     return np.allclose(x, 0, rtol=0, atol=0)
#
#
# def _all_ones(x):
#     return np.allclose(x, 1, rtol=0, atol=0)
#
#
# def permuted_cvxopt(D):
#     # TODO treat infinity
#     # TODO use less variable & constraints
#     n = len(D)
#     if n == 1:  # relaxed
#         return [0]
#     if n == 2:
#         return [1, 0]
#     if n == 3:
#         return [1, 2, 0]
#
#     mask = np.ones(D.shape, dtype=bool)
#     np.fill_diagonal(mask, 0)
#     flattened = D[mask].flatten()
#     inf_indices = np.isinf(flattened)
#     flattened[inf_indices] = 0
#     f = matrix(flattened)  #
#
#     nn = n * n
#     A_eq = np.zeros((n + n - 1, nn - n))
#     b_eq = np.ones((n + n - 1, 1))
#
#     for r in range(n):
#         offset = (n - 1) * r
#         A_eq[r, offset: offset + (n - 1)] = 1
#     for c in range(n - 1):
#         # first block (c-1, c-1 + n-1, ... ,c-1 + (n-1)*(c-1))
#         A_eq[c + n, (c - 1):(c - 1 + (n - 1) * (c - 1) + 1):(n - 1)] = 1  # +1 for inclusive
#         A_eq[c + n, ((c + 1) * (n - 1) + c)::(n - 1)] = 1
#
#     A_eq = np.vstack([A_eq, np.zeros([1, nn - n])])
#     b_eq = np.vstack([b_eq, np.zeros([1, 1])])
#     A_eq[-1, inf_indices] = 1
#
#     # minimize
#     # with Ax >= b
#     x = variable(nn - n)
#     A = matrix(A_eq)
#     b = matrix(b_eq)
#     inequality1 = (A * x >= b)
#     inequality2 = (A * x <= b)
#     positivity = (x >= 0)
#     c = matrix(f, tc='d')
#     lp2 = op(dot(c, x), constraints=[inequality1, inequality2, positivity])
#     lp2.solve(solver='glpk')
#
#     if x.value:
#         P_ = np.reshape(x.value, (n, n - 1))
#         Pu = np.triu(P_, 0)
#         Pu = np.hstack((np.zeros((n, 1)), Pu))
#
#         Pl = np.tril(P_, -1)
#         Pl = np.hstack((Pl, np.zeros((n, 1))))
#
#         P = Pu + Pl
#         outP = np.round(P).astype(int)
#         # TODO refine solution
#         return outP if is_valid_P(outP) else None
#     else:
#         return None
#
#
# def sparse_permuted_cvxopt(D):
#     # TODO treat infinity
#     # TODO use less variable & constraints
#     n = len(D)
#     if n == 1:  # relaxed
#         return [0]
#     if n == 2:
#         return [1, 0]
#     if n == 3:
#         return [1, 2, 0]
#
#     mask = np.ones(D.shape, dtype=bool)
#     np.fill_diagonal(mask, 0)
#     flattened = D[mask].flatten()
#     inf_indices = np.isinf(flattened)
#     flattened[inf_indices] = 0
#     f = matrix(flattened)  #
#
#     nn = n * n
#     A_eq = np.zeros((n + n - 1, nn - n))
#     b_eq = np.ones((n + n - 1, 1))
#
#     for r in range(n):
#         offset = (n - 1) * r
#         A_eq[r, offset: offset + (n - 1)] = 1
#     for c in range(n - 1):
#         # first block (c-1, c-1 + n-1, ... ,c-1 + (n-1)*(c-1))
#         A_eq[c + n, (c - 1):(c - 1 + (n - 1) * (c - 1) + 1):(n - 1)] = 1  # +1 for inclusive
#         A_eq[c + n, ((c + 1) * (n - 1) + c)::(n - 1)] = 1
#
#     A_eq = np.vstack([A_eq, np.zeros([1, nn - n])])
#     b_eq = np.vstack([b_eq, np.zeros([1, 1])])
#     A_eq[-1, inf_indices] = 1
#
#     # minimize
#     # with Ax >= b
#     x = variable(nn - n)
#     A = matrix(A_eq)
#     b = matrix(b_eq)
#     inequality1 = (A * x >= b)
#     inequality2 = (A * x <= b)
#     positivity = (x >= 0)
#     c = matrix(f, tc='d')
#     lp2 = op(dot(c, x), constraints=[inequality1, inequality2, positivity])
#     lp2.solve(solver='glpk')
#
#     if x.value:
#         P_ = np.reshape(x.value, (n, n - 1))
#         Pu = np.triu(P_, 0)
#         Pu = np.hstack((np.zeros((n, 1)), Pu))
#
#         Pl = np.tril(P_, -1)
#         Pl = np.hstack((Pl, np.zeros((n, 1))))
#
#         P = Pu + Pl
#         outP = np.round(P).astype(int)
#         # TODO refine solution
#         return outP if is_valid_P(outP) else None
#     else:
#         return None
#
#
# if __name__ == '__main__':
#     cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
#
#     np.random.seed(0)
#     for n in [100]:
#         x = np.random.randn(n, n)
#         x = np.abs(x)
#         # for i in range(len(x)):
#         #     x[np.random.randint(len(x)), np.random.randint(len(x))] = np.inf
#         x = x + x.T
#         for i in range(len(x)):
#             x[i, i] = 0.0
#
#         xx = time.time()
#         outp = permuted_cvxopt(x)
#         rows, cols = np.nonzero(outp)
#         p = [c for r, c in sorted(zip(rows, cols))]
#         print(n, time.time() - xx, p)
