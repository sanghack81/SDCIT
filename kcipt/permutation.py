import numpy as np

from kcipt.cython_impl.cy_kcipt import cy_split_permutation


def permuted(D):
    out = np.zeros((len(D),), 'int32')
    cy_split_permutation(D, out)
    return out
