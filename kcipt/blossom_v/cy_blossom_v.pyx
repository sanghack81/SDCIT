cimport numpy as np
import cython
import numpy as np

cdef extern from "c_cy_blossom_v.h":
    void c_cy_blossom_v(double*D, int*output, int n) nogil
    void c_cy_post_2_2_2_to_3_3(double*D, int*comps_of_2, int*abcdefs, int m, int n) nogil
    void c_cy_post_2_3_to_5(double*D, int*comps_of_2, int*comps_of_3, int*abcdes, int m2, int m3, int n) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_blossom_v(np.ndarray[double, ndim=2, mode="c"] D not None, np.ndarray[int, ndim=1, mode="c"] output not None):
    c_cy_blossom_v(&D[0, 0], &output[0], D.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_post_2_2_2_to_3_3(np.ndarray[double, ndim=2, mode="c"] D not None,
                         np.ndarray[int, ndim=1, mode="c"] comps_of_2 not None,
                         np.ndarray[int, ndim=2, mode="c"] abcdefs not None):
    c_cy_post_2_2_2_to_3_3(&D[0, 0], &comps_of_2[0], &abcdefs[0, 0], len(comps_of_2) / 2, D.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_post_2_3_to_5(np.ndarray[double, ndim=2, mode="c"] D not None,
                     np.ndarray[int, ndim=1, mode="c"] comps_of_2 not None,
                     np.ndarray[int, ndim=1, mode="c"] comps_of_3 not None,
                     np.ndarray[int, ndim=2, mode="c"] abcdes not None):
    c_cy_post_2_3_to_5(&D[0, 0], &comps_of_2[0], &comps_of_3[0], &abcdes[0, 0], len(comps_of_2) / 2,
                       len(comps_of_3) / 3, D.shape[0])
