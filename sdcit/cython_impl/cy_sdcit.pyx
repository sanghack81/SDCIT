import cython
import numpy as np
cimport numpy as np

cdef extern from "KCIPT.h":
    void c_kcipt(const double *K_X, const double *K_Y, const double *K_Z, const double *D_Z,
                 const int n, const int B, const int b,
                 double *const inner_null, double *const mmds, const int seed, const int n_threads,
                 double *const outer_null, const int M);

cdef extern from "permutation.h":
    void split_permutation_interface(double *D, const int full_n, int*perm);
    void dense_2n_permutation_interface(const double *D, const int full_n, int *perm);

cdef extern from "SDCIT.h":
    void c_sdcit(const double * const K_XZ, const double * const K_Y, const double * const K_Z, const double * const D_Z_, const int n,
              const int b, const int seed, const int n_threads,
              double *const mmsd, double *const error_mmsd, double *const null, double *const error_null);

cdef extern from "HSIC.h":
    void c_hsic(const double *const K_X, const double *const K_Y, const int n, const int b, const int seed, const int n_threads, double *const test_statistic, double *const null);


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_kcipt(np.ndarray[double, ndim=2, mode="c"] K_X not None,
             np.ndarray[double, ndim=2, mode="c"] K_Y not None,
             np.ndarray[double, ndim=2, mode="c"] K_Z not None,
             np.ndarray[double, ndim=2, mode="c"]  D_Z,
             int B,
             int b,
             np.ndarray[double, ndim=2, mode="c"]  inner_null not None,
             np.ndarray[double, ndim=1, mode="c"]  mmds not None,
             int seed,
             int n_threads,
             np.ndarray[double, ndim=1, mode="c"]  outer_null not None,
             int how_many
             ):
    cdef int ll
    ll = K_X.shape[0]

    c_kcipt(&K_X[0, 0], &K_Y[0, 0], &K_Z[0, 0], &D_Z[0, 0] if D_Z is not None else NULL, ll, B, b, &inner_null[0, 0], &mmds[0], seed, n_threads, &outer_null[0], how_many)

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_split_permutation(np.ndarray[double, ndim=2, mode="c"]  D not None,
                         np.ndarray[int, ndim=1, mode="c"]  perm not None
                         ):
    cdef int ll
    ll = D.shape[0]

    split_permutation_interface(&D[0, 0], ll, &perm[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_dense_permutation(np.ndarray[double, ndim=2, mode="c"]  D not None,
                         np.ndarray[int, ndim=1, mode="c"]  perm not None
                         ):
    cdef int ll
    ll = D.shape[0]

    dense_2n_permutation_interface(&D[0, 0], ll, &perm[0])

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_sdcit(np.ndarray[double, ndim=2, mode="c"] K_XZ not None,
              np.ndarray[double, ndim=2, mode="c"] K_Y not None,
              np.ndarray[double, ndim=2, mode="c"] K_Z not None,
              np.ndarray[double, ndim=2, mode="c"]  D_Z,
              int b,
              int seed,
              int n_threads,
              np.ndarray[double, ndim=1, mode="c"]  mmsd not None,
              np.ndarray[double, ndim=1, mode="c"]  error_mmsd not None,
              np.ndarray[double, ndim=1, mode="c"]  null not None,
              np.ndarray[double, ndim=1, mode="c"]  error_null not None
              ):
    cdef int ll
    ll = K_XZ.shape[0]

    c_sdcit(&K_XZ[0, 0], &K_Y[0, 0], &K_Z[0, 0], &D_Z[0, 0], ll, b, seed, n_threads, &mmsd[0], &error_mmsd[0], &null[0], &error_null[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_hsic(np.ndarray[double, ndim=2, mode="c"] Kc not None,
              np.ndarray[double, ndim=2, mode="c"] Lc not None,
              int b,
              int seed,
              int n_threads,
              np.ndarray[double, ndim=1, mode="c"]  test_statistic not None,
              np.ndarray[double, ndim=1, mode="c"]  null not None
              ):
    cdef int ll
    ll = Kc.shape[0]

    c_hsic(&Kc[0, 0], &Lc[0, 0], ll, b, seed, n_threads, &test_statistic[0], &null[0])
