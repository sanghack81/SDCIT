cimport numpy as np
import cython
import numpy as np

cdef extern from "KCIPT.h":
    void c_adj_kcipt(const int seed, const double * K_X, const double *K_Y, const double *K_Z, const double *K_XYZ, const double *D_Z,
                     const int n, const int B, const int b, const int M, const int n_threads,
                     double *const test_statistic, double *const inner_null, double *const outer_null, const int variance_reduced);
    void c_kcipt(const double *K_X, const double *K_Y, const double *K_Z, const double *D_Z,
                 const int n, const int B, const int b,
                 double *const inner_null, double *const mmds, const int seed, const int n_threads,
                 double *const outer_null, const int M);
    void threaded_null_distribution(const unsigned int seed, const double *K_XYZ, const int n, double *nulls, const int b, const int n_threads);
    void bootstrap_null(const int seed, const int B, const int b, const int M, double *const inner_null, double *const outer_null);
    void null_distribution(const unsigned int seed, const double *K_XYZ, const int n, double *nulls, const int offset, const int sub_b);
    void bootstrap_single_null(const int seed, const int B, const int b, const int M, double *const inner_null, double *const outer_null);

cdef extern from "permutation.h":
    void split_permutation_interface(double *D, const int full_n, int*perm);


cdef extern from "SDCIT.h":
    void c_sdcit(const double *K_XZ, const double *K_Y, const double *D_Z_, const int n,
             const int b, const int seed, const int n_threads,
             double *const mmsd, double *const null);



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

    c_kcipt(&K_X[0, 0], &K_Y[0, 0],&K_Z[0, 0],&D_Z[0, 0] if D_Z is not None else NULL, ll, B, b, &inner_null[0, 0], &mmds[0], seed, n_threads, &outer_null[0], how_many)



@cython.boundscheck(False)
@cython.wraparound(False)
def cy_adj_kcipt(np.ndarray[double, ndim=2, mode="c"] K_X not None,
             np.ndarray[double, ndim=2, mode="c"] K_Y not None,
             np.ndarray[double, ndim=2, mode="c"]  K_Z not None,
             np.ndarray[double, ndim=2, mode="c"]  K_XYZ not None,
             np.ndarray[double, ndim=2, mode="c"]  D_Z,
             int B,
             int b,
             np.ndarray[double, ndim=1, mode="c"]  inner_null not None,
             np.ndarray[double, ndim=1, mode="c"]  mmds not None,
             int seed,
             int n_threads,
             np.ndarray[double, ndim=1, mode="c"]  outer_null not None,
             int how_many,
             int variance_reduced,
             ):
    cdef int ll
    ll = K_X.shape[0]

    c_adj_kcipt(seed, &K_X[0, 0], &K_Y[0, 0],&K_Z[0, 0], &K_XYZ[0, 0], &D_Z[0, 0] if D_Z is not None else NULL,
                ll, B, b, how_many, n_threads,
                &mmds[0], &inner_null[0], &outer_null[0], variance_reduced)




@cython.boundscheck(False)
@cython.wraparound(False)
def cy_null_distribution(int seed,
                         np.ndarray[double, ndim=2, mode="c"] K_XYZ not None,
                         int n,
                         np.ndarray[double, ndim=1, mode="c"] nulls,
                         int b
                         ):
    cdef int ll
    ll = K_XYZ.shape[0]
    null_distribution(seed, &K_XYZ[0, 0], ll, &nulls[0], 0, b)

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_bootstrap_single_null(int seed,
                             int B,
                             int b,
                             int M,
                             np.ndarray[double, ndim=1, mode="c"] inner_null,
                             np.ndarray[double, ndim=1, mode="c"] outer_null,
                             ):
    bootstrap_single_null(seed, B, b, M, &inner_null[0], &outer_null[0])

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def cy_bootstrap_null(int seed,
#                              int B,
#                              int b,
#                              int M,
#                              np.ndarray[double, ndim=1, mode="c"] inner_null,
#                              np.ndarray[double, ndim=1, mode="c"] outer_null,
#                              ):
#     bootstrap_null(seed, B, b, M, &inner_null[0], &outer_null[0])


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
def cy_sdcit(np.ndarray[double, ndim=2, mode="c"] K_XZ not None,
             np.ndarray[double, ndim=2, mode="c"] K_Y not None,
             np.ndarray[double, ndim=2, mode="c"]  D_Z,
             int b,
             int seed,
             int n_threads,
             np.ndarray[double, ndim=1, mode="c"]  mmsd not None,
             np.ndarray[double, ndim=1, mode="c"]  null not None
             ):
    cdef int ll
    ll = K_XZ.shape[0]

    c_sdcit(&K_XZ[0, 0], &K_Y[0, 0], &D_Z[0, 0], ll, b, seed, n_threads, &mmsd[0], &null[0])


