import cython
import numpy as np
cimport numpy as np

cdef extern from "KCIPT.h":
    void c_kcipt(const double *K_X, const double *K_Y, const double *K_Z, const double *D_Z,
                 const int n, const int B, const int b,
                 double *const inner_null, double *const mmds, const int seed, const int n_threads,
                 double *const outer_null, const int M);

cdef extern from "permutation.h":
    void split_permutation_interface(const double *D, const int full_n, int*perm, const int seed);
    void dense_2n_permutation_interface(const double *D, const int full_n, int *perm, const int seed);

cdef extern from "SDCIT.h":
    void c_sdcit(const double * const K_XZ, const double * const K_Y, const double * const K_Z, const double * const D_Z_, const int n,
              const int b, const int seed, const int n_threads,
              double *const mmsd, double *const error_mmsd, double *const null, double *const error_null);

cdef extern from "HSIC.h":
    void c_hsic(const double *const K_X, const double *const K_Y, const int n, const int b, const int seed, const int n_threads, double *const test_statistic, double *const null);


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_kcipt(double[:, ::1] K_X,
             double[:, ::1] K_Y,
             double[:, ::1] K_Z,
             double[:, ::1] D_Z,
             int B,
             int b,
             double[:, ::1] inner_null,
             double[::1] mmds,
             int seed,
             int n_threads,
             double[::1] outer_null,
             int how_many
             ):
    cdef int ll
    ll = K_X.shape[0]

    c_kcipt(&K_X[0, 0], &K_Y[0, 0], &K_Z[0, 0], &D_Z[0, 0] if D_Z is not None else NULL, ll, B, b, &inner_null[0, 0], &mmds[0], seed, n_threads, &outer_null[0], how_many)

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_split_permutation(double[:, ::1] D,
                         int[::1] perm,
                         int seed
                         ):
    cdef int ll
    ll = D.shape[0]

    split_permutation_interface(&D[0, 0], ll, &perm[0], seed)

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_dense_permutation(double[:, ::1] D,
                         int[::1] perm,
                         int seed
                         ):
    cdef int ll
    ll = D.shape[0]

    dense_2n_permutation_interface(&D[0, 0], ll, &perm[0], seed)

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_sdcit(double[:, ::1] K_XZ,
             double[:, ::1] K_Y,
             double[:, ::1] K_Z,
             double[:, ::1] D_Z,
             int b,
             int seed,
             int n_threads,
             double[::1] mmsd,
             double[::1] error_mmsd,
             double[::1] null,
             double[::1] error_null
             ):
    cdef int ll
    ll = K_XZ.shape[0]

    c_sdcit(&K_XZ[0, 0], &K_Y[0, 0], &K_Z[0, 0], &D_Z[0, 0] if D_Z is not None else NULL, ll, b, seed, n_threads, &mmsd[0], &error_mmsd[0], &null[0], &error_null[0])


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_hsic(double[:, ::1] Kc,
            double[:, ::1] Lc,
            int b,
            int seed,
            int n_threads,
            double[::1] test_statistic,
            double[::1] null
            ):
    cdef int ll
    ll = Kc.shape[0]

    c_hsic(&Kc[0, 0], &Lc[0, 0], ll, b, seed, n_threads, &test_statistic[0], &null[0])
