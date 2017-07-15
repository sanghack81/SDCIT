//
// Created by Sanghack Lee on 8/21/16.
//

#ifndef C_KCIPT_KCIPT_H
#define C_KCIPT_KCIPT_H

void c_kcipt(const double *K_X, const double *K_Y, const double *K_Z, const double *D_Z, const int n, const int B,
             const int b, double *const inner_null, double *const mmds, const int seed, const int n_threads,
             double *const outer_null, const int M);


void threaded_null_distribution(const unsigned int seed, const double *K_XYZ, const int n, double *nulls, const int b, const int n_threads);

void bootstrap_null(const int seed, const int B, const int b, const int M, double *const inner_null, double *const outer_null);

void null_distribution(const unsigned int seed, const double *K_XYZ, const int n, double *nulls, const int offset, const int sub_b);

void bootstrap_single_null(const int seed, const int B, const int b, const int M, double *const inner_null, double *const outer_null);

#endif //C_KCIPT_KCIPT_H
