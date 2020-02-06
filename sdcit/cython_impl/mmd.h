//
// Created by Sanghack Lee on 8/19/16.
//

#ifndef C_KCIPT_MMD_H
#define C_KCIPT_MMD_H


#include "TwoSamples.h"

double u_mmd(const double *K_XYZ, const int full_n, const std::vector<int> &idx1, const std::vector<int> &idx2) {
    double k11 = 0.0, k22 = 0.0, k12 = 0.0;

    for (auto const &row_idx : idx1) {
        const double *k_row = &K_XYZ[full_n * row_idx];
        for (auto const &col_idx : idx1) {
            k11 += k_row[col_idx];
        }
        k11 -= k_row[row_idx];

        for (auto const &col_idx : idx2) {
            k12 += k_row[col_idx];
        }
    }

    for (auto const &row_idx : idx2) {
        const double *k_row = &K_XYZ[full_n * row_idx];
        for (auto const &col_idx : idx2) {
            k22 += k_row[col_idx];
        }
        k22 -= k_row[row_idx];
    }

    return k11 / (idx1.size() * idx1.size() - idx1.size())
           + k22 / (idx2.size() * idx2.size() - idx2.size())
           - 2 * k12 / (idx1.size() * idx2.size());
}


double u_mmd(const double *K_X, const double *K_Y, const double *K_Z, const int full_n,
             const std::vector<int> &idx1x, const std::vector<int> &idx1y, const std::vector<int> &idx1z,
             const std::vector<int> &idx2x, const std::vector<int> &idx2y, const std::vector<int> &idx2z) {
    double k11 = 0.0, k22 = 0.0, k12 = 0.0;

    auto idx1_size = idx1x.size();
    auto idx2_size = idx2x.size();

    for (size_t i = 0; i < idx1_size; i++) {
        const double *k_row_x = &K_X[full_n * idx1x[i]];
        const double *k_row_y = &K_Y[full_n * idx1y[i]];
        const double *k_row_z = &K_Z[full_n * idx1z[i]];
        for (size_t j = 0; j < idx1_size; j++) {
            k11 += k_row_x[idx1x[j]] * k_row_y[idx1y[j]] * k_row_z[idx1z[j]];
        }
        k11 -= k_row_x[idx1x[i]] * k_row_y[idx1y[i]] * k_row_z[idx1z[i]];

        for (size_t j = 0; j < idx2_size; j++) {
            k12 += k_row_x[idx2x[j]] * k_row_y[idx2y[j]] * k_row_z[idx2z[j]];
        }
    }

    for (size_t i = 0; i < idx2_size; i++) {
        const double *k_row_x = &K_X[full_n * idx2x[i]];
        const double *k_row_y = &K_Y[full_n * idx2y[i]];
        const double *k_row_z = &K_Z[full_n * idx2z[i]];

        for (size_t j = 0; j < idx2_size; j++) {
            k22 += k_row_x[idx2x[j]] * k_row_y[idx2y[j]] * k_row_z[idx2z[j]];
        }
        k22 -= k_row_x[idx2x[i]] * k_row_y[idx2y[i]] * k_row_z[idx2z[i]];
    }
    return k11 / (idx1_size * idx1_size - idx1_size)
           + k22 / (idx2_size * idx2_size - idx2_size)
           - 2 * k12 / (idx1_size * idx2_size);
}


double u_mmd(const double *K_X, const double *K_Y, const double *K_Z, const int full_n, const TwoSamples &ts) {
    return u_mmd(K_X, K_Y, K_Z, full_n, ts.idx1x, ts.idx1y, ts.idx1z, ts.idx2x, ts.idx2y, ts.idx2z);
}



#endif //C_KCIPT_MMD_H
