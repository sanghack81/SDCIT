//
// Created by Sanghack Lee on 8/19/16.
//

#ifndef C_KCIPT_PERMUTATION_H
#define C_KCIPT_PERMUTATION_H

#include <vector>
#include <random>


std::vector<int> split_permutation(const double *D, const int full_n, const std::vector<int> &idxs, std::mt19937 &generator);
std::vector<int> dense_2n_permutation(const double *D, const int full_n, const std::vector<int> &idxs, std::mt19937 &generator);

void split_permutation_interface(const double *D, const int full_n, int *perm);
void dense_2n_permutation_interface(const double *D, const int full_n, int *perm);

#endif //C_KCIPT_PERMUTATION_H
