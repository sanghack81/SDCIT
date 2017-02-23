//
// Created by Sanghack Lee on 8/19/16.
//

#ifndef C_KCIPT_PERMUTATION_H
#define C_KCIPT_PERMUTATION_H

#include <vector>
#include <random>

using std::vector;

vector<int> split_permutation(const double *D, const int full_n, const vector<int> &idxs, std::mt19937 &generator);

void split_permutation_interface(const double *D, const int full_n, int *perm);

#endif //C_KCIPT_PERMUTATION_H
