//
// Created by Sanghack Lee on 3/14/17.
//
//
#include <random>
#include <thread>
#include <set>
#include <iostream>
#include <algorithm>
#include "SDCIT.h"
#include "permutation.h"
//
using std::thread;
using std::vector;
using std::pair;
using std::mt19937;



std::pair<vector<int>, vector<std::pair<int, int> >> perm_and_mask(const vector<double> &D_Z, const int n, const vector<int> &sample, mt19937 &generator) {
    const vector<int>& permutation = split_permutation(&D_Z[0], n, sample, generator);

    // mask to hide!
    std::set<pair<int, int> > setmask;  // relative index
    const int sample_size = sample.size();
    for (int i = 0; i < sample_size; i++) {
        setmask.insert(std::make_pair(i, i));
        setmask.insert(std::make_pair(i, permutation[i]));
        setmask.insert(std::make_pair(permutation[i], i));
    }
    vector<pair<int, int> > mask(setmask.begin(), setmask.end());
    return std::move(std::make_pair(std::move(permutation), std::move(mask)));
}

std::tuple<double, vector<int>, vector<std::pair<int, int> >> MMSD(const double *K_XZ, const double *K_Y, const vector<double> &D_Z, const int n, const vector<int> &sample, mt19937 generator) {
    double test_statistic = 0.0;
    vector<int> permutation;
    vector<std::pair<int, int> > mask;

    std::tie(permutation, mask) = perm_and_mask(D_Z, n, sample, generator);

    size_t sample_size = sample.size();
    for (size_t i = 0; i < sample_size; i++) {
        const double* K_XZsin = &K_XZ[sample[i] * n];
        const double* K_Ysin = &K_Y[sample[i] * n];
        const int spi = sample[permutation[i]];
        for (size_t j = 0; j < sample_size; j++) {
            test_statistic += K_XZsin[sample[j]] * (K_Ysin[sample[j]] + K_Y[spi * n + sample[permutation[j]]] - 2*K_Ysin[sample[permutation[j]]]);
        }
    }
    for (const auto &rc: mask) {
        const int i = rc.first;
        const int j = rc.second;
        test_statistic -= K_XZ[sample[i] * n + sample[j]] * (K_Y[sample[i] * n + sample[j]] + K_Y[sample[permutation[i]] * n + sample[permutation[j]]] - 2*K_Y[sample[i] * n + sample[permutation[j]]]);
    }
    test_statistic /= (sample_size * sample_size) - mask.size();

    return std::move(std::make_tuple(test_statistic, std::move(permutation), std::move(mask)));
}


// Returns a copy of distance matrix with max value added.
vector<double> penalized_distance(const vector<double> &D_Z, const int n, const vector<std::pair<int, int> > mask) {
//    const double inf = std::numeric_limits<double>::infinity();
    vector<double> copied_D_Z(D_Z);
    double max_val = *std::max_element(copied_D_Z.begin(), copied_D_Z.end());

    for (const auto &rc : mask) {
        copied_D_Z[rc.first * n + rc.second] += max_val;
    }
    return std::move(copied_D_Z);
}


void multi_mmsd(const double *K_XZ, const double *K_Y, const vector<double> &D_Z, const int n, const int sub_b, double *const subnull, const unsigned int subseed) {
    mt19937 generator(subseed);

    vector<int> samples(n);
    std::iota(std::begin(samples), std::end(samples), 0);

    for (int i = 0; i < sub_b; i++) {
        std::shuffle(samples.begin(), samples.end(), generator);
        const vector<int> half_sample = vector<int>(samples.begin(), samples.begin() + (n / 2));

        double mmsd;
        std::tie(mmsd, std::ignore, std::ignore) = MMSD(K_XZ, K_Y, D_Z, n, half_sample, generator);

        subnull[i] = mmsd;
    }
}


vector<double> shuffle_matrix(const double *mat, const int n, const vector<int> &perm) {
    vector<double> newmat;
    newmat.reserve(n * n);
    for (int i = 0; i < n; i++) {
        const int pin = perm[i] * n;
        const int in = i * n;
        for (int j = 0; j < n; j++) {
            newmat[in + j] = mat[pin + perm[j]];
        }
    }
    return std::move(newmat);
}


void c_sdcit(const double *K_XZ, const double *K_Y, const double *D_Z_, const int n,
             const int b, const int seed, const int n_threads,
             double *const mmsd, double *const null) {
    mt19937 generator(seed);

    double test_statistic;
    vector<int> permutation;
    vector<std::pair<int, int> > mask;
    const vector<double> D_Z(D_Z_, D_Z_ + (n * n));
    vector<int> full_idx(n);
    std::iota(std::begin(full_idx), std::end(full_idx), 0);

    std::tie(test_statistic, permutation, mask) = MMSD(K_XZ, K_Y, D_Z, n, full_idx, generator);

    std::tie(permutation, mask) = perm_and_mask(penalized_distance(D_Z, n, mask), n, full_idx, generator);
    const auto& D_Z_for_null = penalized_distance(D_Z, n, mask);
    const auto& K_Y_null = shuffle_matrix(K_Y, n, permutation);
    vector<thread> threads;
    int b_offset = 0;
    for (int i = 0; i < n_threads; i++) {
        const unsigned int sub_seed = generator();
        const int remnant_b = (i < (b % n_threads)) ? 1 : 0;
        const int sub_b = remnant_b + b / n_threads;

        threads.push_back(thread(multi_mmsd, K_XZ, &K_Y_null[0], D_Z_for_null, n, sub_b, null + b_offset, sub_seed));

        b_offset += sub_b;
    }
    for (int i = 0; i < n_threads; i++) { threads[i].join(); }

    *mmsd = test_statistic;
}
