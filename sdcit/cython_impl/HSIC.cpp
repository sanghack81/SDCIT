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

void multi_hsic(const double *const K, const double *const L, const int n, const int sub_b, double *const subnull, const unsigned int subseed) {
    mt19937 generator(subseed);

    vector<int> samples(n);
    std::iota(std::begin(samples), std::end(samples), 0);

    for (int i = 0; i < sub_b; i++) {
        std::shuffle(samples.begin(), samples.end(), generator);

        double ss = 0.0;
        for(int i = 0; i < n; i++){
            const int in = i*n;
            const int in2 = samples[i]*n;

            for(int j = 0; j < n; j++){
                ss += K[in + j] * L[in2 + samples[j]];
            }
        }
        subnull[i] = ss / n;

    }
}

double c_hsic_stat(const double *const K,const double *const L, const int n){
    double ss = 0.0;
    const int nn = n*n;
    for(int i = 0; i < nn; i++){
        ss += K[i]*L[i];
    }
    return ss / n;
}

void c_hsic(const double *const K_X, const double *const K_Y, const int n, const int b, const int seed, const int n_threads, double *const test_statistic, double *const null) {
    mt19937 generator(seed);

    vector<thread> threads;
    int b_offset = 0;
    for (int i = 0; i < n_threads; i++) {
        const unsigned int sub_seed = generator();
        const int remnant_b = (i < (b % n_threads)) ? 1 : 0;
        const int sub_b = remnant_b + b / n_threads;

        threads.push_back(thread(multi_hsic, K_X, K_Y, n, sub_b, null + b_offset, sub_seed));
        b_offset += sub_b;
    }
    for (int i = 0; i < n_threads; i++) {
        threads[i].join();
    }

    *test_statistic = c_hsic_stat(K_X, K_Y, n);
}
