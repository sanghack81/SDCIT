//
// Created by Sanghack Lee on 8/21/16.
//

#include <random>
#include <thread>
#include <iostream>
#include "KCIPT.h"
#include "mmd.h"
#include "permutation.h"
#include <valarray>
#include <condition_variable>
#include <functional>

using std::thread;
using std::vector;
using std::valarray;
using std::mt19937;

class Barrier {
private:
    std::mutex _mutex;
    std::condition_variable _cv;
    std::size_t _count;
public:
    explicit Barrier(std::size_t count) : _count{count} {}

    void Wait() {
        std::unique_lock<std::mutex> lock{_mutex};
        if (--_count == 0) {
            _cv.notify_all();
        } else {
            _cv.wait(lock, [this] { return _count == 0; });
        }
    }
};

void threaded_null_distribution(const unsigned int seed, const double *K_XYZ, const int n, double *nulls, const int b, const int n_threads) {
    mt19937 generator(seed);
    vector<thread> threads;

    // create threads
    int offset = 0;
    for (int i = 0; i < n_threads; i++) {
        const int remnant = (i < (b % n_threads)) ? 1 : 0;
        const int sub_b = remnant + b / n_threads;
        threads.push_back(thread(null_distribution, generator(), K_XYZ, n, nulls, offset, sub_b));

        offset += sub_b;
    }

    // join threads
    for (int i = 0; i < n_threads; i++) {
        threads[i].join();
    }
}


void null_distribution(mt19937 generator, const double *K_XYZ, const int n, double *nulls, const int offset, const int sub_b) {
    // Compute inner null distribution
    vector<int> samples(n);
    std::iota(std::begin(samples), std::end(samples), 0);
    for (int i = offset; i < offset + sub_b; i++) {
        std::shuffle(samples.begin(), samples.end(), generator);
        const vector<int> idx1 = vector<int>(samples.begin(), samples.begin() + (n / 2));
        const vector<int> idx2 = vector<int>(samples.begin() + (n / 2), samples.end());

        nulls[i] = u_mmd(K_XYZ, n, idx1, idx2);
    }
}


void null_distribution(const unsigned int seed, const double *K_XYZ, const int n, double *nulls, const int offset, const int sub_b) {
    mt19937 generator(seed);
    null_distribution(generator, K_XYZ, n, nulls, offset, sub_b);
}


// Monte Carlo Simulation for outer null distribution (original KCIPT)
void bootstrap_null(double *const inner_null, const int B, const int b, double *const outer_null, mt19937 &generator, const int M) {
    std::uniform_int_distribution<> dist(0, b - 1);
    auto gen = std::bind(dist, generator);  // will generate random numbers 0 <= ... <= b-1

    // B*how_many
    for (int i = 0; i < M; i++) {
        double avg = 0.0;
        for (int j = 0; j < B; j++) {
            avg += inner_null[b * j + gen()];
        }
        avg /= B;
        outer_null[i] = avg;
    }
}

// Monte Carlo Simulation for outer null distribution (original KCIPT)
void bootstrap_null(const int seed, const int B, const int b, const int M, double *const inner_null, double *const outer_null) {
    mt19937 generator(seed);
    bootstrap_null(inner_null, B, b, outer_null, generator, M);
}


void bootstrap_single_null(mt19937 &generator, const int B, const int b, const int M, double *const inner_null, double *const outer_null) {
    std::uniform_int_distribution<> dist(0, b - 1);
    auto gen = std::bind(dist, generator);

    // B*how_many
    for (int i = 0; i < M; i++) {
        double avg = 0.0;
        for (int j = 0; j < B; j++) {
            avg += inner_null[gen()];
        }
        avg /= B;
        outer_null[i] = avg;
    }
}


// Monte Carlo Simulation for outer null distribution for single identical inner null distribution
void bootstrap_single_null(const int seed, const int B, const int b, const int M, double *const inner_null, double *const outer_null) {
    mt19937 generator(seed);
    bootstrap_single_null(generator, B, b, M, inner_null, outer_null);
}


void sub_kcipt(const unsigned int sub_seed, const double *K_X, const double *K_Y, const double *K_Z, const double *D_Z,
               const int n, const int B, const int sub_B, const int b,
               double *mmds, double *inner_null, int inner_null_offset, Barrier *wait_for_bootstrap, double *outer_null, const int sub_M) {
    mt19937 generator(sub_seed);

    double *inner_null_write = &inner_null[inner_null_offset];
    for (int curr_B = 0; curr_B < sub_B; curr_B++) {
        TwoSamples ts_1c(n, generator);
        ts_1c.permute_by(ts_1c.idx2y, split_permutation(D_Z, n, ts_1c.idx2y, generator));

        *mmds++ = u_mmd(K_X, K_Y, K_Z, n, ts_1c);   // 1c
        for (int i = 0; i < b; i++) {
            *inner_null_write++ = u_mmd(K_X, K_Y, K_Z, n, ts_1c.resplit(generator));
        }
    }

    wait_for_bootstrap->Wait();

    bootstrap_null(inner_null, B, b, outer_null, generator, sub_M);
}

void c_kcipt(const double *K_X, const double *K_Y, const double *K_Z, const double *D_Z, const int n, const int B,
             const int b, double *const inner_null, double *const mmds, const int seed, const int n_threads,
             double *const outer_null, const int M) {
    mt19937 generator(seed);
    vector<thread> threads;

    Barrier barrier(n_threads);
    // create threads
    int offset = 0;
    int outer_null_offset = 0;
    for (int i = 0; i < n_threads; i++) {
        const int remnant = (i < (B % n_threads)) ? 1 : 0;
        const int sub_B = remnant + B / n_threads;

        const int remnant_M = (i < (M % n_threads)) ? 1 : 0;
        const int sub_M = remnant_M + M / n_threads;
        unsigned int sub_seed = generator();

        // Run threaded-kcipt with a random seed and outer bootstrap -- sub_B
        threads.push_back(thread(sub_kcipt,
                                 sub_seed, K_X, K_Y, K_Z, D_Z,
                                 n, B, sub_B, b,
                                 mmds + offset, inner_null, b * offset, &barrier, &outer_null[outer_null_offset], sub_M));

        offset += sub_B;
        outer_null_offset += sub_M;
    }

    // join threads
    for (int i = 0; i < n_threads; i++) {
        threads[i].join();
    }

}


int main() {
    return 0;
}