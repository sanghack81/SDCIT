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


void
sub_adj_kcipt(const unsigned int sub_seed, const double *K_X, const double *K_Y, const double *K_Z, const double *K_XYZ,
              const double *D_Z, const int n, const int B, const int b,
              const int sub_B, int sub_b, int sub_M,
              double *mmds,
              double *inner_null, int inner_null_offset, double *bootstrap_null, Barrier *wait_for_bootstrap) {
    mt19937 generator(sub_seed);
    // Compute test statistic
    double mmd00 = 0.0, mmd0y = 0.0, mmd0z = 0.0, mmdyy = 0.0, mmdzz = 0.0;
    for (int curr_B = 0; curr_B < sub_B; curr_B++) {
        TwoSamples ts_00(n, generator);
        TwoSamples ts_0y = ts_00;   // calls copy constructor
        TwoSamples ts_0z = ts_00;   // calls copy constructor
        TwoSamples ts_yy = ts_00;   // calls copy constructor
        TwoSamples ts_zz = ts_00;   // calls copy constructor

        const vector<int> perm1 = split_permutation(D_Z, n, ts_0y.idx1z, generator);   // any idx2*
        const vector<int> perm2 = split_permutation(D_Z, n, ts_0y.idx2z, generator);   // any idx2*

        ts_0y.permute_by(ts_0y.idx2y, perm2);

        ts_0z.permute_by(ts_0z.idx2z, perm2);

        ts_yy.permute_by(ts_yy.idx1y, perm1);
        ts_yy.permute_by(ts_yy.idx2y, perm2);

        ts_zz.permute_by(ts_zz.idx1z, perm1);
        ts_zz.permute_by(ts_zz.idx2z, perm2);

        // Compute test statistic
        mmd00 += u_mmd(K_X, K_Y, K_Z, n, ts_00);
        mmd0y += u_mmd(K_X, K_Y, K_Z, n, ts_0y);
        mmd0z += u_mmd(K_X, K_Y, K_Z, n, ts_0z);
        mmdyy += u_mmd(K_X, K_Y, K_Z, n, ts_yy);
        mmdzz += u_mmd(K_X, K_Y, K_Z, n, ts_zz);
    }
    mmds[0] = mmd00;
    mmds[1] = mmd0y;
    mmds[2] = mmd0z;
    mmds[3] = mmdyy;
    mmds[4] = mmdzz;

    // Compute inner null distribution
    null_distribution(generator, K_XYZ, n, inner_null, inner_null_offset, sub_b);

    // Compute outer null distribution
    wait_for_bootstrap->Wait();

    bootstrap_single_null(generator, B, b, sub_M, inner_null, bootstrap_null);
}


// This is an MMD-based adjusted method
void c_adj_kcipt(const int seed,
                 const double *K_X, const double *K_Y, const double *K_Z, const double *K_XYZ, const double *D_Z,
                 const int n,
                 const int B, const int b, const int M, const int n_threads,
                 double *const test_statistic, double *const inner_null, double *const outer_null, const int variance_reduced) {
    mt19937 generator(seed);
    vector<thread> threads;

    // record 5 MMD values
    vector<double> mmds(5 * n_threads);

    Barrier barrier(n_threads);
    // create threads
    int B_offset = 0, b_offset = 0, M_offset = 0;
    for (int i = 0; i < n_threads; i++) {
        const int remnant_B = (i < (B % n_threads)) ? 1 : 0;
        const int sub_B = remnant_B + B / n_threads;

        const int remnant_b = (i < (b % n_threads)) ? 1 : 0;
        const int sub_b = remnant_b + b / n_threads;

        const int remnant_M = (i < (M % n_threads)) ? 1 : 0;
        const int sub_M = remnant_M + M / n_threads;

        const unsigned int sub_seed = generator();

        // Run threaded-kcipt with a random seed and outer bootstrap -- sub_B
        threads.push_back(
                thread(sub_adj_kcipt, sub_seed, K_X, K_Y, K_Z, K_XYZ, D_Z, n, B, b, sub_B, sub_b, sub_M, &mmds[i * 5], inner_null, b_offset, &outer_null[M_offset], &barrier));

        B_offset += sub_B;
        b_offset += sub_b;
        M_offset += sub_M;
    }

    // join threads
    for (int i = 0; i < n_threads; i++) { threads[i].join(); }

    double mmd00 = 0.0, mmd0y = 0.0, mmd0z = 0.0, mmdyy = 0.0, mmdzz = 0.0;
    for (int i = 0; i < n_threads; i++) {
        mmd00 += mmds[i * 5];
        mmd0y += mmds[i * 5 + 1];
        mmd0z += mmds[i * 5 + 2];
        mmdyy += mmds[i * 5 + 3];
        mmdzz += mmds[i * 5 + 4];
    }
    mmd00 /= B;
    mmd0y /= B;
    mmd0z /= B;
    mmdyy /= B;
    mmdzz /= B;

    double error_factor = 0.0;
    if (mmdzz - mmd00 == 0.0) {
        error_factor = 1.0;
    } else {
        error_factor = (mmdyy - mmd00) / std::max(mmd0z - mmd00, mmdzz - mmd00);
    }

    if (variance_reduced > 0) {
        *test_statistic = (mmd0y - mmd00) - (mmd0z - mmd00) * error_factor;
    } else {
        *test_statistic = (mmd0y) - (mmd0z - mmd00) * error_factor;
    }
}

int main() {
    return 0;
}