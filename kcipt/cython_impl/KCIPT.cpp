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
#include <mutex>
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
               double *mmds, double *inner_null, int inner_null_offset, Barrier* wait_for_bootstrap, double *outer_null, const int sub_M) {
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
              double *inner_null, int inner_null_offset, double *bootstrap_null, Barrier* wait_for_bootstrap) {
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

    double mmd00=0.0, mmd0y=0.0, mmd0z=0.0, mmdyy=0.0, mmdzz=0.0;
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
    if(mmdzz - mmd00 == 0.0){
        error_factor = 1.0;
    }else{
        error_factor = (mmdyy - mmd00) / std::max(mmd0z-mmd00, mmdzz - mmd00);
    }

    if(variance_reduced>0){
        *test_statistic = (mmd0y - mmd00) - (mmd0z - mmd00) * error_factor;
    }else{
        *test_statistic = (mmd0y) - (mmd0z - mmd00) * error_factor;
    }
}

//
//double mean(const valarray<double> &arr) {
//    return arr.sum() / arr.size();
//}
//
//double mean_kk_between(const double *K_XYZ, const int full_n, const vector<int> idx1, const vector<int> idx2) {
//    double m = 0.0;
//    const int n = idx1.size();
//    for (int i = 0; i < n; i++) {
//        const double *temp_row1 = &K_XYZ[full_n * idx1[i]];
//        for (int j = 0; j < n; j++) {
//            m += temp_row1[idx2[j]];
//        }
//    }
//    return m / (n * n);
//}
//
//
//double mean_kk_between(const double *K_X, const double *K_Y, const double *K_Z, const int full_n,
//                       const vector<int> idx1x, const vector<int> idx1y, const vector<int> idx1z,
//                       const vector<int> idx2x, const vector<int> idx2y, const vector<int> idx2z) {
//    double m = 0.0;
//    const int n = idx1x.size();
//
//    for (int i = 0; i < n; i++) {
//        const double *temp_row1 = &K_X[full_n * idx1x[i]];
//        const double *temp_row2 = &K_Y[full_n * idx1y[i]];
//        const double *temp_row3 = &K_Z[full_n * idx1z[i]];
//
//        for (int j = 0; j < n; j++) {
//            m += temp_row1[idx2x[j]] * temp_row2[idx2y[j]] * temp_row3[idx2z[j]];
//        }
//    }
//    return m / (n * n);
//}
//
//
//double mean_kk_within(const double *K_XYZ, const int full_n, const vector<int> idx) {
//    double m = 0.0;
//    const int n = idx.size();
//
//    for (int i = 0; i < n; i++) {
//        const double *temp_row1 = &K_XYZ[full_n * idx[i]];
//        for (int j = 0; j < n; j++) {
//            m += temp_row1[idx[j]];
//        }
//        m -= temp_row1[idx[i]];
//    }
//    return m / (n * (n - 1));
//}
//
//double mean_kk_within(const double *K_X, const double *K_Y, const double *K_Z, const int full_n,
//                      const vector<int> idxx, const vector<int> idxy, const vector<int> idxz) {
//    double m = 0.0;
//    const int n = idxx.size();
//
//    for (int i = 0; i < n; i++) {
//        const double *temp_row1 = &K_X[full_n * idxx[i]];
//        const double *temp_row2 = &K_Y[full_n * idxy[i]];
//        const double *temp_row3 = &K_Z[full_n * idxz[i]];
//
//        for (int j = 0; j < n; j++) {
//            m += temp_row1[idxx[j]] * temp_row2[idxy[j]] * temp_row3[idxz[j]];
//        }
//        m -= temp_row1[idxx[i]] * temp_row2[idxy[i]] * temp_row3[idxz[i]];
//    }
//    return m / (n * (n - 1));
//}
//
//
//vector<int> permute_by(const vector<int> &to_change, const vector<int> &perm) {
//    const int n = to_change.size();
//    vector<int> new_idx(n);
//    for (int i = 0; i < n; i++) {
//        new_idx[i] = to_change[perm[i]];
//    }
//    return new_idx;
//}
//
//// 2. //////////////////////////////////////////////////////////////////////////////////////////////////////////////
//void
//sub_adj_kcipt2(const unsigned int sub_seed, const double *K_XYZ, const double *K_X, const double *K_Y, const double *K_Z, const double *D_Z,
//               const int n, const int sub_B, const int b, double *kks, double *nulls_1a) {
//    double kk_11_00 = 0.0;
//    double kk_22_00 = 0.0;
//    double kk_12_00 = 0.0;
//    double kk_22_yy = 0.0;
//    double kk_22_zz = 0.0;
//    double kk_12_0y = 0.0;
//    double kk_12_0z = 0.0;
//
//    const int length_of_sample = n / 2;
//    vector<int> samples(length_of_sample * 2);
//    std::iota(std::begin(samples), std::end(samples), 0);
//
//    mt19937 generator(sub_seed);
//    for (int curr_B = 0; curr_B < sub_B; curr_B++) {
//        std::shuffle(samples.begin(), samples.end(), generator);
//
//        const vector<int> idx1 = vector<int>(samples.begin(), samples.begin() + length_of_sample);
//        const vector<int> idx2 = vector<int>(samples.begin() + length_of_sample, samples.end());
//        const vector<int> perm2 = split_permutation(D_Z, n, idx2, generator);   // any idx2*
//        const vector<int> Pidx2 = permute_by(idx2, perm2);
//
//        kk_11_00 += mean_kk_within(K_XYZ, n, idx1);
//        kk_22_00 += mean_kk_within(K_XYZ, n, idx2);
//        kk_12_00 += mean_kk_between(K_XYZ, n, idx1, idx2);
//        kk_22_yy += mean_kk_within(K_X, K_Y, K_Z, n, idx2, Pidx2, idx2);
//        kk_22_zz += mean_kk_within(K_X, K_Y, K_Z, n, idx2, idx2, Pidx2);
//        kk_12_0y += mean_kk_between(K_X, K_Y, K_Z, n, idx1, idx1, idx1, idx2, Pidx2, idx2);
//        kk_12_0z += mean_kk_between(K_X, K_Y, K_Z, n, idx1, idx1, idx1, idx2, idx2, Pidx2);
//
//        // Compute inner-null distribution
//        for (int i = 0; i < b; i++) {
//            std::shuffle(samples.begin(), samples.end(), generator);
//            const vector<int> idx1 = vector<int>(samples.begin(), samples.begin() + length_of_sample);
//            const vector<int> idx2 = vector<int>(samples.begin() + length_of_sample, samples.end());
//
//            *nulls_1a++ = mean_kk_within(K_XYZ, n, idx1) + mean_kk_within(K_XYZ, n, idx2) - 2.0 * mean_kk_between(K_XYZ, n, idx1, idx2);
//        }
//    }
//
//    kks[0] = kk_11_00;
//    kks[1] = kk_22_00;
//    kks[2] = kk_12_00;
//    kks[3] = kk_22_yy;
//    kks[4] = kk_22_zz;
//    kks[5] = kk_12_0y;
//    kks[6] = kk_12_0z;
//}
//
//void c_adj_kcipt2(const double *K_XYZ, const double *K_X, const double *K_Y, const double *K_Z, const double *D_Z, const int n, const int B,
//                  const int b, double *const inner_null, double *const mmds, const int seed, const int n_threads,
//                  double *const outer_null, const int how_many) {
//    mt19937 generator(seed);
//    vector<thread> threads;
//
//    vector<double> nulls_1a(B * b);
//    vector<double> kks(7 * n_threads);
//
//    // create threads
//    int offset = 0;
//    for (int i = 0; i < n_threads; i++) {
//        const int remnant = (i < (B % n_threads)) ? 1 : 0;
//        const int sub_B = remnant + B / n_threads;
//        const unsigned int sub_seed = generator();
//        threads.push_back(thread(sub_adj_kcipt2, sub_seed, K_XYZ, K_X, K_Y, K_Z, D_Z, n, sub_B, b, &kks[7 * i], &nulls_1a[b * offset]));
//
//        offset += sub_B;
//    }
//
//    // join threads
//    for (int i = 0; i < n_threads; i++) { threads[i].join(); }
//
//    double kk_11_00 = 0.0;
//    double kk_22_00 = 0.0;
//    double kk_12_00 = 0.0;
//    double kk_22_yy = 0.0;
//    double kk_22_zz = 0.0;
//    double kk_12_0y = 0.0;
//    double kk_12_0z = 0.0;
//
//    for (int i = 0; i < n_threads; i++) {
//        kk_11_00 += kks[i * 7 + 0];
//        kk_22_00 += kks[i * 7 + 1];
//        kk_12_00 += kks[i * 7 + 2];
//        kk_22_yy += kks[i * 7 + 3];
//        kk_22_zz += kks[i * 7 + 4];
//        kk_12_0y += kks[i * 7 + 5];
//        kk_12_0z += kks[i * 7 + 6];
//    }
//    kk_11_00 /= B;
//    kk_22_00 /= B;
//    kk_12_00 /= B;
//    kk_22_yy /= B;
//    kk_22_zz /= B;
//    kk_12_0y /= B;
//    kk_12_0z /= B;
//
//    const double alpha_z = kk_22_zz / kk_22_00;
//    const double alpha_y = kk_22_yy / kk_22_00;
//    const double beta_z = kk_12_0z / kk_12_00;
//    const double beta_y = 1.0 + (alpha_y - 1) * (beta_z - 1) / (alpha_z - 1);   // divide by zero?
//
//    const double test_statistic = kk_11_00 + kk_22_00 - 2.0 * kk_12_0y / beta_y;
//
//    mmds[0] = test_statistic;
//    for (int i = 0; i < B * b; i++) { inner_null[i] = nulls_1a[i]; }
//    bootstrap_single_null(how_many, B, outer_null, 0, nulls_1a, generator());
//}
