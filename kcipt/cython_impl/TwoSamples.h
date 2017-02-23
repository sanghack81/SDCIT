//
// Created by Sanghack Lee on 8/21/16.
//

#ifndef C_KCIPT_TWOSAMPLES_H
#define C_KCIPT_TWOSAMPLES_H

#include <numeric>
#include <random>
#include <algorithm>

using std::vector;
using std::mt19937;

class TwoSamples {
    int length_of_sample;
public:
    vector<int> idx1x;
    vector<int> idx1y;
    vector<int> idx1z;

    vector<int> idx2x;
    vector<int> idx2y;
    vector<int> idx2z;

    // initialize with randomized
    TwoSamples(const int n_sample, std::mt19937 &generator) {
        length_of_sample = n_sample / 2;

        vector<int> samples(n_sample);
        std::iota(std::begin(samples), std::end(samples), 0);
        std::shuffle(samples.begin(), samples.end(), generator);

        idx1x = idx1y = idx1z = vector<int>(samples.begin(), samples.begin() + length_of_sample);
        idx2x = idx2y = idx2z = vector<int>(samples.begin() + length_of_sample, samples.end());
    }

    TwoSamples(TwoSamples &org) {
        length_of_sample = org.length_of_sample;

        idx1x = org.idx1x;  // this is copy
        idx2x = org.idx2x;  // this is copy
        idx1y = org.idx1y;  // this is copy
        idx2y = org.idx2y;  // this is copy
        idx1z = org.idx1z;  // this is copy
        idx2z = org.idx2z;  // this is copy
    }

    int sample_length() const {
        return length_of_sample;
    }

    void permute_by(vector<int> &to_change, const vector<int> &perm) {
        vector<int> new_idx(length_of_sample);
        for (int i = 0; i < length_of_sample; i++) {
            new_idx[i] = to_change[perm[i]];
        }
        to_change = new_idx;
    }

    TwoSamples &resplit(const std::vector<int> &samples) {
        // 1. merge,
        vector<int> xs = idx1x;
        vector<int> ys = idx1y;
        vector<int> zs = idx1z;
        xs.insert(xs.end(), idx2x.begin(), idx2x.end());
        ys.insert(ys.end(), idx2y.begin(), idx2y.end());
        zs.insert(zs.end(), idx2z.begin(), idx2z.end());

        // 3. split
        for (int j = 0; j < length_of_sample; j++) {
            idx1x[j] = xs[samples[j]];
            idx1y[j] = ys[samples[j]];
            idx1z[j] = zs[samples[j]];
        }
        for (int j = 0; j < length_of_sample; j++) {
            idx2x[j] = xs[samples[j + length_of_sample]];
            idx2y[j] = ys[samples[j + length_of_sample]];
            idx2z[j] = zs[samples[j + length_of_sample]];
        }
        return *this;
    }


    TwoSamples &resplit(mt19937 &generator) {
        // 1. merge,
        vector<int> xs = idx1x;
        vector<int> ys = idx1y;
        vector<int> zs = idx1z;
        xs.insert(xs.end(), idx2x.begin(), idx2x.end());
        ys.insert(ys.end(), idx2y.begin(), idx2y.end());
        zs.insert(zs.end(), idx2z.begin(), idx2z.end());

        vector<int> samples(length_of_sample * 2);
        std::iota(std::begin(samples), std::end(samples), 0);
        std::shuffle(samples.begin(), samples.end(), generator);

        // 3. split
        for (int j = 0; j < length_of_sample; j++) {
            idx1x[j] = xs[samples[j]];
            idx1y[j] = ys[samples[j]];
            idx1z[j] = zs[samples[j]];
        }
        for (int j = 0; j < length_of_sample; j++) {
            idx2x[j] = xs[samples[j + length_of_sample]];
            idx2y[j] = ys[samples[j + length_of_sample]];
            idx2z[j] = zs[samples[j + length_of_sample]];
        }
        return *this;
    }

};


#endif //C_KCIPT_TWOSAMPLES_H
