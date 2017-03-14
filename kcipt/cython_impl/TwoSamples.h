//
// Created by Sanghack Lee on 8/21/16.
//

#ifndef C_KCIPT_TWOSAMPLES_H
#define C_KCIPT_TWOSAMPLES_H

#include <numeric>
#include <random>
#include <algorithm>


class TwoSamples {
    int length_of_sample;
public:
    std::vector<int> idx1x;
    std::vector<int> idx1y;
    std::vector<int> idx1z;

    std::vector<int> idx2x;
    std::vector<int> idx2y;
    std::vector<int> idx2z;

    // initialize with randomized
    TwoSamples(const int n_sample, std::mt19937 &generator) {
        length_of_sample = n_sample / 2;

        std::vector<int> samples(n_sample);
        std::iota(std::begin(samples), std::end(samples), 0);
        std::shuffle(samples.begin(), samples.end(), generator);

        idx1x = idx1y = idx1z = std::vector<int>(samples.begin(), samples.begin() + length_of_sample);
        idx2x = idx2y = idx2z = std::vector<int>(samples.begin() + length_of_sample, samples.end());
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

    void permute_by(std::vector<int> &to_change, const std::vector<int> &perm) {
        std::vector<int> new_idx(length_of_sample);
        for (int i = 0; i < length_of_sample; i++) {
            new_idx[i] = to_change[perm[i]];
        }
        to_change = new_idx;
    }

    TwoSamples &resplit(const std::vector<int> &samples) {
        // 1. merge,
        std::vector<int> xs = idx1x;
        std::vector<int> ys = idx1y;
        std::vector<int> zs = idx1z;
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


    TwoSamples &resplit(std::mt19937 &generator) {
        // 1. merge,
        std::vector<int> xs = idx1x;
        std::vector<int> ys = idx1y;
        std::vector<int> zs = idx1z;
        xs.insert(xs.end(), idx2x.begin(), idx2x.end());
        ys.insert(ys.end(), idx2y.begin(), idx2y.end());
        zs.insert(zs.end(), idx2z.begin(), idx2z.end());

        std::vector<int> samples(length_of_sample * 2);
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
