#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "SDCIT.h"
#include "permutation.h"

using std::vector;

// Enumerate all pairings independently of Blossom and the production scaling.
double minimum_pairing(const vector<double> &distance, int n,
                       const vector<int> &remaining) {
    if (remaining.empty()) return 0.0;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t j = 1; j < remaining.size(); ++j) {
        vector<int> rest;
        for (std::size_t k = 1; k < remaining.size(); ++k) {
            if (k != j) rest.push_back(remaining[k]);
        }
        best = std::min(best, 2 * distance[remaining[0] * n + remaining[j]] +
                             minimum_pairing(distance, n, rest));
    }
    return best;
}

double check_permutation(const vector<double> &distance, int n,
                         const vector<int> &permutation) {
    vector<int> sorted = permutation;
    std::sort(sorted.begin(), sorted.end());
    double cost = 0.0;
    for (int i = 0; i < n; ++i) {
        if (sorted[i] != i || permutation[i] == i) {
            throw std::runtime_error("invalid derangement");
        }
        cost += distance[i * n + permutation[i]];
    }
    if (!std::isfinite(cost)) throw std::runtime_error("used a forbidden edge");
    return cost;
}

void check_matching(const vector<double> &distance, int n, std::mt19937 &rng) {
    vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    const double largest = *std::max_element(distance.begin(), distance.end());
    vector<double> normalized = distance;
    if (largest > 0) {
        for (auto &value : normalized) value /= largest;
    }
    const double optimum = minimum_pairing(normalized, n, indices);
    // The integer conversion rounds each matching edge down by less than one
    // unit. This bound also covers the original cost of a better postprocessed
    // permutation; it does not assert that the heuristic is globally optimal.
    const double tolerance = n * (n * (n - 1.0) / 2 + n) /
                             (std::numeric_limits<int>::max() / 8) + 1e-12;
    for (bool split : {false, true}) {
        const auto permutation = split
            ? split_permutation(distance.data(), n, indices, rng)
            : dense_2n_permutation(distance.data(), n, indices, rng);
        if (check_permutation(normalized, n, permutation) > optimum + tolerance) {
            throw std::runtime_error("matching exceeds independent oracle bound");
        }
    }
}

void original_n8_regression() {
    const vector<double> z = {
        0.49981651916309294, 2.3120875219151036, -0.8553031968163819,
        0.8889168210259178, 0.25464513895549346, -1.8884212540610696,
        0.48237236923884624, 0.8892314367755281,
    };
    vector<double> kz(64), distance(64), ones(64, 1.0);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            const double delta = z[i] - z[j];
            kz[8 * i + j] = std::exp(-delta * delta);
            distance[8 * i + j] = std::sqrt(2 - 2 * kz[8 * i + j]);
        }
    }
    vector<double> null(37), error_null(37);
    double statistic, error;
    c_sdcit(kz.data(), ones.data(), kz.data(), distance.data(), 8, 37, 3, 1,
            &statistic, &error, null.data(), error_null.data());
    if (statistic != 0 || error != 0 || null != vector<double>(37, 0.0) ||
        error_null != vector<double>(37, 0.0)) {
        throw std::runtime_error("constant Y must yield zero discrepancies");
    }
}

int main() {
    try {
        original_n8_regression();
        std::mt19937 rng(2017);
        int cases = 0;
        for (int n : {2, 4, 6, 8}) {
            for (int trial = 0; trial < 80; ++trial) {
                vector<double> distance(n * n);
                for (int i = 0; i < n; ++i) {
                    for (int j = i + 1; j < n; ++j) {
                        distance[i * n + j] = distance[j * n + i] =
                            static_cast<double>(rng() % 10001) / 10000;
                    }
                }
                check_matching(distance, n, rng);
                ++cases;
            }
        }
        // For n<=4 there are no 3+3 or 2+3 heuristic moves, so very large
        // distances isolate conversion into Blossom's integer representation.
        for (double scale : {1e-300, 1.0, 1e300}) {
            check_matching({0, scale, scale, 0}, 2, rng);
            check_matching({0, scale, 2 * scale, 3 * scale,
                            scale, 0, 4 * scale, 5 * scale,
                            2 * scale, 4 * scale, 0, 6 * scale,
                            3 * scale, 5 * scale, 6 * scale, 0}, 4, rng);
            cases += 2;
        }
        check_matching({0, 1e308, 1e308, 0}, 2, rng);
        check_matching({0, 1e308, 1e308, 1e308, 1e308, 0, 1e308, 1e308,
                        1e308, 1e308, 0, 1e308, 1e308, 1e308, 1e308, 0}, 4, rng);
        cases += 2;

        // Two odd components force six dummy edges in the split path. Each
        // component's only derangements are the two orientations of its triangle.
        const double inf = std::numeric_limits<double>::infinity();
        vector<double> blocked(36, inf);
        for (int i = 0; i < 6; ++i) {
            blocked[i * 6 + i] = 0;
            for (int j = i + 1; j < 6; ++j) {
                if (i / 3 == j / 3) blocked[i * 6 + j] = blocked[j * 6 + i] = i + j + 1;
            }
        }
        vector<int> indices(6);
        std::iota(indices.begin(), indices.end(), 0);
        const auto permutation = split_permutation(blocked.data(), 6, indices, rng);
        if (check_permutation(blocked, 6, permutation) != 36) {
            throw std::runtime_error("wrong triangle-component cost");
        }
        std::cout << "matching costs: " << cases
                  << " oracle cases and n8/dummy-edge regressions passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
