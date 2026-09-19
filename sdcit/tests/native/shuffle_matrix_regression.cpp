#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "SDCIT.h"

// This executable links the actual SDCIT.cpp translation unit. Matching is not
// needed by shuffle_matrix; fail loudly if the isolated test ever calls it.
std::vector<int> dense_2n_permutation(const double *, const int,
                                    const std::vector<int> &, std::mt19937 &) {
    throw std::logic_error("matching is outside the shuffle_matrix unit test");
}

void check(const std::vector<double> &input, const std::vector<int> &permutation,
           const std::vector<double> &expected) {
    const auto original = input;
    const auto output = shuffle_matrix(input.data(),
                                       static_cast<int>(permutation.size()),
                                       permutation);
    if (output.size() != expected.size()) {
        throw std::runtime_error("wrong output size: " +
                                 std::to_string(output.size()) + " != " +
                                 std::to_string(expected.size()));
    }
    if (output != expected) {
        throw std::runtime_error("wrong permuted matrix contents");
    }
    if (input != original) {
        throw std::runtime_error("input matrix was changed");
    }
}

int main() {
    try {
        check({1, 2, 3, 4}, {1, 0}, {4, 3, 2, 1});
        check({0, 1, 2, 3, 4, 5, 6, 7, 8}, {2, 0, 1},
              {8, 6, 7, 2, 0, 1, 5, 3, 4});
        check({}, {}, {});
        check({4.5}, {0}, {4.5});

        std::mt19937 generator(1729);
        int cases = 4;
        for (int n = 2; n <= 64; ++n) {
            std::vector<double> input(n * n);
            std::iota(input.begin(), input.end(), 0.0);
            std::vector<int> permutation(n);
            std::iota(permutation.begin(), permutation.end(), 0);
            for (int trial = 0; trial < 12; ++trial) {
                std::shuffle(permutation.begin(), permutation.end(), generator);
                // Independent oracle: matrix multiplication P A P^T, evaluated
                // sparsely from the positions of the permutation matrix's ones.
                std::vector<double> rows;
                for (const auto index : permutation) {
                    rows.insert(rows.end(), input.begin() + index * n,
                                input.begin() + (index + 1) * n);
                }
                std::vector<double> expected;
                for (int row = 0; row < n; ++row) {
                    for (const auto col : permutation) {
                        expected.push_back(rows.at(row * n + col));
                    }
                }
                check(input, permutation, expected);
                ++cases;
            }
        }
        std::cout << "shuffle_matrix: " << cases << " cases passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
