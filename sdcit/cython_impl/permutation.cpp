#include <vector>
#include <set>
#include <numeric>
#include <cmath>
#include <random>
#include <algorithm>
#include "../../blossom5/PerfectMatching.h"

using std::vector;
// using std::isinf;


template<typename T>
vector<size_t> sort_indexes(const vector<T> &v) {
    // initialize original index locations
    vector<size_t> idx(v.size());
    for (auto i = 0; i < v.size(); i++) {
        idx[i] = i;
    }
    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
}


void post_2_2_2_to_3_3(const double *DDDD, const int full_n, const vector<int> &idxs, vector<int> &comps_of_2, vector<int> &comps_of_3, vector<int> &perm_array) {
    const int n = full_n;
    const int m = comps_of_2.size() / 2;
    vector<bool> merged(m, false);
    vector<double> neg_dists_of_2(m);
    for (long i = 0; i < m; i++) {
        auto k = comps_of_2[2 * i];
        auto l = comps_of_2[2 * i + 1];
        neg_dists_of_2[i] = -DDDD[n * idxs[k] + idxs[l]];
    }

    for (auto i : sort_indexes(neg_dists_of_2)) {
        if (merged[i]) {
            continue;
        }
        int a = comps_of_2[2 * i];
        int b = comps_of_2[2 * i + 1];

        vector<int> nears_a;
        vector<int> nears_b;
        for (int j = 0; j < m; j++) {
            if (merged[j]) {
                continue;
            }
            if (i == j) {
                continue;
            }
            int c = comps_of_2[2 * j];
            int d = comps_of_2[2 * j + 1];
            if (DDDD[idxs[a] * n + idxs[c]] + DDDD[idxs[a] * n + idxs[d]] - DDDD[idxs[a] * n + idxs[b]] - DDDD[idxs[c] * n + idxs[d]] < 0) {
                nears_a.push_back(j);
            }
            if (DDDD[idxs[b] * n + idxs[c]] + DDDD[idxs[b] * n + idxs[d]] - DDDD[idxs[a] * n + idxs[b]] - DDDD[idxs[c] * n + idxs[d]] < 0) {
                nears_b.push_back(j);
            }
        }

        if (!nears_a.empty() && !nears_b.empty()) {
            double max_gain = 0.0;
            int max_near_a_at;
            int max_near_b_at;

            for (int near_a_at : nears_a) {
                int c = comps_of_2[2 * near_a_at];
                int d = comps_of_2[2 * near_a_at + 1];

                for (int near_b_at : nears_b) {
                    if (near_b_at == near_a_at) {
                        continue;
                    }
                    int e = comps_of_2[2 * near_b_at];
                    int f = comps_of_2[2 * near_b_at + 1];

                    double previous = 2.0 * (DDDD[idxs[a] * n + idxs[b]] + DDDD[idxs[c] * n + idxs[d]] + DDDD[idxs[e] * n + idxs[f]]);
                    double after =
                            DDDD[idxs[a] * n + idxs[c]] + DDDD[idxs[a] * n + idxs[d]] + DDDD[idxs[c] * n + idxs[d]] +
                            DDDD[idxs[b] * n + idxs[e]] + DDDD[idxs[b] * n + idxs[f]] + DDDD[idxs[e] * n + idxs[f]];
                    if (max_gain < previous - after) {
                        max_near_a_at = near_a_at;
                        max_near_b_at = near_b_at;
                        max_gain = previous - after;
                    }
                }
            }
            if (max_gain > 0.0) {
                int c = comps_of_2[2 * max_near_a_at];
                int d = comps_of_2[2 * max_near_a_at + 1];
                int e = comps_of_2[2 * max_near_b_at];
                int f = comps_of_2[2 * max_near_b_at + 1];

                comps_of_3.push_back(a);
                comps_of_3.push_back(c);
                comps_of_3.push_back(d);

                comps_of_3.push_back(b);
                comps_of_3.push_back(e);
                comps_of_3.push_back(f);

                perm_array[a] = c;
                perm_array[c] = d;
                perm_array[d] = a;

                perm_array[b] = e;
                perm_array[e] = f;
                perm_array[f] = b;

                merged[i] = true;
                merged[max_near_a_at] = true;
                merged[max_near_b_at] = true;
            }
        }
    }

    comps_of_2.clear();
    auto perm_size = perm_array.size();
    for (size_t i = 0; i < perm_size; i++) {
        if (perm_array[perm_array[i]] == i && i < perm_array[i]) {
            comps_of_2.push_back(i);
            comps_of_2.push_back(perm_array[i]);
        }
    }
}


inline double max3(double a, double b, double c) {
    if (a <= c && b <= c) { return c; }
    if (a <= b && c <= b) { return b; }
    return a;
}

inline double min3(double a, double b, double c) {
    if (a >= c && b >= c) { return c; }
    if (a >= b && c >= b) { return b; }
    return a;
}

void find_cyc(int v, int w, int x, int y, int z, int cyc_id, int *result_at) {
    switch (cyc_id) {
        case 0:
            *result_at++ = v;
            *result_at++ = w;
            *result_at++ = x;
            *result_at++ = y;
            *result_at++ = z;
            *result_at = v;
            return;
        case 1:
            *result_at++ = v;
            *result_at++ = w;
            *result_at++ = x;
            *result_at++ = z;
            *result_at++ = y;
            *result_at = v;
            return;
        case 2:
            *result_at++ = v;
            *result_at++ = w;
            *result_at++ = y;
            *result_at++ = x;
            *result_at++ = z;
            *result_at = v;
            return;
        case 3:
            *result_at++ = v;
            *result_at++ = w;
            *result_at++ = y;
            *result_at++ = z;
            *result_at++ = x;
            *result_at = v;
            return;
        case 4:
            *result_at++ = v;
            *result_at++ = w;
            *result_at++ = z;
            *result_at++ = x;
            *result_at++ = y;
            *result_at = v;
            return;
        case 5:
            *result_at++ = v;
            *result_at++ = w;
            *result_at++ = z;
            *result_at++ = y;
            *result_at++ = x;
            *result_at = v;
            return;
        case 6:
            *result_at++ = v;
            *result_at++ = x;
            *result_at++ = w;
            *result_at++ = y;
            *result_at++ = z;
            *result_at = v;
            return;
        case 7:
            *result_at++ = v;
            *result_at++ = x;
            *result_at++ = w;
            *result_at++ = z;
            *result_at++ = y;
            *result_at = v;
            return;
        case 8:
            *result_at++ = v;
            *result_at++ = x;
            *result_at++ = y;
            *result_at++ = w;
            *result_at++ = z;
            *result_at = v;
            return;
        case 9:
            *result_at++ = v;
            *result_at++ = x;
            *result_at++ = z;
            *result_at++ = w;
            *result_at++ = y;
            *result_at = v;
            return;
        case 10:
            *result_at++ = v;
            *result_at++ = y;
            *result_at++ = w;
            *result_at++ = x;
            *result_at++ = z;
            *result_at = v;
            return;
        case 11:
            *result_at++ = v;
            *result_at++ = y;
            *result_at++ = x;
            *result_at++ = w;
            *result_at++ = z;
            *result_at = v;
            return;
    }
}

struct Comp235 {
    int c1;
    int c2;
    int cyc_id;
    double gain;
};

void data_analysis(const double *D, const int full_n, const int len, vector<vector<int>> &odd_components, const vector<int> &idxs, int &n_edges, double &max_distance, double &sum_distance);
//void data_analysis(const double *D, const int full_n, const int len, const vector<int> &idxs, int &n_edges, double &max_distance, double &sum_distance);

bool acompare(Comp235 lhs, Comp235 rhs) { return lhs.gain > rhs.gain; }


void post_2_3_to_5(const double *DDDD, const int full_n, const vector<int> &idxs, const vector<int> &comps_of_2, const vector<int> &comps_of_3, vector<int> &perm_array) {
    const int n = full_n;
    const int m2 = comps_of_2.size() / 2;
    const int m3 = comps_of_3.size() / 3;

    vector<Comp235> gains;
    for (int c2 = 0; c2 < m3; c2++) {
        int c = comps_of_3[c2 * 3];
        int d = comps_of_3[c2 * 3 + 1];
        int e = comps_of_3[c2 * 3 + 2];
        double max_side_triangle = max3(DDDD[idxs[c] * n + idxs[d]], DDDD[idxs[d] * n + idxs[e]], DDDD[idxs[e] * n + idxs[c]]);

        for (int c1 = 0; c1 < m2; c1++) {
            int a = comps_of_2[c1 * 2];
            int b = comps_of_2[c1 * 2 + 1];

            if ((max_side_triangle + DDDD[idxs[a] * n + idxs[b]]) <=
                (min3(DDDD[idxs[a] * n + idxs[c]], DDDD[idxs[a] * n + idxs[d]], DDDD[idxs[a] * n + idxs[e]]) +
                 min3(DDDD[idxs[b] * n + idxs[c]], DDDD[idxs[b] * n + idxs[d]], DDDD[idxs[b] * n + idxs[e]]))) {
                continue;
            }
            double current = 2.0 * DDDD[idxs[a] * n + idxs[b]] + DDDD[idxs[c] * n + idxs[d]] + DDDD[idxs[d] * n + idxs[e]] + DDDD[idxs[e] * n + idxs[c]];

            int v = a;
            int w = b;
            int x = c;
            int y = d;
            int z = e;

            double d_vw = DDDD[idxs[v] * n + idxs[w]];
            double d_vx = DDDD[idxs[v] * n + idxs[x]];
            double d_vy = DDDD[idxs[v] * n + idxs[y]];
            double d_vz = DDDD[idxs[v] * n + idxs[z]];
            double d_wx = DDDD[idxs[w] * n + idxs[x]];
            double d_wy = DDDD[idxs[w] * n + idxs[y]];
            double d_wz = DDDD[idxs[w] * n + idxs[z]];
            double d_xy = DDDD[idxs[x] * n + idxs[y]];
            double d_xz = DDDD[idxs[x] * n + idxs[z]];
            double d_yz = DDDD[idxs[y] * n + idxs[z]];

            double dd[12] = {
                    d_vw + d_wx + d_xy + d_yz + d_vz, d_vw + d_wx + d_xz + d_yz + d_vy,
                    d_vw + d_wy + d_xy + d_xz + d_vz, d_vw + d_wy + d_yz + d_xz + d_vx,
                    d_vw + d_wz + d_xz + d_xy + d_vy, d_vw + d_wz + d_yz + d_xy + d_vx,
                    d_vx + d_wx + d_wy + d_yz + d_vz, d_vx + d_wx + d_wz + d_yz + d_vy,
                    d_vx + d_xy + d_wy + d_wz + d_vz, d_vx + d_xz + d_wz + d_wy + d_vy,
                    d_vy + d_wy + d_wx + d_xz + d_vz, d_vy + d_xy + d_wx + d_wz + d_vz
            };

            int min_dd_at = 0;
            for (int j = 1; j < 12; j++) {
                if (dd[j] < dd[min_dd_at]) {
                    min_dd_at = j;
                }
            }

            if (dd[min_dd_at] < current) {
                Comp235 result;
                result.c1 = c1;
                result.c2 = c2;
                result.cyc_id = min_dd_at;
                result.gain = current - dd[min_dd_at];
                gains.push_back(result);
            }
        }
    }
    std::sort(gains.begin(), gains.end(), acompare);

    std::set<int> merge2;
    std::set<int> merge3;
    int cyc_array[6];
    for (Comp235 comp235 : gains) {
        int c1 = comp235.c1;
        int c2 = comp235.c2;
        if (merge2.find(c1) != merge2.end() || merge3.find(c2) != merge3.end()) {
            continue;
        }
        int a = comps_of_2[c1 * 2];
        int b = comps_of_2[c1 * 2 + 1];
        int c = comps_of_3[c2 * 3];
        int d = comps_of_3[c2 * 3 + 1];
        int e = comps_of_3[c2 * 3 + 2];

        find_cyc(a, b, c, d, e, comp235.cyc_id, cyc_array);
        // 0-1, ... 4-5,
        for (int i = 1; i < 6; i++) {
            perm_array[cyc_array[i - 1]] = cyc_array[i];
        }

        merge2.insert(c1);
        merge3.insert(c2);
    }
}


vector<int> random_permutation(const vector<int> &idxs, std::mt19937 &generator) {
    int len = idxs.size();

    // random permutation
    if (len == 1) {
        return {0};
    } else if (len == 2) {
        return {1, 0};
    } else if (len == 3) {
        return {1, 2, 0};
    }
    vector<int> samples(len);
    std::iota(std::begin(samples), std::end(samples), 0);
    std::shuffle(samples.begin(), samples.end(), generator);

    vector<int> unpermuted;
    for (int i = 0; i < len; i++) {
        if (samples[i] == i) {
            unpermuted.push_back(i);
        }
    }
    if (unpermuted.size() == 1) {
        int at = unpermuted[0];
        int to_swap = (at == 0) ? 1 : (at - 1);

        int temp = samples[at];
        samples[at] = samples[to_swap];
        samples[to_swap] = temp;
    } else if (unpermuted.size() > 1) {
        // rotation among unpermuted
        int temp = samples[unpermuted[0]];
        unsigned int up_size = unpermuted.size();
        for (size_t i = 1; i < up_size; i++) {
            samples[unpermuted[i - 1]] = samples[unpermuted[i]];
        }
        samples[unpermuted.back()] = temp;
    }

    return samples;
}


// Returns a permutation array for "idxs"
// Hence, the length of permutation array matches to the length of "idxs"
// Further, a permutation array contains [0, |idxs|-1].
vector<int> split_permutation(const double *D, const int full_n, const vector<int> &idxs, std::mt19937 &generator) {
    const int len = idxs.size();
    if (len == 1) {
        return {0};
    }
    if (D == NULL) {
        return random_permutation(idxs, generator);
    }

    // Preparation
    vector<int> perm_array(len);
    int n_edges = 0;
    double max_distance = 0.0;
    double sum_distance = 0.0;
    vector<vector<int>> odd_components;
    data_analysis(D, full_n, len, odd_components, idxs, n_edges, max_distance, sum_distance);

    if (sum_distance == 0.0) {
        return random_permutation(idxs, generator);
    }


    struct PerfectMatching::Options options;
    PerfectMatching *pm = new PerfectMatching(len + odd_components.size(), n_edges);
    {
        double factor = 1.0;
        if (sum_distance < INT_MAX) {
            factor = INT_MAX / sum_distance;
        }
        for (int i = 0; i < len; i++) {
            const double *D_i = D + idxs[i] * full_n;
            for (int j = i + 1; j < len; j++) {
                double d = D_i[idxs[j]];
                if (!std::isinf(d)) {
                    pm->AddEdge(i, j, (int) (d * factor));
                }
            }
        }

        int dummy_node_id = len;
        for (vector<int> component: odd_components) {
            for (int idxs_index: component) {
                pm->AddEdge(dummy_node_id, idxs_index, (int) (factor * max_distance));
            }
            dummy_node_id++;
        }

        options.verbose = false;
        pm->options = options;
        pm->Solve();
        for (int i = 0; i < len; i++) {
            int j = pm->GetMatch(i);
            if (j < len) {
                if (i < j) {
                    perm_array[i] = j;
                    perm_array[j] = i;
                }
            } else {
                perm_array[i] = i;  // matched to dummy, not permuted.
            }
        }
        delete pm;
    }


    vector<int> comps_of_3;
    // 1 + 2 = 3
    // there exists exactly one singleton (i.e., unpermuted) for an "odd" component
    for (auto odd_comp: odd_components) {
        if (odd_comp.size() >= 3) {
            for (auto i: odd_comp) {
                if (perm_array[i] == i) {
                    double min_d = INFINITY;
                    int min_d_at = -1;
                    for (int j: odd_comp) {
                        if (j < perm_array[j]) {
                            double d = D[idxs[i] * full_n + idxs[j]] + D[idxs[i] * full_n + idxs[perm_array[j]]];
                            if (d <= min_d) {
                                min_d = d;
                                min_d_at = j;
                            }
                        }
                    }
                    if (min_d_at != -1) {
                        comps_of_3.push_back(i);
                        comps_of_3.push_back(min_d_at);
                        comps_of_3.push_back(perm_array[min_d_at]);

                        perm_array[i] = min_d_at;
                        perm_array[perm_array[min_d_at]] = i;
                    }
                    break;
                }
            }
        }
    }

    vector<int> comps_of_2;
    for (int i = 0; i < len; i++) {
        if (perm_array[perm_array[i]] == i) {
            if (i < perm_array[i]) {
                comps_of_2.push_back(i);
                comps_of_2.push_back(perm_array[i]);
            }
        }
    }

    post_2_2_2_to_3_3(D, full_n, idxs, comps_of_2, comps_of_3, perm_array);
    post_2_3_to_5(D, full_n, idxs, comps_of_2, comps_of_3, perm_array);

    return perm_array;
}


void dense_data_analysis(const double *D, const int full_n, const int len, const vector<int> &idxs, int &n_edges, double &max_distance, double &sum_distance) {
    for (int i = 0; i < len; i++) {
        const double *D_i = D + idxs[i] * full_n;
        for (int j = i + 1; j < len; j++) {
            double d = D_i[idxs[j]];
            if (std::isinf(d)) {

            } else {
                if (max_distance < d) {
                    max_distance = d;
                }
                sum_distance += d;
                n_edges += 1;
            }
        }
    }
}


vector<int> dense_2n_permutation(const double *D, const int full_n, const vector<int> &idxs, std::mt19937 &generator) {
    const int len = idxs.size();
    if (len == 1) {
        return {0};
    }
    if (D == NULL) {
        return random_permutation(idxs, generator);
    }

    // Preparation
    vector<int> perm_array(len);
    int n_edges = 0;
    double max_distance = 0.0;
    double sum_distance = 0.0;
    dense_data_analysis(D, full_n, len, idxs, n_edges, max_distance, sum_distance);

    if (sum_distance == 0.0) {
        return random_permutation(idxs, generator);
    }


    struct PerfectMatching::Options options;
    PerfectMatching *pm = new PerfectMatching(len, n_edges);
    {
        double factor = 1.0;
        if (sum_distance < INT_MAX) {
            factor = INT_MAX / sum_distance;
        }
        for (int i = 0; i < len; i++) {
            const double *D_i = D + idxs[i] * full_n;
            for (int j = i + 1; j < len; j++) {
                double d = D_i[idxs[j]];
                if (!std::isinf(d)) {
                    pm->AddEdge(i, j, (int) (d * factor));
                }
            }
        }

        options.verbose = false;
        pm->options = options;
        pm->Solve();
        for (int i = 0; i < len; i++) {
            int j = pm->GetMatch(i);
            if (i < j) {
                perm_array[i] = j;
                perm_array[j] = i;
            }
        }
        delete pm;
    }

    vector<int> comps_of_3;
    vector<int> comps_of_2;
    for (int i = 0; i < len; i++) {
        if (perm_array[perm_array[i]] == i) {
            if (i < perm_array[i]) {
                comps_of_2.push_back(i);
                comps_of_2.push_back(perm_array[i]);
            }
        }
    }

    post_2_2_2_to_3_3(D, full_n, idxs, comps_of_2, comps_of_3, perm_array);
    post_2_3_to_5(D, full_n, idxs, comps_of_2, comps_of_3, perm_array);

    return perm_array;
}


void data_analysis(const double *D, const int full_n, const int len, vector<vector<int>> &odd_components, const vector<int> &idxs, int &n_edges, double &max_distance, double &sum_distance) {
    std::vector<bool> visited(len);
    for (int i = 0; i < len;) {
        vector<int> component;  // indices of idxs

        component.push_back(i);
        visited[i] = true;
        int next_component_start = -1;
        const double *D_i = D + idxs[i] * full_n;
        for (int j = i + 1; j < len; j++) {
            double d = D_i[idxs[j]];
            if (std::isinf(d)) {
                if (next_component_start == -1 && !visited[j]) {
                    next_component_start = j;
                }
            } else {
                if (max_distance < d) {
                    max_distance = d;
                }
                sum_distance += d;
                component.push_back(j);
                visited[j] = true;
            }
        }
        if (component.size() % 2) {
            odd_components.push_back(component);
        }
        int evened = component.size() + (component.size() % 2);
        n_edges += evened * (evened - 1);

        if (next_component_start == -1) {
            break;
        } else {
            i = next_component_start;
        }
    }
}


void split_permutation_interface(const double *D, const int full_n, int *perm) {
    std::mt19937 generator;

    vector<int> samples(full_n);
    std::iota(std::begin(samples), std::end(samples), 0);

    const vector<int>& output = split_permutation(D, full_n, samples, generator);
    for (int i: output) {
        *perm++ = i;
    }
}


void dense_2n_permutation_interface(const double *D, const int full_n, int *perm) {
    std::mt19937 generator;

    vector<int> samples(full_n);
    std::iota(std::begin(samples), std::end(samples), 0);

    const vector<int>& output = dense_2n_permutation(D, full_n, samples, generator);
    for (int i: output) {
        *perm++ = i;
    }
}

