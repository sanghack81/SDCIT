#include <vector>
#include <set>
#include <numeric>
#include <algorithm>
#include <stdio.h>
#include "PerfectMatching.h"

void c_cy_blossom_v(double* D, int* output, int n)
{
    int i, j;
    struct PerfectMatching::Options options;
    PerfectMatching* pm = new PerfectMatching(n, n * n - n);
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            pm->AddEdge(i, j, (int)D[i * n + j]);
        }
    }
    options.verbose = false;
    pm->options = options;
    pm->Solve();
    for (i = 0; i < n; i++) {
        j = pm->GetMatch(i);
        output[i] = j;
        output[j] = i;
    }
    delete pm;
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T>& v)
{
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    for (int i = 0; i < v.size(); i++) {
        idx[i] = i;
    }
    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
}

void c_cy_post_2_2_2_to_3_3(double* D, int* comps_of_2, int* abcdefs, int m,
    int n)
{
    std::vector<bool> merged(m, false);
    std::vector<double> neg_dists_of_2(m);
    for (long i = 0; i < m; i++) {
        auto k = comps_of_2[2 * i];
        auto l = comps_of_2[2 * i + 1];
        neg_dists_of_2[i] = -D[n * k + l];
    }
    for (int i : sort_indexes(neg_dists_of_2)) {
        if (merged[i]) {
            continue;
        }
        int a = comps_of_2[2 * i];
        int b = comps_of_2[2 * i + 1];

        std::vector<int> nears_a;
        std::vector<int> nears_b;
        for (int j = 0; j < m; j++) {
            if (merged[j]) {
                continue;
            }
            if (i == j) {
                continue;
            }
            int c = comps_of_2[2 * j];
            int d = comps_of_2[2 * j + 1];
            if (D[a * n + c] + D[a * n + d] - D[a * n + b] - D[c * n + d] < 0) {
                nears_a.push_back(j);
            }
            if (D[b * n + c] + D[b * n + d] - D[a * n + b] - D[c * n + d] < 0) {
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

                    double previous = 2.0 * (D[a * n + b] + D[c * n + d] + D[e * n + f]);
                    double after = D[a * n + c] + D[a * n + d] + D[c * n + d] + D[b * n + e] + D[b * n + f] + D[e * n + f];
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

                *abcdefs++ = a;
                *abcdefs++ = b;
                *abcdefs++ = c;
                *abcdefs++ = d;
                *abcdefs++ = e;
                *abcdefs++ = f;

                merged[i] = true;
                merged[max_near_a_at] = true;
                merged[max_near_b_at] = true;
            }
        }
    }
}

double max3(double a, double b, double c)
{
    if (a <= c && b <= c) {
        return c;
    }
    if (a <= b && c <= b) {
        return b;
    }
    return a;
}

double min3(double a, double b, double c)
{
    if (a >= c && b >= c) {
        return c;
    }
    if (a >= b && c >= b) {
        return b;
    }
    return a;
}

void find_cyc(int v, int w, int x, int y, int z, int cyc_id, int* result_at)
{
    switch (cyc_id) {
    case 0:
        *result_at++ = v;
        *result_at++ = w;
        *result_at++ = x;
        *result_at++ = y;
        *result_at++ = z;
        *result_at++ = v;
        return;
    case 1:
        *result_at++ = v;
        *result_at++ = w;
        *result_at++ = x;
        *result_at++ = z;
        *result_at++ = y;
        *result_at++ = v;
        return;
    case 2:
        *result_at++ = v;
        *result_at++ = w;
        *result_at++ = y;
        *result_at++ = x;
        *result_at++ = z;
        *result_at++ = v;
        return;
    case 3:
        *result_at++ = v;
        *result_at++ = w;
        *result_at++ = y;
        *result_at++ = z;
        *result_at++ = x;
        *result_at++ = v;
        return;
    case 4:
        *result_at++ = v;
        *result_at++ = w;
        *result_at++ = z;
        *result_at++ = x;
        *result_at++ = y;
        *result_at++ = v;
        return;
    case 5:
        *result_at++ = v;
        *result_at++ = w;
        *result_at++ = z;
        *result_at++ = y;
        *result_at++ = x;
        *result_at++ = v;
        return;
    case 6:
        *result_at++ = v;
        *result_at++ = x;
        *result_at++ = w;
        *result_at++ = y;
        *result_at++ = z;
        *result_at++ = v;
        return;
    case 7:
        *result_at++ = v;
        *result_at++ = x;
        *result_at++ = w;
        *result_at++ = z;
        *result_at++ = y;
        *result_at++ = v;
        return;
    case 8:
        *result_at++ = v;
        *result_at++ = x;
        *result_at++ = y;
        *result_at++ = w;
        *result_at++ = z;
        *result_at++ = v;
        return;
    case 9:
        *result_at++ = v;
        *result_at++ = x;
        *result_at++ = z;
        *result_at++ = w;
        *result_at++ = y;
        *result_at++ = v;
        return;
    case 10:
        *result_at++ = v;
        *result_at++ = y;
        *result_at++ = w;
        *result_at++ = x;
        *result_at++ = z;
        *result_at++ = v;
        return;
    case 11:
        *result_at++ = v;
        *result_at++ = y;
        *result_at++ = x;
        *result_at++ = w;
        *result_at++ = z;
        *result_at++ = v;
        return;
    }
}

struct Comp235 {
    int c1;
    int c2;
    int cyc_id;
    double gain;
};

bool acompare(Comp235 lhs, Comp235 rhs) { return lhs.gain > rhs.gain; }

void c_cy_post_2_3_to_5(double* D, int* comps_of_2, int* comps_of_3,
    int* abcdes, int m2, int m3, int n)
{
    int* abcdes_at = abcdes;
    std::vector<Comp235> gains;
    for (int c2 = 0; c2 < m3; c2++) {
        int c = comps_of_3[c2 * 3];
        int d = comps_of_3[c2 * 3 + 1];
        int e = comps_of_3[c2 * 3 + 2];
        double max_side_triangle = max3(D[c * n + d], D[d * n + e], D[e * n + c]);

        for (int c1 = 0; c1 < m2; c1++) {
            int a = comps_of_2[c1 * 2];
            int b = comps_of_2[c1 * 2 + 1];

            if ((max_side_triangle + D[a * n + b]) <= (min3(D[a * n + c], D[a * n + d], D[a * n + e]) + min3(D[b * n + c], D[b * n + d], D[b * n + e]))) {
                continue;
            }
            double current = 2.0 * D[a * n + b] + D[c * n + d] + D[d * n + e] + D[e * n + c];

            int v = a;
            int w = b;
            int x = c;
            int y = d;
            int z = e;

            double d_vw = D[v * n + w];
            double d_vx = D[v * n + x];
            double d_vy = D[v * n + y];
            double d_vz = D[v * n + z];
            double d_wx = D[w * n + x];
            double d_wy = D[w * n + y];
            double d_wz = D[w * n + z];
            double d_xy = D[x * n + y];
            double d_xz = D[x * n + z];
            double d_yz = D[y * n + z];

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

        *abcdes_at++ = a;
        *abcdes_at++ = b;
        *abcdes_at++ = c;
        *abcdes_at++ = d;
        *abcdes_at++ = e;
        find_cyc(a, b, c, d, e, comp235.cyc_id, cyc_array);
        *abcdes_at++ = cyc_array[0];
        *abcdes_at++ = cyc_array[1];
        *abcdes_at++ = cyc_array[2];
        *abcdes_at++ = cyc_array[3];
        *abcdes_at++ = cyc_array[4];
        *abcdes_at++ = cyc_array[5];

        merge2.insert(c1);
        merge3.insert(c2);
    }
}