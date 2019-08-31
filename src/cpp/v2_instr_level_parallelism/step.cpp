/*
 * From http://ppc.cs.aalto.fi/ch2/v2/
 */
#include <limits>
#include <algorithm>
#include <vector>
#include "step.hpp"

constexpr float infty = std::numeric_limits<float>::infinity();

void step(float* r, const float* d_input, int n) {
    constexpr int nb = 4;
    int na = (n + nb - 1) / nb;
    int nab = na*nb;

    std::vector<float> d(n*nab, infty);
    std::vector<float> t(n*nab, infty);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            d[nab*i + j] = d_input[n*i + j];
            t[nab*i + j] = d_input[n*j + i];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float vv[nb];
            for (int kb = 0; kb < nb; ++kb) {
                vv[kb] = infty;
            }
            for (int ka = 0; ka < na; ++ka) {
                for (int kb = 0; kb < nb; ++kb) {
                    float x = d[nab*i + ka * nb + kb];
                    float y = t[nab*j + ka * nb + kb];
                    float z = x + y;
                    vv[kb] = std::min(vv[kb], z);
                }
            }
            float v = infty;
            for (int kb = 0; kb < nb; ++kb) {
                v = std::min(vv[kb], v);
            }
            r[n*i + j] = v;
        }
    }
}

