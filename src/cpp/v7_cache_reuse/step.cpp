/*
 * From http://ppc.cs.aalto.fi/ch2/v7/
 */
#include <algorithm>
#if !NO_MULTI_THREAD
#include <parallel/algorithm>
#endif
#include <vector>
#include <tuple>
#include "step.hpp"
#include "simd.hpp"


void step(float* r, const float* d_, int n) {
    constexpr int cols_per_stripe = 500;
    int na = (n + 8 - 1) / 8;

    // Build a Z-order curve memory access pattern for vd and vt
    std::vector<std::tuple<int,int,int>> rows(na*na);
    #pragma omp parallel for
    for (int ia = 0; ia < na; ++ia) {
        for (int ja = 0; ja < na; ++ja) {
            int ija = _pdep_u32(ia, 0x55555555) | _pdep_u32(ja, 0xAAAAAAAA);
            rows[ia*na + ja] = std::make_tuple(ija, ia, ja);
        }
    }
#if !NO_MULTI_THREAD
    __gnu_parallel::sort(rows.begin(), rows.end());
#else
    std::sort(rows.begin(), rows.end());
#endif

    // One stripe of rows and cols packed into float8_ts
    float8_t* vd = float8_alloc(na*cols_per_stripe);
    float8_t* vt = float8_alloc(na*cols_per_stripe);
    // Partial results
    float8_t* vr = float8_alloc(na*na*8);

    for (int stripe = 0; stripe < n; stripe += cols_per_stripe) {
        int stripe_end = std::min(n - stripe, cols_per_stripe);

        // Load one row and column stripe from d
        #pragma omp parallel for
        for (int ja = 0; ja < na; ++ja) {
            for (int id = 0; id < stripe_end; ++id) {
                int t = cols_per_stripe * ja + id;
                int i = stripe + id;
                for (int jb = 0; jb < 8; ++jb) {
                    int j = ja * 8 + jb;
                    vd[t][jb] = j < n ? d_[n*j + i] : infty;
                    vt[t][jb] = j < n ? d_[n*i + j] : infty;
                }
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < na * na; ++i) {
            // Get corresponding pair (x, y) for Z-order value ija
            int ija, ia, ja;
            std::tie(ija, ia, ja) = rows[i];
            (void)ija;
            // If we are not at column 0, then the partial results contain something useful, else the partial results are uninitialized
            float8_t vv000 = stripe ? vr[8*i + 0] : f8infty;
            float8_t vv001 = stripe ? vr[8*i + 1] : f8infty;
            float8_t vv010 = stripe ? vr[8*i + 2] : f8infty;
            float8_t vv011 = stripe ? vr[8*i + 3] : f8infty;
            float8_t vv100 = stripe ? vr[8*i + 4] : f8infty;
            float8_t vv101 = stripe ? vr[8*i + 5] : f8infty;
            float8_t vv110 = stripe ? vr[8*i + 6] : f8infty;
            float8_t vv111 = stripe ? vr[8*i + 7] : f8infty;

            // Compute partial results for this column chunk
            for (int k = 0; k < stripe_end; ++k) {
                float8_t a000 = vd[cols_per_stripe*ia + k];
                float8_t b000 = vt[cols_per_stripe*ja + k];
                float8_t a100 = swap4(a000);
                float8_t a010 = swap2(a000);
                float8_t a110 = swap2(a100);
                float8_t b001 = swap1(b000);
                vv000 = min8(vv000, a000 + b000);
                vv001 = min8(vv001, a000 + b001);
                vv010 = min8(vv010, a010 + b000);
                vv011 = min8(vv011, a010 + b001);
                vv100 = min8(vv100, a100 + b000);
                vv101 = min8(vv101, a100 + b001);
                vv110 = min8(vv110, a110 + b000);
                vv111 = min8(vv111, a110 + b001);
            }
            vr[8*i + 0] = vv000;
            vr[8*i + 1] = vv001;
            vr[8*i + 2] = vv010;
            vr[8*i + 3] = vv011;
            vr[8*i + 4] = vv100;
            vr[8*i + 5] = vv101;
            vr[8*i + 6] = vv110;
            vr[8*i + 7] = vv111;
        }
    }

    // Unpack final results from partial results
    #pragma omp parallel for
    for (int i = 0; i < na * na; ++i) {
        int ija, ia, ja;
        std::tie(ija, ia, ja) = rows[i];
        (void)ija;
        float8_t vv[8];
        for (int kb = 0; kb < 8; ++kb) {
            vv[kb] = vr[8*i + kb];
        }
        for (int kb = 1; kb < 8; kb += 2) {
            vv[kb] = swap1(vv[kb]);
        }
        for (int jb = 0; jb < 8; ++jb) {
            for (int ib = 0; ib < 8; ++ib) {
                int i = ib + ia*8;
                int j = jb + ja*8;
                if (j < n && i < n) {
                    r[n*i + j] = vv[ib^jb][jb];
                }
            }
        }
    }

    std::free(vd);
    std::free(vt);
    std::free(vr);
}
