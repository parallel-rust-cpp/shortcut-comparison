#include "step.hpp"
#include "simd.hpp"

void step(float* r, const float* d_, int n) {
    constexpr int nb = 8;
    int na = (n + nb - 1) / nb;

    constexpr int nd = 3;
    int nc = (n + nd - 1) / nd;
    int ncd = nc * nd;

    float8_t* vd = float8_alloc(ncd*na);
    float8_t* vt = float8_alloc(ncd*na);

#if !NO_MULTI_THREAD
    #pragma omp parallel for
#endif
    for (int j = 0; j < n; ++j) {
        for (int ka = 0; ka < na; ++ka) {
            for (int kb = 0; kb < nb; ++kb) {
                int i = ka * nb + kb;
                vd[na*j + ka][kb] = i < n ? d_[n*j + i] : infty;
                vt[na*j + ka][kb] = i < n ? d_[n*i + j] : infty;
            }
        }
    }
    for (int j = n; j < ncd; ++j) {
        for (int ka = 0; ka < na; ++ka) {
            for (int kb = 0; kb < nb; ++kb) {
                vd[na*j + ka][kb] = infty;
                vt[na*j + ka][kb] = infty;
            }
        }
    }

#if !NO_MULTI_THREAD
    #pragma omp parallel for
#endif
    for (int ic = 0; ic < nc; ++ic) {
        for (int jc = 0; jc < nc; ++jc) {
            float8_t vv[nd][nd];
            for (int id = 0; id < nd; ++id) {
                for (int jd = 0; jd < nd; ++jd) {
                    vv[id][jd] = f8infty;
                }
            }
            for (int ka = 0; ka < na; ++ka) {
                float8_t y0 = vt[na*(jc * nd + 0) + ka];
                float8_t y1 = vt[na*(jc * nd + 1) + ka];
                float8_t y2 = vt[na*(jc * nd + 2) + ka];
                float8_t x0 = vd[na*(ic * nd + 0) + ka];
                float8_t x1 = vd[na*(ic * nd + 1) + ka];
                float8_t x2 = vd[na*(ic * nd + 2) + ka];
                vv[0][0] = min8(vv[0][0], x0 + y0);
                vv[0][1] = min8(vv[0][1], x0 + y1);
                vv[0][2] = min8(vv[0][2], x0 + y2);
                vv[1][0] = min8(vv[1][0], x1 + y0);
                vv[1][1] = min8(vv[1][1], x1 + y1);
                vv[1][2] = min8(vv[1][2], x1 + y2);
                vv[2][0] = min8(vv[2][0], x2 + y0);
                vv[2][1] = min8(vv[2][1], x2 + y1);
                vv[2][2] = min8(vv[2][2], x2 + y2);
            }
            for (int id = 0; id < nd; ++id) {
                for (int jd = 0; jd < nd; ++jd) {
                    int i = ic * nd + id;
                    int j = jc * nd + jd;
                    if (i < n && j < n) {
                        r[n*i + j] = hmin8(vv[id][jd]);
                    }
                }
            }
        }
    }

    std::free(vt);
    std::free(vd);
}
