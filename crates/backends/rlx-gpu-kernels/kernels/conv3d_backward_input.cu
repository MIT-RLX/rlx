// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// 3D conv backward-input (dx) NCDHW. Weight: [c_out, c_in/groups, kd, kh, kw].
// One thread per dx element; gather over output windows (no atomics).
extern "C" __global__ void conv3d_backward_input(
    float* arena,
    unsigned int n, unsigned int c_in, unsigned int c_out,
    unsigned int d, unsigned int h, unsigned int w,
    unsigned int d_out, unsigned int h_out, unsigned int w_out,
    unsigned int kd, unsigned int kh, unsigned int kw,
    unsigned int sd, unsigned int sh, unsigned int sw,
    unsigned int pd, unsigned int ph, unsigned int pw,
    unsigned int dd, unsigned int dh, unsigned int dw,
    unsigned int groups,
    unsigned int dy_off,
    unsigned int w_off,
    unsigned int dx_off
) {
    unsigned int total = n * c_in * d * h * w;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int iw = i % w;
    unsigned int q1 = i / w;
    unsigned int ih = q1 % h;
    unsigned int q2 = q1 / h;
    unsigned int id = q2 % d;
    unsigned int q3 = q2 / d;
    unsigned int ci = q3 % c_in;
    unsigned int nn = q3 / c_in;

    unsigned int c_in_per_g = c_in / groups;
    unsigned int c_out_per_g = c_out / groups;
    unsigned int g = ci / c_in_per_g;
    unsigned int ci_off = ci - g * c_in_per_g;
    unsigned int co_start = g * c_out_per_g;

    float hi = 0.0f, lo = 0.0f;
    for (unsigned int kz = 0; kz < kd; ++kz) {
        int num_d = (int)(id + pd) - (int)(kz * dd);
        if (num_d < 0 || (unsigned int)num_d % sd != 0) continue;
        unsigned int do_ = (unsigned int)num_d / sd;
        if (do_ >= d_out) continue;
        for (unsigned int ki = 0; ki < kh; ++ki) {
            int num_h = (int)(ih + ph) - (int)(ki * dh);
            if (num_h < 0 || (unsigned int)num_h % sh != 0) continue;
            unsigned int ho = (unsigned int)num_h / sh;
            if (ho >= h_out) continue;
            for (unsigned int kj = 0; kj < kw; ++kj) {
                int num_w = (int)(iw + pw) - (int)(kj * dw);
                if (num_w < 0 || (unsigned int)num_w % sw != 0) continue;
                unsigned int wo = (unsigned int)num_w / sw;
                if (wo >= w_out) continue;
                for (unsigned int co_off = 0; co_off < c_out_per_g; ++co_off) {
                    unsigned int co = co_start + co_off;
                    float dyv = arena[dy_off + ((((nn * c_out + co) * d_out + do_) * h_out + ho) * w_out + wo)];
                    float wv = arena[w_off + (((((co * c_in_per_g + ci_off) * kd + kz) * kh + ki) * kw + kj))];
                    float p = dyv * wv;
                    float ep = __fmaf_rn(dyv, wv, -p);
                    float s = hi + p;
                    float bb = s - hi;
                    float es = (hi - (s - bb)) + (p - bb);
                    lo += ep + es;
                    float t = s + lo;
                    lo -= t - s;
                    hi = t;
                }
            }
        }
    }
    arena[dx_off + i] = hi;
}
