// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// 2D convolution backward-input (dx) NCHW. Weight: [c_out, c_in/groups, kh, kw]
// — the SAME layout as the forward `conv2d.cu`. One thread owns one dx element
// and GATHERS the contributions from every output position (ho,wo) and output
// channel (in this input channel's group) whose receptive field covers it, so
// no atomics / col2im are needed. Inverse of the forward index map
// `ih = ho*sh + ki*dh - ph`: `ho = (ih + ph - ki*dh) / sh` (must be exact and in
// range). Pure host-parity fallback for when cuDNN is unavailable.
extern "C" __global__ void conv2d_backward_input(
    float* arena,
    unsigned int n, unsigned int c_in, unsigned int c_out,
    unsigned int h, unsigned int w,
    unsigned int h_out, unsigned int w_out,
    unsigned int kh, unsigned int kw,
    unsigned int sh, unsigned int sw,
    unsigned int ph, unsigned int pw,
    unsigned int dh, unsigned int dw,
    unsigned int groups,
    unsigned int dy_off, // grad_output [n, c_out, h_out, w_out]
    unsigned int w_off,  // weight       [c_out, c_in/groups, kh, kw]
    unsigned int dx_off  // grad_input   [n, c_in, h, w]
) {
    unsigned int total = n * c_in * h * w;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int iw = i % w;
    unsigned int q1 = i / w;
    unsigned int ih = q1 % h;
    unsigned int q2 = q1 / h;
    unsigned int ci = q2 % c_in;
    unsigned int nn = q2 / c_in;

    unsigned int c_in_per_g = c_in / groups;
    unsigned int c_out_per_g = c_out / groups;
    unsigned int g = ci / c_in_per_g;
    unsigned int ci_off = ci - g * c_in_per_g;
    unsigned int co_start = g * c_out_per_g;

    // DOUBLE-SINGLE (f32+f32) accumulation — see conv2d_backward_weight.cu.
    float hi = 0.0f, lo = 0.0f;
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
                float dyv = arena[dy_off + ((nn * c_out + co) * h_out + ho) * w_out + wo];
                float wv = arena[w_off + (((co * c_in_per_g + ci_off) * kh + ki) * kw + kj)];
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
    arena[dx_off + i] = hi;
}
