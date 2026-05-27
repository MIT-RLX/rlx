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

// 2D convolution NCHW. Weight: [c_out, c_in/groups, kh, kw].

extern "C" __global__ void conv2d(
    float* arena,
    unsigned int n, unsigned int c_in, unsigned int c_out,
    unsigned int h, unsigned int w,
    unsigned int h_out, unsigned int w_out,
    unsigned int kh, unsigned int kw,
    unsigned int sh, unsigned int sw,
    unsigned int ph, unsigned int pw,
    unsigned int dh, unsigned int dw,
    unsigned int groups,
    unsigned int in_off,
    unsigned int w_off,
    unsigned int out_off
) {
    unsigned int total = n * c_out * h_out * w_out;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int wo = i % w_out;
    unsigned int q1 = i / w_out;
    unsigned int ho = q1 % h_out;
    unsigned int q2 = q1 / h_out;
    unsigned int co = q2 % c_out;
    unsigned int nn = q2 / c_out;

    unsigned int c_in_per_g = c_in / groups;
    unsigned int c_out_per_g = c_out / groups;
    unsigned int g = co / c_out_per_g;
    unsigned int ci_start = g * c_in_per_g;

    float acc = 0.0f;
    for (unsigned int ci_off = 0; ci_off < c_in_per_g; ++ci_off) {
        unsigned int ci = ci_start + ci_off;
        for (unsigned int ki = 0; ki < kh; ++ki) {
            for (unsigned int kj = 0; kj < kw; ++kj) {
                int ih = (int)(ho * sh + ki * dh) - (int)ph;
                int iw = (int)(wo * sw + kj * dw) - (int)pw;
                if (ih < 0 || iw < 0 || ih >= (int)h || iw >= (int)w) continue;
                float xv = arena[in_off + ((nn * c_in + ci) * h + ih) * w + iw];
                float wv = arena[w_off + (((co * c_in_per_g + ci_off) * kh + ki) * kw + kj)];
                acc += xv * wv;
            }
        }
    }
    arena[out_off + i] = acc;
}
