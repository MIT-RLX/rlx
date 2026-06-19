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

// 3D conv NCDHW. Weight: [c_out, c_in/groups, kd, kh, kw].
extern "C" __global__ void conv3d(
    float* arena,
    unsigned int n, unsigned int c_in, unsigned int c_out,
    unsigned int d, unsigned int h, unsigned int w,
    unsigned int d_out, unsigned int h_out, unsigned int w_out,
    unsigned int kd, unsigned int kh, unsigned int kw,
    unsigned int sd, unsigned int sh, unsigned int sw,
    unsigned int pd, unsigned int ph, unsigned int pw,
    unsigned int dd, unsigned int dh, unsigned int dw,
    unsigned int groups,
    unsigned int in_off,
    unsigned int w_off,
    unsigned int out_off
) {
    unsigned int total = n * c_out * d_out * h_out * w_out;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int wo = i % w_out;
    unsigned int q1 = i / w_out;
    unsigned int ho = q1 % h_out;
    unsigned int q2 = q1 / h_out;
    unsigned int do_ = q2 % d_out;
    unsigned int q3 = q2 / d_out;
    unsigned int co = q3 % c_out;
    unsigned int nn = q3 / c_out;
    unsigned int c_in_per_g = c_in / groups;
    unsigned int c_out_per_g = c_out / groups;
    unsigned int g = co / c_out_per_g;
    unsigned int ci_start = g * c_in_per_g;
    float acc = 0.0f;
    for (unsigned int ci_off = 0; ci_off < c_in_per_g; ++ci_off) {
        unsigned int ci = ci_start + ci_off;
        for (unsigned int ki = 0; ki < kd; ++ki)
        for (unsigned int kj = 0; kj < kh; ++kj)
        for (unsigned int kk = 0; kk < kw; ++kk) {
            int id = (int)(do_ * sd + ki * dd) - (int)pd;
            int ih = (int)(ho  * sh + kj * dh) - (int)ph;
            int iw = (int)(wo  * sw + kk * dw) - (int)pw;
            if (id < 0 || ih < 0 || iw < 0
                || id >= (int)d || ih >= (int)h || iw >= (int)w) continue;
            float xv = arena[in_off + (((nn * c_in + ci) * d + id) * h + ih) * w + iw];
            float wv = arena[w_off + ((((co * c_in_per_g + ci_off) * kd + ki) * kh + kj) * kw + kk)];
            acc += xv * wv;
        }
    }
    arena[out_off + i] = acc;
}
