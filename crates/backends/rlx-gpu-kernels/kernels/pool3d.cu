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

// 3D pool NCDHW.
extern "C" __global__ void pool3d(
    float* arena,
    unsigned int n, unsigned int c,
    unsigned int d, unsigned int h, unsigned int w,
    unsigned int d_out, unsigned int h_out, unsigned int w_out,
    unsigned int kd, unsigned int kh, unsigned int kw,
    unsigned int sd, unsigned int sh, unsigned int sw,
    unsigned int pd, unsigned int ph, unsigned int pw,
    unsigned int op,
    unsigned int in_off,
    unsigned int out_off
) {
    unsigned int total = n * c * d_out * h_out * w_out;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int wo = i % w_out;
    unsigned int q1 = i / w_out;
    unsigned int ho = q1 % h_out;
    unsigned int q2 = q1 / h_out;
    unsigned int do_ = q2 % d_out;
    unsigned int q3 = q2 / d_out;
    unsigned int cc = q3 % c;
    unsigned int nn = q3 / c;
    float acc;
    switch (op) {
        case 0: acc = -3.4e38f; break;
        case 3: acc =  3.4e38f; break;
        case 4: acc = 1.0f; break;
        default: acc = 0.0f;
    }
    unsigned int count = 0;
    for (unsigned int ki = 0; ki < kd; ++ki)
    for (unsigned int kj = 0; kj < kh; ++kj)
    for (unsigned int kk = 0; kk < kw; ++kk) {
        int id = (int)(do_ * sd + ki) - (int)pd;
        int ih = (int)(ho  * sh + kj) - (int)ph;
        int iw = (int)(wo  * sw + kk) - (int)pw;
        if (id < 0 || ih < 0 || iw < 0
            || id >= (int)d || ih >= (int)h || iw >= (int)w) continue;
        float v = arena[in_off + (((nn * c + cc) * d + id) * h + ih) * w + iw];
        switch (op) {
            case 0: acc = fmaxf(acc, v); break;
            case 1: case 2: acc += v; break;
            case 3: acc = fminf(acc, v); break;
            case 4: acc *= v; break;
        }
        count++;
    }
    if (op == 1 && count > 0) acc /= (float)count;
    arena[out_off + i] = acc;
}
