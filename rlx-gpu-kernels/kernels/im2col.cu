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

// RLX — versatile ML compiler + runtime.
//
// NCHW im2col → [M, C·kH·kW] row layout (M = N · H_out · W_out), arena f32 offsets.

extern "C" __global__ void im2col(
    float* arena,
    unsigned int n,
    unsigned int c_in,
    unsigned int h,
    unsigned int w,
    unsigned int h_out,
    unsigned int w_out,
    unsigned int kh,
    unsigned int kw,
    unsigned int sh,
    unsigned int sw,
    unsigned int ph,
    unsigned int pw,
    unsigned int dh,
    unsigned int dw,
    unsigned int x_off,
    unsigned int col_off
) {
    unsigned int k = c_in * kh * kw;
    unsigned int m = n * h_out * w_out;
    unsigned int total = m * k;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    unsigned int elem = i % k;
    unsigned int row = i / k;

    unsigned int wo = row % w_out;
    unsigned int q1 = row / w_out;
    unsigned int ho = q1 % h_out;
    unsigned int ni = q1 / h_out;

    unsigned int kj = elem % kw;
    unsigned int q2 = elem / kw;
    unsigned int ki = q2 % kh;
    unsigned int ci = q2 / kh;

    int hi = (int)(ho * sh + ki * dh) - (int)ph;
    int wi = (int)(wo * sw + kj * dw) - (int)pw;

    float v = 0.0f;
    if (hi >= 0 && wi >= 0 && hi < (int)h && wi < (int)w) {
        unsigned int x_base = ni * c_in * h * w;
        v = arena[x_off + x_base + (ci * h + (unsigned int)hi) * w + (unsigned int)wi];
    }
    arena[col_off + i] = v;
}
