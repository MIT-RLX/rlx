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

// 2D convolution backward-weight (dw). Weight: [c_out, c_in/groups, kh, kw] —
// the SAME layout as the forward `conv2d.cu`. One thread owns one dw element and
// GATHERS over the batch and output spatial positions (no atomics), accumulating
// `dy[n,co,ho,wo] * x[n,ci,ih,iw]` with `ih = ho*sh + ki*dh - ph`. Pure
// host-parity fallback for when cuDNN is unavailable.
extern "C" __global__ void conv2d_backward_weight(
    float* arena,
    unsigned int n, unsigned int c_in, unsigned int c_out,
    unsigned int h, unsigned int w,
    unsigned int h_out, unsigned int w_out,
    unsigned int kh, unsigned int kw,
    unsigned int sh, unsigned int sw,
    unsigned int ph, unsigned int pw,
    unsigned int dh, unsigned int dw,
    unsigned int groups,
    unsigned int x_off,  // input       [n, c_in, h, w]
    unsigned int dy_off, // grad_output [n, c_out, h_out, w_out]
    unsigned int dw_off  // grad_weight [c_out, c_in/groups, kh, kw]
) {
    unsigned int c_in_per_g = c_in / groups;
    unsigned int c_out_per_g = c_out / groups;
    unsigned int total = c_out * c_in_per_g * kh * kw;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int kj = i % kw;
    unsigned int q1 = i / kw;
    unsigned int ki = q1 % kh;
    unsigned int q2 = q1 / kh;
    unsigned int ci_off = q2 % c_in_per_g;
    unsigned int co = q2 / c_in_per_g;

    unsigned int g = co / c_out_per_g;
    unsigned int ci = g * c_in_per_g + ci_off;

    float acc = 0.0f;
    for (unsigned int nn = 0; nn < n; ++nn) {
        for (unsigned int ho = 0; ho < h_out; ++ho) {
            int ih = (int)(ho * sh + ki * dh) - (int)ph;
            if (ih < 0 || ih >= (int)h) continue;
            for (unsigned int wo = 0; wo < w_out; ++wo) {
                int iw = (int)(wo * sw + kj * dw) - (int)pw;
                if (iw < 0 || iw >= (int)w) continue;
                float dyv = arena[dy_off + ((nn * c_out + co) * h_out + ho) * w_out + wo];
                float xv = arena[x_off + ((nn * c_in + ci) * h + ih) * w + iw];
                acc += dyv * xv;
            }
        }
    }
    arena[dw_off + i] = acc;
}
