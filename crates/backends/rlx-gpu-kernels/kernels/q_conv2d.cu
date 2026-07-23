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

// Real INT8 `Op::QConv2d` matching `rlx_cpu` `Thunk::QConv2d` /
// `exec_dispatch` (NCHW, grouped).
//
//   x[N,C_in,H,W] i8, w[C_out,C_in/groups,kH,kW] i8, bias[C_out] i32
//     → out[N,C_out,H_out,W_out] i8
//   out = clamp(round((bias + Σ (x−x_zp)(w−w_zp)) · mult) + out_zp, -128, 127)
//
// Arena layout (f32 buffer base):
//   - x / w / out: packed i8 at byte offsets (same as QMatMul / QuantizeI8)
//   - bias: f32-lane I32 convention (value stored as float, cast to int)

__device__ __forceinline__ float round_half_away(float x) {
    float sgn = (x > 0.0f) - (x < 0.0f);
    return sgn * floorf(fabsf(x) + 0.5f);
}

extern "C" __global__ void q_conv2d(
    float* arena,
    unsigned int n,
    unsigned int c_in,
    unsigned int c_out,
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
    unsigned int groups,
    unsigned int x_byte_off,
    unsigned int w_byte_off,
    unsigned int bias_off,
    unsigned int out_byte_off,
    int x_zp,
    int w_zp,
    int out_zp,
    unsigned int mult_bits
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

    const signed char* x = reinterpret_cast<const signed char*>(arena) + x_byte_off;
    const signed char* wt = reinterpret_cast<const signed char*>(arena) + w_byte_off;
    signed char* out = reinterpret_cast<signed char*>(arena) + out_byte_off;

    int acc = (int)truncf(arena[bias_off + co]);
    for (unsigned int ci_off = 0u; ci_off < c_in_per_g; ++ci_off) {
        unsigned int ci = ci_start + ci_off;
        unsigned int in_chan = ((nn * c_in) + ci) * h * w;
        unsigned int wt_chan = ((co * c_in_per_g) + ci_off) * kh * kw;
        for (unsigned int ki = 0u; ki < kh; ++ki) {
            for (unsigned int kj = 0u; kj < kw; ++kj) {
                int ih = (int)(ho * sh + ki * dh) - (int)ph;
                int iw = (int)(wo * sw + kj * dw) - (int)pw;
                if (ih < 0 || iw < 0 || ih >= (int)h || iw >= (int)w) continue;
                int xv = (int)x[in_chan + (unsigned int)ih * w + (unsigned int)iw] - x_zp;
                int wv = (int)wt[wt_chan + ki * kw + kj] - w_zp;
                acc += xv * wv;
            }
        }
    }
    float mult = __uint_as_float(mult_bits);
    int r = (int)round_half_away((float)acc * mult) + out_zp;
    if (r < -128) r = -128;
    if (r > 127) r = 127;
    out[i] = (signed char)r;
}
