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

// 3D conv backward-weight (dw) NCDHW. Weight: [c_out, c_in/groups, kd, kh, kw].
extern "C" __global__ void conv3d_backward_weight(
    float* arena,
    unsigned int n, unsigned int c_in, unsigned int c_out,
    unsigned int d, unsigned int h, unsigned int w,
    unsigned int d_out, unsigned int h_out, unsigned int w_out,
    unsigned int kd, unsigned int kh, unsigned int kw,
    unsigned int sd, unsigned int sh, unsigned int sw,
    unsigned int pd, unsigned int ph, unsigned int pw,
    unsigned int dd, unsigned int dh, unsigned int dw,
    unsigned int groups,
    unsigned int x_off,
    unsigned int dy_off,
    unsigned int dw_off
) {
    unsigned int c_in_per_g = c_in / groups;
    unsigned int c_out_per_g = c_out / groups;
    unsigned int total = c_out * c_in_per_g * kd * kh * kw;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    unsigned int kj = i % kw;
    unsigned int q1 = i / kw;
    unsigned int ki = q1 % kh;
    unsigned int q2 = q1 / kh;
    unsigned int kz = q2 % kd;
    unsigned int q3 = q2 / kd;
    unsigned int ci_off = q3 % c_in_per_g;
    unsigned int co = q3 / c_in_per_g;

    unsigned int g = co / c_out_per_g;
    unsigned int ci = g * c_in_per_g + ci_off;

    float hi = 0.0f, lo = 0.0f;
    for (unsigned int nn = 0; nn < n; ++nn) {
        for (unsigned int do_ = 0; do_ < d_out; ++do_) {
            int id = (int)(do_ * sd + kz * dd) - (int)pd;
            if (id < 0 || id >= (int)d) continue;
            for (unsigned int ho = 0; ho < h_out; ++ho) {
                int ih = (int)(ho * sh + ki * dh) - (int)ph;
                if (ih < 0 || ih >= (int)h) continue;
                for (unsigned int wo = 0; wo < w_out; ++wo) {
                    int iw = (int)(wo * sw + kj * dw) - (int)pw;
                    if (iw < 0 || iw >= (int)w) continue;
                    float dyv = arena[dy_off + ((((nn * c_out + co) * d_out + do_) * h_out + ho) * w_out + wo)];
                    float xv = arena[x_off + ((((nn * c_in + ci) * d + (unsigned)id) * h + (unsigned)ih) * w + (unsigned)iw)];
                    float p = dyv * xv;
                    float ep = __fmaf_rn(dyv, xv, -p);
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
    arena[dw_off + i] = hi;
}
