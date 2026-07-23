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

// MaxPool2d backward (NCHW): one thread per input element. For each input
// location, sum `dy` over output windows that selected this location as the
// argmax (ties keep the first max, matching the forward pool2d kernel).

extern "C" __global__ void maxpool2d_backward(
    float* arena, unsigned n, unsigned c, unsigned h, unsigned w,
    unsigned h_out, unsigned w_out, unsigned kh, unsigned kw,
    unsigned sh, unsigned sw, unsigned ph, unsigned pw,
    unsigned x_off, unsigned dy_off, unsigned dx_off)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = n*c*h*w;
    if (idx >= total) return;
    unsigned iw = idx % w;
    unsigned ih = (idx / w) % h;
    unsigned cc = (idx / (w*h)) % c;
    unsigned nn = idx / (w*h*c);
    const float* x = arena + x_off;
    unsigned base_nc = (nn*c + cc)*h*w;
    int ph_i = (int)ph, pw_i = (int)pw, sh_i = (int)sh, sw_i = (int)sw;
    // output windows (ho,wo) whose receptive field covers input (ih,iw)
    int ho_lo = (int)ih + ph_i - (int)kh + 1;
    ho_lo = ho_lo <= 0 ? 0 : (ho_lo + sh_i - 1) / sh_i;
    int ho_hi = ((int)ih + ph_i) / sh_i;
    int wo_lo = (int)iw + pw_i - (int)kw + 1;
    wo_lo = wo_lo <= 0 ? 0 : (wo_lo + sw_i - 1) / sw_i;
    int wo_hi = ((int)iw + pw_i) / sw_i;
    float acc = 0.0f;
    for (int ho = ho_lo; ho <= ho_hi && ho < (int)h_out; ho++) {
        int hstart = ho*sh_i - ph_i;
        for (int wo = wo_lo; wo <= wo_hi && wo < (int)w_out; wo++) {
            int wstart = wo*sw_i - pw_i;
            float best = -3.402823466e+38f; int best_idx = -1;
            for (unsigned i=0;i<kh;i++){
                int ir = hstart + (int)i;
                if (ir < 0 || ir >= (int)h) continue;
                for (unsigned j=0;j<kw;j++){
                    int ic = wstart + (int)j;
                    if (ic < 0 || ic >= (int)w) continue;
                    unsigned id2 = base_nc + (unsigned)ir*w + (unsigned)ic;
                    float v = x[id2];
                    if (v > best){ best = v; best_idx = (int)id2; }
                }
            }
            if (best_idx == (int)idx) {
                acc += arena[dy_off + ((nn*c+cc)*h_out + (unsigned)ho)*w_out + (unsigned)wo];
            }
        }
    }
    arena[dx_off + idx] = acc;
}
