// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// MaxPool3d backward (NCDHW): one thread per input element. Ties keep the
// first max, matching forward pool3d.
extern "C" __global__ void maxpool3d_backward(
    float* arena,
    unsigned n, unsigned c,
    unsigned d, unsigned h, unsigned w,
    unsigned d_out, unsigned h_out, unsigned w_out,
    unsigned kd, unsigned kh, unsigned kw,
    unsigned sd, unsigned sh, unsigned sw,
    unsigned pd, unsigned ph, unsigned pw,
    unsigned x_off, unsigned dy_off, unsigned dx_off)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = n * c * d * h * w;
    if (idx >= total) return;
    unsigned iw = idx % w;
    unsigned q1 = idx / w;
    unsigned ih = q1 % h;
    unsigned q2 = q1 / h;
    unsigned id = q2 % d;
    unsigned q3 = q2 / d;
    unsigned cc = q3 % c;
    unsigned nn = q3 / c;
    const float* x = arena + x_off;
    unsigned base_nc = (nn * c + cc) * d * h * w;
    int pd_i = (int)pd, ph_i = (int)ph, pw_i = (int)pw;
    int sd_i = (int)sd, sh_i = (int)sh, sw_i = (int)sw;

    int do_lo = (int)id + pd_i - (int)kd + 1;
    do_lo = do_lo <= 0 ? 0 : (do_lo + sd_i - 1) / sd_i;
    int do_hi = ((int)id + pd_i) / sd_i;
    int ho_lo = (int)ih + ph_i - (int)kh + 1;
    ho_lo = ho_lo <= 0 ? 0 : (ho_lo + sh_i - 1) / sh_i;
    int ho_hi = ((int)ih + ph_i) / sh_i;
    int wo_lo = (int)iw + pw_i - (int)kw + 1;
    wo_lo = wo_lo <= 0 ? 0 : (wo_lo + sw_i - 1) / sw_i;
    int wo_hi = ((int)iw + pw_i) / sw_i;

    float acc = 0.0f;
    for (int do_ = do_lo; do_ <= do_hi && do_ < (int)d_out; do_++) {
        int dstart = do_ * sd_i - pd_i;
        for (int ho = ho_lo; ho <= ho_hi && ho < (int)h_out; ho++) {
            int hstart = ho * sh_i - ph_i;
            for (int wo = wo_lo; wo <= wo_hi && wo < (int)w_out; wo++) {
                int wstart = wo * sw_i - pw_i;
                float best = -3.402823466e+38f;
                int best_idx = -1;
                for (unsigned kz = 0; kz < kd; kz++) {
                    int irz = dstart + (int)kz;
                    if (irz < 0 || irz >= (int)d) continue;
                    for (unsigned i = 0; i < kh; i++) {
                        int ir = hstart + (int)i;
                        if (ir < 0 || ir >= (int)h) continue;
                        for (unsigned j = 0; j < kw; j++) {
                            int ic = wstart + (int)j;
                            if (ic < 0 || ic >= (int)w) continue;
                            unsigned id3 = base_nc + ((unsigned)irz * h + (unsigned)ir) * w + (unsigned)ic;
                            float v = x[id3];
                            if (v > best) { best = v; best_idx = (int)id3; }
                        }
                    }
                }
                if (best_idx == (int)idx) {
                    acc += arena[dy_off + ((((nn * c + cc) * d_out + (unsigned)do_) * h_out + (unsigned)ho) * w_out + (unsigned)wo)];
                }
            }
        }
    }
    arena[dx_off + idx] = acc;
}
