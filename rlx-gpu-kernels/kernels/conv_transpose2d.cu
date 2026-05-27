// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// NCHW transposed convolution (PyTorch ConvTranspose2d, no bias).
// Weight layout [C_in, C_out/groups, kH, kW]. One thread per output.

extern "C" __global__ void conv_transpose2d(
    float* arena,
    unsigned int src_off,
    unsigned int w_off,
    unsigned int dst_off,
    unsigned int n,
    unsigned int c_in,
    unsigned int h,
    unsigned int w_in,
    unsigned int c_out,
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
    unsigned int groups
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
    unsigned int oc_off = co % c_out_per_g;

    float acc = 0.0f;
    for (unsigned int ci_off = 0; ci_off < c_in_per_g; ++ci_off) {
        unsigned int ci = g * c_in_per_g + ci_off;
        for (unsigned int ky = 0; ky < kh; ++ky) {
            int t_h = (int)ho + (int)ph - (int)ky * (int)dh;
            if (t_h < 0 || t_h % (int)sh != 0) continue;
            int iy = t_h / (int)sh;
            if (iy < 0 || iy >= (int)h) continue;
            for (unsigned int kx = 0; kx < kw; ++kx) {
                int t_w = (int)wo + (int)pw - (int)kx * (int)dw;
                if (t_w < 0 || t_w % (int)sw != 0) continue;
                int ix = t_w / (int)sw;
                if (ix < 0 || ix >= (int)w_in) continue;
                unsigned int w_idx = ((ci * c_out_per_g + oc_off) * kh + ky) * kw + kx;
                float v = arena[src_off + ((nn * c_in + ci) * h + (unsigned int)iy) * w_in + (unsigned int)ix];
                acc += v * arena[w_off + w_idx];
            }
        }
    }
    arena[dst_off + i] = acc;
}
