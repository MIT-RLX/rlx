// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// 3D NCDHW transposed conv (output-centric gather).
// Weight: [c_in, c_out/groups, kd, kh, kw] (PyTorch ConvTranspose3d).
// Mirrors wgpu `conv_transpose3d.wgsl` / Metal `conv_transpose3d`.

extern "C" __global__ void conv_transpose3d(
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
    unsigned int oc_off = co % c_out_per_g;
    unsigned int ci_start = g * c_in_per_g;

    float acc = 0.0f;
    for (unsigned int kz = 0; kz < kd; ++kz) {
        int num_d = (int)do_ + (int)pd - (int)(kz * dd);
        if (num_d < 0 || (num_d % (int)sd) != 0) continue;
        unsigned int id = (unsigned int)(num_d / (int)sd);
        if (id >= d) continue;
        for (unsigned int ky = 0; ky < kh; ++ky) {
            int num_h = (int)ho + (int)ph - (int)(ky * dh);
            if (num_h < 0 || (num_h % (int)sh) != 0) continue;
            unsigned int ih = (unsigned int)(num_h / (int)sh);
            if (ih >= h) continue;
            for (unsigned int kx = 0; kx < kw; ++kx) {
                int num_w = (int)wo + (int)pw - (int)(kx * dw);
                if (num_w < 0 || (num_w % (int)sw) != 0) continue;
                unsigned int iw = (unsigned int)(num_w / (int)sw);
                if (iw >= w) continue;
                for (unsigned int ci_off = 0; ci_off < c_in_per_g; ++ci_off) {
                    unsigned int ci = ci_start + ci_off;
                    unsigned int in_idx =
                        (((nn * c_in + ci) * d + id) * h + ih) * w + iw;
                    unsigned int w_idx =
                        (((ci * c_out_per_g + oc_off) * kd + kz) * kh + ky) * kw + kx;
                    acc += arena[in_off + in_idx] * arena[w_off + w_idx];
                }
            }
        }
    }
    arena[out_off + i] = acc;
}
