// NCHW transposed convolution (PyTorch ConvTranspose2d, no bias).
// Weight layout [C_in, C_out/groups, kH, kW]. One work-item per output.
// Mirrors Vulkan `conv_transpose2d.comp` / CUDA `conv_transpose2d.cu`.

__kernel void conv_transpose2d(__global float* arena,
                     uint nn, uint cin, uint hh, uint ww,
                     uint cout, uint oh, uint ow,
                     uint kh, uint kw,
                     uint sh, uint sw, uint ph, uint pw, uint dh, uint dw,
                     uint groups,
                     uint x_off, uint w_off, uint out_off) {
    uint total = nn * cout * oh * ow;
    uint gid = get_global_id(0);
    if (gid >= total) return;

    uint wo = gid % ow;
    uint q1 = gid / ow;
    uint ho = q1 % oh;
    uint q2 = q1 / oh;
    uint co = q2 % cout;
    uint n = q2 / cout;

    uint cin_pg = cin / groups;
    uint cout_pg = cout / groups;
    uint g = co / cout_pg;
    uint oc_off = co % cout_pg;
    uint ci_start = g * cin_pg;

    float acc = 0.0f;
    for (uint ci_off = 0u; ci_off < cin_pg; ci_off++) {
        uint ci = ci_start + ci_off;
        for (uint ky = 0u; ky < kh; ky++) {
            int t_h = (int)ho + (int)ph - (int)(ky * dh);
            if (t_h < 0 || (t_h % (int)sh) != 0) continue;
            uint iy = (uint)(t_h / (int)sh);
            if (iy >= hh) continue;
            for (uint kx = 0u; kx < kw; kx++) {
                int t_w = (int)wo + (int)pw - (int)(kx * dw);
                if (t_w < 0 || (t_w % (int)sw) != 0) continue;
                uint ix = (uint)(t_w / (int)sw);
                if (ix >= ww) continue;
                uint w_idx = ((ci * cout_pg + oc_off) * kh + ky) * kw + kx;
                float v = arena[x_off + ((n * cin + ci) * hh + iy) * ww + ix];
                acc += v * arena[w_off + w_idx];
            }
        }
    }
    arena[out_off + gid] = acc;
}
