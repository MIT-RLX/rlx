// 3D NCDHW transposed conv (output-centric gather).
// Weight: [c_in, c_out/groups, kd, kh, kw] (PyTorch ConvTranspose3d).
// Mirrors wgpu `conv_transpose3d.wgsl` / CUDA `conv_transpose3d.cu`.

__kernel void conv_transpose3d(__global float* arena,
                     uint n, uint c_in, uint c_out,
                     uint d, uint h, uint w,
                     uint d_out, uint h_out, uint w_out,
                     uint kd, uint kh, uint kw,
                     uint sd, uint sh, uint sw,
                     uint pd, uint ph, uint pw,
                     uint dd, uint dh, uint dw,
                     uint groups,
                     uint in_off, uint w_off, uint out_off) {
    uint total = n * c_out * d_out * h_out * w_out;
    uint i = get_global_id(0);
    if (i >= total) return;

    uint wo = i % w_out;
    uint q1 = i / w_out;
    uint ho = q1 % h_out;
    uint q2 = q1 / h_out;
    uint do_ = q2 % d_out;
    uint q3 = q2 / d_out;
    uint co = q3 % c_out;
    uint nn = q3 / c_out;

    uint c_in_per_g = c_in / groups;
    uint c_out_per_g = c_out / groups;
    uint g = co / c_out_per_g;
    uint oc_off = co % c_out_per_g;
    uint ci_start = g * c_in_per_g;

    float acc = 0.0f;
    for (uint kz = 0u; kz < kd; kz++) {
        int num_d = (int)do_ + (int)pd - (int)(kz * dd);
        if (num_d < 0 || (num_d % (int)sd) != 0) continue;
        uint id = (uint)(num_d / (int)sd);
        if (id >= d) continue;
        for (uint ky = 0u; ky < kh; ky++) {
            int num_h = (int)ho + (int)ph - (int)(ky * dh);
            if (num_h < 0 || (num_h % (int)sh) != 0) continue;
            uint ih = (uint)(num_h / (int)sh);
            if (ih >= h) continue;
            for (uint kx = 0u; kx < kw; kx++) {
                int num_w = (int)wo + (int)pw - (int)(kx * dw);
                if (num_w < 0 || (num_w % (int)sw) != 0) continue;
                uint iw = (uint)(num_w / (int)sw);
                if (iw >= w) continue;
                for (uint ci_off = 0u; ci_off < c_in_per_g; ci_off++) {
                    uint ci = ci_start + ci_off;
                    uint in_idx =
                        (((nn * c_in + ci) * d + id) * h + ih) * w + iw;
                    uint w_idx =
                        (((ci * c_out_per_g + oc_off) * kd + kz) * kh + ky) * kw + kx;
                    acc += arena[in_off + in_idx] * arena[w_off + w_idx];
                }
            }
        }
    }
    arena[out_off + i] = acc;
}
