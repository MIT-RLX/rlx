// 3D conv NCDHW. Weight: [c_out, c_in/groups, kd, kh, kw].
// One work-item per output element.
__kernel void conv3d(__global float* arena,
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
    uint ci_start = g * c_in_per_g;
    float acc = 0.0f;
    for (uint ci_off = 0; ci_off < c_in_per_g; ci_off++) {
        uint ci = ci_start + ci_off;
        for (uint ki = 0; ki < kd; ki++)
        for (uint kj = 0; kj < kh; kj++)
        for (uint kk = 0; kk < kw; kk++) {
            int id = (int)(do_ * sd + ki * dd) - (int)pd;
            int ih = (int)(ho  * sh + kj * dh) - (int)ph;
            int iw = (int)(wo  * sw + kk * dw) - (int)pw;
            if (id < 0 || ih < 0 || iw < 0
                || id >= (int)d || ih >= (int)h || iw >= (int)w) continue;
            float xv = arena[in_off + (((nn * c_in + ci) * d + (uint)id) * h + (uint)ih) * w + (uint)iw];
            float wv = arena[w_off + ((((co * c_in_per_g + ci_off) * kd + ki) * kh + kj) * kw + kk)];
            acc += xv * wv;
        }
    }
    arena[out_off + i] = acc;
}
