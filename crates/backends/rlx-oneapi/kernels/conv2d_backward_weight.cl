// 2D convolution backward-weight (dw). Weight: [c_out, c_in/groups, kh, kw].
// One work-item per dw element; gathers over batch × output spatial.
// Mirrors CUDA `conv2d_backward_weight.cu`.

__kernel void conv2d_backward_weight(__global float* arena,
                     uint n, uint c_in, uint c_out,
                     uint h, uint w,
                     uint h_out, uint w_out,
                     uint kh, uint kw,
                     uint sh, uint sw,
                     uint ph, uint pw,
                     uint dh, uint dw,
                     uint groups,
                     uint x_off, uint dy_off, uint dw_off) {
    uint c_in_per_g = c_in / groups;
    uint c_out_per_g = c_out / groups;
    uint total = c_out * c_in_per_g * kh * kw;
    uint i = get_global_id(0);
    if (i >= total) return;
    uint kj = i % kw;
    uint q1 = i / kw;
    uint ki = q1 % kh;
    uint q2 = q1 / kh;
    uint ci_off = q2 % c_in_per_g;
    uint co = q2 / c_in_per_g;

    uint g = co / c_out_per_g;
    uint ci = g * c_in_per_g + ci_off;

    float acc = 0.0f;
    for (uint nn = 0u; nn < n; nn++) {
        for (uint ho = 0u; ho < h_out; ho++) {
            int ih = (int)(ho * sh + ki * dh) - (int)ph;
            if (ih < 0 || ih >= (int)h) continue;
            for (uint wo = 0u; wo < w_out; wo++) {
                int iw = (int)(wo * sw + kj * dw) - (int)pw;
                if (iw < 0 || iw >= (int)w) continue;
                float dyv = arena[dy_off + ((nn * c_out + co) * h_out + ho) * w_out + wo];
                float xv = arena[x_off + ((nn * c_in + ci) * h + (uint)ih) * w + (uint)iw];
                acc += dyv * xv;
            }
        }
    }
    arena[dw_off + i] = acc;
}
