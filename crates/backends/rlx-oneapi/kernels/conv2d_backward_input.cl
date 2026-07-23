// 2D convolution backward-input (dx) NCHW. Weight: [c_out, c_in/groups, kh, kw].
// One work-item per dx element; gathers from every (ho,wo,co) whose receptive
// field covers it. Mirrors CUDA `conv2d_backward_input.cu`.

__kernel void conv2d_backward_input(__global float* arena,
                     uint n, uint c_in, uint c_out,
                     uint h, uint w,
                     uint h_out, uint w_out,
                     uint kh, uint kw,
                     uint sh, uint sw,
                     uint ph, uint pw,
                     uint dh, uint dw,
                     uint groups,
                     uint dy_off, uint w_off, uint dx_off) {
    uint total = n * c_in * h * w;
    uint i = get_global_id(0);
    if (i >= total) return;
    uint iw = i % w;
    uint q1 = i / w;
    uint ih = q1 % h;
    uint q2 = q1 / h;
    uint ci = q2 % c_in;
    uint nn = q2 / c_in;

    uint c_in_per_g = c_in / groups;
    uint c_out_per_g = c_out / groups;
    uint g = ci / c_in_per_g;
    uint ci_off = ci - g * c_in_per_g;
    uint co_start = g * c_out_per_g;

    float acc = 0.0f;
    for (uint ki = 0u; ki < kh; ki++) {
        int num_h = (int)(ih + ph) - (int)(ki * dh);
        if (num_h < 0 || (uint)num_h % sh != 0u) continue;
        uint ho = (uint)num_h / sh;
        if (ho >= h_out) continue;
        for (uint kj = 0u; kj < kw; kj++) {
            int num_w = (int)(iw + pw) - (int)(kj * dw);
            if (num_w < 0 || (uint)num_w % sw != 0u) continue;
            uint wo = (uint)num_w / sw;
            if (wo >= w_out) continue;
            for (uint co_off = 0u; co_off < c_out_per_g; co_off++) {
                uint co = co_start + co_off;
                float dyv = arena[dy_off + ((nn * c_out + co) * h_out + ho) * w_out + wo];
                float wv = arena[w_off + (((co * c_in_per_g + ci_off) * kh + ki) * kw + kj)];
                acc += dyv * wv;
            }
        }
    }
    arena[dx_off + i] = acc;
}
