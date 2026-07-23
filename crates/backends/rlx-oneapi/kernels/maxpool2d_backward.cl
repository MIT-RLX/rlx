// MaxPool2d backward (NCHW). One work-item per input spatial location
// (n,c,ih,iw); accumulates dy from every pool window whose argmax lands here.
// Mirrors wgpu `maxpool2d_backward.wgsl`.

__kernel void maxpool2d_backward(__global float* arena,
                     uint n, uint c,
                     uint h, uint w,
                     uint h_out, uint w_out,
                     uint kh, uint kw,
                     uint sh, uint sw,
                     uint ph, uint pw,
                     uint x_off, uint dy_off, uint dx_off) {
    uint total = n * c * h * w;
    uint i = get_global_id(0);
    if (i >= total) return;
    uint iw = i % w;
    uint q1 = i / w;
    uint ih = q1 % h;
    uint nc = q1 / h;
    if (nc >= n * c) return;

    int p_h = (int)ih + (int)ph;
    int p_w = (int)iw + (int)pw;
    int oh_max = p_h / (int)sh;
    int ow_max = p_w / (int)sw;
    if (oh_max >= (int)h_out) oh_max = (int)h_out - 1;
    if (ow_max >= (int)w_out) ow_max = (int)w_out - 1;
    int oh_min = 0;
    if (p_h - (int)kh >= 0)
        oh_min = (p_h - (int)kh) / (int)sh + 1;
    int ow_min = 0;
    if (p_w - (int)kw >= 0)
        ow_min = (p_w - (int)kw) / (int)sw + 1;

    uint in_chan = nc * h * w;
    uint out_chan = nc * h_out * w_out;
    float acc = 0.0f;
    for (int oh = oh_min; oh <= oh_max; oh++) {
        for (int ow = ow_min; ow <= ow_max; ow++) {
            float best_v = -3.402823466e+38f;
            int best_h = -1;
            int best_w = -1;
            for (uint ki = 0u; ki < kh; ki++) {
                int hh = oh * (int)sh + (int)ki - (int)ph;
                if (hh < 0 || hh >= (int)h) continue;
                for (uint kj = 0u; kj < kw; kj++) {
                    int ww = ow * (int)sw + (int)kj - (int)pw;
                    if (ww < 0 || ww >= (int)w) continue;
                    float v = arena[x_off + in_chan + (uint)hh * w + (uint)ww];
                    if (v > best_v) {
                        best_v = v;
                        best_h = hh;
                        best_w = ww;
                    }
                }
            }
            if (best_h == (int)ih && best_w == (int)iw)
                acc += arena[dy_off + out_chan + (uint)oh * w_out + (uint)ow];
        }
    }
    arena[dx_off + in_chan + ih * w + iw] = acc;
}
