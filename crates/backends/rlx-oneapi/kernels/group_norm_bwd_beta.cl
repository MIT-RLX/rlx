// GroupNorm backward w.r.t. beta. Single serial work-item (matches CUDA).
__kernel void group_norm_bwd_beta(__global float* arena,
                                  uint dy_off, uint out_off,
                                  uint n, uint c, uint h, uint w) {
    if (get_global_id(0) != 0) return;
    uint spatial = h * w;
    uint plane = c * spatial;
    for (uint ch = 0; ch < c; ch++) arena[out_off + ch] = 0.0f;
    for (uint bn = 0; bn < n; bn++) {
        uint b_base = bn * plane;
        for (uint ch = 0; ch < c; ch++) {
            uint dy_base = dy_off + b_base + ch * spatial;
            float acc = arena[out_off + ch];
            for (uint s = 0; s < spatial; s++) acc += arena[dy_base + s];
            arena[out_off + ch] = acc;
        }
    }
}
