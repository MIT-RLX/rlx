// GroupNorm backward w.r.t. gamma. Single serial work-item (matches CUDA).
__kernel void group_norm_bwd_gamma(__global float* arena,
                                   uint x_off, uint dy_off, uint out_off,
                                   uint n, uint c, uint h, uint w, uint num_groups,
                                   float eps) {
    if (get_global_id(0) != 0) return;
    uint spatial = h * w;
    uint plane = c * spatial;
    uint cpg = c / num_groups;
    float n_inv = 1.0f / (float)(cpg * spatial);

    for (uint ch = 0; ch < c; ch++) arena[out_off + ch] = 0.0f;

    for (uint bn = 0; bn < n; bn++) {
        uint b_base = bn * plane;
        for (uint g = 0; g < num_groups; g++) {
            uint c0 = g * cpg;
            float mean = 0.0f;
            for (uint ci = 0; ci < cpg; ci++) {
                uint base = x_off + b_base + (c0 + ci) * spatial;
                for (uint s = 0; s < spatial; s++) mean += arena[base + s];
            }
            mean *= n_inv;
            float var_ = 0.0f;
            for (uint ci = 0; ci < cpg; ci++) {
                uint base = x_off + b_base + (c0 + ci) * spatial;
                for (uint s = 0; s < spatial; s++) {
                    float d = arena[base + s] - mean;
                    var_ += d * d;
                }
            }
            float inv_std = rsqrt(var_ * n_inv + eps);
            for (uint ci = 0; ci < cpg; ci++) {
                uint gi = c0 + ci;
                uint x_base = x_off + b_base + gi * spatial;
                uint dy_base = dy_off + b_base + gi * spatial;
                float acc = arena[out_off + gi];
                for (uint s = 0; s < spatial; s++) {
                    float xh = (arena[x_base + s] - mean) * inv_std;
                    acc += arena[dy_base + s] * xh;
                }
                arena[out_off + gi] = acc;
            }
        }
    }
}
