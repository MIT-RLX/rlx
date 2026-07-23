// GroupNorm (NCHW) backward w.r.t. input. One work-item per (batch, group).
// Matches CUDA `group_norm_bwd_input` / CPU training_bwd.
__kernel void group_norm_bwd_input(__global float* arena,
                                   uint x_off, uint gamma_off, uint dy_off, uint out_off,
                                   uint n, uint c, uint h, uint w, uint num_groups,
                                   float eps) {
    uint ng = get_global_id(0);
    if (ng >= n * num_groups) return;
    uint bn = ng / num_groups;
    uint g = ng % num_groups;
    uint cpg = c / num_groups;
    uint c0 = g * cpg;
    uint spatial = h * w;
    uint count = cpg * spatial;
    float n_inv = 1.0f / (float)count;
    uint plane = c * spatial;
    uint b_base = bn * plane;

    float sum = 0.0f;
    for (uint i = 0; i < count; i++) {
        uint ch = c0 + i / spatial;
        uint s = i % spatial;
        sum += arena[x_off + b_base + ch * spatial + s];
    }
    float mean = sum * n_inv;

    float var_ = 0.0f;
    for (uint i = 0; i < count; i++) {
        uint ch = c0 + i / spatial;
        uint s = i % spatial;
        float d = arena[x_off + b_base + ch * spatial + s] - mean;
        var_ += d * d;
    }
    float inv_std = rsqrt(var_ * n_inv + eps);

    float sum_sy = 0.0f;
    float sum_sxh = 0.0f;
    for (uint i = 0; i < count; i++) {
        uint gi = c0 + i / spatial;
        uint s = i % spatial;
        float xh = (arena[x_off + b_base + gi * spatial + s] - mean) * inv_std;
        float sy = arena[dy_off + b_base + gi * spatial + s] * arena[gamma_off + gi];
        sum_sy += sy;
        sum_sxh += sy * xh;
    }
    float m_sy = sum_sy * n_inv;
    float m_sxh = sum_sxh * n_inv;

    for (uint i = 0; i < count; i++) {
        uint gi = c0 + i / spatial;
        uint s = i % spatial;
        float xh = (arena[x_off + b_base + gi * spatial + s] - mean) * inv_std;
        float sy = arena[dy_off + b_base + gi * spatial + s] * arena[gamma_off + gi];
        arena[out_off + b_base + gi * spatial + s] = inv_std * (sy - m_sy - xh * m_sxh);
    }
}
