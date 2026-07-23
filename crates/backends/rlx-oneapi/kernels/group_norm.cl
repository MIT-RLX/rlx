// NCHW group norm: normalize each (C/G)×H×W block. One work-item per
// (batch, group); sequential two-pass mean / var (stable; matches CUDA/CPU).
__kernel void group_norm(__global float* arena,
                         uint src_off, uint g_off, uint b_off, uint dst_off,
                         uint n, uint c, uint h, uint w, uint num_groups,
                         float eps) {
    uint ng = get_global_id(0);
    if (ng >= n * num_groups) return;
    uint bn = ng / num_groups;
    uint g = ng % num_groups;
    uint cpg = c / num_groups;
    uint c0 = g * cpg;
    uint plane = h * w;
    uint count = cpg * plane;
    float n_inv = 1.0f / (float)count;

    float sum = 0.0f;
    for (uint i = 0; i < count; i++) {
        uint ch = c0 + i / plane;
        uint s = i % plane;
        sum += arena[src_off + ((bn * c + ch) * plane) + s];
    }
    float mean = sum * n_inv;

    float var_ = 0.0f;
    for (uint i = 0; i < count; i++) {
        uint ch = c0 + i / plane;
        uint s = i % plane;
        float d = arena[src_off + ((bn * c + ch) * plane) + s] - mean;
        var_ += d * d;
    }
    float inv = rsqrt(var_ * n_inv + eps);

    for (uint i = 0; i < count; i++) {
        uint ch = c0 + i / plane;
        uint s = i % plane;
        uint idx = ((bn * c + ch) * plane) + s;
        float v = (arena[src_off + idx] - mean) * inv;
        arena[dst_off + idx] = v * arena[g_off + ch] + arena[b_off + ch];
    }
}
