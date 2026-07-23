// BatchNormInference (channels-last): y = γ · x̂ + β.
__kernel void batch_norm_inference(__global float* arena,
                                   uint n, uint channels, float eps,
                                   uint src_off, uint g_off, uint b_off,
                                   uint mean_off, uint var_off, uint dst_off) {
    uint i = get_global_id(0);
    if (i >= n || channels == 0u) return;
    uint c = i % channels;
    float inv = 1.0f / sqrt(arena[var_off + c] + eps);
    float xhat = (arena[src_off + i] - arena[mean_off + c]) * inv;
    arena[dst_off + i] = arena[g_off + c] * xhat + arena[b_off + c];
}
