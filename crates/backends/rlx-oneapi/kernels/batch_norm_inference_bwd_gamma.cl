// BatchNormInferenceBackwardGamma: one work-item per channel.
__kernel void batch_norm_inference_bwd_gamma(__global float* arena,
                                             uint count, uint channels, float eps,
                                             uint x_off, uint mean_off, uint var_off,
                                             uint dy_off, uint out_off) {
    uint c = get_global_id(0);
    if (c >= channels) return;
    float inv = 1.0f / sqrt(arena[var_off + c] + eps);
    float mean = arena[mean_off + c];
    float acc = 0.0f;
    for (uint row = 0u; row < count; row++) {
        uint idx = row * channels + c;
        float xhat = (arena[x_off + idx] - mean) * inv;
        acc += arena[dy_off + idx] * xhat;
    }
    arena[out_off + c] = acc;
}
