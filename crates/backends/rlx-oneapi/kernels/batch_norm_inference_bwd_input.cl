// BatchNormInferenceBackwardInput: dx = dy · γ · inv_std.
__kernel void batch_norm_inference_bwd_input(__global float* arena,
                                             uint n, uint channels, float eps,
                                             uint gamma_off, uint var_off,
                                             uint dy_off, uint out_off) {
    uint i = get_global_id(0);
    if (i >= n || channels == 0u) return;
    uint c = i % channels;
    float inv = 1.0f / sqrt(arena[var_off + c] + eps);
    arena[out_off + i] = arena[dy_off + i] * arena[gamma_off + c] * inv;
}
