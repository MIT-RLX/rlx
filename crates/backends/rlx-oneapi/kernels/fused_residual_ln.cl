// Fused (residual add + optional bias) + LayerNorm.
// One work-item per outer row; two-pass variance (stable; matches wgpu).
__kernel void fused_residual_ln(__global float* arena,
                                uint outer, uint inner,
                                uint in_off, uint residual_off, uint bias_off,
                                uint gamma_off, uint beta_off, uint out_off,
                                float eps, uint has_bias) {
    uint row = get_global_id(0);
    if (row >= outer || inner == 0u) return;
    uint in_base = in_off + row * inner;
    uint res_base = residual_off + row * inner;
    uint out_base = out_off + row * inner;
    float n_inv = 1.0f / (float)inner;

    float sum_x = 0.0f;
    for (uint i = 0; i < inner; i++) {
        float v = arena[in_base + i] + arena[res_base + i];
        if (has_bias != 0u) v += arena[bias_off + i];
        arena[out_base + i] = v;
        sum_x += v;
    }
    float mean = sum_x * n_inv;

    float sum_sq = 0.0f;
    for (uint i = 0; i < inner; i++) {
        float d = arena[out_base + i] - mean;
        sum_sq += d * d;
    }
    float inv_std = rsqrt(sum_sq * n_inv + eps);

    for (uint i = 0; i < inner; i++) {
        float g = arena[gamma_off + i];
        float b = arena[beta_off + i];
        arena[out_base + i] = (arena[out_base + i] - mean) * inv_std * g + b;
    }
}
