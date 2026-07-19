// Packed AdaLayerNorm backward: out = [dx ∥ dscale ∥ dshift] (1-D floats).
// One work-item per unique modulation row (DiT [B,S,D] / [B,1,D]).
__kernel void ada_layer_norm_backward(__global float* arena,
                                      uint mod_rows, uint seq_per_mod, uint inner,
                                      uint x_off, uint scale_off, uint dy_off, uint out_off,
                                      uint layer_norm, float eps) {
    uint m = get_global_id(0);
    if (m >= mod_rows || inner == 0u) return;

    uint nx = mod_rows * seq_per_mod * inner;
    uint mod_len = mod_rows * inner;
    uint mod_base = m * inner;
    uint dscale_base = out_off + nx + mod_base;
    uint dshift_base = out_off + nx + mod_len + mod_base;
    float n_inv = 1.0f / (float)inner;

    for (uint i = 0; i < inner; i++) {
        arena[dscale_base + i] = 0.0f;
        arena[dshift_base + i] = 0.0f;
    }

    for (uint seq = 0; seq < seq_per_mod; seq++) {
        uint row = m * seq_per_mod + seq;
        uint x_base = x_off + row * inner;
        uint dy_base = dy_off + row * inner;
        uint dx_base = out_off + row * inner;

        float sum_x = 0.0f;
        float sum_x2 = 0.0f;
        for (uint i = 0; i < inner; i++) {
            float v = arena[x_base + i];
            sum_x += v;
            sum_x2 += v * v;
        }

        float mean = 0.0f;
        float inv;
        if (layer_norm != 0u) {
            mean = sum_x * n_inv;
            float var_ = fmax(sum_x2 * n_inv - mean * mean, 0.0f);
            inv = rsqrt(var_ + eps);
        } else {
            inv = rsqrt(sum_x2 * n_inv + eps);
        }

        float sum_sy = 0.0f;
        float sum_sxh = 0.0f;
        for (uint i = 0; i < inner; i++) {
            float n = (arena[x_base + i] - mean) * inv;
            float d = arena[dy_base + i];
            float sc = arena[scale_off + mod_base + i];
            float sy = d * (1.0f + sc);
            arena[dscale_base + i] += d * n;
            arena[dshift_base + i] += d;
            sum_sy += sy;
            sum_sxh += sy * n;
        }
        float m_sy = sum_sy * n_inv;
        float m_sxh = sum_sxh * n_inv;

        for (uint i = 0; i < inner; i++) {
            float d = arena[dy_base + i];
            float sc = arena[scale_off + mod_base + i];
            float sy = d * (1.0f + sc);
            if (layer_norm != 0u) {
                float n = (arena[x_base + i] - mean) * inv;
                arena[dx_base + i] = inv * (sy - m_sy - n * m_sxh);
            } else {
                float n_rms = arena[x_base + i] * inv;
                arena[dx_base + i] = inv * (sy - n_rms * m_sxh);
            }
        }
    }
}
