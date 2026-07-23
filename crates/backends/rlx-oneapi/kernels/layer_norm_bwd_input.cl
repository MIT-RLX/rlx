// LayerNorm backward w.r.t. input (last-axis rows). One work-item per row.
// Matches CUDA `layer_norm_bwd_input` / CPU training_bwd (serial row form).
__kernel void layer_norm_bwd_input(__global float* arena,
                                   uint outer, uint inner,
                                   uint x_off, uint gamma_off, uint dy_off, uint out_off,
                                   float eps) {
    uint row = get_global_id(0);
    if (row >= outer || inner == 0u) return;
    uint x_base = x_off + row * inner;
    uint dy_base = dy_off + row * inner;
    uint out_base = out_off + row * inner;
    float n_inv = 1.0f / (float)inner;

    float sum = 0.0f;
    for (uint i = 0; i < inner; i++) sum += arena[x_base + i];
    float mean = sum * n_inv;

    float var_ = 0.0f;
    for (uint i = 0; i < inner; i++) {
        float d = arena[x_base + i] - mean;
        var_ += d * d;
    }
    float inv_std = 1.0f / sqrt(var_ * n_inv + eps);

    float sum_sy = 0.0f;
    float sum_sxh = 0.0f;
    for (uint i = 0; i < inner; i++) {
        float xh = (arena[x_base + i] - mean) * inv_std;
        float sy = arena[dy_base + i] * arena[gamma_off + i];
        sum_sy += sy;
        sum_sxh += sy * xh;
    }
    float m_sy = sum_sy * n_inv;
    float m_sxh = sum_sxh * n_inv;

    for (uint i = 0; i < inner; i++) {
        float xh = (arena[x_base + i] - mean) * inv_std;
        float sy = arena[dy_base + i] * arena[gamma_off + i];
        arena[out_base + i] = inv_std * (sy - m_sy - xh * m_sxh);
    }
}
