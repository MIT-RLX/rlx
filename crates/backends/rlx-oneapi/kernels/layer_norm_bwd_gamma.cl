// LayerNorm backward w.r.t. gamma. Single serial work-item (matches CUDA).
__kernel void layer_norm_bwd_gamma(__global float* arena,
                                   uint outer, uint inner,
                                   uint x_off, uint dy_off, uint out_off,
                                   float eps) {
    if (get_global_id(0) != 0 || inner == 0u) return;
    float n_inv = 1.0f / (float)inner;

    for (uint i = 0; i < inner; i++) arena[out_off + i] = 0.0f;

    for (uint row = 0; row < outer; row++) {
        uint x_base = x_off + row * inner;
        uint dy_base = dy_off + row * inner;
        float sum = 0.0f;
        for (uint i = 0; i < inner; i++) sum += arena[x_base + i];
        float mean = sum * n_inv;
        float var_ = 0.0f;
        for (uint i = 0; i < inner; i++) {
            float d = arena[x_base + i] - mean;
            var_ += d * d;
        }
        float inv_std = 1.0f / sqrt(var_ * n_inv + eps);
        for (uint i = 0; i < inner; i++) {
            float xh = (arena[x_base + i] - mean) * inv_std;
            arena[out_off + i] += arena[dy_base + i] * xh;
        }
    }
}
