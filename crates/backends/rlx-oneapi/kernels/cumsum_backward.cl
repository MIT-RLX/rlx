// Cumsum backward along last axis (one work-item per row).
// Mirrors wgpu cumsum_backward.wgsl / CUDA rlx_cumsum_bwd.

__kernel void cumsum_backward(__global float* arena,
                     uint outer, uint inner,
                     uint dy_off, uint dx_off,
                     uint exclusive) {
    uint row = get_global_id(0);
    if (row >= outer) return;
    uint dy_base = dy_off + row * inner;
    uint dx_base = dx_off + row * inner;
    float suffix = 0.0f;
    for (int i = (int)inner - 1; i >= 0; --i) {
        uint ui = (uint)i;
        if (exclusive != 0u) {
            arena[dx_base + ui] = suffix;
            suffix += arena[dy_base + ui];
        } else {
            suffix += arena[dy_base + ui];
            arena[dx_base + ui] = suffix;
        }
    }
}
