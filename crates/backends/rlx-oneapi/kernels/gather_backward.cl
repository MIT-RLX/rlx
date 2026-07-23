// Gather-axis backward: zero dst slice then scatter-add dy (one work-item
// per outer). Mirrors wgpu gather_backward.wgsl (zero + acc fused).

__kernel void gather_backward(__global float* arena,
                     uint outer, uint axis_dim,
                     uint num_idx, uint trailing,
                     uint dy_off, uint idx_off, uint dst_off) {
    uint o = get_global_id(0);
    if (o >= outer) return;
    uint dst_base = dst_off + o * axis_dim * trailing;
    for (uint t = 0u; t < axis_dim * trailing; t++) {
        arena[dst_base + t] = 0.0f;
    }
    for (uint k = 0u; k < num_idx; k++) {
        uint row = (uint)arena[idx_off + k];
        if (row >= axis_dim) continue;
        for (uint j = 0u; j < trailing; j++) {
            float v = arena[dy_off + (o * num_idx + k) * trailing + j];
            arena[dst_off + (o * axis_dim + row) * trailing + j] += v;
        }
    }
}
