// Packed GatedResidual backward: out = [dx ∥ dy ∥ dgate] (1-D floats).
// One work-item per unique gate row.
__kernel void gated_residual_backward(__global float* arena,
                                      uint mod_rows, uint seq_per_mod, uint inner,
                                      uint y_off, uint gate_off, uint dy_off, uint out_off) {
    uint m = get_global_id(0);
    if (m >= mod_rows || inner == 0u) return;

    uint nx = mod_rows * seq_per_mod * inner;
    uint gate_base = m * inner;

    for (uint i = 0; i < inner; i++) {
        float acc = 0.0f;
        float g = arena[gate_off + gate_base + i];
        for (uint seq = 0; seq < seq_per_mod; seq++) {
            uint row = m * seq_per_mod + seq;
            uint idx = row * inner + i;
            float d = arena[dy_off + idx];
            arena[out_off + idx] = d;
            arena[out_off + nx + idx] = d * g;
            acc += d * arena[y_off + idx];
        }
        arena[out_off + 2u * nx + gate_base + i] = acc;
    }
}
