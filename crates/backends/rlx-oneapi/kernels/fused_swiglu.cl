// Fused SwiGLU: y = up * silu(gate) with concatenated [..., 2*n_half] input.
// One work-item per output element.
__kernel void fused_swiglu(__global float* arena,
                           uint n_half, uint total, uint gate_first,
                           uint in_off, uint out_off) {
    uint i = get_global_id(0);
    if (i >= total) return;
    uint row = i / n_half;
    uint col = i % n_half;
    uint base = row * (2u * n_half);
    float up, gate;
    if (gate_first != 0u) {
        gate = arena[in_off + base + col];
        up = arena[in_off + base + n_half + col];
    } else {
        up = arena[in_off + base + col];
        gate = arena[in_off + base + n_half + col];
    }
    arena[out_off + i] = up * (gate / (1.0f + exp(-gate)));
}
