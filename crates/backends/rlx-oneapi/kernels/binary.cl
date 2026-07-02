// Elementwise binary op over the f32-uniform arena.
//
// Arg 0 is the whole arena (single `__global float*`); every operand is an
// element offset into it — the Level Zero analog of rlx-vulkan's single bound
// storage buffer + push constants. `a_mod`/`b_mod` are 0 for a full-size
// operand or the operand's element count for a trailing-broadcast operand.
__kernel void binary(__global float* arena,
                     uint n, uint off_a, uint off_b, uint off_out,
                     uint a_mod, uint b_mod, uint op) {
    uint gid = get_global_id(0);
    if (gid >= n) return;
    uint ai = (a_mod == 0u) ? gid : (gid % a_mod);
    uint bi = (b_mod == 0u) ? gid : (gid % b_mod);
    float a = arena[off_a + ai];
    float b = arena[off_b + bi];
    float r;
    switch (op) {
        case 0u: r = a + b; break;        // Add
        case 1u: r = a - b; break;        // Sub
        case 2u: r = a * b; break;        // Mul
        case 3u: r = a / b; break;        // Div
        case 4u: r = fmax(a, b); break;   // Max
        case 5u: r = fmin(a, b); break;   // Min
        case 6u: r = pow(a, b); break;    // Pow
        default: r = a + b; break;
    }
    arena[off_out + gid] = r;
}
