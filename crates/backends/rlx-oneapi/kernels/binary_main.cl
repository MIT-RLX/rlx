// Plumbing half of the standalone `binary` kernel. The per-op scalar math
// (`rlx_binary_apply`) is @generated once from the shared rlxsl manifest and
// prepended to this file by build.rs — so the op set (and the 64-bit bitwise +
// negative-base `pow` semantics matching the CPU oracle) live in a single source.
//
// Arg 0 is the whole arena (single `__global float*`); every operand is an
// element offset into it. `a_mod`/`b_mod` are 0 for a full-size operand or the
// operand's element count for a trailing-broadcast operand.
__kernel void binary(__global float* arena,
                     uint n, uint off_a, uint off_b, uint off_out,
                     uint a_mod, uint b_mod, uint op) {
    uint gid = get_global_id(0);
    if (gid >= n) return;
    uint ai = (a_mod == 0u) ? gid : (gid % a_mod);
    uint bi = (b_mod == 0u) ? gid : (gid % b_mod);
    float a = arena[off_a + ai];
    float b = arena[off_b + bi];
    arena[off_out + gid] = rlx_binary_apply(op, a, b);
}
