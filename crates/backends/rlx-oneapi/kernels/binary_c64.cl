// Element-wise C64 binary op over the f32-uniform arena, dispatched over the
// complex-element index `k in [0, n)`. C64 = 2 f32 lanes [re, im] (8 B/elem),
// so each thread reads BOTH lanes of its operands — the reason this cannot ride
// the scalar-per-thread `binary` kernel (that model can't reach the partner
// `im` lane). Formulas mirror rlx-cpu `exec_binary_full_c64`:
//   Add: (ar+br, ai+bi)   Sub: (ar-br, ai-bi)
//   Mul: (ar*br - ai*bi, ar*bi + ai*br)
//   Div: d = br*br + bi*bi; ((ar*br + ai*bi)/d, (ai*br - ar*bi)/d)
// Max/Min/Pow are rejected at lowering (undefined for complex), as is C128
// arithmetic (rlx-cpu has none either).
//
// Broadcast: `n_a` / `n_b` are the operands' complex-element counts (>= 1).
// Indexing uses `k % n_a` / `k % n_b` in complex-element units, matching the
// CPU modulo fallback — a scalar operand (count 1) reads element 0 for every k.
// Arg 0 is the whole arena; `a_off`/`b_off`/`c_off` are f32-ELEMENT offsets
// (lane j of complex element m is `off + 2*m + j`), `uint` to match the host
// `KArg::U32` from `arena.elem_offset()`. Mirrors rlx-cuda `binary_c64.cu` /
// rlx-wgpu `binary_c64.wgsl` / rlx-vulkan `binary_c64.comp`.
__kernel void binary_c64(__global float* arena,
                         uint n, uint a_off, uint b_off, uint c_off,
                         uint op, uint n_a, uint n_b) {
    uint k = get_global_id(0);
    if (k >= n) return;
    uint ka = k % n_a;
    uint kb = k % n_b;
    float ar = arena[a_off + 2u * ka];
    float ai = arena[a_off + 2u * ka + 1u];
    float br = arena[b_off + 2u * kb];
    float bi = arena[b_off + 2u * kb + 1u];
    float cr = 0.0f;
    float ci = 0.0f;
    switch (op) {
        case 0u: cr = ar + br; ci = ai + bi; break;   // Add
        case 1u: cr = ar - br; ci = ai - bi; break;   // Sub
        case 2u:                                       // Mul
            cr = ar * br - ai * bi;
            ci = ar * bi + ai * br;
            break;
        case 3u: {                                     // Div
            float d = br * br + bi * bi;
            cr = (ar * br + ai * bi) / d;
            ci = (ai * br - ar * bi) / d;
            break;
        }
        default: break;
    }
    arena[c_off + 2u * k]      = cr;
    arena[c_off + 2u * k + 1u] = ci;
}
