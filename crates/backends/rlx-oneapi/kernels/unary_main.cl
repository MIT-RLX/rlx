// Plumbing half of the `unary` kernel. The activation math (act < 100) is
// @generated once in `rlxsl::opencl_activation_module` (gelu-first ids, matching
// `act_id` in src/backend.rs) and prepended to this file by build.rs — so the
// erf polynomial, softplus stability trick, etc. live in a single source shared
// with every other backend, not a hand-maintained copy here.
//
// Cast op ids (see `classify_cast` in src/backend.rs, kept in sync with
// unary.cu / unary.comp). In the f32-uniform arena the dst dtype's value is
// written back as an f32 lane:
//   100=f32->i8  101=f32->i16 102=f32->i32 103=f32->i64
//   104=f32->u8  105=f32->u32 106=(x!=0)->bool
// float->int truncates toward zero + saturates (Rust `as` / rlx-cpu); NaN->0.
__kernel void unary(__global float* arena,
                    uint n, uint off_x, uint off_out, uint act) {
    uint gid = get_global_id(0);
    if (gid >= n) return;
    float x = arena[off_x + gid];
    float r;
    switch (act) {
        // f32 -> int: truncate toward zero, saturate to dst range, NaN -> 0.
        case 100u: r = isnan(x) ? 0.0f : clamp(trunc(x), -128.0f, 127.0f); break;
        case 101u: r = isnan(x) ? 0.0f : clamp(trunc(x), -32768.0f, 32767.0f); break;
        case 102u: r = isnan(x) ? 0.0f : clamp(trunc(x), -2147483648.0f, 2147483647.0f); break;
        case 103u: r = isnan(x) ? 0.0f : clamp(trunc(x), -9223372036854775808.0f, 9223372036854775807.0f); break;
        case 104u: r = isnan(x) ? 0.0f : clamp(trunc(x), 0.0f, 255.0f); break;
        case 105u: r = isnan(x) ? 0.0f : clamp(trunc(x), 0.0f, 4294967295.0f); break;
        // -> Bool: x != 0 ? 1 : 0 (NaN is non-zero -> 1, matching Rust).
        case 106u: r = (x != 0.0f) ? 1.0f : 0.0f; break;
        // Activations (gelu-first ids) — dispatched by the generated function.
        default: r = rlx_activation_apply(act, x); break;
    }
    arena[off_out + gid] = r;
}
