// Standalone complex `Op::Cast` on the f32-uniform arena, dispatched over the
// complex-element index `k in [0, n)`. Representation:
//   C64  = 2 f32 lanes [re, im]                          (8 B/elem)
//   C128 = 4 f32 lanes [re_hi, re_lo, im_hi, im_lo] df64  (16 B/elem)
//
// Every source of a real->complex cast comes from an f32 real (lo=0), so all
// six directions are pure lane MOVES — no compensated df64 arithmetic. The
// C128->C64 narrow drops the `lo` lanes (keeps `hi`); the widen sets them 0.
//
//   mode 0 real->C64 : out[2k]=in[k];   out[2k+1]=0
//   mode 1 C64->real : out[k]=in[2k]
//   mode 2 real->C128: out[4k]=in[k];   out[4k+1..3]=0
//   mode 3 C128->real: out[k]=in[4k]
//   mode 4 C64->C128 : out[4k]=in[2k]; out[4k+1]=0; out[4k+2]=in[2k+1]; out[4k+3]=0
//   mode 5 C128->C64 : out[2k]=in[4k]; out[2k+1]=in[4k+2]
//
// Like the other oneAPI kernels, arg 0 is the whole arena (single
// `__global float*`); `in_off` / `out_off` are f32-ELEMENT offsets (the start
// lane of each tensor). They are `uint` — the host passes `KArg::U32` from
// `arena.elem_offset()`, so host and kernel widths match (rlx-cuda uses u64 for
// its > 4 GiB arenas; the oneAPI arena stays u32). Mirrors rlx-cuda
// `complex_cast.cu` / rlx-wgpu `complex_cast.wgsl` / rlx-vulkan `complex_cast.comp`.
__kernel void complex_cast(__global float* arena,
                           uint n, uint in_off, uint out_off, uint mode) {
    uint k = get_global_id(0);
    if (k >= n) return;
    uint i = in_off;
    uint o = out_off;
    switch (mode) {
        case 0u: // real -> C64
            arena[o + 2u * k]      = arena[i + k];
            arena[o + 2u * k + 1u] = 0.0f;
            break;
        case 1u: // C64 -> real
            arena[o + k] = arena[i + 2u * k];
            break;
        case 2u: // real -> C128
            arena[o + 4u * k]      = arena[i + k];
            arena[o + 4u * k + 1u] = 0.0f;
            arena[o + 4u * k + 2u] = 0.0f;
            arena[o + 4u * k + 3u] = 0.0f;
            break;
        case 3u: // C128 -> real
            arena[o + k] = arena[i + 4u * k];
            break;
        case 4u: // C64 -> C128
            arena[o + 4u * k]      = arena[i + 2u * k];
            arena[o + 4u * k + 1u] = 0.0f;
            arena[o + 4u * k + 2u] = arena[i + 2u * k + 1u];
            arena[o + 4u * k + 3u] = 0.0f;
            break;
        case 5u: // C128 -> C64
            arena[o + 2u * k]      = arena[i + 4u * k];
            arena[o + 2u * k + 1u] = arena[i + 4u * k + 2u];
            break;
        default:
            break;
    }
}
