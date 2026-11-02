// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// Portable cooperative-matrix matmul for Vulkan / DX12 (8×8 f32 tiles).
// Matches the wgpu `cooperative_matrix/shader.wgsl` load/store pattern:
// row-major `coopLoad`, dispatch (M/8, N/8).


//
// ── COOPERATIVE-MATRIX LAYOUT CONVENTION (read before editing) ──────────────
//
// naga's shared WGSL frontend sets `row_major = function_name.ends_with("T")`,
// so `coopLoad`/`coopLoadT` differ by one bool. That bool reaches the two
// backends as DIFFERENT THINGS: the MSL backend passes it to Metal's
// `simdgroup_load(..., transpose_matrix)`, the SPIR-V backend turns it into
// `RowMajorKHR`/`ColumnMajorKHR` on `OpCooperativeMatrixLoadKHR`. Identical
// WGSL therefore does NOT mean identical behaviour on Metal and Vulkan.
//
// This is measured, not inferred. The Metal kernels
// (`matmul_coop_f32.wgsl`, `matmul_qkv_coop_f32.wgsl`, `matmul_coop16.wgsl`)
// needed `coopLoadT` + `coopStoreT` on EVERY operand; with the plain forms they
// computed `b·a` instead of `a·b` (near-orthogonal for unrelated matrices —
// the cos 0.016 "orthogonal garbage" that kept Metal CoopF32 opt-in for a long
// time). The Vulkan convention in this file — `coopLoad` on A, `coopLoadT` on B
// — was then tried on Metal and is WRONG there: it leaves only the first column
// of each 8x8 tile correct and zeroes the rest. So the two conventions are
// genuinely different and must NOT be unified. These Vulkan kernels were
// deliberately left alone.
//
// ⚠ UNRESOLVED INCONSISTENCY — this file contradicts its own sibling.
//
// `matmul_coop_f16_vulkan.wgsl` and `matmul_coop_f16_vulkan_widen.wgsl` differ
// by EXACTLY ONE TOKEN (`coopLoadT` vs `coopLoad` on B) and nothing else — same
// `b_ptr = b_base + k_off * params.n + tile_col`, same `params.n` stride, same
// body. Which one runs is chosen by SIZE: `use_wide_matmul` in
// `src/coop_f16_vk.rs` picks the `coopLoad` variant when `n > 768` (or when the
// weight is in the `wide_b` allowlist).
//
// A memory layout is a property of the bytes, not of N. The same buffer cannot
// be row-major at n=768 and column-major at n=769, so one of these two kernels
// is describing its operand wrongly and returning a transposed product for its
// half of the N range. The same split exists in the `matmul_qkv_coop_f16_vk*`
// pair, and `matmul_coop_f32_portable.wgsl` uses the plain `coopLoad` on BOTH
// operands, matching neither.
//
// It cannot be settled on any machine here: neither Mesa lavapipe nor MoltenVK
// exposes a cooperative-matrix extension, so `just test-wgpu-linux` and an
// Apple box both fall short of reaching this code. It needs a device reporting
// VK_KHR_cooperative_matrix.
//
// The decisive probe, on such a device — it multiplies a column vector by a row
// vector, so `a·b` is a dense rank-1 outer product while `b·a` collapses to one
// value per fragment (1008/1024 exact zeros):
//
//     RLX_WGPU_FORCE_COOP_F32=1 cargo test -p rlx-wgpu \
//         --test coop_f32_correctness coop_f32_operand_order_is_not_commuted
//
// Run it at an N on BOTH sides of the 768 threshold. The existing
// `coop_f16_vk_correctness.rs` cannot catch this: its K=8 probe multiplies
// all-ones by all-ones, which is commutation-blind.
//
// HOW THIS IS HANDLED NOW: `src/coop_probe.rs` runs a non-commuting GEMM probe
// through this kernel on the actual device, once per process, and
// `derive_matmul_compute` / `pick_coop_f16_vk_matmul` believe the result. If
// this shader computes `a·b` here it is used; if it does not, it is reported
// ineligible and the caller falls back to the portable tiled matmul — slower,
// correct. So the open question above can no longer produce wrong numbers on
// any device, including ones no developer machine can reach. It is still worth
// settling on real hardware, because the answer decides whether the fast path
// is available at all; `tests/coop_self_check.rs` prints the verdict per kernel.
enable wgpu_cooperative_matrix;

struct Params {
    m: u32,
    k: u32,
    n: u32,
    a_off: u32,
    b_off: u32,
    c_off: u32,
    batch: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    has_bias: u32,
    bias_off: u32,
    act_id: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

const TILE: u32 = 8u;

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;

// Correctly-rounded `a / b`: a verbatim copy of rlxsl's `rlx_div_rn` (pinned
// by tests/epilogue_exact_math.rs). WGSL `/` is not correctly rounded, and
// Metal compiles it differently for native wgpu and for a browser's WebGPU,
// so a raw `/` in sigmoid / SiLU moved activations by an ULP between them.
fn epi_div_rn(a: f32, b: f32) -> f32 {
    let k = abs(a / b) < 0x1p-100f;
    let a2 = select(a, a * 0x1p64f, k);
    let q = a2 / b;
    let ok = q != 0.0 && abs(q) <= 0x1.fffffep+127f;
    var c = select(q, q + fma(-b, q, a2) / b, ok);
    // The step rounds its correction before adding it, so near a midpoint
    // `c` depends on the first quotient. Move to the neighbour on the
    // quotient's side when the quotient lies past their midpoint (ties to
    // even): `r = a - b·c` is exact, and `r - b·u/2` has the sign of
    // `a - b·(c + n)/2`.
    if (ok && c != 0.0 && abs(c) < 0x1.fffffep+127f) {
        let r = fma(-b, c, a2);
        if (r != 0.0) {
            let up = (r > 0.0) == (b > 0.0);
            let cb = bitcast<u32>(c);
            let nb = select(cb - 1u, cb + 1u, up == (c > 0.0));
            let n = bitcast<f32>(nb);
            let t = fma(-b, (n - c) * 0.5, r);
            if ((t != 0.0 && ((t > 0.0) == (b > 0.0)) == up) || (t == 0.0 && (nb & 1u) == 0u)) {
                c = n;
            }
        }
    }
    return select(c, c * 0x1p-64f, k);
}

// `rlxsl::exact`'s `rlx_exp` (the builtin `exp` rounds by math mode).
fn epi_exp(x: f32) -> f32 {
    let bx = bitcast<u32>(x);
    if ((bx & 0x7fffffffu) > 0x7f800000u) { return x; }
    let xc = clamp(x, -110.0, 89.0);
    let n = round(xc * 1.44269504);
    let r = fma(n, -1.428606765330187e-6, fma(n, -0.693145751953125, xc));
    var p = 1.9875691500e-4;
    p = fma(p, r, 1.3981999507e-3);
    p = fma(p, r, 8.3334519073e-3);
    p = fma(p, r, 4.1665795894e-2);
    p = fma(p, r, 1.6666665459e-1);
    p = fma(p, r, 5.0000001201e-1);
    return ldexp(1.0 + fma(p, r * r, r), i32(n));
}

fn gelu_erf(x: f32) -> f32 {
    let arg = x * 0.70710678118654752;
    let s = select(-1.0, 1.0, arg >= 0.0);
    let xa = abs(arg);
    let t = epi_div_rn(1.0, fma(0.3275911, xa, 1.0));
    // explicit fma: an uncontracted chain is contracted differently by
    // different shader compilers
    let poly = t * fma(t, fma(t, fma(t, fma(t, 1.061405429, -1.453152027), 1.421413741),
                -0.284496736), 0.254829592);
    let e = s * fma(-poly, epi_exp(-xa * xa), 1.0);
    return 0.5 * x * (1.0 + e);
}

fn apply_act(v_in: f32) -> f32 {
    var v = v_in;
    if (params.act_id == 0xFFFFu) { return v; }
    switch (params.act_id) {
        case 0u: { v = max(v, 0.0); }
        case 1u: { v = epi_div_rn(1.0, 1.0 + epi_exp(-clamp(v, -88.0, 88.0))); }
        case 2u: { v = tanh(clamp(v, -15.0, 15.0)); }
        case 5u: { v = sqrt(v); }
        case 7u: { v = -v; }
        case 8u: { v = abs(v); }
        case 9u: { v = gelu_erf(v); }
        case 11u: {
            let c = 0.7978845608028654;
            let x3 = v * v * v;
            let inner = clamp(c * fma(0.044715, x3, v), -15.0, 15.0);
            v = 0.5 * v * (1.0 + tanh(inner));
        }
        case 10u: {
            let nx = clamp(-v, -88.0, 88.0);
            v = epi_div_rn(v, 1.0 + epi_exp(nx));
        }
        default: {}
    }
    return v;
}

@compute @workgroup_size(8, 8, 1)
fn matmul_coop_f32_portable(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bz = wid.z;
    if (bz >= params.batch) { return; }

    let tile_row = wid.x * TILE;
    let tile_col = wid.y * TILE;

    let a_base = params.a_off + bz * params.a_batch_stride;
    let b_base = params.b_off + bz * params.b_batch_stride;
    let c_base = params.c_off + bz * params.c_batch_stride;

    var acc: coop_mat8x8<f32, C> = coop_mat8x8<f32, C>();

    let n_tiles = (params.k + TILE - 1u) / TILE;
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let k_off = t * TILE;
        let a_ptr = a_base + tile_row * params.k + k_off;
        let b_ptr = b_base + k_off * params.n + tile_col;
        let a_tile: coop_mat8x8<f32, A> = coopLoad<coop_mat8x8<f32, A>>(&arena[a_ptr], params.k);
        let b_tile: coop_mat8x8<f32, B> = coopLoad<coop_mat8x8<f32, B>>(&arena[b_ptr], params.n);
        acc = coopMultiplyAdd(a_tile, b_tile, acc);
    }

    let c_ptr = c_base + tile_row * params.n + tile_col;
    coopStore(acc, &arena[c_ptr], params.n);

    let gr = tile_row + lid.y;
    let gc = tile_col + lid.x;
    if (gr >= params.m || gc >= params.n) { return; }
    let out_idx = c_base + gr * params.n + gc;
    var v = arena[out_idx];
    if (params.has_bias != 0u) {
        v = v + arena[params.bias_off + gc];
    }
    v = apply_act(v);
    arena[out_idx] = v;
}
