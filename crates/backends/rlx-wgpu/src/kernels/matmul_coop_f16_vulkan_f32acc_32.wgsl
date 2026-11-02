// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// RLX — f16 A/B with f32 cooperative accumulator, 32×32 output block per
// workgroup (2×2 register-blocked 16×16 tiles). Same bindings, params and
// epilogue as `matmul_coop_f16_vulkan_f32acc.wgsl`; picked over it when M and N
// are multiples of 32.
//
// Why a second kernel: the 16×16 one runs 64 invocations — two 32-wide
// subgroups on NVIDIA — and cooperative-matrix ops are per subgroup, so both
// compute (and store) the same tile. Each tile also reloads its A and B
// fragments from global memory for a single multiply-add. Here one subgroup
// owns a 32×32 block: per K step it loads two A and two B fragments and issues
// four multiply-adds, so every fragment is used twice.

enable f16;

//
// ── COOPERATIVE-MATRIX LAYOUT CONVENTION (read before editing) ──────────────
//
// naga sets `row_major = function_name.ends_with("T")`: on SPIR-V `coopLoadT` /
// `coopStoreT` are `RowMajorKHR` and the plain forms `ColumnMajorKHR`. A, B and
// the output tile are all row-major here (`a_ptr + r*k`, `b_ptr + r*n`,
// `tile[r*TILE + c]`), so every load and store is the `T` form — the same
// spelling the Metal kernels (`matmul_coop_f32.wgsl`, `matmul_coop16.wgsl`) use.
//
// Measured on an RTX 3080 Ti (Vulkan, VK_KHR_cooperative_matrix) with the
// non-commuting probe in `src/coop_probe.rs`, over all eight load/store
// combinations: only `coopLoadT` on A and B with `coopStoreT` computes `a·b`.
// The spelling this family shipped with (`coopLoad` on A, `coopStore` on the
// tile) failed it in every variant, so the runtime probe had been refusing the
// whole path on that device.
//
// The `_widen` siblings read B with `coopLoad` and were picked for `n > 768`.
// A layout does not depend on N, and that form computes `a·bᵀ`. They now match
// their base kernels token for token (save the entry point) and stay only
// because the size-based selection still names them.
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
    a_off_f16: u32,  // A in the f16 window (`a_off` is the f32 arena's)
    _p1: u32,
    _p2: u32,
};

const TILE: u32 = 16u;
const BLOCK: u32 = 32u;

@group(0) @binding(0) var<storage, read>         arena_f16: array<f16>;
@group(0) @binding(1) var<storage, read_write>   arena: array<f32>;
@group(0) @binding(2) var<uniform>               params: Params;

var<workgroup> f32_tile: array<f32, 1024>;

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

// Exact GELU — A&S 7.1.26 erf, the same as `matmul.wgsl` / `matmul_coop_f32.wgsl`
// and rlx-cpu's `scalar_gelu`. Opcode 9 (`Activation::Gelu`) is the EXACT form;
// only opcode 11 (`GeluApprox`) is the tanh form. This kernel used to map both
// to tanh, so a fused matmul+bias+GELU here drifted ~3e-4 from CPU at |x|≈2–3.
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

@compute @workgroup_size(32, 1, 1)
fn matmul_coop_f16_vulkan_f32acc_32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bz = wid.z;
    if (bz >= params.batch) { return; }

    let row0 = wid.x * BLOCK;
    let col0 = wid.y * BLOCK;

    let a_base = params.a_off_f16 + bz * params.a_batch_stride;
    let b_base = params.b_off + bz * params.b_batch_stride;
    let c_base = params.c_off + bz * params.c_batch_stride;

    var acc00: coop_mat16x16<f32, C> = coop_mat16x16<f32, C>();
    var acc01: coop_mat16x16<f32, C> = coop_mat16x16<f32, C>();
    var acc10: coop_mat16x16<f32, C> = coop_mat16x16<f32, C>();
    var acc11: coop_mat16x16<f32, C> = coop_mat16x16<f32, C>();

    var k_off: u32 = 0u;
    while (k_off < params.k) {
        let a_ptr = a_base + row0 * params.k + k_off;
        let b_ptr = b_base + k_off * params.n + col0;
        let a0: coop_mat16x16<f16, A> =
            coopLoadT<coop_mat16x16<f16, A>>(&arena_f16[a_ptr], params.k);
        let a1: coop_mat16x16<f16, A> =
            coopLoadT<coop_mat16x16<f16, A>>(&arena_f16[a_ptr + TILE * params.k], params.k);
        let b0: coop_mat16x16<f16, B> =
            coopLoadT<coop_mat16x16<f16, B>>(&arena_f16[b_ptr], params.n);
        let b1: coop_mat16x16<f16, B> =
            coopLoadT<coop_mat16x16<f16, B>>(&arena_f16[b_ptr + TILE], params.n);
        acc00 = coopMultiplyAdd(a0, b0, acc00);
        acc01 = coopMultiplyAdd(a0, b1, acc01);
        acc10 = coopMultiplyAdd(a1, b0, acc10);
        acc11 = coopMultiplyAdd(a1, b1, acc11);
        k_off = k_off + TILE;
    }

    coopStoreT(acc00, &f32_tile[0], BLOCK);
    coopStoreT(acc01, &f32_tile[TILE], BLOCK);
    coopStoreT(acc10, &f32_tile[TILE * BLOCK], BLOCK);
    coopStoreT(acc11, &f32_tile[TILE * BLOCK + TILE], BLOCK);
    workgroupBarrier();

    for (var i: u32 = 0u; i < BLOCK; i = i + 1u) {
        let idx = lid.x + i * BLOCK;
        let gr = row0 + idx / BLOCK;
        let gc = col0 + idx % BLOCK;
        if (gr >= params.m || gc >= params.n) { continue; }
        var v = f32_tile[idx];
        if (params.has_bias != 0u) {
            v = v + arena[params.bias_off + gc];
        }
        v = apply_act(v);
        arena[c_base + gr * params.n + gc] = v;
    }
}
