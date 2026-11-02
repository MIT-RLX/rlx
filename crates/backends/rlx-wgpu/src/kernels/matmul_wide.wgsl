// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Wide-tile matmul variant: 32×64 output tile, 4×8 register block.
//
// Same 8×8 = 64-thread workgroup as `matmul.wgsl`, but each thread holds
// a 4×8 (32-float) accumulator block instead of 4×4 (16-float). Per
// K-step inner loop: 4 a-loads + 8 b-loads → 32 FMAs (2.67 FMA / register
// load, vs the small variant's 2.0). Each thread holds 32 acc + 12 load
// registers ≈ 44 f32 — within Apple M-series per-thread budget. The
// 8×8 / 64-acc variant (TILE_M=64, TILE_N=64, RM=8, RN=8) was tested
// and spilled hard, measuring 2-3× slower than this variant.
//
// Use criterion: when M ≥ 32 AND N ≥ 64, dispatch this kernel for
// ~7-13 % speedup on BERT FFN/QKV matmuls at batch ≥ 64. For smaller
// shapes the partial-tile waste in N (1 wg covers 64 cols) outweighs
// the AI win, so fall back to the 4×4-block kernel.

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

const TILE_M: u32 = 32u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 16u;
const RM: u32 = 4u;
const RN: u32 = 8u;
const WG_M: u32 = 8u;     // TILE_M / RM
const WG_N: u32 = 8u;     // TILE_N / RN

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

var<workgroup> tile_a: array<array<f32, 16>, 32>;  // [TILE_M][TILE_K]
var<workgroup> tile_b: array<array<f32, 64>, 16>;  // [TILE_K][TILE_N]

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
    switch (params.act_id) {
        case 0xFFFFu: {}
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

@compute @workgroup_size(8, 8)
fn matmul_wide(
    @builtin(local_invocation_id)    lid: vec3<u32>,
    @builtin(workgroup_id)           wid: vec3<u32>,
) {
    // No early return before workgroupBarrier — FXC X4026 on DX12.
    let bz = wid.z;
    let in_batch = bz < params.batch;
    let bz_safe = select(0u, bz, in_batch);

    let lr = lid.y;
    let lc = lid.x;
    let row_base = wid.y * TILE_M + lr * RM;
    let col_base = wid.x * TILE_N + lc * RN;

    let a_base = params.a_off + bz_safe * params.a_batch_stride;
    let b_base = params.b_off + bz_safe * params.b_batch_stride;
    let c_base = params.c_off + bz_safe * params.c_batch_stride;

    var acc: array<array<f32, 8>, 4>;
    for (var i: u32 = 0u; i < RM; i = i + 1u) {
        for (var j: u32 = 0u; j < RN; j = j + 1u) {
            acc[i][j] = 0.0;
        }
    }

    let n_tiles = (params.k + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        // tile_a (32×16): each thread writes 4 rows × 2 cols = 8 elem.
        for (var i: u32 = 0u; i < RM; i = i + 1u) {
            let m_local = lr * RM + i;
            let global_row = wid.y * TILE_M + m_local;
            for (var j: u32 = 0u; j < 2u; j = j + 1u) {
                let k_local = lc * 2u + j;
                let global_k = t * TILE_K + k_local;
                if (in_batch && global_row < params.m && global_k < params.k) {
                    tile_a[m_local][k_local] = arena[a_base + global_row * params.k + global_k];
                } else {
                    tile_a[m_local][k_local] = 0.0;
                }
            }
        }
        // tile_b (16×64): each thread writes 2 rows × 8 cols = 16 elem.
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let k_local = lr * 2u + i;
            let global_k = t * TILE_K + k_local;
            for (var j: u32 = 0u; j < RN; j = j + 1u) {
                let n_local = lc * RN + j;
                let global_col = wid.x * TILE_N + n_local;
                if (in_batch && global_k < params.k && global_col < params.n) {
                    tile_b[k_local][n_local] = arena[b_base + global_k * params.n + global_col];
                } else {
                    tile_b[k_local][n_local] = 0.0;
                }
            }
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            var a_reg: array<f32, 4>;
            var b_reg: array<f32, 8>;
            for (var i: u32 = 0u; i < RM; i = i + 1u) {
                a_reg[i] = tile_a[lr * RM + i][k];
            }
            for (var j: u32 = 0u; j < RN; j = j + 1u) {
                b_reg[j] = tile_b[k][lc * RN + j];
            }
            for (var i: u32 = 0u; i < RM; i = i + 1u) {
                for (var j: u32 = 0u; j < RN; j = j + 1u) {
                    acc[i][j] = acc[i][j] + a_reg[i] * b_reg[j];
                }
            }
        }

        workgroupBarrier();
    }

    for (var i: u32 = 0u; i < RM; i = i + 1u) {
        let global_row = row_base + i;
        if (in_batch && global_row < params.m) {
            for (var j: u32 = 0u; j < RN; j = j + 1u) {
                let global_col = col_base + j;
                if (global_col < params.n) {
                    var v = acc[i][j];
                    if (params.has_bias != 0u) {
                        v = v + arena[params.bias_off + global_col];
                    }
                    v = apply_act(v);
                    arena[c_base + global_row * params.n + global_col] = v;
                }
            }
        }
    }
}
