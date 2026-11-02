// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

enable wgpu_cooperative_matrix;

// Edge-tolerant twin of `matmul_coop_f32.wgsl` for shapes that kernel cannot
// take: K not a multiple of 8 and/or N not a multiple of 32 (e.g. attention
// with head_dim 25 — scores `[S,25]·[25,S]`, context `[S,S]·[S,25]` — and
// d_model 200 projections). Same 32×32 tile, same `simdgroup_float8x8`
// accumulation, same Metal row-major (`...T`) fragment convention.
//
// Difference: A is staged with zero padding past M/K; B is staged (zero-padded
// past K/N) only on a ragged tile, so a ragged tile neither reads past the
// operand nor multiplies neighbouring data into a valid output (0 × NaN would
// poison it, hence B is zeroed too, not just A). Full B tiles load directly.
// B may be any arena tensor (activation or Param). The aligned kernel stays
// the fast path; this one is only selected when `k % 8 != 0 || n % 32 != 0`
// (see `matmul_coop_f32_kernel_for`).

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
const TILE_N: u32 = 32u;
const TILE_K: u32 = 8u;

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;

var<workgroup> acc_scratch: array<f32, 1024>;
var<workgroup> a_stage:     array<f32, 256>;     // 32 × 8
var<workgroup> b_stage:     array<f32, 256>;     // 8 × 32
var<workgroup> b_full_wg:   u32;

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

// Exact GELU — A&S 7.1.26 erf, matches rlx-cpu's scalar_gelu.
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

@compute @workgroup_size(32)
fn matmul_coop_f32_edge(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let bz = wid.z;
    let row_base = wid.y * TILE_M;
    let col_base = wid.x * TILE_N;

    let a_base = params.a_off + bz * params.a_batch_stride;
    let b_base = params.b_off + bz * params.b_batch_stride;
    let c_base = params.c_off + bz * params.c_batch_stride;

    for (var s: u32 = 0u; s < 32u; s = s + 1u) {
        acc_scratch[lid + s * 32u] = 0.0;
    }
    workgroupBarrier();

    var acc_00: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_01: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_02: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_03: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_10: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_11: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_12: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_13: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_20: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_21: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_22: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_23: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_30: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_31: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_32: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);
    var acc_33: coop_mat8x8<f32, C> = coopLoadT<coop_mat8x8<f32, C>>(&acc_scratch[0], 8u);

    let n_tiles = (params.k + TILE_K - 1u) / TILE_K;
    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        let k_off = t * TILE_K;
        // A tile 32×8, zero outside [M,K] (always staged: it is also the M guard).
        for (var s: u32 = 0u; s < 8u; s = s + 1u) {
            let idx = lid + s * 32u;
            let ar = row_base + idx / 8u;
            let ac = k_off + idx % 8u;
            var av: f32 = 0.0;
            if (ar < params.m && ac < params.k) {
                av = arena[a_base + ar * params.k + ac];
            }
            a_stage[idx] = av;
        }
        // B: a FULL 8×32 tile (inside K and N) is read straight from the arena,
        // as the aligned kernel does; only the ragged column tile / K tail goes
        // through zero-padded staging. Workgroup-uniform (wid.x, t), as coop
        // ops require.
        // Built from uniform values only, but naga's uniformity analysis does not
        // accept a coop load under a plain `if` on it; `workgroupUniformLoad`
        // is the WGSL-sanctioned way to hand it a provably uniform value.
        if (lid == 0u) {
            b_full_wg = select(0u, 1u, (k_off + TILE_K <= params.k) && (col_base + TILE_N <= params.n));
        }
        let b_full = workgroupUniformLoad(&b_full_wg) == 1u;
        if (!b_full) {
            for (var s: u32 = 0u; s < 8u; s = s + 1u) {
                let idx = lid + s * 32u;
                let br = k_off + idx / 32u;
                let bc = col_base + idx % 32u;
                var bv: f32 = 0.0;
                if (br < params.k && bc < params.n) {
                    bv = arena[b_base + br * params.n + bc];
                }
                b_stage[idx] = bv;
            }
        }
        workgroupBarrier();

        // Row-major (`T`) loads — see the convention note in matmul_coop_f32.wgsl.
        let a_0: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[0u], 8u);
        let a_1: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[64u], 8u);
        let a_2: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[128u], 8u);
        let a_3: coop_mat8x8<f32, A> = coopLoadT<coop_mat8x8<f32, A>>(&a_stage[192u], 8u);
        if (b_full) {
            let b_row = b_base + k_off * params.n + col_base;
            let b_0: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 0u], params.n);
            let b_1: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 8u], params.n);
            let b_2: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 16u], params.n);
            let b_3: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&arena[b_row + 24u], params.n);
            acc_00 = coopMultiplyAdd(a_0, b_0, acc_00);
            acc_01 = coopMultiplyAdd(a_0, b_1, acc_01);
            acc_02 = coopMultiplyAdd(a_0, b_2, acc_02);
            acc_03 = coopMultiplyAdd(a_0, b_3, acc_03);
            acc_10 = coopMultiplyAdd(a_1, b_0, acc_10);
            acc_11 = coopMultiplyAdd(a_1, b_1, acc_11);
            acc_12 = coopMultiplyAdd(a_1, b_2, acc_12);
            acc_13 = coopMultiplyAdd(a_1, b_3, acc_13);
            acc_20 = coopMultiplyAdd(a_2, b_0, acc_20);
            acc_21 = coopMultiplyAdd(a_2, b_1, acc_21);
            acc_22 = coopMultiplyAdd(a_2, b_2, acc_22);
            acc_23 = coopMultiplyAdd(a_2, b_3, acc_23);
            acc_30 = coopMultiplyAdd(a_3, b_0, acc_30);
            acc_31 = coopMultiplyAdd(a_3, b_1, acc_31);
            acc_32 = coopMultiplyAdd(a_3, b_2, acc_32);
            acc_33 = coopMultiplyAdd(a_3, b_3, acc_33);
        } else {
            let b_0: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&b_stage[0u], 32u);
            let b_1: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&b_stage[8u], 32u);
            let b_2: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&b_stage[16u], 32u);
            let b_3: coop_mat8x8<f32, B> = coopLoadT<coop_mat8x8<f32, B>>(&b_stage[24u], 32u);
            acc_00 = coopMultiplyAdd(a_0, b_0, acc_00);
            acc_01 = coopMultiplyAdd(a_0, b_1, acc_01);
            acc_02 = coopMultiplyAdd(a_0, b_2, acc_02);
            acc_03 = coopMultiplyAdd(a_0, b_3, acc_03);
            acc_10 = coopMultiplyAdd(a_1, b_0, acc_10);
            acc_11 = coopMultiplyAdd(a_1, b_1, acc_11);
            acc_12 = coopMultiplyAdd(a_1, b_2, acc_12);
            acc_13 = coopMultiplyAdd(a_1, b_3, acc_13);
            acc_20 = coopMultiplyAdd(a_2, b_0, acc_20);
            acc_21 = coopMultiplyAdd(a_2, b_1, acc_21);
            acc_22 = coopMultiplyAdd(a_2, b_2, acc_22);
            acc_23 = coopMultiplyAdd(a_2, b_3, acc_23);
            acc_30 = coopMultiplyAdd(a_3, b_0, acc_30);
            acc_31 = coopMultiplyAdd(a_3, b_1, acc_31);
            acc_32 = coopMultiplyAdd(a_3, b_2, acc_32);
            acc_33 = coopMultiplyAdd(a_3, b_3, acc_33);
        }
        workgroupBarrier();
    }

    coopStoreT(acc_00, &acc_scratch[0u   * 32u + 0u ], 32u);
    coopStoreT(acc_01, &acc_scratch[0u   * 32u + 8u ], 32u);
    coopStoreT(acc_02, &acc_scratch[0u   * 32u + 16u], 32u);
    coopStoreT(acc_03, &acc_scratch[0u   * 32u + 24u], 32u);
    coopStoreT(acc_10, &acc_scratch[8u   * 32u + 0u ], 32u);
    coopStoreT(acc_11, &acc_scratch[8u   * 32u + 8u ], 32u);
    coopStoreT(acc_12, &acc_scratch[8u   * 32u + 16u], 32u);
    coopStoreT(acc_13, &acc_scratch[8u   * 32u + 24u], 32u);
    coopStoreT(acc_20, &acc_scratch[16u  * 32u + 0u ], 32u);
    coopStoreT(acc_21, &acc_scratch[16u  * 32u + 8u ], 32u);
    coopStoreT(acc_22, &acc_scratch[16u  * 32u + 16u], 32u);
    coopStoreT(acc_23, &acc_scratch[16u  * 32u + 24u], 32u);
    coopStoreT(acc_30, &acc_scratch[24u  * 32u + 0u ], 32u);
    coopStoreT(acc_31, &acc_scratch[24u  * 32u + 8u ], 32u);
    coopStoreT(acc_32, &acc_scratch[24u  * 32u + 16u], 32u);
    coopStoreT(acc_33, &acc_scratch[24u  * 32u + 24u], 32u);
    workgroupBarrier();

    for (var s: u32 = 0u; s < 32u; s = s + 1u) {
        let idx = lid + s * 32u;
        let r = idx / 32u;
        let c = idx % 32u;
        let global_row = row_base + r;
        let global_col = col_base + c;
        // Bounds-check on M: when params.m isn't a multiple of 32, the
        // dispatcher rounds up the y-dim and the kernel computes garbage
        // for the padded rows (it reads OOB in A and accumulates garbage
        // into acc). Skipping the write keeps the padded compute harmless
        // — the output rows past `params.m` are left untouched.
        if (global_row >= params.m || global_col >= params.n) { continue; }
        var v: f32 = acc_scratch[idx];
        if (params.has_bias != 0u) {
            v = v + arena[params.bias_off + global_col];
        }
        v = apply_act(v);
        arena[c_base + global_row * params.n + global_col] = v;
    }
}
