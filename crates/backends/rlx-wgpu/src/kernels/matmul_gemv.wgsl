// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Matrix-vector product for decode: `m ≤ 4` rows of A against a row-major
// `[K, N]` B, same `Params`, bias + activation epilogue and batch strides as
// `matmul.wgsl` (so it shares its bind groups).
//
// The 32×32-tile `matmul` spends 31/32 of its work on padding rows at m = 1,
// and launches only N/32 workgroups of 64 threads — a GPU-wide stall for a
// 1024-wide layer. Here a 256-thread workgroup owns 32 adjacent columns and
// splits K eight ways: a 32-lane group reads 32 consecutive B values (one
// coalesced row segment) per k, and the eight partial sums meet in shared
// memory. Up to four A rows share each B load.
//
// Pure WGSL — no extensions, no subgroup ops.

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

const COLS: u32 = 32u;
const KSPLIT: u32 = 8u;
const MAX_M: u32 = 4u;

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// [KSPLIT * MAX_M][COLS] partial sums.
var<workgroup> partial: array<array<f32, 32>, 32>;

// Exact GELU via Abramowitz & Stegun 7.1.26 erf approximation.
// Mirrors `scalar_gelu` in rlx-cpu/src/kernels.rs so wgpu output matches
// the CPU baseline to f32 precision (~1e-7) instead of the ~5e-4 drift
// the tanh approximation introduces.
fn gelu_erf(x: f32) -> f32 {
    let arg = x * 0.70710678118654752;       // x / sqrt(2)
    let s = select(-1.0, 1.0, arg >= 0.0);
    let xa = abs(arg);
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))));
    let e = s * (1.0 - poly * exp(-xa * xa));
    return 0.5 * x * (1.0 + e);
}

fn apply_act(v_in: f32) -> f32 {
    // No early return — FXC flags `return` inside helpers when the caller
    // later sits behind workgroup barriers (uninit / varying-flow noise).
    var v = v_in;
    switch (params.act_id) {
        case 0xFFFFu: {}                                                   // none
        case 0u: { v = max(v, 0.0); }                                      // relu
        case 1u: { v = 1.0 / (1.0 + exp(-clamp(v, -88.0, 88.0))); }        // sigmoid
        case 2u: { v = tanh(clamp(v, -15.0, 15.0)); }                      // tanh
        case 5u: { v = sqrt(v); }
        case 7u: { v = -v; }
        case 8u: { v = abs(v); }
        case 9u: { v = gelu_erf(v); }                                      // exact gelu
        case 11u: {
            // GeluApprox: the tanh form (~5e-4 max diff vs erf form).
            let c = 0.7978845608028654;
            let x3 = v * v * v;
            let inner = clamp(c * (v + 0.044715 * x3), -15.0, 15.0);
            v = 0.5 * v * (1.0 + tanh(inner));
        }
        case 10u: {
            // SiLU = x · sigmoid(x)
            let nx = clamp(-v, -88.0, 88.0);
            v = v / (1.0 + exp(nx));
        }
        default: {}
    }
    return v;
}

@compute @workgroup_size(32, 8)
fn matmul_gemv(
    @builtin(local_invocation_id)    lid: vec3<u32>,
    @builtin(workgroup_id)           wid: vec3<u32>,
) {
    let bz = wid.z;
    let in_batch = bz < params.batch;
    let bz_safe = select(0u, bz, in_batch);
    let col = wid.x * COLS + lid.x;
    let ks = lid.y;
    let col_ok = in_batch && col < params.n;

    let a_base = params.a_off + bz_safe * params.a_batch_stride;
    let b_base = params.b_off + bz_safe * params.b_batch_stride;
    let c_base = params.c_off + bz_safe * params.c_batch_stride;
    let rows = min(params.m, MAX_M);

    var acc: array<f32, 4>;
    for (var i: u32 = 0u; i < MAX_M; i = i + 1u) {
        acc[i] = 0.0;
    }
    if (col_ok) {
        for (var k: u32 = ks; k < params.k; k = k + KSPLIT) {
            let b = arena[b_base + k * params.n + col];
            for (var i: u32 = 0u; i < rows; i = i + 1u) {
                acc[i] = acc[i] + arena[a_base + i * params.k + k] * b;
            }
        }
    }
    for (var i: u32 = 0u; i < MAX_M; i = i + 1u) {
        partial[ks * MAX_M + i][lid.x] = acc[i];
    }

    workgroupBarrier();

    if (ks == 0u && col_ok) {
        for (var i: u32 = 0u; i < rows; i = i + 1u) {
            var v = 0.0;
            for (var s: u32 = 0u; s < KSPLIT; s = s + 1u) {
                v = v + partial[s * MAX_M + i][lid.x];
            }
            if (params.has_bias != 0u) {
                v = v + arena[params.bias_off + col];
            }
            v = apply_act(v);
            arena[c_base + i * params.n + col] = v;
        }
    }
}
