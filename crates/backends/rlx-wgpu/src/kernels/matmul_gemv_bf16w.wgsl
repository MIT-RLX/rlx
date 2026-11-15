// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// `rlx_div_rn`, `rlx_exp`, …: rlxsl's correctly-rounded helpers, appended by
// `kernel_module` (`with_numeric_helpers`).

// Decode (m <= 4) form of `matmul_bf16w.wgsl`: the same packed-BF16
// weight B (two bf16 per u32 word, `b_off` in bf16 elements), A and C f32
// in the arena, same bias + activation epilogue and bindings, so it shares
// that kernel's bind groups.
//
// The tiled kernel spends 31/32 of its work on padding rows at m = 1. Here
// one thread owns two adjacent output columns, read from one u32 word of B
// per k (coalesced, half the bytes of the f32 `matmul_gemv`), and adds K in
// order, as the tiled kernel does, so a product is bit-identical to it.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    a_off: u32,
    b_off: u32,        // bf16-element offset (== the param's arena byte offset / 2)
    c_off: u32,
    batch: u32,
    a_batch_stride: u32,
    b_batch_stride: u32, // bf16 elements
    c_batch_stride: u32,
    has_bias: u32,
    bias_off: u32,
    act_id: u32,
    _p0: u32, _p1: u32, _p2: u32,
};

const COLS: u32 = 128u;  // two per thread: a u32 word of B each

const MAX_M: u32 = 4u;

// Bind group ordering matches `build_kernel_3` in kernels/mod.rs:
// 0=storage(rw), 1=uniform, 2=storage(ro).
@group(0) @binding(0) var<storage, read_write> arena:   array<f32>;
@group(0) @binding(1) var<uniform>             params:  Params;
@group(0) @binding(2) var<storage, read>       weights: array<u32>;


// bf16 element `e` → f32. bf16 is the high 16 bits of f32.
fn unpack_bf16(e: u32) -> f32 {
    let word = weights[e >> 1u];
    let bits = select(word >> 16u, word & 0xFFFFu, (e & 1u) == 0u);
    return bitcast<f32>(bits << 16u);
}

fn gelu_erf(x: f32) -> f32 {
    let arg = x * 0.70710678118654752;
    let s = select(-1.0, 1.0, arg >= 0.0);
    let xa = abs(arg);
    let t = rlx_div_rn(1.0, fma(0.3275911, xa, 1.0));
    // explicit fma: an uncontracted chain is contracted differently by
    // different shader compilers
    let poly = t * fma(t, fma(t, fma(t, fma(t, 1.061405429, -1.453152027), 1.421413741),
                -0.284496736), 0.254829592);
    let e = s * fma(-poly, rlx_exp(-xa * xa), 1.0);
    return 0.5 * x * (1.0 + e);
}

fn apply_act(v_in: f32) -> f32 {
    var v = v_in;
    switch (params.act_id) {
        case 0xFFFFu: {}
        case 0u: { v = max(v, 0.0); }
        case 1u: { v = rlx_div_rn(1.0, 1.0 + rlx_exp(-clamp(v, -88.0, 88.0))); }
        case 2u: { v = rlx_tanh(clamp(v, -15.0, 15.0)); }
        case 5u: { v = rlx_sqrt_rn(v); }
        case 7u: { v = -v; }
        case 8u: { v = abs(v); }
        case 9u: { v = gelu_erf(v); }
        case 11u: {
            let c = 0.7978845608028654;
            let x3 = v * v * v;
            let inner = clamp(c * fma(0.044715, x3, v), -15.0, 15.0);
            v = 0.5 * v * (1.0 + rlx_tanh(inner));
        }
        case 10u: {
            let nx = clamp(-v, -88.0, 88.0);
            v = rlx_div_rn(v, 1.0 + rlx_exp(nx));
        }
        default: {}
    }
    return v;
}

// Columns `col`, `col + 1` (`col` even) from one word per k.
fn pair(w: u32) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(w << 16u), bitcast<f32>(w & 0xFFFF0000u));
}

@compute @workgroup_size(64)
fn matmul_gemv_bf16w(
    @builtin(local_invocation_id)    lid: vec3<u32>,
    @builtin(workgroup_id)           wid: vec3<u32>,
) {
    let bz = wid.z;
    let in_batch = bz < params.batch;
    let bz_safe = select(0u, bz, in_batch);
    let col = wid.x * COLS + lid.x * 2u;
    if (!in_batch || col >= params.n) {
        return;
    }
    let two = col + 1u < params.n;
    let a_base = params.a_off + bz_safe * params.a_batch_stride;
    let b_base = params.b_off + bz_safe * params.b_batch_stride + col;  // bf16 elements
    let c_base = params.c_off + bz_safe * params.c_batch_stride;
    let rows = min(params.m, MAX_M);

    var acc: array<vec2<f32>, 4>;
    for (var i: u32 = 0u; i < MAX_M; i = i + 1u) {
        acc[i] = vec2<f32>(0.0);
    }
    // A pair shares one word when it starts on one (even element offset and
    // an even row width); otherwise each half is unpacked on its own.
    let aligned = ((b_base | params.n) & 1u) == 0u;
    if (aligned) {
        // Four words in flight per step, the sums still taken in k order.
        let k4 = params.k & ~3u;
        let w0 = b_base >> 1u;
        let nw = params.n >> 1u;
        for (var k: u32 = 0u; k < k4; k = k + 4u) {
            let b0 = pair(weights[w0 + k * nw]);
            let b1 = pair(weights[w0 + (k + 1u) * nw]);
            let b2 = pair(weights[w0 + (k + 2u) * nw]);
            let b3 = pair(weights[w0 + (k + 3u) * nw]);
            for (var i: u32 = 0u; i < rows; i = i + 1u) {
                let ar = a_base + i * params.k + k;
                var v = acc[i];
                v = v + arena[ar] * b0;
                v = v + arena[ar + 1u] * b1;
                v = v + arena[ar + 2u] * b2;
                v = v + arena[ar + 3u] * b3;
                acc[i] = v;
            }
        }
        for (var k: u32 = k4; k < params.k; k = k + 1u) {
            let b = pair(weights[w0 + k * nw]);
            for (var i: u32 = 0u; i < rows; i = i + 1u) {
                acc[i] = acc[i] + arena[a_base + i * params.k + k] * b;
            }
        }
    } else {
        for (var k: u32 = 0u; k < params.k; k = k + 1u) {
            let e = b_base + k * params.n;
            let b = vec2<f32>(unpack_bf16(e), select(0.0, unpack_bf16(e + 1u), two));
            for (var i: u32 = 0u; i < rows; i = i + 1u) {
                acc[i] = acc[i] + arena[a_base + i * params.k + k] * b;
            }
        }
    }
    for (var i: u32 = 0u; i < rows; i = i + 1u) {
        var v = acc[i];
        if (params.has_bias != 0u) {
            v.x = v.x + arena[params.bias_off + col];
            if (two) {
                v.y = v.y + arena[params.bias_off + col + 1u];
            }
        }
        arena[c_base + i * params.n + col] = apply_act(v.x);
        if (two) {
            arena[c_base + i * params.n + col + 1u] = apply_act(v.y);
        }
    }
}
