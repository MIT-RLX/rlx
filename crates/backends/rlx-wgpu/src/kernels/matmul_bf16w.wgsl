// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Tiled fp32 matmul reading a PACKED-BF16 weight B in-shader.
//
// Same 32×32 output tile / 4×4 register block / f32 accumulator as
// `matmul.wgsl::matmul`, but the weight (B) input is bound as a raw
// `array<u32>` holding two bf16 values per word (bf16 = the high 16
// bits of f32). Each B element `e` is unpacked with:
//   word = weights[e / 2]
//   bits = (e & 1) == 0 ? (word & 0xFFFF) : (word >> 16)
//   w    = bitcast<f32>(bits << 16)
// so the kernel reads HALF the B bytes vs the widened-f32 arena path,
// and is BIT-EXACT to an f32 matmul whose weights were bf16-rounded
// (bf16→f32 is exact — it only zero-extends the mantissa).
//
// `b_off` / `b_batch_stride` are in bf16 ELEMENTS. The packed buffer is
// indexed parallel to the f32 arena (bf16 element i ↔ f32 element i), so
// these are the same numeric values the plain f32 kernel would use.
//
// A (activation) and C (output) stay f32 in the arena (binding 0).
// Pure WGSL u32 ops — no `enable f16`, runs on any wgpu backend.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    a_off: u32,
    b_off: u32,        // bf16-element offset (== global arena f32-word index)
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

const TILE_M: u32 = 32u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 16u;
const RM: u32 = 4u;
const RN: u32 = 4u;

// Bind group ordering matches `build_kernel_3` in kernels/mod.rs:
// 0=storage(rw), 1=uniform, 2=storage(ro).
@group(0) @binding(0) var<storage, read_write> arena:   array<f32>;
@group(0) @binding(1) var<uniform>             params:  Params;
@group(0) @binding(2) var<storage, read>       weights: array<u32>;

var<workgroup> tile_a: array<array<f32, 16>, 32>;  // [TILE_M][TILE_K]
var<workgroup> tile_b: array<array<f32, 32>, 16>;  // [TILE_K][TILE_N]

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
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))));
    let e = s * (1.0 - poly * exp(-xa * xa));
    return 0.5 * x * (1.0 + e);
}

fn apply_act(v_in: f32) -> f32 {
    var v = v_in;
    switch (params.act_id) {
        case 0xFFFFu: {}
        case 0u: { v = max(v, 0.0); }
        case 1u: { v = 1.0 / (1.0 + exp(-clamp(v, -88.0, 88.0))); }
        case 2u: { v = tanh(clamp(v, -15.0, 15.0)); }
        case 5u: { v = sqrt(v); }
        case 7u: { v = -v; }
        case 8u: { v = abs(v); }
        case 9u: { v = gelu_erf(v); }
        case 11u: {
            let c = 0.7978845608028654;
            let x3 = v * v * v;
            let inner = clamp(c * (v + 0.044715 * x3), -15.0, 15.0);
            v = 0.5 * v * (1.0 + tanh(inner));
        }
        case 10u: {
            let nx = clamp(-v, -88.0, 88.0);
            v = v / (1.0 + exp(nx));
        }
        default: {}
    }
    return v;
}

@compute @workgroup_size(8, 8)
fn matmul_bf16w(
    @builtin(local_invocation_id)    lid: vec3<u32>,
    @builtin(workgroup_id)           wid: vec3<u32>,
) {
    let bz = wid.z;
    let in_batch = bz < params.batch;
    let bz_safe = select(0u, bz, in_batch);

    let lr = lid.y;
    let lc = lid.x;
    let row_base = wid.y * TILE_M + lr * RM;
    let col_base = wid.x * TILE_N + lc * RN;

    let a_base = params.a_off + bz_safe * params.a_batch_stride;
    let b_base = params.b_off + bz_safe * params.b_batch_stride;  // bf16 elements
    let c_base = params.c_off + bz_safe * params.c_batch_stride;

    var acc: array<array<f32, 4>, 4>;
    for (var i: u32 = 0u; i < RM; i = i + 1u) {
        for (var j: u32 = 0u; j < RN; j = j + 1u) {
            acc[i][j] = 0.0;
        }
    }

    let n_tiles = (params.k + TILE_K - 1u) / TILE_K;

    for (var t: u32 = 0u; t < n_tiles; t = t + 1u) {
        // tile_a (32×16) — f32 arena reads. Unchanged from matmul.wgsl.
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
        // tile_b (16×32) — unpacked from the packed bf16 weight buffer.
        for (var i: u32 = 0u; i < 2u; i = i + 1u) {
            let k_local = lr * 2u + i;
            let global_k = t * TILE_K + k_local;
            for (var j: u32 = 0u; j < RN; j = j + 1u) {
                let n_local = lc * RN + j;
                let global_col = wid.x * TILE_N + n_local;
                if (in_batch && global_k < params.k && global_col < params.n) {
                    let e = b_base + global_k * params.n + global_col;
                    tile_b[k_local][n_local] = unpack_bf16(e);
                } else {
                    tile_b[k_local][n_local] = 0.0;
                }
            }
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_K; k = k + 1u) {
            var a_reg: array<f32, 4>;
            var b_reg: array<f32, 4>;
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
