// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// PLAN L2 — interpreted N-ary element-wise chain kernel.
//
// One thread per output element. Each thread walks the chain encoding
// (compile-time fixed-size array, runtime length via num_steps) and
// computes intermediate values into a private scratch register array.
// The final step's result is written to dst.
//
// Encoding (4 u32s per step):
//   chain[k*4 + 0] = op_kind  (0=Activation, 1=Cast, 2=Binary, 3=Compare)
//   chain[k*4 + 1] = op_sub   (discriminant of the inner op variant)
//   chain[k*4 + 2] = lhs_enc  (bit 31 = src kind: 0=Input, 1=Step;
//                              bits 0..30 = index)
//   chain[k*4 + 3] = rhs_enc  (same; ignored for unary ops)
//
// Per-input data is read from `arena[input_offs[idx] + i]`.
// Output is written at `arena[dst_off + i]`.

const MAX_STEPS: u32 = 32u;
const MAX_INPUTS: u32 = 16u;

struct Params {
    len: u32,
    num_inputs: u32,
    num_steps: u32,
    dst_off: u32,
    input_offs: array<u32, MAX_INPUTS>,
    chain: array<u32, 128>,   // MAX_STEPS * 4
    scalar_input_mask: u32,
    prologue: u32,
    out_n: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    prologue_input: u32,
    input_modulus: array<u32, MAX_INPUTS>,
};

fn region_input_row_resize2x_nchw(
    gid: u32,
    out_n: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
) -> u32 {
    let plane = out_c * out_h * out_w;
    let local = gid % plane;
    let batch = gid / plane;
    let w_pos = local % out_w;
    let tmp = local / out_w;
    let h_pos = tmp % out_h;
    let c_pos = tmp / out_h;
    let in_w = out_w / 2u;
    let in_h = out_h / 2u;
    let in_plane = out_c * in_h * in_w;
    return batch * in_plane + c_pos * in_h * in_w + (h_pos / 2u) * in_w + (w_pos / 2u);
}

fn region_resolve_row(
    gid: u32,
    kind: u32,
    idx: u32,
    prologue_row0: u32,
    has_prologue_row0: u32,
) -> u32 {
    if (kind != 0u) { return 0u; }
    if (has_prologue_row0 != 0u && idx == params.prologue_input) {
        return prologue_row0;
    }
    if ((params.scalar_input_mask & (1u << idx)) != 0u) { return 0u; }
    if (params.input_modulus[idx] != 0u) { return gid % params.input_modulus[idx]; }
    return gid;
}

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
// Storage buffer (read-only) instead of uniform: WGSL uniform-storage
// requires 16-byte stride for array elements, which doesn't fit our
// `array<u32, N>` packed layout. Storage allows any stride.
@group(0) @binding(1) var<storage, read>        params: Params;

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

fn resolve_operand(
    enc: u32,
    i: u32,
    scratch: ptr<function, array<f32, 32>>,
    prologue_row0: u32,
    has_prologue_row0: u32,
) -> f32 {
    let kind = enc >> 31u;
    let idx  = enc & 0x7FFFFFFFu;
    if (kind == 0u) {
        let off = params.input_offs[idx];
        let row = region_resolve_row(i, kind, idx, prologue_row0, has_prologue_row0);
        return arena[off + row];
    } else {
        return (*scratch)[idx];
    }
}

fn apply_activation(sub: u32, x: f32) -> f32 {
    if (sub == 3u) { return max(x, 0.0); }                // Relu
    if (sub == 0u) { return gelu_erf(x); }                   // Gelu (exact erf)
    if (sub == 1u) {                                          // GeluApprox
        let c = 0.7978845608028654;
        let x3 = x * x * x;
        let inner = clamp(c * (x + 0.044715 * x3), -15.0, 15.0);
        return 0.5 * x * (1.0 + tanh(inner));
    }
    if (sub == 2u) { return x / (1.0f + exp(-x)); }       // Silu
    if (sub == 4u) { return 1.0f / (1.0f + exp(-x)); }    // Sigmoid
    if (sub == 5u) { return tanh(x); }                     // Tanh
    if (sub == 6u) { return exp(x); }                      // Exp
    if (sub == 7u) { return log(x); }                      // Log
    if (sub == 8u) { return sqrt(x); }                     // Sqrt
    if (sub == 9u) { return 1.0f / sqrt(x); }              // Rsqrt
    if (sub == 10u) { return -x; }                         // Neg
    if (sub == 11u) { return abs(x); }                     // Abs
    if (sub == 13u) { return sin(x); }                     // Sin
    if (sub == 14u) { return cos(x); }                     // Cos
    if (sub == 15u) { return tan(x); }                     // Tan
    if (sub == 16u) { return atan(x); }                    // Atan
    return x;
}

fn apply_binary(sub: u32, a: f32, b: f32) -> f32 {
    if (sub == 0u) { return a + b; }   // Add
    if (sub == 1u) { return a - b; }   // Sub
    if (sub == 2u) { return a * b; }   // Mul
    if (sub == 3u) { return a / b; }   // Div
    if (sub == 4u) { return max(a, b); } // Max
    if (sub == 5u) { return min(a, b); } // Min
    if (sub == 6u) { return pow(a, b); } // Pow
    return a;
}

// Fused `ChainStep::Cast` on the f32-uniform arena. `sub` is the target-class
// code injected by wgpu's `cast_region_sub`: 0=identity (float target), 1=Bool,
// 2..=7 = int targets. Int casts truncate toward zero and SATURATE to the
// destination range (matches rlx-cpu `x as iN`); NaN → 0.
fn apply_cast(sub: u32, x: f32) -> f32 {
    if (sub == 0u) { return x; }                             // → float: identity
    if (sub == 1u) { return select(0.0f, 1.0f, x != 0.0f); } // → Bool
    var lo: f32 = 0.0f;
    var hi: f32 = 0.0f;
    if (sub == 2u) { lo = -128.0f; hi = 127.0f; }                       // I8
    else if (sub == 3u) { lo = -32768.0f; hi = 32767.0f; }             // I16
    else if (sub == 4u) { lo = -2147483648.0f; hi = 2147483647.0f; }   // I32
    else if (sub == 5u) { lo = -9223372036854775808.0f; hi = 9223372036854775807.0f; } // I64
    else if (sub == 6u) { lo = 0.0f; hi = 255.0f; }                    // U8
    else if (sub == 7u) { lo = 0.0f; hi = 4294967295.0f; }             // U32
    else { return x; }
    let t = clamp(trunc(x), lo, hi);
    return select(t, 0.0f, x != x); // NaN → 0
}

fn apply_compare(sub: u32, a: f32, b: f32) -> f32 {
    if (sub == 0u) { return select(0.0f, 1.0f, a == b); } // Eq
    if (sub == 1u) { return select(0.0f, 1.0f, a != b); } // Ne
    if (sub == 2u) { return select(0.0f, 1.0f, a <  b); } // Lt
    if (sub == 3u) { return select(0.0f, 1.0f, a <= b); } // Le
    if (sub == 4u) { return select(0.0f, 1.0f, a >  b); } // Gt
    if (sub == 5u) { return select(0.0f, 1.0f, a >= b); } // Ge
    return 0.0f;
}

fn run_region(i: u32) {
    var prologue_row0: u32 = 0u;
    var has_prologue_row0: u32 = 0u;
    if (params.prologue == 1u) {
        prologue_row0 = region_input_row_resize2x_nchw(
            i, params.out_n, params.out_c, params.out_h, params.out_w);
        has_prologue_row0 = 1u;
    }

    var scratch: array<f32, 32>;
    var last_idx: u32 = 0u;
    for (var k: u32 = 0u; k < params.num_steps; k = k + 1u) {
        let base = k * 4u;
        let op_kind = params.chain[base + 0u];
        let op_sub  = params.chain[base + 1u];
        let lhs_enc = params.chain[base + 2u];
        let rhs_enc = params.chain[base + 3u];

        let lhs = resolve_operand(lhs_enc, i, &scratch, prologue_row0, has_prologue_row0);
        var result: f32;
        if (op_kind == 0u) {
            result = apply_activation(op_sub, lhs);
        } else if (op_kind == 1u) {
            // Cast. op_sub carries the target-class code (cast_region_sub).
            result = apply_cast(op_sub, lhs);
        } else if (op_kind == 2u) {
            let rhs = resolve_operand(rhs_enc, i, &scratch, prologue_row0, has_prologue_row0);
            result = apply_binary(op_sub, lhs, rhs);
        } else if (op_kind == 3u) {
            let rhs = resolve_operand(rhs_enc, i, &scratch, prologue_row0, has_prologue_row0);
            result = apply_compare(op_sub, lhs, rhs);
        } else {
            // op_kind == 4u: Where (3-operand select). op_sub carries
            // cond_enc; lhs already resolved is on_true; rhs is on_false.
            let cond = resolve_operand(op_sub, i, &scratch, prologue_row0, has_prologue_row0);
            let on_false = resolve_operand(rhs_enc, i, &scratch, prologue_row0, has_prologue_row0);
            result = select(on_false, lhs, cond != 0.0f);
        }
        scratch[k] = result;
        last_idx = k;
    }
    arena[params.dst_off + i] = scratch[last_idx];
}

@compute @workgroup_size(64)
fn elementwise_region(@builtin(global_invocation_id) gid: vec3<u32>,
                      @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.len) { return; }
    run_region(i);
}

// NCHW spatial grid for resize-nearest prologue (8×8×1 workgroups).
@compute @workgroup_size(8, 8, 1)
fn elementwise_region_spatial(@builtin(global_invocation_id) gid: vec3<u32>) {
    let wo = gid.x;
    let ho = gid.y;
    let nc = gid.z;
    if (nc >= params.out_n * params.out_c || ho >= params.out_h || wo >= params.out_w) {
        return;
    }
    let i = nc * params.out_h * params.out_w + ho * params.out_w + wo;
    run_region(i);
}

// FKL batch horizontal fusion: one dispatch, workgroup Z = slice index.
struct BatchParams {
    slice_len: u32,
    num_batch: u32,
    num_steps: u32,
    base_dst_off: u32,
    slice_elems: u32,
    batch_input_offs: array<u32, 64>,
    chain: array<u32, 128>,
    scalar_input_mask: u32,
    input_modulus: array<u32, MAX_INPUTS>,
}

@group(0) @binding(1) var<storage, read> batch_params: BatchParams;

fn region_resolve_row_batch(gid: u32, kind: u32, idx: u32) -> u32 {
    if (kind != 0u) { return 0u; }
    if ((batch_params.scalar_input_mask & (1u << idx)) != 0u) { return 0u; }
    if (batch_params.input_modulus[idx] != 0u) { return gid % batch_params.input_modulus[idx]; }
    return gid;
}

fn resolve_operand_batch(
    enc: u32,
    i: u32,
    batch_idx: u32,
    scratch: ptr<function, array<f32, 32>>,
) -> f32 {
    let kind = enc >> 31u;
    let idx  = enc & 0x7FFFFFFFu;
    if (kind == 0u) {
        let off = batch_params.batch_input_offs[batch_idx];
        let row = region_resolve_row_batch(i, kind, idx);
        return arena[off + row];
    } else {
        return (*scratch)[idx];
    }
}

fn run_batch_region(i: u32, batch_idx: u32) {
    var scratch: array<f32, 32>;
    var last_idx: u32 = 0u;
    let dst = batch_params.base_dst_off + batch_idx * batch_params.slice_elems;
    for (var k: u32 = 0u; k < batch_params.num_steps; k = k + 1u) {
        let base = k * 4u;
        let op_kind = batch_params.chain[base + 0u];
        let op_sub  = batch_params.chain[base + 1u];
        let lhs_enc = batch_params.chain[base + 2u];
        let rhs_enc = batch_params.chain[base + 3u];

        let lhs = resolve_operand_batch(lhs_enc, i, batch_idx, &scratch);
        var result: f32;
        if (op_kind == 0u) {
            result = apply_activation(op_sub, lhs);
        } else if (op_kind == 1u) {
            result = apply_cast(op_sub, lhs);
        } else if (op_kind == 2u) {
            let rhs = resolve_operand_batch(rhs_enc, i, batch_idx, &scratch);
            result = apply_binary(op_sub, lhs, rhs);
        } else if (op_kind == 3u) {
            let rhs = resolve_operand_batch(rhs_enc, i, batch_idx, &scratch);
            result = apply_compare(op_sub, lhs, rhs);
        } else {
            let cond = resolve_operand_batch(op_sub, i, batch_idx, &scratch);
            let on_false = resolve_operand_batch(rhs_enc, i, batch_idx, &scratch);
            result = select(on_false, lhs, cond != 0.0f);
        }
        scratch[k] = result;
        last_idx = k;
    }
    arena[dst + i] = scratch[last_idx];
}

@compute @workgroup_size(64, 1, 1)
fn batch_elementwise_region(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.z >= batch_params.num_batch) { return; }
    if (gid.x >= batch_params.slice_len) { return; }
    run_batch_region(gid.x, gid.z);
}
