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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    input_modulus: array<u32, MAX_INPUTS>,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
// Storage buffer (read-only) instead of uniform: WGSL uniform-storage
// requires 16-byte stride for array elements, which doesn't fit our
// `array<u32, N>` packed layout. Storage allows any stride.
@group(0) @binding(1) var<storage, read>        params: Params;

fn resolve_operand(enc: u32, i: u32, scratch: ptr<function, array<f32, 32>>) -> f32 {
    let kind = enc >> 31u;
    let idx  = enc & 0x7FFFFFFFu;
    if (kind == 0u) {
        // Input. Scalar-broadcast inputs read element 0 (fast path);
        // trailing-shape broadcast tiles by `i % input_modulus[idx]`;
        // `input_modulus[idx] == 0` ⇒ read by `i` directly.
        let off = params.input_offs[idx];
        var row: u32;
        if ((params.scalar_input_mask & (1u << idx)) != 0u) {
            row = 0u;
        } else if (params.input_modulus[idx] != 0u) {
            row = i % params.input_modulus[idx];
        } else {
            row = i;
        }
        return arena[off + row];
    } else {
        // Prior step result
        return (*scratch)[idx];
    }
}

fn apply_activation(sub: u32, x: f32) -> f32 {
    if (sub == 3u) { return max(x, 0.0); }                // Relu
    if (sub == 0u || sub == 1u) {                          // Gelu / GeluApprox
        // GELU via the sigmoid-form identity:
        //   gelu_approx(x) = 0.5 · x · (1 + tanh(c·(x + 0.044715·x³)))
        //                  = x · sigmoid(2·c·(x + 0.044715·x³))
        // The tanh form hits `0·∞ = NaN` on Apple Metal's wgsl `tanh`
        // for some specific x values that produce huge intermediates
        // (observed reproducer: BERT MiniLM6 FFN1, 1 NaN per ~5000
        // outputs — see commit message for the bisect). The sigmoid
        // form has no such trap because sigmoid clamps in [0, 1] and
        // x · sigmoid never multiplies a finite by a sign-flipping
        // zero.
        let c2 = 2.0f * 0.7978845608f;
        let inner = c2 * (x + 0.044715f * x * x * x);
        let s = 1.0f / (1.0f + exp(-inner));
        return x * s;
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

fn apply_compare(sub: u32, a: f32, b: f32) -> f32 {
    if (sub == 0u) { return select(0.0f, 1.0f, a == b); } // Eq
    if (sub == 1u) { return select(0.0f, 1.0f, a != b); } // Ne
    if (sub == 2u) { return select(0.0f, 1.0f, a <  b); } // Lt
    if (sub == 3u) { return select(0.0f, 1.0f, a <= b); } // Le
    if (sub == 4u) { return select(0.0f, 1.0f, a >  b); } // Gt
    if (sub == 5u) { return select(0.0f, 1.0f, a >= b); } // Ge
    return 0.0f;
}

@compute @workgroup_size(64)
fn elementwise_region(@builtin(global_invocation_id) gid: vec3<u32>,
                      @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = gid.x + gid.y * ngs.x * 64u;
    if (i >= params.len) { return; }

    var scratch: array<f32, 32>;
    var last_idx: u32 = 0u;
    for (var k: u32 = 0u; k < params.num_steps; k = k + 1u) {
        let base = k * 4u;
        let op_kind = params.chain[base + 0u];
        let op_sub  = params.chain[base + 1u];
        let lhs_enc = params.chain[base + 2u];
        let rhs_enc = params.chain[base + 3u];

        let lhs = resolve_operand(lhs_enc, i, &scratch);
        var result: f32;
        if (op_kind == 0u) {
            result = apply_activation(op_sub, lhs);
        } else if (op_kind == 1u) {
            // Cast at f32-arena layer is identity.
            result = lhs;
        } else if (op_kind == 2u) {
            let rhs = resolve_operand(rhs_enc, i, &scratch);
            result = apply_binary(op_sub, lhs, rhs);
        } else if (op_kind == 3u) {
            let rhs = resolve_operand(rhs_enc, i, &scratch);
            result = apply_compare(op_sub, lhs, rhs);
        } else {
            // op_kind == 4u: Where (3-operand select). op_sub carries
            // cond_enc; lhs already resolved is on_true; rhs is on_false.
            let cond = resolve_operand(op_sub, i, &scratch);
            let on_false = resolve_operand(rhs_enc, i, &scratch);
            result = select(on_false, lhs, cond != 0.0f);
        }
        scratch[k] = result;
        last_idx = k;
    }
    arena[params.dst_off + i] = scratch[last_idx];
}
