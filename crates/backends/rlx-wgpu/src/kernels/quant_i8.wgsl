// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// INT8 Quantize / Dequantize — `Op::Quantize` and `Op::Dequantize` with an
// affine (scale, zero-point) table, per-tensor or per-channel. Mirrors
// `rlx_cpu::thunk::ops::quant::exec_quantize` / `exec_dequantize`, the oracle,
// and the Vulkan twins `quantize_i8.comp` / `dequantize_i8.comp`.
//
// I8 tensors are BYTE-PACKED in the f32-uniform arena (`plan_f32_uniform` sizes
// an I8 slot `elems`, not `elems * 4`), so all code offsets here are byte
// offsets into the f32 word array.
//
// Three details that each silently change the result:
//
//   * **`array<vec4<u32>>`, not `array<u32>`.** A scalar array in a WGSL
//     *uniform* buffer has a 16-byte stride, so a tightly packed host-side table
//     would be read from the wrong offsets. That is not hypothetical — the
//     Vulkan twin shipped exactly that bug (naga gives GLSL push arrays std140
//     stride) and returned zeros for every quantized tensor.
//   * **`x * (1/s)`, not `x / s`.** The CPU kernel computes `inv_scale =
//     1.0 / scales[c]` once and multiplies. Dividing instead differs by an ULP,
//     which is enough to move a value across a rounding boundary and change the
//     code. `div_rn` supplies the correctly-rounded reciprocal, since Metal's
//     fast math would otherwise approximate even that.
//   * **round-half-AWAY, not WGSL's `round`.** Rust's `f32::round` breaks ties
//     away from zero; WGSL's `round` breaks them to even. Quantization lands on
//     exact halves constantly (any `x` that is an odd multiple of `s/2`), so the
//     two disagree on real data, not just in principle.

struct Params {
    n: u32,
    chan_dim: u32,
    inner: u32,
    /// `x` f32-element offset (quantize) or code byte offset (dequantize).
    a_off: u32,
    /// Code byte offset (quantize) or `out` f32-element offset (dequantize).
    out_off: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    /// `[scale_bits, zero_point]` per channel, two per `vec4`.
    affine: array<vec4<u32>, 6>,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;

fn affine_at(i: u32) -> u32 { return params.affine[i / 4u][i % 4u]; }

fn rd_byte(byte_off: u32) -> u32 {
    let word = byte_off / 4u;
    let shift = (byte_off % 4u) * 8u;
    return (bitcast<u32>(arena[word]) >> shift) & 0xffu;
}

// Signed i8 at `byte_off`.
fn rd_i8(byte_off: u32) -> i32 {
    let b = rd_byte(byte_off);
    return i32(b) - 256 * i32((b >> 7u) & 1u);
}

// WGSL has no byte store: one invocation owns a whole 4-byte word.
fn wr_word(byte_off: u32, bytes: vec4<u32>) {
    let packed = (bytes.x & 0xffu)
        | ((bytes.y & 0xffu) << 8u)
        | ((bytes.z & 0xffu) << 16u)
        | ((bytes.w & 0xffu) << 24u);
    arena[byte_off / 4u] = bitcast<f32>(packed);
}

// Correctly-rounded a/b — Metal's fast math lowers `/` to a reciprocal
// multiply, up to 1 ULP off. `fma` forms the residual exactly.
fn div_rn(a: f32, b: f32) -> f32 {
    let q = a / b;
    let r = fma(-b, q, a);
    return q + r / b;
}

// Rust `f32::round`: ties away from zero.
fn round_half_away(x: f32) -> f32 {
    let sgn = f32(x > 0.0) - f32(x < 0.0);
    return sgn * floor(abs(x) + 0.5);
}

fn chan_of(i: u32) -> u32 {
    if (params.chan_dim <= 1u) { return 0u; }
    return (i / params.inner) % params.chan_dim;
}

fn tid(gid: vec3<u32>, ngs: vec3<u32>) -> u32 {
    return gid.x + gid.y * ngs.x * 64u;
}

fn quantize_one(i: u32) -> u32 {
    let c = chan_of(i);
    let s = bitcast<f32>(affine_at(2u * c));
    let zp = i32(affine_at(2u * c + 1u));
    let inv = div_rn(1.0, s);
    // Clamp in FLOAT before the int conversion. WGSL leaves `i32(f)` undefined
    // for an out-of-range `f`, and even where it saturates, adding `zp` to
    // `i32::MAX` wraps. ±1e9 is far outside the i8 range, so anything beyond it
    // clamps to ±127/-128 either way — this just makes the intermediate honest.
    // (rlx-cpu hits the same hazard and uses `saturating_add`.)
    let r = clamp(round_half_away(arena[params.a_off + i] * inv), -1.0e9, 1.0e9);
    let v = clamp(i32(r) + zp, -128, 127);
    return u32(v) & 0xffu;
}

@compute @workgroup_size(64)
fn quantize_i8(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(num_workgroups) ngs: vec3<u32>) {
    let t = tid(gid, ngs);
    let words = (params.n + 3u) / 4u;
    if (t >= words) { return; }
    let base = params.out_off + t * 4u;
    // Bytes past the end of the tensor may belong to whatever the planner put
    // next, so carry them through untouched.
    let existing = bitcast<u32>(arena[base / 4u]);
    var out = vec4<u32>(0u, 0u, 0u, 0u);
    for (var lane = 0u; lane < 4u; lane = lane + 1u) {
        let i = t * 4u + lane;
        if (i >= params.n) {
            out[lane] = (existing >> (lane * 8u)) & 0xffu;
            continue;
        }
        out[lane] = quantize_one(i);
    }
    wr_word(base, out);
}

@compute @workgroup_size(64)
fn dequantize_i8(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = tid(gid, ngs);
    if (i >= params.n) { return; }
    let c = chan_of(i);
    let s = bitcast<f32>(affine_at(2u * c));
    let zp = i32(affine_at(2u * c + 1u));
    let qv = rd_i8(params.a_off + i);
    arena[params.out_off + i] = f32(qv - zp) * s;
}
