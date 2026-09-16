// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// General (all-format, all-scale-layout) low-precision quantize / dequantize /
// decode-GEMM for `Op::ScaledMatMul` and friends. WGSL twin of
// `rlx-gpu-kernels/kernels/scaled_lowp_general.cu`; the decode and encode logic
// mirrors `rlx-ir/src/lowp_codec.rs`, the CPU oracle every backend is checked
// against.
//
// Before this, `ScaledMatMul` / `ScaledQuantize` / `ScaledDequantize` /
// `ScaledQuantScale` all took the generic CPU host route on wgpu — a readback,
// a CPU pass and an upload per op — even though the *harder* grouped (MoE)
// variant already had a native decode kernel next door in
// `scaled_grouped_matmul_decode.wgsl`. That variant is MXFP4-only, though, so
// none of it was reusable: the dense op spans every `ScaledFormat`.
//
// **Why an f32 encoder is allowed here.** `lowp_codec::encode` searches for the
// nearest code with an `f64` error term and an even-LSB tie-break, and WGSL has
// no f64 at all. Rather than assume f32 is close enough,
// `rlx-ir/tests/lowp_encode_f32_feasibility.rs` runs both encoders over every
// grid point, every midpoint between adjacent grid points, ±1 ULP either side of
// those midpoints, and 20k pseudo-random values per format spanning 1e-6× to
// 1e6× the format's range. They agree on every one. That test is the licence for
// this file; if it ever fails, this encoder is wrong too.

const TILE: u32 = 16u;
const CUSTOM_BIT: u32 = 0x80000000u;

struct Params {
    // Byte offsets for U8 code / scale tensors; f32-element offsets for f32.
    a_byte_off: u32,        // lhs codes | x (f32 elem off) | codes
    b_byte_off: u32,        // rhs codes
    a_scale_byte_off: u32,
    b_scale_byte_off: u32,
    out_off: u32,           // f32 elem off, or byte off for quantize
    bias_off: u32,
    m: u32,
    k: u32,
    n: u32,
    rows: u32,
    cols: u32,
    a_fmt: u32,
    b_fmt: u32,
    scale_mode: u32,        // 0 per-tensor f32, 1 block E8M0, 2 NVFP4 E4M3
    block: u32,
    has_bias: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>             params: Params;

// ---------------------------------------------------------------------------
// Byte access on the f32-word arena
// ---------------------------------------------------------------------------

fn rd_byte(byte_off: u32) -> u32 {
    let word = byte_off / 4u;
    let shift = (byte_off % 4u) * 8u;
    return (bitcast<u32>(arena[word]) >> shift) & 0xffu;
}

// WGSL has no byte stores, so writers work a whole word at a time: one thread
// owns one 4-byte word and assembles all four codes itself. A read-modify-write
// per byte would race with the three neighbouring threads sharing the word.
fn wr_word(word_index: u32, bytes: vec4<u32>) {
    let packed = (bytes.x & 0xffu)
        | ((bytes.y & 0xffu) << 8u)
        | ((bytes.z & 0xffu) << 16u)
        | ((bytes.w & 0xffu) << 24u);
    arena[word_index] = bitcast<f32>(packed);
}

// WGSL removed `isNan`/`isInf`. NaN is the only value unequal to itself, and
// only the infinities exceed f32::MAX in magnitude.
// Exact 2^e for the exponent ranges these formats use (|e| <= 16 in the worst
// case, well inside f32's normal range).
//
// NOT `exp2(f32(e))`: Metal compiles with fast math on by default, and its
// approximate `exp2` is 1 ULP off for some integer arguments — which showed up
// as a 1-ULP mismatch in `decode(code) * scale` against the CPU oracle even
// though the code and the scale were provably identical. Constructing the
// exponent field directly is both exact and cheaper.
fn pow2i(e: i32) -> f32 {
    return bitcast<f32>(u32(e + 127) << 23u);
}

// Correctly-rounded a/b.
//
// NOT a bare `/`: Metal compiles with fast math on by default, which lowers f32
// division to a reciprocal-multiply. That is up to 1 ULP off, and it is not a
// cosmetic 1 ULP — the scale it produces multiplies every element of a block, so
// `amax / max_finite` came out 0x3bdb6db8 on the GPU against the oracle's
// 0x3bdb6db7 and the whole tensor shifted with it. (Measured on the wgpu/Metal
// backend; the same hazard exists wherever a driver defaults to fast math.)
//
// One Newton step fixes it: `fma` forms the residual `a - b*q` exactly, and
// correcting by `r/b` lands on the correctly-rounded quotient — the second
// divide's own error is second order.
fn div_rn(a: f32, b: f32) -> f32 {
    let q = a / b;
    let r = fma(-b, q, a);
    return q + r / b;
}

fn is_nan(v: f32) -> bool { return v != v; }
fn is_inf(v: f32) -> bool { return abs(v) > 3.4028235e38; }
fn is_finite(v: f32) -> bool { return !is_nan(v) && !is_inf(v); }
fn quiet_nan() -> f32 { return bitcast<f32>(0x7fc00000u); }
// Built from bits rather than by overflowing a literal: naga's GLSL front end
// rejects a constant-folded infinity outright, and the two ports are kept
// spelled the same on purpose.
fn signed_inf(sign: f32) -> f32 {
    return bitcast<f32>(select(0x7f800000u, 0xff800000u, sign < 0.0));
}

// ---------------------------------------------------------------------------
// Codec — mirrors rlx_decode_lowp / rlx_encode_lowp in the .cu
// ---------------------------------------------------------------------------

fn decode_lowp(fmt: u32, code: u32) -> f32 {
    var e_bits: u32;
    var m_bits: u32;
    var bias: i32;
    var fnuz = false;
    var has_inf = false;
    var e4m3ocp = false;

    if ((fmt & CUSTOM_BIT) != 0u) {
        // Parameterized fNeXmY: all-finite, fields packed in `fmt`.
        e_bits = fmt & 0xFu;
        m_bits = (fmt >> 4u) & 0xFu;
        // (bias >> 8) & 0xFF is a signed byte.
        let b = (fmt >> 8u) & 0xFFu;
        bias = select(i32(b), i32(b) - 256, b >= 128u);
    } else {
        if (fmt == 6u) {
            // FP4 E2M1 LUT — matches rlx_ir FP4_E2M1_LUT bit-for-bit.
            let c = code & 0xFu;
            let mag = array<f32, 8>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
            let v = mag[c & 0x7u];
            if ((c & 0x8u) != 0u) { return -v; }
            return v;
        }
        switch fmt {
            case 0u: { e_bits = 4u; m_bits = 3u; bias = 7;  e4m3ocp = true; }
            case 1u: { e_bits = 5u; m_bits = 2u; bias = 15; has_inf = true; }
            case 2u: { e_bits = 4u; m_bits = 3u; bias = 8;  fnuz = true; }
            case 3u: { e_bits = 5u; m_bits = 2u; bias = 16; fnuz = true; }
            case 4u: { e_bits = 2u; m_bits = 3u; bias = 1;  }
            case 5u: { e_bits = 3u; m_bits = 2u; bias = 3;  }
            default: { return 0.0; }
        }
    }

    let width = e_bits + m_bits;
    let sign_bit = (code >> width) & 1u;
    let exp = (code >> m_bits) & ((1u << e_bits) - 1u);
    let mant = code & ((1u << m_bits) - 1u);
    let sign = select(1.0, -1.0, sign_bit != 0u);
    let max_exp = (1u << e_bits) - 1u;

    if (fnuz) {
        if (sign_bit != 0u && exp == 0u && mant == 0u) { return quiet_nan(); }
    } else if (has_inf) {
        if (exp == max_exp) {
            if (mant == 0u) { return signed_inf(sign); }
            return quiet_nan();
        }
    } else if (e4m3ocp) {
        if (exp == max_exp && mant == ((1u << m_bits) - 1u)) { return quiet_nan(); }
    }

    let m_div = f32(1u << m_bits);
    var val: f32;
    if (exp == 0u) {
        val = (f32(mant) / m_div) * pow2i(1 - bias);
    } else {
        val = (1.0 + f32(mant) / m_div) * pow2i(i32(exp) - bias);
    }
    return sign * val;
}

fn code_width(fmt: u32) -> u32 {
    if ((fmt & CUSTOM_BIT) != 0u) {
        return 1u + (fmt & 0xFu) + ((fmt >> 4u) & 0xFu);
    }
    if (fmt == 6u) { return 4u; }
    if (fmt == 4u || fmt == 5u) { return 6u; }
    return 8u;
}

fn max_finite(fmt: u32) -> f32 {
    if ((fmt & CUSTOM_BIT) != 0u) {
        // Scan the (<=256-code) space like the CPU oracle — matches exactly.
        let n = 1u << code_width(fmt);
        var mx = 0.0;
        for (var c = 0u; c < n; c = c + 1u) {
            let v = abs(decode_lowp(fmt, c));
            if (is_finite(v)) { mx = max(mx, v); }
        }
        return mx;
    }
    switch fmt {
        case 0u: { return 448.0; }
        case 1u: { return 57344.0; }
        case 2u: { return 240.0; }
        case 3u: { return 57344.0; }
        case 4u: { return 7.5; }
        case 5u: { return 28.0; }
        default: { return 6.0; }
    }
}

// Nearest-representable encode by exhaustive search — round-half-to-even,
// saturating, NaN -> 0. See the f64 note at the top of this file.
fn encode_lowp(fmt: u32, x_in: f32) -> u32 {
    if (is_nan(x_in)) { return 0u; }
    var x = x_in;
    // ±inf saturates to ±max_finite: a value far outside the grid is equidistant
    // from every code, so the tie-break would otherwise pick code 0.
    if (is_inf(x)) {
        let mf = max_finite(fmt);
        x = select(-mf, mf, x > 0.0);
    }
    let n_codes = 1u << code_width(fmt);
    var best = 0u;
    var best_err = 3.4028235e38;
    var best_lsb = 1u;
    for (var c = 0u; c < n_codes; c = c + 1u) {
        let v = decode_lowp(fmt, c);
        if (!is_finite(v)) { continue; }
        let err = abs(v - x);
        let lsb = c & 1u;
        if (err < best_err || (err == best_err && lsb < best_lsb)) {
            best_err = err;
            best = c;
            best_lsb = lsb;
        }
    }
    return best;
}

fn e8m0(b: u32) -> f32 {
    if (b == 0xFFu) { return quiet_nan(); }
    // Same as `bitcast<f32>(b << 23)`; written via pow2i to keep the
    // "no transcendentals in the codec" rule visible in one place.
    return pow2i(i32(b) - 127);
}

// `ceil(log2(s))`, computed from the bits rather than from `log2`.
//
// The same fast-math reasoning as `pow2i`, but the stakes are higher: this
// picks a whole E8M0 scale byte, so a 1-ULP `log2` near a power of two would
// shift every value in the block by a factor of two, not by an ULP. For a
// normal `s`, `floor(log2(s))` is exactly the unbiased exponent field, and
// `ceil` is that plus one whenever the mantissa is non-zero.
fn f32_to_e8m0(s: f32) -> u32 {
    if (!(s > 0.0) || !is_finite(s)) { return 0u; }
    let bits = bitcast<u32>(s);
    let exp_field = i32((bits >> 23u) & 0xFFu);
    let mant = bits & 0x7FFFFFu;
    var e: i32;
    if (exp_field == 0) {
        // Subnormal: fall back to the scaled form. s = m * 2^-149, so
        // ceil(log2 s) = ceil(log2 m) - 149 with m an integer < 2^23.
        var m = mant;
        var lz = 0;
        while (m > 1u) { m = m >> 1u; lz = lz + 1; }
        let is_pow2 = (mant & (mant - 1u)) == 0u;
        e = lz - 149 + select(1, 0, is_pow2);
    } else {
        e = exp_field - 127 + select(1, 0, mant == 0u);
    }
    var b = e + 127;
    if (b < 0) { b = 0; }
    if (b > 254) { b = 254; }
    return u32(b);
}

/// Scale for element `(r, c)` under `scale_mode`, from a scale tensor at
/// `scale_byte_off` with `nblk` blocks per row.
fn read_scale(scale_byte_off: u32, r: u32, c: u32, nblk: u32) -> f32 {
    if (params.scale_mode == 0u) {
        return arena[scale_byte_off / 4u];
    }
    let si = r * nblk + c / params.block;
    let b = rd_byte(scale_byte_off + si);
    if (params.scale_mode == 1u) { return e8m0(b); }
    return decode_lowp(0u, b); // NVFP4 E4M3 scale
}

fn nblk_of(cols: u32) -> u32 {
    if (params.scale_mode == 0u) { return 1u; }
    return (cols + params.block - 1u) / params.block;
}

fn tid(gid: vec3<u32>, ngs: vec3<u32>) -> u32 {
    return gid.x + gid.y * ngs.x * 64u;
}

// ---------------------------------------------------------------------------
// amax -> scale
// ---------------------------------------------------------------------------

// Per-tensor mode writes one f32; block modes write one scale BYTE per block,
// so a thread owns a whole 4-byte word and computes its four blocks itself.
@compute @workgroup_size(64)
fn scaled_quant_scale(@builtin(global_invocation_id) gid: vec3<u32>,
                      @builtin(num_workgroups) ngs: vec3<u32>) {
    let t = tid(gid, ngs);
    let maxf = max_finite(params.a_fmt);

    if (params.scale_mode == 0u) {
        if (t != 0u) { return; }
        var amax = 0.0;
        let n = params.rows * params.cols;
        for (var i = 0u; i < n; i = i + 1u) {
            amax = max(amax, abs(arena[params.a_byte_off + i]));
        }
        arena[params.a_scale_byte_off / 4u] = select(1.0, div_rn(amax, maxf), amax > 0.0);
        return;
    }

    let nblk = nblk_of(params.cols);
    let total = params.rows * nblk;
    let words = (total + 3u) / 4u;
    if (t >= words) { return; }

    var out = vec4<u32>(0u, 0u, 0u, 0u);
    // Preserve bytes past the end of the scale tensor: the word may be shared
    // with whatever the planner put next in the arena.
    let base = params.a_scale_byte_off + t * 4u;
    let word = base / 4u;
    let existing = bitcast<u32>(arena[word]);
    for (var lane = 0u; lane < 4u; lane = lane + 1u) {
        let idx = t * 4u + lane;
        if (idx >= total) {
            out[lane] = (existing >> (lane * 8u)) & 0xffu;
            continue;
        }
        let r = idx / nblk;
        let b = idx % nblk;
        let lo = b * params.block;
        let hi = min(lo + params.block, params.cols);
        var amax = 0.0;
        for (var c = lo; c < hi; c = c + 1u) {
            amax = max(amax, abs(arena[params.a_byte_off + r * params.cols + c]));
        }
        let s = select(1.0, div_rn(amax, maxf), amax > 0.0);
        if (params.scale_mode == 1u) {
            out[lane] = f32_to_e8m0(s);
        } else {
            out[lane] = encode_lowp(0u, s);
        }
    }
    wr_word(word, out);
}

// ---------------------------------------------------------------------------
// quantize: x / scale(block) -> codes
// ---------------------------------------------------------------------------

// One thread per output WORD (four codes) — see `wr_word`.
@compute @workgroup_size(64)
fn scaled_quantize(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(num_workgroups) ngs: vec3<u32>) {
    let t = tid(gid, ngs);
    let total = params.rows * params.cols;
    let words = (total + 3u) / 4u;
    if (t >= words) { return; }

    let nblk = nblk_of(params.cols);
    let base = params.out_off + t * 4u;
    let word = base / 4u;
    let existing = bitcast<u32>(arena[word]);

    var out = vec4<u32>(0u, 0u, 0u, 0u);
    for (var lane = 0u; lane < 4u; lane = lane + 1u) {
        let i = t * 4u + lane;
        if (i >= total) {
            out[lane] = (existing >> (lane * 8u)) & 0xffu;
            continue;
        }
        let r = i / params.cols;
        let c = i % params.cols;
        let s = read_scale(params.a_scale_byte_off, r, c, nblk);
        let v = select(0.0, div_rn(arena[params.a_byte_off + i], s), s != 0.0);
        out[lane] = encode_lowp(params.a_fmt, v);
    }
    wr_word(word, out);
}

// ---------------------------------------------------------------------------
// dequantize: codes -> f32
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn scaled_dequantize(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(num_workgroups) ngs: vec3<u32>) {
    let i = tid(gid, ngs);
    let total = params.rows * params.cols;
    if (i >= total) { return; }
    let r = i / params.cols;
    let c = i % params.cols;
    let nblk = nblk_of(params.cols);
    let s = read_scale(params.a_scale_byte_off, r, c, nblk);
    let code = rd_byte(params.a_byte_off + i);
    arena[params.out_off + i] = decode_lowp(params.a_fmt, code) * s;
}

// ---------------------------------------------------------------------------
// decode-GEMM (TN: out = lhs · rhsᵀ, both operands K-last)
// ---------------------------------------------------------------------------

var<workgroup> a_tile: array<array<f32, 16>, 16>;
var<workgroup> b_tile: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn scaled_matmul_decode(@builtin(local_invocation_id) lid: vec3<u32>,
                        @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let i = wid.y * TILE + ty; // output row (m)
    let j = wid.x * TILE + tx; // output col (n)

    let nblk = nblk_of(params.k);
    var acc = 0.0;
    let ntiles = (params.k + TILE - 1u) / TILE;

    for (var t = 0u; t < ntiles; t = t + 1u) {
        // Stage A[i, t*TILE+tx] (decoded × its scale).
        let pa = t * TILE + tx;
        if (i < params.m && pa < params.k) {
            let ls = read_scale(params.a_scale_byte_off, i, pa, nblk);
            a_tile[ty][tx] =
                decode_lowp(params.a_fmt, rd_byte(params.a_byte_off + i * params.k + pa)) * ls;
        } else {
            a_tile[ty][tx] = 0.0;
        }
        // Stage B[j, t*TILE+ty] — b_tile[p][tx] maps to rhs[j, ·].
        let pb = t * TILE + ty;
        if (j < params.n && pb < params.k) {
            let rs = read_scale(params.b_scale_byte_off, j, pb, nblk);
            b_tile[ty][tx] =
                decode_lowp(params.b_fmt, rd_byte(params.b_byte_off + j * params.k + pb)) * rs;
        } else {
            b_tile[ty][tx] = 0.0;
        }
        workgroupBarrier();
        for (var p = 0u; p < TILE; p = p + 1u) {
            acc = acc + a_tile[ty][p] * b_tile[p][tx];
        }
        workgroupBarrier();
    }

    if (i < params.m && j < params.n) {
        if (params.has_bias != 0u) { acc = acc + arena[params.bias_off + j]; }
        arena[params.out_off + i * params.n + j] = acc;
    }
}
