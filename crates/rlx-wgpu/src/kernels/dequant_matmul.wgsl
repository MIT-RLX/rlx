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

// Dequant-on-the-fly matmul. Reads packed int8 weight bytes from the
// f32-typed arena via bitcast<u32>, then dequantizes per-block before
// accumulating into the output.
//
// The arena binding is `array<f32>` like every other kernel — for the
// w_q tensor we bitcast each f32 word to u32 and extract one of four
// bytes via shift+mask. The user uploads packed int8 bytes through
// `set_param_bytes`, which writes them tight-packed at the slot's
// f32 offset.
//
// Layout:
//   x  [m, k]                            f32
//   w_q[k, n]   tightly packed i8         (k*n bytes inside k*n*4-byte slot)
//   scale[k/block, n]                    f32
//   zp   [k/block, n]                    f32 (read only when is_asym != 0)
// Output:
//   out[m, n]                            f32

// scheme_id selects the unpack path:
//   0 = Int8Block      (bits=8, signed, per-block scale)
//   1 = Int8BlockAsym  (bits=8, signed, per-block scale + zero point)
//   2 = Int4Block      (bits=4, signed, per-block scale)
//   3 = Fp8E4m3        (bits=8, no scale, OCP E4M3 bit decode)
//   4 = Fp8E5m2        (bits=8, no scale, OCP E5M2 bit decode)
//   5 = Nvfp4Block     (E2M1 nibbles + FP8 E4M3 block scales + f32 global_scale @ zp_off)

struct Params {
    m: u32,
    k: u32,
    n: u32,
    block_size: u32,
    scheme_id: u32,
    x_off: u32,
    w_off: u32,        // f32-element offset of w_q's slot
    scale_off: u32,
    zp_off: u32,
    out_off: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

// Read one byte from the f32-packed weight stream at flat element index.
fn read_byte(elem_idx: u32) -> u32 {
    let word_idx = elem_idx / 4u;
    let byte_shift = (elem_idx % 4u) * 8u;
    let bits = bitcast<u32>(arena[params.w_off + word_idx]);
    return (bits >> byte_shift) & 0xffu;
}

// Read one signed nibble (4-bit, two-per-byte, low nibble first).
fn read_nibble_signed(elem_idx: u32) -> i32 {
    let word_idx = elem_idx / 8u;
    let nib_shift = (elem_idx % 8u) * 4u;
    let bits = bitcast<u32>(arena[params.w_off + word_idx]);
    let nib: u32 = (bits >> nib_shift) & 0xfu;
    var q = i32(nib);
    if (q >= 8) { q = q - 16; }
    return q;
}

// Read one byte from the f32-packed scale stream at flat byte index.
fn read_scale_byte(byte_idx: u32) -> u32 {
    let word_idx = byte_idx / 4u;
    let byte_shift = (byte_idx % 4u) * 8u;
    let bits = bitcast<u32>(arena[params.scale_off + word_idx]);
    return (bits >> byte_shift) & 0xffu;
}

// Read one unsigned nibble (FP4 E2M1 code 0..15).
fn read_nibble_u4(elem_idx: u32) -> u32 {
    let word_idx = elem_idx / 8u;
    let nib_shift = (elem_idx % 8u) * 4u;
    let bits = bitcast<u32>(arena[params.w_off + word_idx]);
    return (bits >> nib_shift) & 0xfu;
}

const FP4_E2M1: array<f32, 16> = array<f32, 16>(
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
);

// OCP E4M3: 1 sign + 4 exp + 3 mantissa, exp bias = 7, no infinity.
// The all-ones exp + max mantissa pattern (0x7F / 0xFF) is reserved for NaN.
fn decode_e4m3(byte: u32) -> f32 {
    let sign = (byte >> 7u) & 1u;
    let exp  = (byte >> 3u) & 0xfu;
    let mant = byte & 0x7u;
    var v: f32;
    if (exp == 0u) {
        // Subnormal: value = mant/8 * 2^-6 (smallest subnormal = 2^-9).
        v = (f32(mant) / 8.0) * exp2(-6.0);
    } else if (exp == 15u && mant == 7u) {
        // NaN — coerce to 0 so downstream matmul stays finite.
        v = 0.0;
    } else {
        let m = 1.0 + f32(mant) / 8.0;
        v = m * exp2(f32(i32(exp) - 7));
    }
    if (sign != 0u) { v = -v; }
    return v;
}

// OCP E5M2: 1 sign + 5 exp + 2 mantissa, exp bias = 15, has infinity.
// exp=31 + mant=0 → ±inf; exp=31 + mant>0 → NaN.
fn decode_e5m2(byte: u32) -> f32 {
    let sign = (byte >> 7u) & 1u;
    let exp  = (byte >> 2u) & 0x1fu;
    let mant = byte & 0x3u;
    var v: f32;
    if (exp == 0u) {
        // Subnormal: value = mant/4 * 2^-14.
        v = (f32(mant) / 4.0) * exp2(-14.0);
    } else if (exp == 31u) {
        // Inf or NaN — coerce to 0 to keep the matmul finite.
        v = 0.0;
    } else {
        let m = 1.0 + f32(mant) / 4.0;
        v = m * exp2(f32(i32(exp) - 15));
    }
    if (sign != 0u) { v = -v; }
    return v;
}

@compute @workgroup_size(8, 8)
fn dequant_matmul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.m || col >= params.n) { return; }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < params.k; k = k + 1u) {
        let elem_idx = k * params.n + col;
        var w_dq: f32 = 0.0;
        if (params.scheme_id == 0u) {
            // Int8Block (symmetric).
            let byte = read_byte(elem_idx);
            var q = i32(byte);
            if (q >= 128) { q = q - 256; }
            let block = k / params.block_size;
            let scale = arena[params.scale_off + block * params.n + col];
            w_dq = f32(q) * scale;
        } else if (params.scheme_id == 1u) {
            // Int8BlockAsym (with zero-point).
            let byte = read_byte(elem_idx);
            var q = i32(byte);
            if (q >= 128) { q = q - 256; }
            let block = k / params.block_size;
            let scale = arena[params.scale_off + block * params.n + col];
            let zp    = arena[params.zp_off    + block * params.n + col];
            w_dq = (f32(q) - zp) * scale;
        } else if (params.scheme_id == 2u) {
            // Int4Block (symmetric).
            let q = read_nibble_signed(elem_idx);
            let block = k / params.block_size;
            let scale = arena[params.scale_off + block * params.n + col];
            w_dq = f32(q) * scale;
        } else if (params.scheme_id == 3u) {
            // Fp8E4m3 — direct bit decode, no scale.
            let byte = read_byte(elem_idx);
            w_dq = decode_e4m3(byte);
        } else if (params.scheme_id == 4u) {
            // Fp8E5m2 — direct bit decode, no scale.
            let byte = read_byte(elem_idx);
            w_dq = decode_e5m2(byte);
        } else {
            // Nvfp4Block — E2M1 nibble × FP8 block scale × global f32 scale.
            let nib = read_nibble_u4(elem_idx);
            let block = k / params.block_size;
            let scale_byte = read_scale_byte(block * params.n + col);
            let scale = decode_e4m3(scale_byte);
            let gs = arena[params.zp_off];
            w_dq = FP4_E2M1[nib] * scale * gs;
        }
        acc = acc + arena[params.x_off + row * params.k + k] * w_dq;
    }
    arena[params.out_off + row * params.n + col] = acc;
}
