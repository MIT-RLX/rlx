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

//! Quantization parameters and helpers.
//!
//! A quantized tensor stores `i8` codes alongside a single `QParams`
//! `{ scale, zero_point }`. Real value = `scale * (q - zero_point)`.

#[derive(Debug, Clone, Copy)]
pub struct QParams {
    pub scale: f32,
    pub zero_point: i32,
}

impl QParams {
    pub const fn new(scale: f32, zero_point: i32) -> Self {
        Self { scale, zero_point }
    }

    /// Symmetric quantization: zero_point = 0.
    pub const fn symmetric(scale: f32) -> Self {
        Self {
            scale,
            zero_point: 0,
        }
    }
}

/// Saturating cast i32 → i8.
#[inline(always)]
pub fn sat_i8(v: i32) -> i8 {
    if v > i8::MAX as i32 {
        i8::MAX
    } else if v < i8::MIN as i32 {
        i8::MIN
    } else {
        v as i8
    }
}

/// Round-half-to-even f32 → i32. The Cortex-M4F FPU has VCVT for this
/// in hardware; on the host we fall back to the standard library.
#[inline(always)]
pub fn round_to_i32(x: f32) -> i32 {
    // `round_ties_even` would be ideal but isn't stable; plain `round`
    // (half-away-from-zero) is what most quantization toolchains
    // assume anyway.
    let r = x + if x >= 0.0 { 0.5 } else { -0.5 };
    r as i32
}

/// Requantize an i32 accumulator to i8 with a single f32 multiplier
/// and an output zero point. This is the hot spot in every layer's
/// epilogue.
#[inline(always)]
pub fn requantize(acc: i32, multiplier: f32, out_zp: i32) -> i8 {
    sat_i8(round_to_i32(acc as f32 * multiplier) + out_zp)
}

/// Dequantize: q-code → real f32. Useful for fp32 cross-checks.
#[inline(always)]
pub fn dequant(q: i8, p: QParams) -> f32 {
    (q as i32 - p.zero_point) as f32 * p.scale
}

/// Quantize: real f32 → i8 q-code, saturating.
#[inline(always)]
pub fn quant(x: f32, p: QParams) -> i8 {
    sat_i8(round_to_i32(x / p.scale) + p.zero_point)
}

/// Read one logical weight from a packed byte slice.
///
/// `bits` ∈ {8, 4, 2}. For 8 the value is just `w[idx]`. For 4 the
/// byte at `idx/2` holds two nibbles (low = idx even, high = idx
/// odd), each a 2's-complement signed 4-bit value sign-extended to
/// i32. For 2 the byte at `idx/4` holds four 2-bit lanes, indexed
/// from the LSB up, each sign-extended to i32.
///
/// The trainer emits values within the bit-width's signed range
/// (i4: `[-7, 7]`, i2: `[-1, 1]` ternary), so the unused 2's-complement
/// codepoint at each width (`-8` for i4, `-2` for i2) never appears
/// in practice — but the unpack still sign-extends it correctly.
#[inline(always)]
pub fn read_weight(w: &[i8], idx: usize, bits: u8) -> i32 {
    match bits {
        8 => w[idx] as i32,
        4 => {
            let byte = w[idx >> 1] as u8;
            let nibble = if idx & 1 == 0 { byte & 0x0F } else { byte >> 4 } as i32;
            // sign-extend the low 4 bits
            (nibble << 28) >> 28
        }
        2 => {
            let byte = w[idx >> 2] as u8;
            let lane = (idx & 3) * 2;
            let crumb = ((byte >> lane) & 0x03) as i32;
            // sign-extend the low 2 bits
            (crumb << 30) >> 30
        }
        _ => panic!("read_weight: unsupported bits {bits}"),
    }
}

/// Read the raw 2-bit code at `idx` from a ternary-packed buffer.
///
/// Returns the 2-bit value as `u8` ∈ `{0, 1, 2, 3}`. The trainer's
/// ternary scheme only emits codes `0` (= 0), `1` (= +1), and `3`
/// (= -1, two's-complement) — code `2` is never emitted, so the
/// ternary kernel can match those three cases and skip a multiply.
/// This is the hot helper used by `conv2d_ternary` / `dense_ternary`.
#[inline(always)]
pub fn ternary_code(w: &[i8], idx: usize) -> u8 {
    let byte = w[idx >> 2] as u8;
    let lane = (idx & 3) * 2;
    (byte >> lane) & 0x03
}

/// Number of logical weights per packed byte.
#[inline(always)]
pub const fn weights_per_byte(bits: u8) -> usize {
    (8 / bits) as usize
}

/// Length in bytes of a packed-weight tensor with `n_weights` logical
/// elements at `bits` bits per element. Rounds up to a whole byte.
#[inline(always)]
pub const fn packed_byte_len(n_weights: usize, bits: u8) -> usize {
    let w_per_b = weights_per_byte(bits);
    n_weights.div_ceil(w_per_b)
}
