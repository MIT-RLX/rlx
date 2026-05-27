// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! NVIDIA FP4 (E2M1) block layout shared by FLUX / MLX `nvfp4` mode.
//!
//! Fixed group size 16 along the contracting (K) axis. Each group stores
//! eight packed bytes (two 4-bit codes per byte) plus one FP8 E4M3 block
//! scale per output column. An optional per-tensor `global_scale` (f32)
//! lives in the fourth `Op::DequantMatMul` input (the legacy `zp` slot).

/// Elements sharing one FP8 E4M3 block scale.
pub const NVFP4_GROUP_SIZE: usize = 16;

/// OCP E2M1 FP4 decode table (MLX / Blackwell NVFP4, indices 0..15).
pub const FP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[inline]
pub fn fp4_e2m1_to_f32(nibble: u8) -> f32 {
    FP4_E2M1_LUT[(nibble & 0x0F) as usize]
}

/// Decode one FP8 E4M3 scale byte (OCP, exp bias 7).
#[inline]
pub fn fp8_e4m3_scale_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;
    let v = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            (mant as f32 / 8.0) * 2f32.powi(-6)
        }
    } else if exp == 0x0F && mant == 0x07 {
        0.0 // NaN → 0 for matmul stability
    } else {
        (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    sign * v
}

/// Packed weight bytes for `[k, n]` FP4 weights (two nibbles per byte).
#[inline]
pub const fn nvfp4_weight_bytes(k: usize, n: usize) -> usize {
    (k * n).div_ceil(2)
}

/// Block-scale bytes for `[k, n]` with groups along K (`[k/16, n]` FP8 scales).
#[inline]
pub const fn nvfp4_scale_bytes(k: usize, n: usize) -> usize {
    k.div_ceil(NVFP4_GROUP_SIZE) * n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_lut_matches_ocp() {
        assert_eq!(fp4_e2m1_to_f32(2), 1.0);
        assert_eq!(fp4_e2m1_to_f32(14), -4.0);
    }

    #[test]
    fn fp8_scale_one_is_unity() {
        // E4M3 encoding of 1.0: exp=7 → biased 14 (0x38), mant=0.
        assert!((fp8_e4m3_scale_to_f32(0x38) - 1.0).abs() < 1e-6);
    }
}
