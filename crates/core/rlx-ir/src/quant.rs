// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Quantization metadata as graph annotations (plan #57).
//!
//! Schemes attach per-tensor via [`QuantMap`] on [`crate::Graph`], not inside
//! [`crate::Node`], so the common case stays lean.
//!
//! # GGUF schemes and backends
//!
//! All `Gguf*` variants use two-input `Op::DequantMatMul` (`x`, packed weights).
//! GPU backends share integer **scheme ids** (0–23) in `dequant_gguf` kernels;
//! legacy tail: Q4_0 = 19, Q8_0 = 20, Q4_1 = 21, Q5_0 = 22, Q5_1 = 23.
//!
//! Per-backend dispatch (GPU dequant, fused GEMV, ANE constexpr, TPU compile-time
//! bake, env toggles): [`docs/gguf-backend-paths.md`](../../docs/gguf-backend-paths.md).

use crate::NodeId;
use std::collections::HashMap;

/// How a tensor is quantized. Mirrors the schemes RLX needs for LLM
/// inference on Apple Silicon: blockwise int8 (GPTQ-style),
/// blockwise int4 (Q4_K), and per-tensor fp8 (e4m3 / e5m2).
///
/// Each variant carries the parameters the dequantizer needs to read
/// at runtime — scale, zero-point, block size. Where these live in
/// the actual weight tensor is up to the loader (#56).
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantScheme {
    /// Symmetric int8 with one scale per `block_size` elements.
    Int8Block { block_size: u32 },
    /// Asymmetric int8 with scale + zero-point per `block_size` elements.
    Int8BlockAsym { block_size: u32 },
    /// Int4 packed two-per-byte, scale per `block_size` elements
    /// (Q4_K-ish; matches GGUF block layout).
    Int4Block { block_size: u32 },
    /// FP8 e4m3 (no scale; same domain as half).
    Fp8E4m3,
    /// FP8 e5m2 (no scale; wider range than e4m3).
    Fp8E5m2,
    /// GGUF / llama.cpp Q4_K super-block (256 elements / 144 bytes).
    /// Packs an f16 super-scale + f16 super-min + 8 sub-block 6-bit
    /// scales + 8 sub-block 6-bit mins + 128 nibbles. Block layout is
    /// fixed by the format — there's no `block_size` knob.
    GgufQ4K,
    /// GGUF Q5_K (256 / 176 bytes). Adds a 32-byte high-bit plane on
    /// top of Q4_K.
    GgufQ5K,
    /// GGUF Q6_K (256 / 210 bytes). Per-sub-block signed scales,
    /// no min term.
    GgufQ6K,
    /// GGUF Q8_K (256 / 276 bytes). Per-super-block f32 scale plus
    /// i8 quants and a 32-byte sum-of-blocks table that's only used
    /// by Q8_K × Q8_K matmul accumulation paths.
    GgufQ8K,
    /// GGUF Q2_K (256 / 84 bytes). 2-bit quants with per-sub-block scale/min.
    GgufQ2K,
    /// GGUF Q3_K (256 / 110 bytes). 3-bit quants with hmask high bit plane.
    GgufQ3K,
    /// GGUF Q4_0 (32 / 18 bytes). Legacy llama.cpp block: f16 scale + signed nibbles (−8..7).
    GgufQ4_0,
    /// GGUF Q4_1 (32 / 20 bytes). Legacy block: f16 scale + f16 min + unsigned nibbles (0..15).
    /// GPU kernel scheme id **21** (shared with Metal/CUDA/ROCm/WGPU).
    GgufQ4_1,
    /// GGUF Q5_0 (32 / 22 bytes). Legacy block: f16 scale + 32-bit high plane + signed 5-bit quants.
    /// GPU kernel scheme id **22**.
    GgufQ5_0,
    /// GGUF Q5_1 (32 / 24 bytes). Legacy block: f16 scale + f16 min + high plane + unsigned 5-bit quants.
    /// GPU kernel scheme id **23**.
    GgufQ5_1,
    /// GGUF Q8_0 (32 / 34 bytes). Legacy block: f16 scale + 32×i8 quants.
    GgufQ8_0,
    /// NVIDIA FP4 (E2M1) block — fixed 16-element groups, FP8 E4M3 block
    /// scales, optional f32 global scale on input 3 (legacy `zp` slot).
    /// Used by FLUX.2 / MLX `nvfp4` checkpoints.
    Nvfp4Block,
    /// MLX affine quantization (`nn.quantize` / mlx-lm default): packed
    /// uint codes + per-group f32 scale and bias. Inputs to
    /// `DequantMatMul`: `x`, `w_q`, `scales`, `biases` (biases in the
    /// legacy `zp` slot). Typical `bits=4`, `group_size=64`.
    MlxAffine { bits: u8, group_size: u32 },
    /// MLX `mxfp4` mode — E2M1 nibbles × per-group scale (E8M0 or FP8).
    /// Inputs: `x`, `w_q`, `scales`, unused `zp` (pass ones).
    MlxMxfp4 { group_size: u32 },
    /// MLX `mxfp8` mode — FP8 E4M3 codes × per-group scale.
    MlxMxfp8 { group_size: u32 },
    /// **MxFp4x2** — two-level residual FP4 (E2M1): `value ≈ s0·q0 + s1·q1`,
    /// each level an E2M1 nibble plane with its own per-group scale. Roughly
    /// doubles the effective mantissa of MXFP4 (~3.3 → ~6.7 bits) at 2× storage.
    /// The low-precision analog of double-word; see [`crate::residual`].
    MxFp4x2Block { group_size: u32 },
    // ── GGUF IQ-family (sub-byte LUT-coded) ─────────────────────
    /// IQ4_NL: 4.5 bpw non-linear. 32-element block (18 bytes).
    GgufIQ4NL,
    /// IQ4_XS: 4.25 bpw. 256-element super-block (136 bytes).
    GgufIQ4XS,
    /// IQ2_XXS: 2.0625 bpw. 256 / 66.
    GgufIQ2XXS,
    /// IQ2_XS: 2.3125 bpw. 256 / 74.
    GgufIQ2XS,
    /// IQ2_S: 2.5625 bpw. 256 / 82.
    GgufIQ2S,
    /// IQ3_XXS: 3.0625 bpw. 256 / 98.
    GgufIQ3XXS,
    /// IQ3_S: 3.4375 bpw. 256 / 110.
    GgufIQ3S,
    /// IQ1_S: 1.5625 bpw. 256 / 50.
    GgufIQ1S,
    /// IQ1_M: 1.75 bpw. 256 / 56.
    GgufIQ1M,
    /// TQ1_0: 1.6875 bpw ternary. 256 / 54.
    GgufTQ1_0,
    /// TQ2_0: 2.0625 bpw ternary. 256 / 66.
    GgufTQ2_0,
    /// MXFP4: OCP microscaling FP4 with E8M0 scale. 32 / 17.
    GgufMXFP4,
    /// NVFP4 GGUF variant: E4M3 scale + E2M1 nibbles. 16 / 9.
    GgufNVFP4,
    /// Custom 1-bit format (PrismML `Bonsai-27B`, ggml type 41): one
    /// f16 group scale + 128 sign bits selecting `±d`. 128 / 18 bytes
    /// (1.125 bpw). See [`crate`] consumers `rlx_gguf::q1_dequant`.
    GgufQ1_0,
    /// PrismML / upstream `Q2_0` (ggml type 42): f16 group scale + 128×2-bit
    /// codes → `(q−1)·d`. 128 / 34 bytes (2.125 bpw). Used by Ternary Bonsai
    /// (`prism-ml/Ternary-Bonsai-*-gguf`). See `rlx_gguf::q2_dequant`.
    GgufQ2_0,
}

/// Single source of truth for GGUF → GPU `dequant_gguf` scheme ids.
///
/// Expanding this macro inside `impl QuantScheme` defines:
/// - [`QuantScheme::gpu_dequant_scheme_id`]
/// - [`QuantScheme::from_gpu_dequant_scheme_id`]
/// - test helper [`QuantScheme::GPU_DEQUANT_SCHEME_ID_PAIRS`]
///
/// GPU backends (`gguf_host::gguf_scheme_id` / `scheme_from_id`) must
/// delegate here — do not re-list variants by hand.
macro_rules! define_gguf_gpu_dequant_ids {
    ($(($variant:ident, $id:literal)),+ $(,)?) => {
        /// Canonical numeric id for the shared on-device GGUF dequant kernel
        /// (`dequant_gguf` in MSL / CU / WGSL / GLSL).
        ///
        /// Generated by `define_gguf_gpu_dequant_ids!` so encode and
        /// [`Self::from_gpu_dequant_scheme_id`] share one table — backends
        /// must only call these helpers (no hand-rolled `match` in
        /// `gguf_host`). Adding a scheme is one macro row + the matching
        /// kernel branch. Guarded by `gpu_dequant_scheme_id_is_stable`.
        pub const fn gpu_dequant_scheme_id(self) -> Option<u32> {
            match self {
                $(Self::$variant => Some($id),)+
                _ => None,
            }
        }

        /// Inverse of [`Self::gpu_dequant_scheme_id`].
        pub const fn from_gpu_dequant_scheme_id(id: u32) -> Option<Self> {
            match id {
                $($id => Some(Self::$variant),)+
                _ => None,
            }
        }

        /// `(scheme, id)` pairs for tests / docs — keep in sync via the macro.
        #[cfg(test)]
        pub(crate) const GPU_DEQUANT_SCHEME_ID_PAIRS: &'static [(Self, u32)] = &[
            $((Self::$variant, $id),)+
        ];
    };
}

impl QuantScheme {
    /// Bits per element after packing (×10 for K-quants since they
    /// have fractional bit budgets — divide by 10 when comparing).
    pub const fn bits_per_element_x10(self) -> u32 {
        match self {
            Self::Int8Block { .. } | Self::Int8BlockAsym { .. } => 80,
            Self::Int4Block { .. } => 40,
            Self::Fp8E4m3 | Self::Fp8E5m2 => 80,
            // GGUF K-quants: header + per-element bits over a 256-element block.
            Self::GgufQ4K => 45,  // 144 bytes / 256 elems × 8 = 4.5 bpe
            Self::GgufQ5K => 55,  // 176 / 256 × 8 ≈ 5.5
            Self::GgufQ6K => 66,  // 210 / 256 × 8 ≈ 6.5625 → 66 (rounded)
            Self::GgufQ8K => 91,  // 292 / 256 × 8 ≈ 9.125 → 91
            Self::GgufQ2K => 26,  // 84 / 256 × 8 ≈ 2.625 → 26
            Self::GgufQ3K => 34,  // 110 / 256 × 8 ≈ 3.4375 → 34
            Self::GgufQ4_0 => 45, // 18 / 32 × 8 = 4.5 bpe
            Self::GgufQ4_1 => 50, // 20 / 32 × 8 = 5.0 bpe
            Self::GgufQ5_0 => 55, // 22 / 32 × 8 = 5.5 bpe
            Self::GgufQ5_1 => 60, // 24 / 32 × 8 = 6.0 bpe
            Self::GgufQ8_0 => 85, // 34 / 32 × 8 = 8.5 bpe
            Self::Nvfp4Block => 40,
            Self::MlxAffine { bits, .. } => (bits as u32) * 10,
            Self::MlxMxfp4 { .. } => 40,
            Self::MlxMxfp8 { .. } => 80,
            Self::MxFp4x2Block { .. } => 84, // 2× E2M1 nibbles + 2 group scales

            Self::GgufIQ4NL => 45,
            Self::GgufIQ4XS => 42, // 136/256 × 8 = 4.25
            Self::GgufIQ2XXS => 20,
            Self::GgufIQ2XS => 23,
            Self::GgufIQ2S => 25,
            Self::GgufIQ3XXS => 30,
            Self::GgufIQ3S => 34,
            Self::GgufIQ1S => 15,
            Self::GgufIQ1M => 17,
            Self::GgufTQ1_0 => 16, // 54/256 × 8 = 1.6875 → 16
            Self::GgufTQ2_0 => 20,
            Self::GgufMXFP4 => 42,
            Self::GgufNVFP4 => 45,
            Self::GgufQ1_0 => 11, // 18 bytes / 128 elems × 8 = 1.125 bpe
            Self::GgufQ2_0 => 21, // 34 bytes / 128 elems × 8 = 2.125 bpe
        }
    }

    /// Bits per element after packing (rounded down). Use
    /// `bits_per_element_x10` for the K-quant fractional values.
    pub const fn bits_per_element(self) -> u32 {
        self.bits_per_element_x10() / 10
    }

    // Table defined once via `define_gguf_gpu_dequant_ids!` above.
    define_gguf_gpu_dequant_ids! {
        (GgufQ4K, 0),
        (GgufQ5K, 1),
        (GgufQ6K, 2),
        (GgufQ8K, 3),
        (GgufQ2K, 4),
        (GgufQ3K, 5),
        (GgufIQ4NL, 6),
        (GgufIQ4XS, 7),
        (GgufTQ1_0, 8),
        (GgufTQ2_0, 9),
        (GgufMXFP4, 10),
        (GgufNVFP4, 11),
        (GgufIQ2XXS, 12),
        (GgufIQ2XS, 13),
        (GgufIQ2S, 14),
        (GgufIQ3XXS, 15),
        (GgufIQ3S, 16),
        (GgufIQ1S, 17),
        (GgufIQ1M, 18),
        (GgufQ4_0, 19),
        (GgufQ8_0, 20),
        (GgufQ4_1, 21),
        (GgufQ5_0, 22),
        (GgufQ5_1, 23),
        (GgufQ1_0, 24),
        (GgufQ2_0, 25),
    }

    /// True if this scheme requires a per-block scale tensor on the side.
    pub const fn has_scale(self) -> bool {
        matches!(
            self,
            Self::Int8Block { .. }
                | Self::Int8BlockAsym { .. }
                | Self::Int4Block { .. }
                | Self::Nvfp4Block
                | Self::MlxAffine { .. }
                | Self::MlxMxfp4 { .. }
                | Self::MlxMxfp8 { .. }
                | Self::MxFp4x2Block { .. }
        )
    }

    /// True for NVFP4 / MLX mxfp block scales stored as FP8 / E8M0 bytes (not f32).
    pub const fn scale_is_fp8(self) -> bool {
        matches!(
            self,
            Self::Nvfp4Block | Self::MlxMxfp4 { .. } | Self::MlxMxfp8 { .. }
        )
    }

    /// Fixed NVFP4 group size along K (0 for other schemes).
    pub const fn nvfp4_group_size(self) -> u32 {
        match self {
            Self::Nvfp4Block => crate::nvfp4::NVFP4_GROUP_SIZE as u32,
            _ => 0,
        }
    }

    /// For [`Self::MxFp4x2Block`]: `(per-element format, residual levels, group
    /// size)`; `None` for every other scheme. Ties the scheme to the shared
    /// residual codec in [`crate::residual`].
    pub const fn mxfp4x2_config(self) -> Option<(ScaledFormat, u32, u32)> {
        match self {
            Self::MxFp4x2Block { group_size } => Some((ScaledFormat::F4E2M1, 2, group_size)),
            _ => None,
        }
    }

    /// True if this scheme requires a per-block zero-point / bias tensor.
    pub const fn has_zero_point(self) -> bool {
        matches!(self, Self::Int8BlockAsym { .. } | Self::MlxAffine { .. })
    }

    /// True for MLX-native affine / mxfp schemes (not GGUF, not legacy int blocks).
    pub const fn is_mlx(self) -> bool {
        matches!(
            self,
            Self::MlxAffine { .. } | Self::MlxMxfp4 { .. } | Self::MlxMxfp8 { .. }
        )
    }

    /// MLX group size along K (0 when not an MLX scheme).
    pub const fn mlx_group_size(self) -> u32 {
        match self {
            Self::MlxAffine { group_size, .. }
            | Self::MlxMxfp4 { group_size }
            | Self::MlxMxfp8 { group_size } => group_size,
            _ => 0,
        }
    }

    /// GPU launch args for MLX dequant-matmul kernels: `(kind, bits, group_size)`.
    ///
    /// `kind`: 0 = [`Self::MlxAffine`], 1 = [`Self::MlxMxfp4`], 2 = [`Self::MlxMxfp8`].
    /// `bits` is 0 for mxfp schemes. Returns `None` for non-MLX schemes.
    pub const fn mlx_gpu_launch(self) -> Option<(u32, u32, u32)> {
        match self {
            Self::MlxAffine { bits, group_size } => Some((0, bits as u32, group_size)),
            Self::MlxMxfp4 { group_size } => Some((1, 0, group_size)),
            Self::MlxMxfp8 { group_size } => Some((2, 0, group_size)),
            _ => None,
        }
    }

    /// GGUF K-quant block size (256 elements) — meaningless for the
    /// non-GGUF schemes (returns 0).
    pub const fn gguf_block_size(self) -> u32 {
        match self {
            Self::GgufQ4K
            | Self::GgufQ5K
            | Self::GgufQ6K
            | Self::GgufQ8K
            | Self::GgufQ2K
            | Self::GgufQ3K
            | Self::GgufIQ4XS
            | Self::GgufIQ2XXS
            | Self::GgufIQ2XS
            | Self::GgufIQ2S
            | Self::GgufIQ3XXS
            | Self::GgufIQ3S
            | Self::GgufIQ1S
            | Self::GgufIQ1M
            | Self::GgufTQ1_0
            | Self::GgufTQ2_0 => 256,
            Self::GgufQ4_0
            | Self::GgufQ4_1
            | Self::GgufQ5_0
            | Self::GgufQ5_1
            | Self::GgufQ8_0
            | Self::GgufIQ4NL
            | Self::GgufMXFP4 => 32,
            Self::GgufNVFP4 => 16,
            Self::GgufQ1_0 => 128,
            Self::GgufQ2_0 => 128,
            _ => 0,
        }
    }

    /// Bytes per GGUF super-block. 0 for non-GGUF schemes.
    pub const fn gguf_block_bytes(self) -> u32 {
        match self {
            Self::GgufQ4K => 144, // f16 d + f16 dmin + 12 packed scales + 128 nibbles
            Self::GgufQ5K => 176, // + 32-byte high-bit plane
            Self::GgufQ6K => 210, // 128 ql + 64 qh + 16 i8 scales + f16 d
            Self::GgufQ8K => 292, // f32 d + 256 i8 + 16 i16 bsums = 4 + 256 + 32
            Self::GgufQ2K => 84,  // f16 d + f16 dmin + 16 scales + 64 qs
            Self::GgufQ3K => 110, // f16 d + 12 scales + 32 hmask + 64 qs
            Self::GgufQ4_0 => 18, // f16 d + 16 packed nibbles
            Self::GgufQ4_1 => 20, // f16 d + f16 min + 16 packed nibbles
            Self::GgufQ5_0 => 22, // f16 d + 32-bit qh + 16 packed nibbles
            Self::GgufQ5_1 => 24, // f16 d + f16 min + 32-bit qh + 16 packed nibbles
            Self::GgufQ8_0 => 34, // f16 d + 32 i8 quants
            Self::GgufIQ4NL => 18,
            Self::GgufIQ4XS => 136,
            Self::GgufIQ2XXS => 66,
            Self::GgufIQ2XS => 74,
            Self::GgufIQ2S => 82,
            Self::GgufIQ3XXS => 98,
            Self::GgufIQ3S => 110,
            Self::GgufIQ1S => 50,
            Self::GgufIQ1M => 56,
            Self::GgufTQ1_0 => 54,
            Self::GgufTQ2_0 => 66,
            Self::GgufMXFP4 => 17,
            Self::GgufNVFP4 => 9,
            Self::GgufQ1_0 => 18, // f16 scale + 128 sign bits
            Self::GgufQ2_0 => 34, // Metal fused Q2_0 block (128 elems)
            _ => 0,
        }
    }

    /// True for any GGUF-format block scheme. GGUF schemes carry
    /// their scales / mins / sub-block metadata *inside* the packed
    /// weight bytes — they don't need separate `scale` / `zp`
    /// tensors fed alongside as the legacy `Int8Block` paths do.
    pub const fn is_gguf(self) -> bool {
        matches!(
            self,
            Self::GgufQ4K
                | Self::GgufQ5K
                | Self::GgufQ6K
                | Self::GgufQ8K
                | Self::GgufQ2K
                | Self::GgufQ3K
                | Self::GgufQ4_0
                | Self::GgufQ4_1
                | Self::GgufQ5_0
                | Self::GgufQ5_1
                | Self::GgufQ8_0
                | Self::GgufIQ4NL
                | Self::GgufIQ4XS
                | Self::GgufIQ2XXS
                | Self::GgufIQ2XS
                | Self::GgufIQ2S
                | Self::GgufIQ3XXS
                | Self::GgufIQ3S
                | Self::GgufIQ1S
                | Self::GgufIQ1M
                | Self::GgufTQ1_0
                | Self::GgufTQ2_0
                | Self::GgufMXFP4
                | Self::GgufNVFP4
                | Self::GgufQ1_0
                | Self::GgufQ2_0
        )
    }
}

impl std::fmt::Display for QuantScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int8Block { block_size } => write!(f, "int8/{block_size}"),
            Self::Int8BlockAsym { block_size } => write!(f, "int8a/{block_size}"),
            Self::Int4Block { block_size } => write!(f, "int4/{block_size}"),
            Self::Fp8E4m3 => write!(f, "fp8e4m3"),
            Self::Fp8E5m2 => write!(f, "fp8e5m2"),
            Self::GgufQ4K => write!(f, "gguf_q4k"),
            Self::GgufQ5K => write!(f, "gguf_q5k"),
            Self::GgufQ6K => write!(f, "gguf_q6k"),
            Self::GgufQ8K => write!(f, "gguf_q8k"),
            Self::GgufQ2K => write!(f, "gguf_q2k"),
            Self::GgufQ3K => write!(f, "gguf_q3k"),
            Self::GgufQ4_0 => write!(f, "gguf_q4_0"),
            Self::GgufQ4_1 => write!(f, "gguf_q4_1"),
            Self::GgufQ5_0 => write!(f, "gguf_q5_0"),
            Self::GgufQ5_1 => write!(f, "gguf_q5_1"),
            Self::GgufQ8_0 => write!(f, "gguf_q8_0"),
            Self::Nvfp4Block => write!(f, "nvfp4/16"),
            Self::MlxAffine { bits, group_size } => {
                write!(f, "mlx_affine/{bits}/{group_size}")
            }
            Self::MlxMxfp4 { group_size } => write!(f, "mlx_mxfp4/{group_size}"),
            Self::MxFp4x2Block { group_size } => write!(f, "mxfp4x2/{group_size}"),
            Self::MlxMxfp8 { group_size } => write!(f, "mlx_mxfp8/{group_size}"),
            Self::GgufIQ4NL => write!(f, "gguf_iq4_nl"),
            Self::GgufIQ4XS => write!(f, "gguf_iq4_xs"),
            Self::GgufIQ2XXS => write!(f, "gguf_iq2_xxs"),
            Self::GgufIQ2XS => write!(f, "gguf_iq2_xs"),
            Self::GgufIQ2S => write!(f, "gguf_iq2_s"),
            Self::GgufIQ3XXS => write!(f, "gguf_iq3_xxs"),
            Self::GgufIQ3S => write!(f, "gguf_iq3_s"),
            Self::GgufIQ1S => write!(f, "gguf_iq1_s"),
            Self::GgufIQ1M => write!(f, "gguf_iq1_m"),
            Self::GgufTQ1_0 => write!(f, "gguf_tq1_0"),
            Self::GgufTQ2_0 => write!(f, "gguf_tq2_0"),
            Self::GgufMXFP4 => write!(f, "gguf_mxfp4"),
            Self::GgufNVFP4 => write!(f, "gguf_nvfp4"),
            Self::GgufQ1_0 => write!(f, "gguf_q1_0"),
            Self::GgufQ2_0 => write!(f, "gguf_q2_0"),
        }
    }
}

/// Element format for a **native low-precision tensor-core GEMM** operand
/// (see [`crate::op::Op::ScaledMatMul`]).
///
/// Distinct from [`QuantScheme`]: a `QuantScheme` is block-scaled *storage*
/// that the CPU/GPU decodes to f32 *before* a normal sgemm. A `ScaledFormat`
/// is the raw element encoding hardware tensor cores consume **directly**, with
/// f32 accumulation — the whole point of FP8/FP6/FP4 on Hopper / Ada /
/// Blackwell / CDNA3 / CDNA4. The per-block/per-tensor scale layout is
/// orthogonal and lives in [`ScaleLayout`].
///
/// Operands flow through the graph as `DType::U8` byte buffers (one code per
/// byte on the CPU oracle; packed on GPU); the format is carried on the op,
/// not the dtype, so no `DType` variant is needed.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaledFormat {
    /// OCP FP8 E4M3 — bias 7, no infinities, only `S.1111.111` is NaN.
    /// Max ±448. Hopper / Ada / Blackwell; weights + forward activations.
    F8E4M3,
    /// OCP FP8 E5M2 — bias 15, IEEE-like inf/NaN at max exponent.
    /// Max ±57344. Wider range — the gradient format.
    F8E5M2,
    /// AMD "FNUZ" FP8 E4M3 — bias 8, single NaN at `0x80`, no inf, no −0.
    /// Max ±240. CDNA3 / MI300.
    F8E4M3Fnuz,
    /// AMD "FNUZ" FP8 E5M2 — bias 16, single NaN at `0x80`, no inf, no −0.
    /// Max ±57344. CDNA3 / MI300.
    F8E5M2Fnuz,
    /// MX FP6 E2M3 — bias 1, all 64 codes finite. Max ±7.5. Blackwell / CDNA4.
    F6E2M3,
    /// MX FP6 E3M2 — bias 3, all 64 codes finite. Max ±28. Blackwell / CDNA4.
    F6E3M2,
    /// FP4 E2M1 — all 16 codes finite. Max ±6. Blackwell / CDNA4.
    /// Shared by NVFP4 and MXFP4 — the [`ScaleLayout`] tells them apart.
    F4E2M1,
    /// Arbitrary sub-byte minifloat parameterized by exponent / mantissa width
    /// — the `fNeXmY` family (`N = 1 + exp_bits + mant_bits` total bits).
    ///
    /// All-finite (no infinities, no NaN, no AMD FNUZ), IEEE-style gradual
    /// underflow (subnormals when `exp == 0`), sign in bit `exp_bits +
    /// mant_bits`, one code per byte. The whole code must fit in a byte, so
    /// `1 + exp_bits + mant_bits <= 8` and `exp_bits >= 1`.
    ///
    /// This is a *research / experimental* encoding: it has no hardware
    /// tensor-core GEMM, so it always runs the decode-and-accumulate path (CPU
    /// oracle + generic GPU kernel). Build one with [`ScaledFormat::custom`]
    /// (IEEE bias `2^(exp_bits-1) - 1`) or [`ScaledFormat::custom_with_bias`],
    /// or parse a name via `"f4e3m0".parse()`.
    ///
    /// Example — `f4e3m0` (`exp_bits: 3, mant_bits: 0, bias: 3`): the 16 codes
    /// are `±0` and `±{0.25, 0.5, 1, 2, 4, 8, 16}` — a signed power-of-two
    /// (near-logarithmic) 4-bit format.
    Custom {
        exp_bits: u8,
        mant_bits: u8,
        bias: i8,
    },
}

impl ScaledFormat {
    /// Total bit width of one element code (4, 6, or 8).
    pub const fn bit_width(self) -> u32 {
        match self {
            Self::F8E4M3 | Self::F8E5M2 | Self::F8E4M3Fnuz | Self::F8E5M2Fnuz => 8,
            Self::F6E2M3 | Self::F6E3M2 => 6,
            Self::F4E2M1 => 4,
            Self::Custom {
                exp_bits,
                mant_bits,
                ..
            } => 1 + exp_bits as u32 + mant_bits as u32,
        }
    }

    /// `(exponent_bits, mantissa_bits, exponent_bias)`.
    pub const fn fields(self) -> (u32, u32, i32) {
        match self {
            Self::F8E4M3 => (4, 3, 7),
            Self::F8E5M2 => (5, 2, 15),
            Self::F8E4M3Fnuz => (4, 3, 8),
            Self::F8E5M2Fnuz => (5, 2, 16),
            Self::F6E2M3 => (2, 3, 1),
            Self::F6E3M2 => (3, 2, 3),
            Self::F4E2M1 => (2, 1, 1),
            Self::Custom {
                exp_bits,
                mant_bits,
                bias,
            } => (exp_bits as u32, mant_bits as u32, bias as i32),
        }
    }

    /// Build an arbitrary all-finite minifloat with `exp_bits` exponent and
    /// `mant_bits` mantissa bits, using the IEEE-style bias `2^(exp_bits-1) - 1`
    /// — the convention every named rlx format follows (E4M3 → 7, E5M2 → 15,
    /// E2M1 → 1, …). The whole code must fit in a byte. `const` — usable in
    /// `const` contexts and pattern-free construction.
    ///
    /// Panics if `exp_bits == 0` or `1 + exp_bits + mant_bits > 8`.
    ///
    /// ```
    /// use rlx_ir::ScaledFormat;
    /// const F4E3M0: ScaledFormat = ScaledFormat::custom(3, 0);
    /// assert_eq!(F4E3M0.to_string(), "f4e3m0");
    /// assert_eq!(F4E3M0.exp_bits(), 3);
    /// assert_eq!(F4E3M0.mant_bits(), 0);
    /// // Nearest representable value ("quantize" a single f32):
    /// assert_eq!(F4E3M0.quantize(1.4), 1.0);
    /// assert_eq!(F4E3M0.max_finite(), 16.0);
    /// ```
    pub const fn custom(exp_bits: u8, mant_bits: u8) -> Self {
        assert!(exp_bits >= 1, "minifloat needs at least one exponent bit");
        assert!(
            1 + exp_bits + mant_bits <= 8,
            "minifloat code must fit in a byte (1 + exp + mant <= 8)"
        );
        let bias = (1i32 << (exp_bits - 1)) - 1;
        Self::Custom {
            exp_bits,
            mant_bits,
            bias: bias as i8,
        }
    }

    /// Like [`custom`](Self::custom) but with an explicit exponent bias
    /// (for formats that don't use the IEEE `2^(e-1)-1` convention). `const`.
    pub const fn custom_with_bias(exp_bits: u8, mant_bits: u8, bias: i8) -> Self {
        assert!(exp_bits >= 1, "minifloat needs at least one exponent bit");
        assert!(
            1 + exp_bits + mant_bits <= 8,
            "minifloat code must fit in a byte (1 + exp + mant <= 8)"
        );
        Self::Custom {
            exp_bits,
            mant_bits,
            bias,
        }
    }

    /// The seven named hardware element formats, for enumeration / format sweeps.
    pub const NAMED: [ScaledFormat; 7] = [
        Self::F8E4M3,
        Self::F8E5M2,
        Self::F8E4M3Fnuz,
        Self::F8E5M2Fnuz,
        Self::F6E2M3,
        Self::F6E3M2,
        Self::F4E2M1,
    ];

    /// Exponent-field bit count (`X` in `fNeXmY`).
    pub const fn exp_bits(self) -> u32 {
        self.fields().0
    }
    /// Mantissa-field bit count (`Y` in `fNeXmY`; `0` for a pure power-of-two
    /// format such as `f4e3m0`).
    pub const fn mant_bits(self) -> u32 {
        self.fields().1
    }
    /// Exponent bias.
    pub const fn bias(self) -> i32 {
        self.fields().2
    }
    /// True for a parameterized [`Custom`](Self::Custom) format (no hardware
    /// tensor core — always runs the decode path).
    pub const fn is_custom(self) -> bool {
        matches!(self, Self::Custom { .. })
    }
    /// True for one of the seven [`NAMED`](Self::NAMED) hardware formats.
    pub const fn is_named(self) -> bool {
        !self.is_custom()
    }

    /// AMD OCP-FNUZ encoding (single NaN at `0x80`, no inf, no −0).
    pub const fn is_fnuz(self) -> bool {
        matches!(self, Self::F8E4M3Fnuz | Self::F8E5M2Fnuz)
    }

    /// True for IEEE-like formats carrying inf / NaN at the max exponent
    /// (OCP E5M2 only — E4M3 is finite-with-one-NaN, FP6/FP4 are all-finite).
    pub const fn has_inf(self) -> bool {
        matches!(self, Self::F8E5M2)
    }

    /// Largest finite magnitude representable (used as the amax divisor when
    /// computing a per-tensor / per-block scale).
    pub fn max_finite(self) -> f32 {
        crate::lowp_codec::max_finite(self)
    }

    /// Decode one packed code to f32 (bit-exact; `±inf` / `NaN` for the formats
    /// that encode them). Method form of [`crate::lowp_codec::decode`].
    pub fn decode(self, code: u8) -> f32 {
        crate::lowp_codec::decode(self, code)
    }

    /// Encode an f32 to its nearest representable code (round-half-to-even,
    /// saturating overflow / `±inf` to `±max_finite`, `NaN → 0`). Method form of
    /// [`crate::lowp_codec::encode`].
    pub fn encode(self, x: f32) -> u8 {
        crate::lowp_codec::encode(self, x)
    }

    /// Round-trip `x` through the format — the nearest value it can represent.
    /// Handy for eyeballing a format's resolution without building a graph
    /// (e.g. `ScaledFormat::custom(3, 0).quantize(1.4) == 1.0`).
    pub fn quantize(self, x: f32) -> f32 {
        self.decode(self.encode(x))
    }

    /// Every finite value the format represents, ascending and deduplicated —
    /// the format's grid. Useful for inspecting a research minifloat, e.g.
    /// `ScaledFormat::custom(3, 0).representable_values()` yields
    /// `[-16, -8, -4, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4, 8, 16]`.
    pub fn representable_values(self) -> Vec<f32> {
        let n = 1u16 << self.bit_width();
        let mut v: Vec<f32> = (0..n)
            .map(|c| self.decode(c as u8))
            .filter(|x| x.is_finite())
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.dedup();
        v
    }

    /// The GPU dispatch word passed to `scaled_lowp_general.cu` (`rlx_decode_lowp`).
    ///
    /// For the seven named formats this is a small stable id matching the
    /// kernel's `switch`: e4m3=0, e5m2=1, e4m3fnuz=2, e5m2fnuz=3, e2m3=4,
    /// e3m2=5, e2m1=6.
    ///
    /// For a [`Custom`](Self::Custom) format there is no fixed id, so this
    /// returns a **packed field descriptor** with the top bit set as a
    /// sentinel: `0x8000_0000 | exp_bits | mant_bits<<4 | (bias as u8)<<8`.
    /// The kernel detects the sentinel bit and decodes generically from the
    /// unpacked `(exp_bits, mant_bits, bias)` — so a new format needs no kernel
    /// edit. The named ids stay in `0..=6` (top bit clear), so the existing
    /// hardware `switch` path is unchanged.
    pub const fn kernel_id(self) -> u32 {
        match self {
            Self::F8E4M3 => 0,
            Self::F8E5M2 => 1,
            Self::F8E4M3Fnuz => 2,
            Self::F8E5M2Fnuz => 3,
            Self::F6E2M3 => 4,
            Self::F6E3M2 => 5,
            Self::F4E2M1 => 6,
            Self::Custom {
                exp_bits,
                mant_bits,
                bias,
            } => {
                0x8000_0000
                    | (exp_bits as u32 & 0xF)
                    | ((mant_bits as u32 & 0xF) << 4)
                    | (((bias as u8) as u32) << 8)
            }
        }
    }

    /// True for the OCP FP8 variants that the native cublasLt / hipBLASLt FP8
    /// GEMM accepts (per-tensor only). Other formats use the decode fallback.
    pub const fn is_native_fp8(self) -> bool {
        matches!(self, Self::F8E4M3 | Self::F8E5M2)
    }
}

impl std::fmt::Display for ScaledFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::F8E4M3 => "f8e4m3",
            Self::F8E5M2 => "f8e5m2",
            Self::F8E4M3Fnuz => "f8e4m3fnuz",
            Self::F8E5M2Fnuz => "f8e5m2fnuz",
            Self::F6E2M3 => "f6e2m3",
            Self::F6E3M2 => "f6e3m2",
            Self::F4E2M1 => "f4e2m1",
            // `fNeXmY`, N = 1 + exp + mant. The bias is implied by the width
            // (IEEE convention) and not rendered; round-trips through `FromStr`.
            Self::Custom {
                exp_bits,
                mant_bits,
                ..
            } => {
                return write!(
                    f,
                    "f{}e{}m{}",
                    1 + exp_bits + mant_bits,
                    exp_bits,
                    mant_bits
                );
            }
        };
        f.write_str(s)
    }
}

impl std::str::FromStr for ScaledFormat {
    type Err = String;

    /// Parse a format name. The seven named formats (`"f8e4m3"`, `"f4e2m1"`,
    /// the `…fnuz` variants, …) keep their exact hardware semantics (FNUZ,
    /// inf/NaN); any other `fNeXmY` string becomes an all-finite
    /// [`Custom`](Self::Custom) with the IEEE bias. `N` must equal `1 + X + Y`
    /// and the code must fit in a byte (`N <= 8`, `X >= 1`).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "f8e4m3" => return Ok(Self::F8E4M3),
            "f8e5m2" => return Ok(Self::F8E5M2),
            "f8e4m3fnuz" => return Ok(Self::F8E4M3Fnuz),
            "f8e5m2fnuz" => return Ok(Self::F8E5M2Fnuz),
            "f6e2m3" => return Ok(Self::F6E2M3),
            "f6e3m2" => return Ok(Self::F6E3M2),
            "f4e2m1" => return Ok(Self::F4E2M1),
            _ => {}
        }
        let err = || format!("invalid minifloat format {s:?} (expected e.g. \"f4e3m0\")");
        let rest = s.strip_prefix('f').ok_or_else(err)?;
        let (total, rest) = take_leading_u32(rest).ok_or_else(err)?;
        let rest = rest.strip_prefix('e').ok_or_else(err)?;
        let (exp, rest) = take_leading_u32(rest).ok_or_else(err)?;
        let rest = rest.strip_prefix('m').ok_or_else(err)?;
        let (mant, rest) = take_leading_u32(rest).ok_or_else(err)?;
        if !rest.is_empty() || exp == 0 || total != 1 + exp + mant || 1 + exp + mant > 8 {
            return Err(err());
        }
        Ok(Self::custom(exp as u8, mant as u8))
    }
}

/// Split the leading run of ASCII digits off `s`, returning `(value, tail)`.
/// `None` if `s` doesn't start with a digit.
fn take_leading_u32(s: &str) -> Option<(u32, &str)> {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    Some((s[..end].parse().ok()?, &s[end..]))
}

/// How scale factors are laid out for an [`crate::op::Op::ScaledMatMul`]
/// operand. The reconstructed value of element `i` is
/// `decode(code[i]) * scale(block_of(i))`.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleLayout {
    /// One f32 amax scale for the whole operand (classic per-tensor FP8 GEMM).
    PerTensor,
    /// OCP microscaling: one power-of-two **E8M0** scale per `block`
    /// consecutive elements along K (MXFP8 / MXFP6 / MXFP4). Scale tensor
    /// is `DType::U8` (E8M0 bytes).
    BlockMxE8M0 { block: u32 },
    /// NVFP4: one FP8 **E4M3** scale per `group` (16) elements along K, plus
    /// an optional per-tensor f32 global. Scale tensor is `DType::U8`.
    Nvfp4 { group: u32 },
}

impl ScaleLayout {
    /// OCP microscaling default (32-element blocks).
    pub const fn mx() -> Self {
        Self::BlockMxE8M0 { block: 32 }
    }
    /// NVFP4 default (16-element groups).
    pub fn nvfp4() -> Self {
        Self::Nvfp4 {
            group: crate::nvfp4::NVFP4_GROUP_SIZE as u32,
        }
    }
    /// Element dtype of the scale tensor for this layout.
    pub const fn scale_dtype(self) -> crate::DType {
        match self {
            Self::PerTensor => crate::DType::F32,
            Self::BlockMxE8M0 { .. } | Self::Nvfp4 { .. } => crate::DType::U8,
        }
    }
    /// Number of consecutive elements sharing one scale (1 for per-tensor).
    pub const fn block(self) -> u32 {
        match self {
            Self::PerTensor => 1,
            Self::BlockMxE8M0 { block } => block,
            Self::Nvfp4 { group } => group,
        }
    }

    /// `(scale_mode, block)` for GPU kernels (`scaled_lowp_general.cu`):
    /// per-tensor=0, block-E8M0=1, NVFP4-E4M3=2.
    pub const fn mode_block(self) -> (u32, u32) {
        match self {
            Self::PerTensor => (0, 1),
            Self::BlockMxE8M0 { block } => (1, block),
            Self::Nvfp4 { group } => (2, group),
        }
    }
}

impl std::fmt::Display for ScaleLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PerTensor => write!(f, "per_tensor"),
            Self::BlockMxE8M0 { block } => write!(f, "mx_e8m0/{block}"),
            Self::Nvfp4 { group } => write!(f, "nvfp4/{group}"),
        }
    }
}

impl std::str::FromStr for ScaleLayout {
    type Err = String;

    /// Parse a scale-layout name: `"per_tensor"`, `"mx"` (32-element E8M0
    /// microscaling), `"nvfp4"` (16-element E4M3), or an explicit block size
    /// `"mx/<block>"` / `"nvfp4/<group>"`. Mirrors [`Display`](Self) round-trip.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "per_tensor" | "pertensor" => return Ok(Self::PerTensor),
            "mx" | "mxfp8" | "block" => return Ok(Self::mx()),
            "nvfp4" => return Ok(Self::nvfp4()),
            _ => {}
        }
        if let Some(rest) = s.strip_prefix("mx/").or_else(|| s.strip_prefix("mx_e8m0/")) {
            if let Ok(block) = rest.parse::<u32>() {
                return Ok(Self::BlockMxE8M0 { block });
            }
        }
        if let Some(rest) = s.strip_prefix("nvfp4/") {
            if let Ok(group) = rest.parse::<u32>() {
                return Ok(Self::Nvfp4 { group });
            }
        }
        Err(format!(
            "unknown scale layout {s:?} (expected \"per_tensor\", \"mx\", \"nvfp4\", or \"mx/<block>\")"
        ))
    }
}

/// Per-graph map of quantized tensors. Lookup is O(1).
#[derive(Debug, Clone, Default)]
pub struct QuantMap {
    map: HashMap<NodeId, QuantScheme>,
}

impl QuantMap {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn get(&self, id: NodeId) -> Option<QuantScheme> {
        self.map.get(&id).copied()
    }
    pub fn insert(&mut self, id: NodeId, scheme: QuantScheme) -> Option<QuantScheme> {
        self.map.insert(id, scheme)
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&NodeId, &QuantScheme)> {
        self.map.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheme_traits() {
        assert_eq!(
            QuantScheme::Int4Block { block_size: 32 }.bits_per_element(),
            4
        );
        assert!(QuantScheme::Int8BlockAsym { block_size: 64 }.has_zero_point());
        assert!(!QuantScheme::Fp8E4m3.has_scale());
    }

    /// Pins the shared GGUF dequant-kernel ids so they can NEVER drift out
    /// from under the compiled MSL/CU/WGSL/GLSL `switch (scheme_id)` in any
    /// backend. Every GPU backend delegates to `gpu_dequant_scheme_id`; if
    /// someone reorders a variant here this test fails before a mis-decode
    /// ships. Non-GGUF schemes must report `None`.
    #[test]
    fn gpu_dequant_scheme_id_is_stable() {
        use QuantScheme::*;
        assert_eq!(QuantScheme::GPU_DEQUANT_SCHEME_ID_PAIRS.len(), 26);
        for &(scheme, id) in QuantScheme::GPU_DEQUANT_SCHEME_ID_PAIRS {
            assert_eq!(scheme.gpu_dequant_scheme_id(), Some(id), "{scheme:?}");
            assert_eq!(
                QuantScheme::from_gpu_dequant_scheme_id(id),
                Some(scheme),
                "id {id}"
            );
        }
        for scheme in [
            Int8Block { block_size: 32 },
            Int8BlockAsym { block_size: 32 },
            Int4Block { block_size: 32 },
            Fp8E4m3,
            Fp8E5m2,
            Nvfp4Block,
            MlxAffine {
                bits: 4,
                group_size: 64,
            },
            MlxMxfp4 { group_size: 32 },
            MlxMxfp8 { group_size: 32 },
        ] {
            assert_eq!(scheme.gpu_dequant_scheme_id(), None, "{scheme:?}");
        }
        assert_eq!(QuantScheme::from_gpu_dequant_scheme_id(999), None);
    }

    #[test]
    fn quant_map_lookup() {
        let mut q = QuantMap::new();
        let id = NodeId(7);
        q.insert(id, QuantScheme::Int8Block { block_size: 32 });
        assert_eq!(q.get(id), Some(QuantScheme::Int8Block { block_size: 32 }));
        assert_eq!(q.get(NodeId(99)), None);
    }

    #[test]
    fn custom_format_ieee_bias_and_width() {
        // IEEE bias 2^(e-1)-1, matching every named format.
        assert_eq!(ScaledFormat::custom(3, 0).fields(), (3, 0, 3)); // f4e3m0
        assert_eq!(ScaledFormat::custom(4, 3).fields(), (4, 3, 7)); // e4m3 bias
        assert_eq!(ScaledFormat::custom(5, 2).fields(), (5, 2, 15)); // e5m2 bias
        assert_eq!(ScaledFormat::custom(2, 1).fields(), (2, 1, 1)); // e2m1 bias
        assert_eq!(ScaledFormat::custom(3, 0).bit_width(), 4);
        assert_eq!(ScaledFormat::custom(3, 4).bit_width(), 8);
    }

    #[test]
    fn custom_format_parse_display_round_trip() {
        let f: ScaledFormat = "f4e3m0".parse().unwrap();
        assert_eq!(f, ScaledFormat::custom(3, 0));
        assert_eq!(f.to_string(), "f4e3m0");
        // A few more splits.
        assert_eq!(
            "f5e2m2".parse::<ScaledFormat>().unwrap().to_string(),
            "f5e2m2"
        );
        assert_eq!(
            "f8e3m4".parse::<ScaledFormat>().unwrap().to_string(),
            "f8e3m4"
        );
        // Named formats parse to their exact hardware variants, not Custom.
        assert_eq!(
            "f8e4m3".parse::<ScaledFormat>().unwrap(),
            ScaledFormat::F8E4M3
        );
        assert_eq!(
            "f4e2m1".parse::<ScaledFormat>().unwrap(),
            ScaledFormat::F4E2M1
        );
        assert_eq!(
            "f8e5m2fnuz".parse::<ScaledFormat>().unwrap(),
            ScaledFormat::F8E5M2Fnuz
        );
    }

    #[test]
    fn custom_format_parse_rejects_invalid() {
        assert!("f4e3m1".parse::<ScaledFormat>().is_err()); // 1+3+1 != 4
        assert!("f9e4m4".parse::<ScaledFormat>().is_err()); // 9 bits > byte
        assert!("f1e0m0".parse::<ScaledFormat>().is_err()); // 0 exponent bits
        assert!("e3m0".parse::<ScaledFormat>().is_err()); // missing 'f'
        assert!("f4e3m0x".parse::<ScaledFormat>().is_err()); // trailing junk
        assert!("garbage".parse::<ScaledFormat>().is_err());
    }

    #[test]
    fn custom_gpu_word_is_packed_descriptor() {
        // Named formats keep their small ids (top bit clear).
        assert_eq!(ScaledFormat::F8E4M3.kernel_id(), 0);
        assert_eq!(ScaledFormat::F4E2M1.kernel_id(), 6);
        assert!(ScaledFormat::F4E2M1.kernel_id() & 0x8000_0000 == 0);
        // Custom packs (exp, mant, bias) with the sentinel bit set.
        let w = ScaledFormat::custom(3, 0).kernel_id(); // e=3, m=0, bias=3
        assert_eq!(w & 0x8000_0000, 0x8000_0000);
        assert_eq!(w & 0xF, 3); // exp_bits
        assert_eq!((w >> 4) & 0xF, 0); // mant_bits
        assert_eq!((w >> 8) & 0xFF, 3); // bias
        // Negative bias round-trips through the u8 reinterpretation.
        let wn = ScaledFormat::custom_with_bias(3, 0, -2).kernel_id();
        assert_eq!(((wn >> 8) & 0xFF) as u8 as i8, -2);
    }

    #[test]
    fn format_is_const_and_introspectable() {
        // Usable in const context.
        const F: ScaledFormat = ScaledFormat::custom(3, 0);
        assert!(F.is_custom() && !F.is_named());
        assert_eq!((F.exp_bits(), F.mant_bits(), F.bias()), (3, 0, 3));
        assert!(ScaledFormat::F8E4M3.is_named() && !ScaledFormat::F8E4M3.is_custom());
        assert_eq!(ScaledFormat::NAMED.len(), 7);
        assert!(ScaledFormat::NAMED.iter().all(|f| f.is_named()));
    }

    #[test]
    fn format_codec_methods_and_quantize() {
        // Method forms match the free functions for every named format.
        for f in ScaledFormat::NAMED {
            for c in 0..(1u16 << f.bit_width()) {
                assert_eq!(
                    f.decode(c as u8).to_bits(),
                    crate::lowp_codec::decode(f, c as u8).to_bits()
                );
            }
        }
        let cf = ScaledFormat::custom(3, 0); // f4e3m0 grid: ±{0.25,..,16}, ±0
        assert_eq!(cf.quantize(1.4), 1.0);
        assert_eq!(cf.quantize(1.6), 2.0);
        assert_eq!(cf.quantize(-3.0), -2.0); // nearer 2 than 4 in magnitude
        assert_eq!(cf.decode(cf.encode(100.0)), 16.0); // saturates
        assert_eq!(cf.decode(cf.encode(f32::INFINITY)), 16.0);
    }

    #[test]
    fn representable_values_grid() {
        assert_eq!(
            ScaledFormat::custom(3, 0).representable_values(),
            vec![
                -16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0,
                16.0
            ]
        );
        // Named formats: the grid size never exceeds the finite code count.
        for f in ScaledFormat::NAMED {
            let g = f.representable_values();
            assert!(!g.is_empty() && g.windows(2).all(|w| w[0] < w[1]));
        }
    }

    #[test]
    fn enumerate_all_fnexmy() {
        // Every valid fNeXmY: exp >= 1, mant >= 0, 1 + exp + mant <= 8.
        let mut all = vec![];
        for exp in 1u8..=7 {
            for mant in 0u8..=(7 - exp) {
                all.push(ScaledFormat::custom(exp, mant));
            }
        }
        assert_eq!(all.len(), 28, "there are exactly 28 fNeXmY formats");
        eprintln!(
            "{:<8} {:>4} {:>3} {:>3} {:>4} {:>12} {:>12} {:>5}",
            "name", "bits", "e", "m", "bias", "max_finite", "min_pos", "vals"
        );
        for f in &all {
            let vals = f.representable_values();
            let min_pos = vals.iter().copied().find(|&x| x > 0.0).unwrap();
            assert!(f.bit_width() <= 8 && f.exp_bits() >= 1 && !vals.is_empty());
            eprintln!(
                "{:<8} {:>4} {:>3} {:>3} {:>4} {:>12.3e} {:>12.3e} {:>5}",
                f.to_string(),
                f.bit_width(),
                f.exp_bits(),
                f.mant_bits(),
                f.bias(),
                f.max_finite(),
                min_pos,
                vals.len()
            );
        }
    }

    #[test]
    fn scale_layout_parse_round_trip() {
        assert_eq!(
            "per_tensor".parse::<ScaleLayout>().unwrap(),
            ScaleLayout::PerTensor
        );
        assert_eq!("mx".parse::<ScaleLayout>().unwrap(), ScaleLayout::mx());
        assert_eq!(
            "nvfp4".parse::<ScaleLayout>().unwrap(),
            ScaleLayout::nvfp4()
        );
        assert_eq!(
            "mx/64".parse::<ScaleLayout>().unwrap(),
            ScaleLayout::BlockMxE8M0 { block: 64 }
        );
        assert!("bogus".parse::<ScaleLayout>().is_err());
        // Display round-trips through FromStr.
        let mx = ScaleLayout::mx();
        assert_eq!(mx.to_string().parse::<ScaleLayout>().unwrap(), mx);
    }
}
