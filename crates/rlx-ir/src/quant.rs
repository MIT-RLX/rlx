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
        }
    }

    /// Bits per element after packing (rounded down). Use
    /// `bits_per_element_x10` for the K-quant fractional values.
    pub const fn bits_per_element(self) -> u32 {
        self.bits_per_element_x10() / 10
    }

    /// True if this scheme requires a per-block scale tensor on the side.
    pub const fn has_scale(self) -> bool {
        matches!(
            self,
            Self::Int8Block { .. }
                | Self::Int8BlockAsym { .. }
                | Self::Int4Block { .. }
                | Self::Nvfp4Block
        )
    }

    /// True for NVFP4 block scales stored as FP8 E4M3 bytes (not f32).
    pub const fn scale_is_fp8(self) -> bool {
        matches!(self, Self::Nvfp4Block)
    }

    /// Fixed NVFP4 group size along K (0 for other schemes).
    pub const fn nvfp4_group_size(self) -> u32 {
        match self {
            Self::Nvfp4Block => crate::nvfp4::NVFP4_GROUP_SIZE as u32,
            _ => 0,
        }
    }

    /// True if this scheme requires a per-block zero-point.
    pub const fn has_zero_point(self) -> bool {
        matches!(self, Self::Int8BlockAsym { .. })
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
}

impl ScaledFormat {
    /// Total bit width of one element code (4, 6, or 8).
    pub const fn bit_width(self) -> u32 {
        match self {
            Self::F8E4M3 | Self::F8E5M2 | Self::F8E4M3Fnuz | Self::F8E5M2Fnuz => 8,
            Self::F6E2M3 | Self::F6E3M2 => 6,
            Self::F4E2M1 => 4,
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
        }
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

    /// Stable integer id passed to GPU kernels (matches `rlx_decode_lowp` in
    /// `scaled_lowp_general.cu`): e4m3=0, e5m2=1, e4m3fnuz=2, e5m2fnuz=3,
    /// e2m3=4, e3m2=5, e2m1=6.
    pub const fn kernel_id(self) -> u32 {
        match self {
            Self::F8E4M3 => 0,
            Self::F8E5M2 => 1,
            Self::F8E4M3Fnuz => 2,
            Self::F8E5M2Fnuz => 3,
            Self::F6E2M3 => 4,
            Self::F6E3M2 => 5,
            Self::F4E2M1 => 6,
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
        };
        f.write_str(s)
    }
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

    #[test]
    fn quant_map_lookup() {
        let mut q = QuantMap::new();
        let id = NodeId(7);
        q.insert(id, QuantScheme::Int8Block { block_size: 32 });
        assert_eq!(q.get(id), Some(QuantScheme::Int8Block { block_size: 32 }));
        assert_eq!(q.get(NodeId(99)), None);
    }
}
