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
//! Borrowed from MAX's `quantization.py` pattern: quantization scheme
//! lives as per-tensor metadata on the IR rather than spawning a
//! parallel "quantized graph" type. Ops can read the scheme and
//! dispatch to fused-dequant kernels (the eventual #5 win) when
//! present, or fall through to the standard f32/f16 path when not.
//!
//! The metadata is held *outside* the [`crate::Node`] type itself, in a
//! [`crate::Graph`]-level [`QuantMap`]. This keeps Node small (every node
//! pays for the rare quantization annotation otherwise) and makes
//! quant info easy to query / clear without rewriting nodes.

use crate::NodeId;
use std::collections::HashMap;

/// How a tensor is quantized. Mirrors the schemes RLX needs for LLM
/// inference on Apple Silicon: blockwise int8 (GPTQ-style),
/// blockwise int4 (Q4_K), and per-tensor fp8 (e4m3 / e5m2).
///
/// Each variant carries the parameters the dequantizer needs to read
/// at runtime — scale, zero-point, block size. Where these live in
/// the actual weight tensor is up to the loader (#56).
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
            Self::Int8Block { .. } | Self::Int8BlockAsym { .. } | Self::Int4Block { .. }
        )
    }

    /// True if this scheme requires a per-block zero-point.
    pub const fn has_zero_point(self) -> bool {
        matches!(self, Self::Int8BlockAsym { .. })
    }

    /// GGUF K-quant block size (256 elements) — meaningless for the
    /// non-GGUF schemes (returns 0).
    pub const fn gguf_block_size(self) -> u32 {
        match self {
            Self::GgufQ4K | Self::GgufQ5K | Self::GgufQ6K | Self::GgufQ8K => 256,
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
            Self::GgufQ4K | Self::GgufQ5K | Self::GgufQ6K | Self::GgufQ8K
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
