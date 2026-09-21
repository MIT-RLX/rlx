// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The element dtypes a torch storage can carry, with the metadata the
//! loader needs: byte width, and how to decode raw little-endian bytes
//! into `f32` (every tensor is materialized as `f32` for the rest of RLX).

/// Storage element type, as named by torch's `*Storage` classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl DType {
    /// Byte width of one element.
    pub fn size(self) -> usize {
        match self {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 | DType::I16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }

    /// Map a torch storage class name (e.g. `FloatStorage`, `BFloat16Storage`)
    /// to its dtype. Accepts the bare class name or a fully qualified one.
    pub fn from_storage_name(name: &str) -> Option<Self> {
        let n = name.rsplit('.').next().unwrap_or(name);
        Some(match n {
            "FloatStorage" => DType::F32,
            "DoubleStorage" => DType::F64,
            "HalfStorage" => DType::F16,
            "BFloat16Storage" => DType::BF16,
            "LongStorage" => DType::I64,
            "IntStorage" => DType::I32,
            "ShortStorage" => DType::I16,
            "CharStorage" => DType::I8,
            "ByteStorage" => DType::U8,
            "BoolStorage" => DType::Bool,
            _ => return None,
        })
    }

    /// Decode a contiguous little-endian byte buffer of this dtype into
    /// `f32`. The buffer length must be a multiple of [`Self::size`].
    pub fn decode_f32(self, bytes: &[u8]) -> Vec<f32> {
        match self {
            DType::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            DType::F64 => bytes
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            DType::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            DType::BF16 => bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            DType::I64 => bytes
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect(),
            DType::I32 => bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect(),
            DType::I16 => bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
                .collect(),
            DType::I8 => bytes.iter().map(|&b| b as i8 as f32).collect(),
            DType::U8 => bytes.iter().map(|&b| b as f32).collect(),
            DType::Bool => bytes.iter().map(|&b| f32::from(b != 0)).collect(),
        }
    }
}
