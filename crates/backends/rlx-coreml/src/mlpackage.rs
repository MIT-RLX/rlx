// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// `.mlpackage` bundle writer. A `.mlpackage` is a directory:
//
//   Model.mlpackage/
//     Manifest.json
//     Data/
//       com.apple.CoreML/
//         model.mlmodel          # the serialized CoreML protobuf Model
//         weights/
//           weight.bin           # large consts, in CoreML MILBlob format
//
// Small consts (< 10 elements) stay inline as MIL immediate values; larger
// weight tensors live in `weight.bin` and the proto references them by
// offset (see `BlobWriter` + `mil.rs`). This keeps the protobuf small —
// real LLM weights would otherwise blow past CoreML's parse limits.

use std::path::{Path, PathBuf};

use prost::Message;

use crate::Result;
use crate::proto;

/// CoreML MILBlob weight file (`weight.bin`).
///
/// Layout (all 64-byte aligned): `[storage_header(64)] [meta_0(64)]
/// [data_0] [meta_1(64)] [data_1] …`. The storage header is `{count: u32,
/// version: u32 = 2, reserved[7]: u64}`. Each metadata record is
/// `{sentinel: u32 = 0xDEADBEEF, dtype: u32, sizeInBytes: u64, offset: u64
/// (→ data), padding_bits: u64, reserved[4]: u64}`. A const references its
/// weight by the **metadata** offset (the value passed to
/// `Value.blobFileValue.offset`).
#[derive(Default)]
pub struct BlobWriter {
    buf: Vec<u8>,
    count: u32,
}

const BLOB_SENTINEL: u32 = 0xDEAD_BEEF;
const BLOB_DTYPE_FLOAT16: u32 = 1;
const BLOB_DTYPE_FLOAT32: u32 = 2;
const BLOB_DTYPE_UINT8: u32 = 3;
const BLOB_DTYPE_INT8: u32 = 4;
/// CoreML MILBlob sub-byte dtype code for 1-bit packed data (palettization
/// indices). Matches coremltools' BlobDataType ordering.
const BLOB_DTYPE_UINT1: u32 = 11;

impl BlobWriter {
    pub fn new() -> Self {
        // Reserve the 64-byte storage header; version = 2 at offset 4.
        let mut buf = vec![0u8; 64];
        buf[4..8].copy_from_slice(&2u32.to_le_bytes());
        BlobWriter { buf, count: 0 }
    }

    fn align64(&mut self) {
        let rem = self.buf.len() % 64;
        if rem != 0 {
            self.buf.resize(self.buf.len() + (64 - rem), 0);
        }
    }

    /// Append an f32 tensor; returns its **metadata** record offset (the
    /// value to store in `Value.blobFileValue.offset`).
    pub fn write_f32(&mut self, data: &[f32]) -> u64 {
        self.align64();
        let meta_off = self.buf.len() as u64;
        let data_off = meta_off + 64; // metadata is 64 bytes, stays aligned
        let size = (data.len() * 4) as u64;

        let mut meta = [0u8; 64];
        meta[0..4].copy_from_slice(&BLOB_SENTINEL.to_le_bytes());
        meta[4..8].copy_from_slice(&BLOB_DTYPE_FLOAT32.to_le_bytes());
        meta[8..16].copy_from_slice(&size.to_le_bytes());
        meta[16..24].copy_from_slice(&data_off.to_le_bytes());
        self.buf.extend_from_slice(&meta);

        self.buf.reserve(data.len() * 4);
        for &v in data {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
        self.count += 1;
        meta_off
    }

    /// Append an f16 tensor; returns its metadata record offset.
    pub fn write_f16(&mut self, data: &[half::f16]) -> u64 {
        self.align64();
        let meta_off = self.buf.len() as u64;
        let data_off = meta_off + 64;
        let size = (data.len() * 2) as u64;

        let mut meta = [0u8; 64];
        meta[0..4].copy_from_slice(&BLOB_SENTINEL.to_le_bytes());
        meta[4..8].copy_from_slice(&BLOB_DTYPE_FLOAT16.to_le_bytes());
        meta[8..16].copy_from_slice(&size.to_le_bytes());
        meta[16..24].copy_from_slice(&data_off.to_le_bytes());
        self.buf.extend_from_slice(&meta);

        self.buf.reserve(data.len() * 2);
        for &v in data {
            self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        self.count += 1;
        meta_off
    }

    /// Append a raw uint8 tensor (packed quant indices / int weights); returns
    /// its metadata record offset. `dtype` is the CoreML blob dtype code
    /// (3 = uint8, 4 = int8). Used by the non-unfolding Q1_0 paths
    /// (`constexpr_lut_to_dense` indices / `constexpr_affine_dequantize` data)
    /// so quant weights never expand to f32 in the blob.
    pub fn write_bytes_dtype(&mut self, data: &[u8], dtype: u32) -> u64 {
        self.align64();
        let meta_off = self.buf.len() as u64;
        let data_off = meta_off + 64;
        let size = data.len() as u64;
        let mut meta = [0u8; 64];
        meta[0..4].copy_from_slice(&BLOB_SENTINEL.to_le_bytes());
        meta[4..8].copy_from_slice(&dtype.to_le_bytes());
        meta[8..16].copy_from_slice(&size.to_le_bytes());
        meta[16..24].copy_from_slice(&data_off.to_le_bytes());
        self.buf.extend_from_slice(&meta);
        self.buf.extend_from_slice(data);
        self.count += 1;
        meta_off
    }
    /// uint8 blob (CoreML blob dtype 3).
    pub fn write_u8(&mut self, data: &[u8]) -> u64 {
        self.write_bytes_dtype(data, BLOB_DTYPE_UINT8)
    }
    /// Packed 1-bit blob (CoreML blob dtype 11 = UInt1): 8 indices per byte,
    /// LSB-first. For `constexpr_lut_to_dense` palettization indices. `data` is
    /// the already-packed bytes (`num_elems / 8`).
    pub fn write_u1_packed(&mut self, packed: &[u8]) -> u64 {
        self.write_bytes_dtype(packed, BLOB_DTYPE_UINT1)
    }
    /// int8 blob (CoreML blob dtype 4).
    pub fn write_i8(&mut self, data: &[i8]) -> u64 {
        let bytes: Vec<u8> = data.iter().map(|&v| v as u8).collect();
        self.write_bytes_dtype(&bytes, BLOB_DTYPE_INT8)
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Patch the header `count` and return the finished file bytes; empty
    /// when no weights were written (so no `weight.bin` is emitted).
    pub fn finish(mut self) -> Vec<u8> {
        if self.count == 0 {
            return Vec::new();
        }
        self.buf[0..4].copy_from_slice(&self.count.to_le_bytes());
        self.buf
    }
}

/// Deterministic placeholder identifier — CoreML only requires the
/// Manifest's `rootModelIdentifier` to match an entry key; the value is
/// otherwise opaque.
const ROOT_ID: &str = "0000000000000000000000000000000000000000";

/// Hard upper bound on the serialized `model.mlmodel` protobuf. A protobuf
/// message must fit in a signed 32-bit length, so CoreML's parser rejects
/// anything ≥ 2 GiB. Weights ≥ 10 elements go to `weight.bin` instead, so
/// hitting this means too many ops or large inline constants.
pub const PROTO_MAX_BYTES: usize = i32::MAX as usize;

/// Error unless the serialized `model` fits within `limit` bytes. Cheap —
/// uses `encoded_len()` without allocating the buffer.
pub fn check_model_size(model: &proto::Model, limit: usize) -> Result<()> {
    let bytes = model.encoded_len();
    if bytes > limit {
        return Err(crate::CoremlError::TooLarge {
            what: "CoreML model.mlmodel proto".to_string(),
            bytes,
            limit,
        });
    }
    Ok(())
}

/// Serialize `model` and write a complete `.mlpackage` directory tree at
/// `package_dir` (which is created, replacing any existing contents).
/// Returns the package path.
pub fn write_mlpackage(model: &proto::Model, blob: &[u8], package_dir: &Path) -> Result<PathBuf> {
    let proto_bytes = encode_model(model)?;
    write_mlpackage_bytes(&proto_bytes, blob, package_dir)
}

/// Serialize `model` to protobuf bytes, enforcing the size cap.
pub fn encode_model(model: &proto::Model) -> Result<Vec<u8>> {
    // Reject oversize models with a clear error rather than letting CoreML
    // fail to parse the protobuf downstream.
    check_model_size(model, PROTO_MAX_BYTES)?;
    let mut buf = Vec::with_capacity(model.encoded_len());
    model
        .encode(&mut buf)
        .map_err(|e| crate::CoremlError::Runtime(format!("proto encode: {e}")))?;
    Ok(buf)
}

/// Write a `.mlpackage` from a pre-encoded `model.mlmodel` + weight blob.
pub fn write_mlpackage_bytes(
    proto_bytes: &[u8],
    blob: &[u8],
    package_dir: &Path,
) -> Result<PathBuf> {
    if package_dir.exists() {
        std::fs::remove_dir_all(package_dir)?;
    }
    let data_dir = package_dir.join("Data").join("com.apple.CoreML");
    std::fs::create_dir_all(&data_dir)?;
    std::fs::write(data_dir.join("model.mlmodel"), proto_bytes)?;

    // weights/weight.bin — referenced by `@model_path/weights/weight.bin`.
    if !blob.is_empty() {
        let wdir = data_dir.join("weights");
        std::fs::create_dir_all(&wdir)?;
        std::fs::write(wdir.join("weight.bin"), blob)?;
    }

    std::fs::write(package_dir.join("Manifest.json"), manifest_json())?;
    Ok(package_dir.to_path_buf())
}

fn manifest_json() -> String {
    // Minimal Manifest the CoreML loader accepts: one item entry pointing
    // at the model file, referenced as the root model.
    format!(
        r#"{{
  "fileFormatVersion": "1.0.0",
  "itemInfoEntries": {{
    "{id}": {{
      "author": "com.apple.CoreML",
      "description": "CoreML Model Specification",
      "name": "model.mlmodel",
      "path": "com.apple.CoreML/model.mlmodel",
      "digestType": "sha256"
    }}
  }},
  "rootModelIdentifier": "{id}"
}}
"#,
        id = ROOT_ID
    )
}
