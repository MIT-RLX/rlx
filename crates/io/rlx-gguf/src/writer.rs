// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF v3 file writer. Serializes a sequence of metadata key/value
//! pairs and tensors. Layout matches the live spec (see lib.rs
//! parser): little-endian, alignment padding before the data
//! segment, tensor data appended in declaration order.
//!
//! Typical use is via [`super::quantize`] feeding raw byte payloads
//! into [`GgufWriter::add_tensor_bytes`], but the writer doesn't
//! quantize itself — keeping float→quant and bytes→file as separate
//! steps lets callers mix schemes per tensor without re-encoding.
//!
//! # Example
//!
//! ```ignore
//! use rlx_gguf::{GgmlType, GgufWriter, MetaValue, quantize};
//!
//! let w_floats: Vec<f32> = /* … */;
//!
//! let mut w = GgufWriter::new();
//! w.set_arch("llama");
//! w.set_meta("general.name", MetaValue::String("my-model".into()));
//! w.add_tensor_bytes(
//!     "token_embd.weight",
//!     vec![4096, 32000],
//!     GgmlType::Q4K,
//!     quantize(&w_floats, GgmlType::Q4K)?,
//! )?;
//! w.write_to_path("out.gguf")?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! The output round-trips through [`super::GgufFile::from_path`] +
//! [`super::GgufFile::dequant_f32`] with no special handling — it's
//! plain v3 GGUF and any GGUF-aware runtime can consume it.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::{DEFAULT_ALIGNMENT, GGUF_MAGIC, GgmlType, MetaValue, bytes_for_public};

/// In-memory description of a tensor to be written. Bytes must
/// already be in the storage layout for `dtype` (call
/// [`crate::quantize`] for floats, pass-through for already-quantized
/// tensors).
pub struct TensorPayload {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgmlType,
    pub bytes: Vec<u8>,
}

impl TensorPayload {
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Validate that `bytes.len()` matches the encoded size for
    /// `shape` × `dtype`.
    pub fn validate(&self) -> Result<()> {
        let n = self.n_elements();
        let expected = bytes_for_public(self.dtype, n).ok_or_else(|| {
            anyhow::anyhow!("tensor {}: bad shape for {:?}", self.name, self.dtype)
        })?;
        if self.bytes.len() != expected {
            bail!(
                "tensor {}: have {} bytes, need {} for {} elements at {:?}",
                self.name,
                self.bytes.len(),
                expected,
                n,
                self.dtype
            );
        }
        Ok(())
    }
}

/// GGUF v3 file writer. Build up metadata + tensors, then call
/// [`GgufWriter::write`] / [`GgufWriter::write_to_path`].
pub struct GgufWriter {
    metadata: BTreeMap<String, MetaValue>,
    tensors: Vec<TensorPayload>,
    alignment: u64,
    version: u32,
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    pub fn new() -> Self {
        Self {
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
            alignment: DEFAULT_ALIGNMENT,
            version: 3,
        }
    }

    /// Set an alternative alignment. Persists as
    /// `general.alignment` in the output file.
    pub fn with_alignment(mut self, alignment: u64) -> Self {
        assert!(alignment.is_power_of_two() && alignment >= 8);
        self.alignment = alignment;
        self
    }

    pub fn set_meta(&mut self, key: impl Into<String>, value: MetaValue) {
        self.metadata.insert(key.into(), value);
    }

    pub fn set_arch(&mut self, arch: &str) {
        self.set_meta("general.architecture", MetaValue::String(arch.into()));
    }

    pub fn add_tensor(&mut self, payload: TensorPayload) -> Result<()> {
        payload.validate()?;
        self.tensors.push(payload);
        Ok(())
    }

    pub fn add_tensor_bytes(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: GgmlType,
        bytes: Vec<u8>,
    ) -> Result<()> {
        self.add_tensor(TensorPayload {
            name: name.into(),
            shape,
            dtype,
            bytes,
        })
    }

    /// Write a complete GGUF v3 file to `path`.
    pub fn write_to_path<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let f = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut bw = BufWriter::new(f);
        self.write(&mut bw)?;
        bw.flush()?;
        Ok(())
    }

    /// Stream the file into any seekable `Write` (e.g. a `Vec<u8>`
    /// via `std::io::Cursor`).
    pub fn write<W: Write + Seek>(&self, w: &mut W) -> Result<()> {
        // 1. Header
        w.write_all(&GGUF_MAGIC.to_le_bytes())?;
        w.write_all(&self.version.to_le_bytes())?;
        w.write_all(&(self.tensors.len() as u64).to_le_bytes())?;
        // Always emit a general.alignment entry when non-default; we
        // also keep whatever the user set explicitly.
        let mut metadata = self.metadata.clone();
        if self.alignment != DEFAULT_ALIGNMENT && !metadata.contains_key("general.alignment") {
            metadata.insert("general.alignment".into(), MetaValue::U64(self.alignment));
        }
        w.write_all(&(metadata.len() as u64).to_le_bytes())?;

        // 2. Metadata
        for (k, v) in &metadata {
            write_string(w, k)?;
            write_value(w, v)?;
        }

        // 3. Tensor descriptors. Compute per-tensor data offsets
        //    relative to the start of the data segment.
        //    Offsets are aligned individually (each tensor's data
        //    starts on an `alignment`-multiple within the segment).
        let mut offsets: Vec<u64> = Vec::with_capacity(self.tensors.len());
        let mut cursor: u64 = 0;
        for t in &self.tensors {
            let pad = (self.alignment - (cursor % self.alignment)) % self.alignment;
            cursor += pad;
            offsets.push(cursor);
            cursor += t.bytes.len() as u64;
        }
        for (t, &off) in self.tensors.iter().zip(&offsets) {
            write_string(w, &t.name)?;
            w.write_all(&(t.shape.len() as u32).to_le_bytes())?;
            for &d in &t.shape {
                w.write_all(&(d as u64).to_le_bytes())?;
            }
            w.write_all(&(t.dtype as u32).to_le_bytes())?;
            w.write_all(&off.to_le_bytes())?;
        }

        // 4. Pad to alignment boundary before the data segment.
        let pos = w.stream_position()?;
        let pad = (self.alignment - (pos % self.alignment)) % self.alignment;
        write_zeros(w, pad as usize)?;

        // 5. Tensor bytes, padded so each tensor starts at its
        //    declared offset within the data segment.
        let data_start = w.stream_position()?;
        let mut written: u64 = 0;
        for (t, &off) in self.tensors.iter().zip(&offsets) {
            let target = data_start + off;
            let cur = w.stream_position()?;
            if cur < target {
                write_zeros(w, (target - cur) as usize)?;
            } else if cur > target {
                bail!(
                    "writer: cursor {cur} past target {target} for tensor {}",
                    t.name
                );
            }
            w.write_all(&t.bytes)?;
            written = (w.stream_position()? - data_start).max(written);
        }
        let _ = written;
        Ok(())
    }
}

fn write_string<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn write_zeros<W: Write>(w: &mut W, n: usize) -> Result<()> {
    const Z: [u8; 64] = [0u8; 64];
    let mut left = n;
    while left > 0 {
        let take = left.min(Z.len());
        w.write_all(&Z[..take])?;
        left -= take;
    }
    Ok(())
}

fn write_value<W: Write>(w: &mut W, v: &MetaValue) -> Result<()> {
    match v {
        MetaValue::U8(x) => {
            w.write_all(&0u32.to_le_bytes())?;
            w.write_all(&[*x])?;
        }
        MetaValue::I8(x) => {
            w.write_all(&1u32.to_le_bytes())?;
            w.write_all(&[*x as u8])?;
        }
        MetaValue::U16(x) => {
            w.write_all(&2u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())?;
        }
        MetaValue::I16(x) => {
            w.write_all(&3u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())?;
        }
        MetaValue::U32(x) => {
            w.write_all(&4u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())?;
        }
        MetaValue::I32(x) => {
            w.write_all(&5u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())?;
        }
        MetaValue::F32(x) => {
            w.write_all(&6u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())?;
        }
        MetaValue::Bool(x) => {
            w.write_all(&7u32.to_le_bytes())?;
            w.write_all(&[u8::from(*x)])?;
        }
        MetaValue::String(s) => {
            w.write_all(&8u32.to_le_bytes())?;
            write_string(w, s)?;
        }
        MetaValue::Array(items) => {
            w.write_all(&9u32.to_le_bytes())?;
            let elem_ty = array_elem_type(items)?;
            w.write_all(&elem_ty.to_le_bytes())?;
            w.write_all(&(items.len() as u64).to_le_bytes())?;
            for it in items {
                write_scalar(w, it)?;
            }
        }
        MetaValue::U64(x) => {
            w.write_all(&10u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())?;
        }
        MetaValue::I64(x) => {
            w.write_all(&11u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())?;
        }
        MetaValue::F64(x) => {
            w.write_all(&12u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())?;
        }
    }
    Ok(())
}

fn write_scalar<W: Write>(w: &mut W, v: &MetaValue) -> Result<()> {
    match v {
        MetaValue::U8(x) => w.write_all(&[*x])?,
        MetaValue::I8(x) => w.write_all(&[*x as u8])?,
        MetaValue::U16(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::I16(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::U32(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::I32(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::F32(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::Bool(x) => w.write_all(&[u8::from(*x)])?,
        MetaValue::String(s) => write_string(w, s)?,
        MetaValue::U64(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::I64(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::F64(x) => w.write_all(&x.to_le_bytes())?,
        MetaValue::Array(_) => bail!("nested arrays not allowed in GGUF metadata"),
    }
    Ok(())
}

fn array_elem_type(items: &[MetaValue]) -> Result<u32> {
    let first = items
        .first()
        .ok_or_else(|| anyhow::anyhow!("empty arrays have no element type"))?;
    let ty = match first {
        MetaValue::U8(_) => 0,
        MetaValue::I8(_) => 1,
        MetaValue::U16(_) => 2,
        MetaValue::I16(_) => 3,
        MetaValue::U32(_) => 4,
        MetaValue::I32(_) => 5,
        MetaValue::F32(_) => 6,
        MetaValue::Bool(_) => 7,
        MetaValue::String(_) => 8,
        MetaValue::U64(_) => 10,
        MetaValue::I64(_) => 11,
        MetaValue::F64(_) => 12,
        MetaValue::Array(_) => bail!("nested arrays not allowed in GGUF metadata"),
    };
    for it in &items[1..] {
        let same = matches!(
            (first, it),
            (MetaValue::U8(_), MetaValue::U8(_))
                | (MetaValue::I8(_), MetaValue::I8(_))
                | (MetaValue::U16(_), MetaValue::U16(_))
                | (MetaValue::I16(_), MetaValue::I16(_))
                | (MetaValue::U32(_), MetaValue::U32(_))
                | (MetaValue::I32(_), MetaValue::I32(_))
                | (MetaValue::F32(_), MetaValue::F32(_))
                | (MetaValue::Bool(_), MetaValue::Bool(_))
                | (MetaValue::String(_), MetaValue::String(_))
                | (MetaValue::U64(_), MetaValue::U64(_))
                | (MetaValue::I64(_), MetaValue::I64(_))
                | (MetaValue::F64(_), MetaValue::F64(_))
        );
        if !same {
            bail!("heterogeneous array element types not allowed");
        }
    }
    Ok(ty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GgufFile;
    use std::io::Cursor;

    #[test]
    fn roundtrip_f32_tensor_via_writer() {
        let mut w = GgufWriter::new();
        w.set_meta("general.architecture", MetaValue::String("test".into()));
        w.set_meta("count", MetaValue::U32(7));
        w.add_tensor_bytes(
            "w",
            vec![2, 3],
            GgmlType::F32,
            (0..6)
                .flat_map(|i| (i as f32).to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        w.add_tensor_bytes(
            "b",
            vec![3],
            GgmlType::F32,
            (10..13)
                .flat_map(|i| (i as f32).to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();

        let mut buf = Cursor::new(Vec::new());
        w.write(&mut buf).unwrap();
        let data = buf.into_inner();

        let mut c = Cursor::new(data);
        let parsed = GgufFile::from_reader(&mut c).unwrap();
        assert_eq!(parsed.tensors.len(), 2);
        let (vw, sw) = parsed.dequant_f32("w").unwrap();
        assert_eq!(sw, vec![2, 3]);
        assert_eq!(vw, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let (vb, sb) = parsed.dequant_f32("b").unwrap();
        assert_eq!(sb, vec![3]);
        assert_eq!(vb, vec![10.0, 11.0, 12.0]);
        assert_eq!(
            parsed
                .metadata
                .get("general.architecture")
                .and_then(MetaValue::as_str),
            Some("test")
        );
        assert_eq!(
            parsed.metadata.get("count").and_then(MetaValue::as_u32),
            Some(7)
        );
    }

    #[test]
    fn roundtrip_quantized_tensor() {
        use crate::quantize::quantize_q8_0;
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let q = quantize_q8_0(&x).unwrap();
        let mut w = GgufWriter::new();
        w.add_tensor_bytes("w", vec![64], GgmlType::Q8_0, q)
            .unwrap();
        let mut buf = Cursor::new(Vec::new());
        w.write(&mut buf).unwrap();
        let mut c = Cursor::new(buf.into_inner());
        let parsed = GgufFile::from_reader(&mut c).unwrap();
        let (out, _) = parsed.dequant_f32("w").unwrap();
        // Q8_0 quant error scales with the per-block max — expect well
        // under 1% relative on this small range.
        let max_err = x
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let max_abs = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
        assert!(
            max_err / max_abs < 0.02,
            "Q8_0 rel err {}",
            max_err / max_abs
        );
    }
}
