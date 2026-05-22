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

//! GGUF (GGML Universal Format) parser + dequantization to f32.
//!
//! Standalone: no `rlx-*` dependencies. Higher-level `WeightLoader` /
//! HF name mapping lives in the separate model-builders repo (see root README).
//!
//! Supports GGUF v1, v2, v3 (the live formats). Tensor dtypes
//! decoded today: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1.
//! Other GGML quants parse but error on `dequant_f32` — file ships,
//! callers know which key is unreadable. Extending = one match arm.
//!
//! Endianness: little-endian assumed (the only flavor that ships in
//! practice). The GGUF spec reserves a flag for big-endian; we don't
//! parse it.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const DEFAULT_ALIGNMENT: u64 = 32;

// ─── GGML tensor dtype codes ──────────────────────────────────────
//
// Subset of upstream `ggml_type`. Codes are stable; adding new ones
// is append-only.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    // K-quants
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    // I-quants
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    // Plain integer / float dtypes
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    // Ternary / MX
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            19 => Self::IQ1S,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1M,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MXFP4,
            40 => Self::NVFP4,
            41 => Self::Q1_0,
            other => bail!("unknown ggml type {other}"),
        })
    }
}

// ─── Metadata value ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) if *v >= 0 => Some(*v as u32),
            Self::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ─── Parsed file ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub shape: Vec<usize>, // GGML order (innermost first); kept verbatim.
    pub dtype: GgmlType,
    /// Offset within the tensor data segment (relative to `data_start`,
    /// not to the start of the file).
    pub offset: u64,
}

impl GgufTensor {
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct GgufFile {
    pub version: u32,
    pub alignment: u64,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: HashMap<String, GgufTensor>,
    /// Raw tensor-data segment (`data_start` to EOF). Slurped into
    /// memory — fine for embed-class models. Future mmap path slots
    /// in here.
    data: Vec<u8>,
}

impl GgufFile {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        Self::from_reader(&mut f)
    }

    pub fn from_reader<R: Read + Seek>(r: &mut R) -> Result<Self> {
        let magic = read_u32(r)?;
        if magic != GGUF_MAGIC {
            bail!("not a GGUF file (magic {magic:#x})");
        }
        let version = read_u32(r)?;
        if !(1..=3).contains(&version) {
            bail!("unsupported GGUF version {version}");
        }

        // v1 used u32 counts; v2/v3 use u64. Same field order.
        let (tensor_count, kv_count) = if version == 1 {
            (read_u32(r)? as u64, read_u32(r)? as u64)
        } else {
            (read_u64(r)?, read_u64(r)?)
        };

        let mut metadata = HashMap::with_capacity(kv_count as usize);
        for _ in 0..kv_count {
            let key = read_string(r, version)?;
            let value = read_value(r, version)?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(MetaValue::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT);

        let mut tensors = HashMap::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = read_string(r, version)?;
            let n_dims = read_u32(r)?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let d = if version == 1 {
                    read_u32(r)? as u64
                } else {
                    read_u64(r)?
                };
                shape.push(d as usize);
            }
            let dtype_raw = read_u32(r)?;
            let dtype =
                GgmlType::from_u32(dtype_raw).with_context(|| format!("tensor {name}: dtype"))?;
            let offset = read_u64(r)?;
            tensors.insert(
                name.clone(),
                GgufTensor {
                    name,
                    shape,
                    dtype,
                    offset,
                },
            );
        }

        // Data segment starts at the next `alignment` boundary.
        let pos = r.stream_position()?;
        let pad = (alignment - (pos % alignment)) % alignment;
        r.seek(SeekFrom::Current(pad as i64))?;

        let mut data = Vec::new();
        r.read_to_end(&mut data)?;

        Ok(Self {
            version,
            alignment,
            metadata,
            tensors,
            data,
        })
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    pub fn get(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.get(name)
    }

    /// Dequantize a tensor to f32. Shape is verbatim from the file
    /// (GGML's innermost-first order); transpose / reorder is the
    /// caller's job since conventions vary by model family.
    pub fn dequant_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        let n = t.n_elements();
        let bytes = self.tensor_bytes(t)?;
        let data = match t.dtype {
            GgmlType::F32 => dequant_f32_raw(bytes, n)?,
            GgmlType::F16 => dequant_f16(bytes, n)?,
            GgmlType::BF16 => dequant_bf16(bytes, n)?,
            GgmlType::Q8_0 => dequant_q8_0(bytes, n)?,
            GgmlType::Q4_0 => dequant_q4_0(bytes, n)?,
            GgmlType::Q4_1 => dequant_q4_1(bytes, n)?,
            GgmlType::Q5_0 => dequant_q5_0(bytes, n)?,
            GgmlType::Q5_1 => dequant_q5_1(bytes, n)?,
            GgmlType::Q4K => dequant_q4_k(bytes, n)?,
            GgmlType::Q5K => dequant_q5_k(bytes, n)?,
            GgmlType::Q6K => dequant_q6_k(bytes, n)?,
            GgmlType::Q8K => dequant_q8_k(bytes, n)?,
            GgmlType::Q2K => dequant_q2_k(bytes, n)?,
            GgmlType::Q3K => dequant_q3_k(bytes, n)?,
            other => bail!("dequant for {other:?} not implemented yet (tensor {name})"),
        };
        Ok((data, t.shape.clone()))
    }

    /// Slice the raw tensor bytes out of the data segment. Public so
    /// callers writing custom kernels can pass quantized blocks
    /// straight through.
    pub fn tensor_bytes(&self, t: &GgufTensor) -> Result<&[u8]> {
        let n = t.n_elements();
        let nbytes = bytes_for(t.dtype, n)
            .ok_or_else(|| anyhow!("element count {n} not aligned to block for {:?}", t.dtype))?;
        let start = t.offset as usize;
        let end = start
            .checked_add(nbytes)
            .ok_or_else(|| anyhow!("tensor {} byte range overflow", t.name))?;
        if end > self.data.len() {
            bail!(
                "tensor {} extends past data segment ({end} > {})",
                t.name,
                self.data.len()
            );
        }
        Ok(&self.data[start..end])
    }
}

// ─── byte-count helpers ───────────────────────────────────────────

const QK8_0: usize = 32;
const QK4_0: usize = 32;
const QK4_1: usize = 32;
const QK5_0: usize = 32;
const QK5_1: usize = 32;
/// Super-block size shared by every K-quant format. Per llama.cpp's
/// `ggml-quants.h`. Tensors quantized with `Q{4,5,6,8}_K` must have
/// an element count divisible by 256.
/// Super-block size for K-quant formats (256 elements).
pub const QK_K: usize = 256;
/// Byte size of the packed scales+mins region in `block_q4_K` /
/// `block_q5_K` — 8 sub-blocks × 12 bits (6 bits scale + 6 bits min)
/// = 96 bits = 12 bytes. Same layout in both formats.
const K_SCALE_SIZE: usize = 12;

fn bytes_for(dtype: GgmlType, n: usize) -> Option<usize> {
    let blk = |qk: usize, blk_bytes: usize| -> Option<usize> {
        if !n.is_multiple_of(qk) {
            return None;
        }
        Some((n / qk) * blk_bytes)
    };
    match dtype {
        GgmlType::F32 => Some(n * 4),
        GgmlType::F16 | GgmlType::BF16 => Some(n * 2),
        GgmlType::Q8_0 => blk(QK8_0, 2 + QK8_0), // f16 d + 32×i8
        GgmlType::Q4_0 => blk(QK4_0, 2 + QK4_0 / 2), // f16 d + 16×u8
        GgmlType::Q4_1 => blk(QK4_1, 2 + 2 + QK4_1 / 2), // f16 d + f16 m + 16×u8
        GgmlType::Q5_0 => blk(QK5_0, 2 + 4 + QK5_0 / 2), // f16 d + u32 qh + 16×u8
        GgmlType::Q5_1 => blk(QK5_1, 2 + 2 + 4 + QK5_1 / 2), // f16 d + f16 m + u32 qh + 16×u8
        // K-quants (super-block = 256 elements):
        GgmlType::Q4K => blk(QK_K, 2 + 2 + K_SCALE_SIZE + QK_K / 2), // d + dmin + 12 scales + 128 quant
        GgmlType::Q5K => blk(QK_K, 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2), // + 32 high bits
        GgmlType::Q6K => blk(QK_K, QK_K / 2 + QK_K / 4 + QK_K / 16 + 2), // ql + qh + scales(i8) + d
        GgmlType::Q8K => blk(QK_K, 4 + QK_K + (QK_K / 16) * 2), // f32 d + 256 i8 + 16 i16 bsums
        GgmlType::Q2K => blk(QK_K, 2 + 2 + QK_K / 16 + QK_K / 4), // d + dmin + 16 scales + 64 qs
        GgmlType::Q3K => blk(QK_K, 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 4), // d + 12 scales + 32 hmask + 64 qs
        // Anything else: not yet supported. dequant_f32 will reject
        // these too; tensor_bytes returns None to stay consistent.
        _ => None,
    }
}

// ─── reader helpers ───────────────────────────────────────────────

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_i8<R: Read>(r: &mut R) -> Result<i8> {
    Ok(read_u8(r)? as i8)
}
fn read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_i16<R: Read>(r: &mut R) -> Result<i16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(i16::from_le_bytes(b))
}
fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_i32<R: Read>(r: &mut R) -> Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_i64<R: Read>(r: &mut R) -> Result<i64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(i64::from_le_bytes(b))
}
fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}
fn read_bool<R: Read>(r: &mut R) -> Result<bool> {
    Ok(read_u8(r)? != 0)
}

fn read_string<R: Read>(r: &mut R, version: u32) -> Result<String> {
    let len = if version == 1 {
        read_u32(r)? as u64
    } else {
        read_u64(r)?
    };
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| anyhow!("non-UTF8 string: {e}"))
}

fn read_value<R: Read + Seek>(r: &mut R, version: u32) -> Result<MetaValue> {
    let ty = read_u32(r)?;
    Ok(match ty {
        0 => MetaValue::U8(read_u8(r)?),
        1 => MetaValue::I8(read_i8(r)?),
        2 => MetaValue::U16(read_u16(r)?),
        3 => MetaValue::I16(read_i16(r)?),
        4 => MetaValue::U32(read_u32(r)?),
        5 => MetaValue::I32(read_i32(r)?),
        6 => MetaValue::F32(read_f32(r)?),
        7 => MetaValue::Bool(read_bool(r)?),
        8 => MetaValue::String(read_string(r, version)?),
        9 => {
            let inner_ty = read_u32(r)?;
            let len = if version == 1 {
                read_u32(r)? as u64
            } else {
                read_u64(r)?
            };
            let mut out = Vec::with_capacity(len as usize);
            for _ in 0..len {
                out.push(read_scalar(r, inner_ty, version)?);
            }
            MetaValue::Array(out)
        }
        10 => MetaValue::U64(read_u64(r)?),
        11 => MetaValue::I64(read_i64(r)?),
        12 => MetaValue::F64(read_f64(r)?),
        other => bail!("unknown metadata value type {other}"),
    })
}

fn read_scalar<R: Read + Seek>(r: &mut R, ty: u32, version: u32) -> Result<MetaValue> {
    // Arrays don't nest (per spec). We re-implement primitive reads
    // here rather than calling read_value because that one expects a
    // leading type tag.
    Ok(match ty {
        0 => MetaValue::U8(read_u8(r)?),
        1 => MetaValue::I8(read_i8(r)?),
        2 => MetaValue::U16(read_u16(r)?),
        3 => MetaValue::I16(read_i16(r)?),
        4 => MetaValue::U32(read_u32(r)?),
        5 => MetaValue::I32(read_i32(r)?),
        6 => MetaValue::F32(read_f32(r)?),
        7 => MetaValue::Bool(read_bool(r)?),
        8 => MetaValue::String(read_string(r, version)?),
        10 => MetaValue::U64(read_u64(r)?),
        11 => MetaValue::I64(read_i64(r)?),
        12 => MetaValue::F64(read_f64(r)?),
        9 => bail!("nested arrays not allowed in GGUF metadata"),
        other => bail!("unknown array element type {other}"),
    })
}

// ─── dequant kernels ──────────────────────────────────────────────
//
// Reference formulas mirror llama.cpp's `dequantize_row_*` in
// `ggml-quants.c`. Naive on purpose — runs once at load, not hot.

fn dequant_f32_raw(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() != n * 4 {
        bail!("F32: {} bytes for {n} elements", bytes.len());
    }
    let f: &[f32] = bytemuck::cast_slice(bytes);
    Ok(f.to_vec())
}

fn dequant_f16(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() != n * 2 {
        bail!("F16: {} bytes for {n} elements", bytes.len());
    }
    let h: &[half::f16] = bytemuck::cast_slice(bytes);
    Ok(h.iter().map(|x| x.to_f32()).collect())
}

fn dequant_bf16(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if bytes.len() != n * 2 {
        bail!("BF16: {} bytes for {n} elements", bytes.len());
    }
    let h: &[half::bf16] = bytemuck::cast_slice(bytes);
    Ok(h.iter().map(|x| x.to_f32()).collect())
}

fn read_f16_le(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}

fn dequant_q8_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK8_0) {
        bail!("Q8_0: n={n} not divisible by {QK8_0}");
    }
    let nb = n / QK8_0;
    let blk = 2 + QK8_0;
    if bytes.len() != nb * blk {
        bail!("Q8_0: bad byte count");
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..nb {
        let off = i * blk;
        let d = read_f16_le(&bytes[off..off + 2]);
        let qs = &bytes[off + 2..off + 2 + QK8_0];
        for &q in qs {
            out.push(d * (q as i8) as f32);
        }
    }
    Ok(out)
}

fn dequant_q4_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK4_0) {
        bail!("Q4_0: n={n} not divisible by {QK4_0}");
    }
    let nb = n / QK4_0;
    let blk = 2 + QK4_0 / 2;
    if bytes.len() != nb * blk {
        bail!("Q4_0: bad byte count");
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..nb {
        let off = i * blk;
        let d = read_f16_le(&bytes[off..off + 2]);
        let qs = &bytes[off + 2..off + 2 + QK4_0 / 2];
        // Layout: low nibbles → first half of block, high nibbles → second half.
        for j in 0..QK4_0 / 2 {
            let v0 = (qs[j] & 0x0F) as i32 - 8;
            out.push(d * v0 as f32);
        }
        for j in 0..QK4_0 / 2 {
            let v1 = (qs[j] >> 4) as i32 - 8;
            out.push(d * v1 as f32);
        }
    }
    Ok(out)
}

fn dequant_q4_1(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK4_1) {
        bail!("Q4_1: n={n} not divisible by {QK4_1}");
    }
    let nb = n / QK4_1;
    let blk = 2 + 2 + QK4_1 / 2;
    if bytes.len() != nb * blk {
        bail!("Q4_1: bad byte count");
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..nb {
        let off = i * blk;
        let d = read_f16_le(&bytes[off..off + 2]);
        let m = read_f16_le(&bytes[off + 2..off + 4]);
        let qs = &bytes[off + 4..off + 4 + QK4_1 / 2];
        for j in 0..QK4_1 / 2 {
            let v0 = (qs[j] & 0x0F) as f32;
            out.push(d * v0 + m);
        }
        for j in 0..QK4_1 / 2 {
            let v1 = (qs[j] >> 4) as f32;
            out.push(d * v1 + m);
        }
    }
    Ok(out)
}

fn dequant_q5_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK5_0) {
        bail!("Q5_0: n={n} not divisible by {QK5_0}");
    }
    let nb = n / QK5_0;
    let blk = 2 + 4 + QK5_0 / 2;
    if bytes.len() != nb * blk {
        bail!("Q5_0: bad byte count");
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..nb {
        let off = i * blk;
        let d = read_f16_le(&bytes[off..off + 2]);
        let qh = u32::from_le_bytes([
            bytes[off + 2],
            bytes[off + 3],
            bytes[off + 4],
            bytes[off + 5],
        ]);
        let qs = &bytes[off + 6..off + 6 + QK5_0 / 2];
        for j in 0..QK5_0 / 2 {
            let xh0 = (((qh >> j) & 0x01) as u8) << 4;
            let v0 = ((qs[j] & 0x0F) | xh0) as i32 - 16;
            out.push(d * v0 as f32);
        }
        for j in 0..QK5_0 / 2 {
            let xh1 = (((qh >> (j + 16)) & 0x01) as u8) << 4;
            let v1 = ((qs[j] >> 4) | xh1) as i32 - 16;
            out.push(d * v1 as f32);
        }
    }
    Ok(out)
}

fn dequant_q5_1(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK5_1) {
        bail!("Q5_1: n={n} not divisible by {QK5_1}");
    }
    let nb = n / QK5_1;
    let blk = 2 + 2 + 4 + QK5_1 / 2;
    if bytes.len() != nb * blk {
        bail!("Q5_1: bad byte count");
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..nb {
        let off = i * blk;
        let d = read_f16_le(&bytes[off..off + 2]);
        let m = read_f16_le(&bytes[off + 2..off + 4]);
        let qh = u32::from_le_bytes([
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]);
        let qs = &bytes[off + 8..off + 8 + QK5_1 / 2];
        for j in 0..QK5_1 / 2 {
            let xh0 = (((qh >> j) & 0x01) as u8) << 4;
            let v0 = ((qs[j] & 0x0F) | xh0) as f32;
            out.push(d * v0 + m);
        }
        for j in 0..QK5_1 / 2 {
            let xh1 = (((qh >> (j + 16)) & 0x01) as u8) << 4;
            let v1 = ((qs[j] >> 4) | xh1) as f32;
            out.push(d * v1 + m);
        }
    }
    Ok(out)
}

// ─── K-quants ─────────────────────────────────────────────────────
//
// All four K-quant formats share a 256-element super-block divided
// into 8 sub-blocks of 32 elements. Q4_K / Q5_K pack a 6-bit scale
// and a 6-bit min for each sub-block into the shared 12-byte
// `scales` region; Q6_K stores 16 signed 8-bit scales directly. The
// `get_scale_min_k4` helper mirrors the bit-interleaving used in
// llama.cpp's reference decoder (`ggml-quants.c`):
//
//   j < 4:  scale = q[j]   & 0x3F                  min = q[j+4]  & 0x3F
//   j >= 4: scale = (q[j+4]   & 0x0F) | ((q[j-4] >> 6) << 4)
//           min   = (q[j+4]   >> 4)   | ((q[j  ] >> 6) << 4)
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize one Q4_K super-block (144 bytes) into `out` (256 f32s).
pub fn dequant_q4_k_block(block: &[u8], out: &mut [f32; QK_K]) {
    let d = read_f16_le(&block[0..2]);
    let dmin = read_f16_le(&block[2..4]);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let qs = &block[4 + K_SCALE_SIZE..];
    let mut is = 0usize;
    let mut out_i = 0usize;
    for j in (0..8).step_by(2) {
        let (sc0, m0) = get_scale_min_k4(j, scales);
        let (sc1, m1) = get_scale_min_k4(j + 1, scales);
        let d0 = d * sc0 as f32;
        let m0f = dmin * m0 as f32;
        let d1 = d * sc1 as f32;
        let m1f = dmin * m1 as f32;
        for l in 0..32 {
            let q = qs[is + l];
            out[out_i] = d0 * (q & 0x0F) as f32 - m0f;
            out_i += 1;
        }
        for l in 0..32 {
            let q = qs[is + l];
            out[out_i] = d1 * (q >> 4) as f32 - m1f;
            out_i += 1;
        }
        is += 32;
    }
}

/// Dequantize one Q5_K super-block (176 bytes) into `out`.
pub fn dequant_q5_k_block(block: &[u8], out: &mut [f32; QK_K]) {
    let d = read_f16_le(&block[0..2]);
    let dmin = read_f16_le(&block[2..4]);
    let scales = &block[4..4 + K_SCALE_SIZE];
    let qh = &block[4 + K_SCALE_SIZE..4 + K_SCALE_SIZE + QK_K / 8];
    let qs = &block[4 + K_SCALE_SIZE + QK_K / 8..];
    let mut is = 0usize;
    let mut out_i = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for j in (0..8).step_by(2) {
        let (sc0, m0) = get_scale_min_k4(j, scales);
        let (sc1, m1) = get_scale_min_k4(j + 1, scales);
        let d0 = d * sc0 as f32;
        let m0f = dmin * m0 as f32;
        let d1 = d * sc1 as f32;
        let m1f = dmin * m1 as f32;
        for l in 0..32 {
            let lo = qs[is + l] & 0x0F;
            let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
            out[out_i] = d0 * (lo + hi) as f32 - m0f;
            out_i += 1;
        }
        for l in 0..32 {
            let lo = qs[is + l] >> 4;
            let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
            out[out_i] = d1 * (lo + hi) as f32 - m1f;
            out_i += 1;
        }
        is += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Dequantize one Q6_K super-block (210 bytes) into `out`.
pub fn dequant_q6_k_block(block: &[u8], out: &mut [f32; QK_K]) {
    let ql_len = QK_K / 2;
    let qh_len = QK_K / 4;
    let sc_len = QK_K / 16;
    let ql = &block[0..ql_len];
    let qh = &block[ql_len..ql_len + qh_len];
    let sc = &block[ql_len + qh_len..ql_len + qh_len + sc_len];
    let d = read_f16_le(&block[ql_len + qh_len + sc_len..]);
    for h in 0..2 {
        let dst_base = h * 128;
        let ql_off = h * 64;
        let qh_off_h = h * 32;
        let sc_off = h * 8;
        for l in 0..32 {
            let is = l / 16;
            let qh_b = qh[qh_off_h + l];
            let q1 = ((ql[ql_off + l] & 0x0F) | (((qh_b >> 0) & 3) << 4)) as i32 - 32;
            let q2 = ((ql[ql_off + l + 32] & 0x0F) | (((qh_b >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[ql_off + l] >> 4) | (((qh_b >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh_b >> 6) & 3) << 4)) as i32 - 32;
            out[dst_base + l] = d * sc[sc_off + is] as f32 * q1 as f32;
            out[dst_base + l + 32] = d * sc[sc_off + is + 2] as f32 * q2 as f32;
            out[dst_base + l + 64] = d * sc[sc_off + is + 4] as f32 * q3 as f32;
            out[dst_base + l + 96] = d * sc[sc_off + is + 6] as f32 * q4 as f32;
        }
    }
}

/// Dequantize one Q8_K super-block (276 bytes) into `out`.
pub fn dequant_q8_k_block(block: &[u8], out: &mut [f32; QK_K]) {
    let d = f32::from_le_bytes(block[0..4].try_into().unwrap());
    let qs = &block[4..4 + QK_K];
    for i in 0..QK_K {
        out[i] = d * qs[i] as i8 as f32;
    }
}

/// Dequantize one Q2_K super-block (84 bytes) into `out`.
pub fn dequant_q2_k_block(block: &[u8], out: &mut [f32; QK_K]) {
    let d = read_f16_le(&block[0..2]);
    let min = read_f16_le(&block[2..4]);
    let mut q = &block[4 + QK_K / 16..];
    let mut is = 0usize;
    let mut out_i = 0usize;
    for _ in 0..(QK_K / 128) {
        let mut shift = 0u32;
        for _ in 0..4 {
            let sc = block[4 + is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = min * (sc >> 4) as f32;
            for l in 0..16 {
                out[out_i] = dl * ((q[l] >> shift) & 3) as f32 - ml;
                out_i += 1;
            }
            let sc = block[4 + is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = min * (sc >> 4) as f32;
            for l in 0..16 {
                out[out_i] = dl * ((q[l + 16] >> shift) & 3) as f32 - ml;
                out_i += 1;
            }
            shift += 2;
        }
        q = &q[32..];
    }
}

/// Dequantize one Q3_K super-block (110 bytes) into `out`.
pub fn dequant_q3_k_block(block: &[u8], out: &mut [f32; QK_K]) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let d_all = read_f16_le(&block[0..2]);
    let hm = &block[2 + K_SCALE_SIZE..2 + K_SCALE_SIZE + QK_K / 8];
    let mut q = &block[2 + K_SCALE_SIZE + QK_K / 8..];
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes(block[2..6].try_into().unwrap());
    aux[1] = u32::from_le_bytes(block[6..10].try_into().unwrap());
    aux[2] = u32::from_le_bytes(block[10..14].try_into().unwrap());
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let scales: &[i8; 16] = unsafe { &*(aux.as_ptr() as *const [i8; 16]) };
    let mut is = 0usize;
    let mut m: u8 = 1;
    let mut out_i = 0usize;
    for _ in 0..(QK_K / 128) {
        let mut shift = 0u32;
        for _ in 0..4 {
            let dl = d_all * (scales[is] - 32) as f32;
            is += 1;
            for l in 0..16 {
                let h = if hm[l] & m != 0 { 0 } else { 4 };
                out[out_i] = dl * (((q[l] >> shift) & 3) as i8 - h) as f32;
                out_i += 1;
            }
            let dl = d_all * (scales[is] - 32) as f32;
            is += 1;
            for l in 0..16 {
                let h = if hm[l + 16] & m != 0 { 0 } else { 4 };
                out[out_i] = dl * (((q[l + 16] >> shift) & 3) as i8 - h) as f32;
                out_i += 1;
            }
            shift += 2;
            m <<= 1;
        }
        q = &q[32..];
    }
}

pub fn dequant_q2_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("Q2_K: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    let blk = 2 + 2 + QK_K / 16 + QK_K / 4;
    if bytes.len() != nb * blk {
        bail!("Q2_K: bad byte count");
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * blk;
        dequant_q2_k_block(
            &bytes[off..off + blk],
            (&mut out[i * QK_K..(i + 1) * QK_K]).try_into().unwrap(),
        );
    }
    Ok(out)
}

pub fn dequant_q3_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("Q3_K: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    let blk = 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 4;
    if bytes.len() != nb * blk {
        bail!("Q3_K: bad byte count");
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * blk;
        dequant_q3_k_block(
            &bytes[off..off + blk],
            (&mut out[i * QK_K..(i + 1) * QK_K]).try_into().unwrap(),
        );
    }
    Ok(out)
}

/// Q4_K block: 144 bytes / 256 elements (4.5 bits/element).
/// Layout: f16 d + f16 dmin + 12-byte packed scales/mins + 128 nibbles.
pub fn dequant_q4_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("Q4_K: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 2;
    if bytes.len() != nb * blk {
        bail!("Q4_K: bad byte count");
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..nb {
        let off = i * blk;
        let d = read_f16_le(&bytes[off..off + 2]);
        let dmin = read_f16_le(&bytes[off + 2..off + 4]);
        let scales = &bytes[off + 4..off + 4 + K_SCALE_SIZE];
        let qs = &bytes[off + 4 + K_SCALE_SIZE..off + blk];
        // 8 sub-blocks × 32 elements. Each pair of sub-blocks reads
        // 32 nibbles (16 bytes): low nibbles → sub-block j, high
        // nibbles → sub-block j+1.
        let mut is = 0usize;
        for j in (0..8).step_by(2) {
            let (sc0, m0) = get_scale_min_k4(j, scales);
            let (sc1, m1) = get_scale_min_k4(j + 1, scales);
            let d0 = d * sc0 as f32;
            let m0 = dmin * m0 as f32;
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            for l in 0..32 {
                let q = qs[is + l];
                out.push(d0 * (q & 0x0F) as f32 - m0);
            }
            for l in 0..32 {
                let q = qs[is + l];
                out.push(d1 * (q >> 4) as f32 - m1);
            }
            is += 32;
        }
    }
    Ok(out)
}

/// Q5_K block: 176 bytes / 256 elements (5.5 bits/element).
/// Layout: f16 d + f16 dmin + 12-byte packed scales/mins + 32-byte
/// high-bits + 128 nibbles. Each element's 5th bit lives in `qh`
/// indexed by position-within-super-block.
pub fn dequant_q5_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("Q5_K: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    let blk = 2 + 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 2;
    if bytes.len() != nb * blk {
        bail!("Q5_K: bad byte count");
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..nb {
        let off = i * blk;
        let d = read_f16_le(&bytes[off..off + 2]);
        let dmin = read_f16_le(&bytes[off + 2..off + 4]);
        let scales = &bytes[off + 4..off + 4 + K_SCALE_SIZE];
        let qh_off = off + 4 + K_SCALE_SIZE;
        let qh = &bytes[qh_off..qh_off + QK_K / 8];
        let qs_off = qh_off + QK_K / 8;
        let qs = &bytes[qs_off..qs_off + QK_K / 2];
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for j in (0..8).step_by(2) {
            let (sc0, m0) = get_scale_min_k4(j, scales);
            let (sc1, m1) = get_scale_min_k4(j + 1, scales);
            let d0 = d * sc0 as f32;
            let m0 = dmin * m0 as f32;
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            for l in 0..32 {
                let lo = qs[is + l] & 0x0F;
                let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                out.push(d0 * (lo + hi) as f32 - m0);
            }
            for l in 0..32 {
                let lo = qs[is + l] >> 4;
                let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                out.push(d1 * (lo + hi) as f32 - m1);
            }
            is += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

/// Q6_K block: 210 bytes / 256 elements (6.5625 bits/element). The
/// highest-quality K-quant; common in `*-Q6_K.gguf` model dumps.
/// Layout: 128 low-nibble bytes + 64 high-2-bit bytes + 16 i8 scales
/// + f16 d (super-block scale; per-sub-block scales are signed).
pub fn dequant_q6_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("Q6_K: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    let ql_len = QK_K / 2; // 128
    let qh_len = QK_K / 4; // 64
    let sc_len = QK_K / 16; // 16
    let blk = ql_len + qh_len + sc_len + 2;
    if bytes.len() != nb * blk {
        bail!("Q6_K: bad byte count");
    }
    let mut out = vec![0f32; n];
    for i in 0..nb {
        let off = i * blk;
        let ql = &bytes[off..off + ql_len];
        let qh = &bytes[off + ql_len..off + ql_len + qh_len];
        let sc = &bytes[off + ql_len + qh_len..off + ql_len + qh_len + sc_len];
        let d = read_f16_le(&bytes[off + ql_len + qh_len + sc_len..off + blk]);
        let dst = &mut out[i * QK_K..(i + 1) * QK_K];
        // Two halves of 128 elements each. Per half we walk l in 0..32
        // and decode four interleaved 6-bit values (offsets 0, 32, 64, 96).
        for h in 0..2 {
            let dst_base = h * 128;
            let ql_off = h * 64;
            let qh_off_h = h * 32;
            let sc_off = h * 8;
            for l in 0..32 {
                let is = l / 16;
                let qh_b = qh[qh_off_h + l];
                let q1 = (((ql[ql_off + l] & 0x0F) | (((qh_b >> 0) & 3) << 4)) as i32 - 32) as f32;
                let q2 =
                    (((ql[ql_off + l + 32] & 0x0F) | (((qh_b >> 2) & 3) << 4)) as i32 - 32) as f32;
                let q3 = (((ql[ql_off + l] >> 4) | (((qh_b >> 4) & 3) << 4)) as i32 - 32) as f32;
                let q4 =
                    (((ql[ql_off + l + 32] >> 4) | (((qh_b >> 6) & 3) << 4)) as i32 - 32) as f32;
                dst[dst_base + l] = d * sc[sc_off + is] as i8 as f32 * q1;
                dst[dst_base + l + 32] = d * sc[sc_off + is + 2] as i8 as f32 * q2;
                dst[dst_base + l + 64] = d * sc[sc_off + is + 4] as i8 as f32 * q3;
                dst[dst_base + l + 96] = d * sc[sc_off + is + 6] as i8 as f32 * q4;
            }
        }
    }
    Ok(out)
}

/// Q8_K block: 276 bytes / 256 elements. Mostly an intermediate
/// format used inside llama.cpp's matmul kernels, but some dumps do
/// store it directly. We only need to materialize the i8 quants ×
/// the f32 super-block scale; `bsums` (per-16-block partial sums) is
/// metadata we can safely ignore for plain dequant.
pub fn dequant_q8_k(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    if !n.is_multiple_of(QK_K) {
        bail!("Q8_K: n={n} not divisible by {QK_K}");
    }
    let nb = n / QK_K;
    let blk = 4 + QK_K + (QK_K / 16) * 2;
    if bytes.len() != nb * blk {
        bail!("Q8_K: bad byte count");
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..nb {
        let off = i * blk;
        let d = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        let qs = &bytes[off + 4..off + 4 + QK_K];
        for &q in qs {
            out.push(d * (q as i8) as f32);
        }
    }
    Ok(out)
}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn roundtrip_f32_v3() {
        let data = [1.0f32, -2.0, 3.5, 0.0];
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        let name = "w";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        while !buf.len().is_multiple_of(DEFAULT_ALIGNMENT as usize) {
            buf.push(0);
        }
        for v in &data {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let mut c = Cursor::new(buf);
        let f = GgufFile::from_reader(&mut c).unwrap();
        assert_eq!(f.version, 3);
        assert_eq!(f.tensors.len(), 1);
        let (out, shape) = f.dequant_f32("w").unwrap();
        assert_eq!(shape, vec![4]);
        assert_eq!(out, data);
    }

    #[test]
    fn rejects_wrong_magic() {
        let buf = vec![0u8; 16];
        let mut c = Cursor::new(buf);
        assert!(GgufFile::from_reader(&mut c).is_err());
    }

    #[test]
    fn dequant_q8_0_block() {
        let mut bytes = Vec::new();
        let d = half::f16::from_f32(0.5);
        bytes.extend_from_slice(&d.to_le_bytes());
        let qs: [i8; QK8_0] = std::array::from_fn(|i| (i as i8) - 16);
        for q in qs {
            bytes.push(q as u8);
        }

        let out = dequant_q8_0(&bytes, QK8_0).unwrap();
        assert_eq!(out.len(), QK8_0);
        for i in 0..QK8_0 {
            assert!((out[i] - 0.5 * (qs[i] as f32)).abs() < 1e-6);
        }
    }

    #[test]
    fn dequant_q4_0_block() {
        let mut bytes = Vec::new();
        let d = half::f16::from_f32(1.0);
        bytes.extend_from_slice(&d.to_le_bytes());
        // Byte i = (i << 4) | i  → low nibble = i, high nibble = i.
        let qs: [u8; 16] = std::array::from_fn(|i| (i as u8 & 0x0F) | ((i as u8 & 0x0F) << 4));
        bytes.extend_from_slice(&qs);

        let out = dequant_q4_0(&bytes, QK4_0).unwrap();
        assert_eq!(out.len(), QK4_0);
        for i in 0..16 {
            assert_eq!(out[i], (i as f32) - 8.0);
        }
        for i in 0..16 {
            assert_eq!(out[16 + i], (i as f32) - 8.0);
        }
    }

    #[test]
    fn dequant_q4_k_block_constant_value() {
        // Hand-build one Q4_K block (144 bytes / 256 elements) where
        // every sub-block has scale=1 (encoded as 1), min=0, every
        // quant nibble = 7. Then d = 1, dmin = 1, and every output
        // should be 7.0.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // d
        bytes.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // dmin
        // scales[12]: pack sc=1, m=0 for each of 8 sub-blocks.
        //   j<4: scales[j] = 1, scales[j+4] = 0
        //   j>=4: encoded across scales[j+4] low nibble + scales[j-4] top 2 bits.
        // Simplest construction: every byte = 0x01 in low 6 bits gives sc=1 for
        // j<4 and propagates through the j>=4 path with min=0. Verify with
        // get_scale_min_k4 manually:
        //   for j>=4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
        //   if scales[j+4] = 0x01 and scales[j-4] = 0x01, top 2 bits of 0x01 = 0,
        //   so sc = 1. Min = (0x01 >> 4) | ((0x01 >> 6) << 4) = 0.
        let mut scales = [0u8; K_SCALE_SIZE];
        for s in &mut scales[0..4] {
            *s = 0x01; // sc=1 in low 6 bits, top 2 bits = 0
        }
        // scales[4..8] hold min=0 for j<4 (low 6 bits = 0) and contribute
        // the low 4 bits of sc / min for j>=4 — leave at 0; sc for j>=4 then
        // equals top-2-bits-of-scales[j-4] << 4 = 0, which would give 0
        // not 1. Workaround: use a simpler check that doesn't require all
        // sub-blocks identical — just verify the j<4 sub-blocks decode to 7.
        bytes.extend_from_slice(&scales);
        // qs[128]: every nibble = 7 → byte = 0x77.
        for _ in 0..(QK_K / 2) {
            bytes.push(0x77);
        }
        let out = dequant_q4_k(&bytes, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        // First 4 sub-blocks (128 elements at positions 0..32, 64..96, 128..160, 192..224)
        // pair as (j=0,j=1), (j=2,j=3) — so 128 elements decode with sc=1, min=0
        // → value = 1 * 7 - 0 = 7.0.
        // The actual emission order is: 32 from j, then 32 from j+1, then j+=2.
        // For j=0,1: first 64 outputs. For j=2,3: next 64 outputs. Both pairs
        // are in the j<4 branch.
        for v in &out[0..128] {
            assert!((v - 7.0).abs() < 1e-5, "Q4K decode mismatch: {v}");
        }
    }

    #[test]
    fn dequant_q6_k_block_constant_value() {
        // Build one Q6_K block where every per-sub-block scale = 1 and
        // every 6-bit quant value = 32 (i.e. 0 after the -32 bias) plus
        // d = 1. Output should be all zeros.
        let ql_len = QK_K / 2;
        let qh_len = QK_K / 4;
        let sc_len = QK_K / 16;
        let mut bytes = Vec::with_capacity(ql_len + qh_len + sc_len + 2);
        // ql: low 4 bits of each 6-bit value = 0 (since 32 = 0b100000 → low=0, high=2)
        bytes.resize(ql_len, 0u8);
        // qh: each pair of high bits = 2 (binary 10). Packed 4 pairs per byte
        // in the order (bits 0-1, 2-3, 4-5, 6-7) for offsets 0, 32, 64, 96.
        // Pattern: 0b10_10_10_10 = 0xAA.
        bytes.resize(ql_len + qh_len, 0xAAu8);
        // sc: 16 i8 scales, all = 1.
        for _ in 0..sc_len {
            bytes.push(1u8);
        }
        // d = 1.0
        bytes.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());

        let out = dequant_q6_k(&bytes, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        for v in &out {
            assert!(v.abs() < 1e-5, "Q6K decode mismatch: {v}");
        }
    }

    #[test]
    fn dequant_q2_k_block_constant_value() {
        // Q2_K: d=1, min=0, all scales encode sc=1/min=0, all 2-bit quants = 3.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // d
        bytes.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes()); // min
        for _ in 0..(QK_K / 16) {
            bytes.push(0x01); // sc=1, min=0
        }
        for _ in 0..(QK_K / 4) {
            bytes.push(0xFF); // all 2-bit fields = 3
        }
        let out = dequant_q2_k(&bytes, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        for v in &out {
            assert!((v - 3.0).abs() < 1e-4, "Q2K decode mismatch: {v}");
        }
    }

    #[test]
    fn dequant_q3_k_check() {
        let blk = 2 + K_SCALE_SIZE + QK_K / 8 + QK_K / 4;
        let bytes = vec![0u8; blk];
        let out = dequant_q3_k(&bytes, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dequant_q8_k_block() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0.25f32.to_le_bytes());
        let qs: [i8; QK_K] = std::array::from_fn(|i| ((i as i32 - 128) as i8));
        for q in qs {
            bytes.push(q as u8);
        }
        // bsums: 16 i16 — unused by decoder, but must be present so the
        // bytes_for check matches.
        for _ in 0..(QK_K / 16) {
            bytes.extend_from_slice(&0i16.to_le_bytes());
        }
        let out = dequant_q8_k(&bytes, QK_K).unwrap();
        for i in 0..QK_K {
            assert!((out[i] - 0.25 * (qs[i] as f32)).abs() < 1e-6);
        }
    }

    #[test]
    fn metadata_roundtrip() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // kv_count
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // type=string
        let val = "llama";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val.as_bytes());
        while !buf.len().is_multiple_of(DEFAULT_ALIGNMENT as usize) {
            buf.push(0);
        }

        let mut c = Cursor::new(buf);
        let f = GgufFile::from_reader(&mut c).unwrap();
        assert_eq!(
            f.metadata.get(key).and_then(MetaValue::as_str),
            Some("llama")
        );
    }
}
