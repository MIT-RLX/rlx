// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`WeightSource`]s over checkpoint files (feature `formats`).
//!
//! - [`SafetensorsSource`]: one `.safetensors` file or a directory of shards,
//!   memory-mapped; F32 / F16 / BF16 / F64 and integer tensors read as f32.
//! - [`GgufSource`]: a `.gguf` file, memory-mapped; tensors dequantized to f32,
//!   or with [`GgufSource::packed`] handed over as quant blocks for
//!   `DequantMatMul`. Shapes come out outermost-first like safetensors (GGUF
//!   stores them innermost-first). Keys are the file's own (`blk.0.attn_q.weight`);
//!   map them with [`WeightSourceExt::renamed`](crate::WeightSourceExt::renamed).
//! - [`open`] picks one by path.
//!
//! Both read without consuming, so one source backs any number of builds.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use rlx_gguf::GgufFile;

use crate::weight::{missing_weight, transpose_2d};
use crate::{GgufPackedBank, GgufPackedLinear, WeightSource};

/// A source for `path`: a `.gguf` file, a `.safetensors` file, or a directory
/// of `.safetensors` shards. GGUF opens unpacked (F32); use [`GgufSource`]
/// directly for packed weights.
pub fn open(path: impl AsRef<Path>) -> Result<Box<dyn WeightSource>> {
    let path = path.as_ref();
    if path.is_dir() || path.extension().is_some_and(|e| e == "safetensors") {
        return Ok(Box::new(SafetensorsSource::open(path)?));
    }
    if path.extension().is_some_and(|e| e == "gguf") {
        return Ok(Box::new(GgufSource::open(path)?));
    }
    bail!(
        "{}: not a .gguf, a .safetensors file or a directory of shards",
        path.display()
    )
}

#[derive(Debug, Clone)]
struct StEntry {
    file: usize,
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// Safetensors file(s) as a [`WeightSource`]. See the [module docs](self).
pub struct SafetensorsSource {
    maps: Vec<(PathBuf, Mmap, usize)>,
    tensors: HashMap<String, StEntry>,
}

impl SafetensorsSource {
    /// `path` is a `.safetensors` file or a directory; a directory contributes
    /// every `*.safetensors` file in it (a sharded checkpoint), and a name in
    /// two shards is an error.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let files = if path.is_dir() {
            let mut f: Vec<PathBuf> = std::fs::read_dir(path)
                .with_context(|| format!("read {}", path.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect();
            f.sort();
            if f.is_empty() {
                bail!("{}: no .safetensors files", path.display());
            }
            f
        } else {
            vec![path.to_path_buf()]
        };
        let mut src = Self {
            maps: Vec::new(),
            tensors: HashMap::new(),
        };
        for p in files {
            src.add_file(p)?;
        }
        Ok(src)
    }

    fn add_file(&mut self, path: PathBuf) -> Result<()> {
        let f = std::fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the mapping is read-only; a checkpoint is not expected to be
        // rewritten while it is being read.
        let map = unsafe { Mmap::map(&f) }.with_context(|| format!("mmap {}", path.display()))?;
        let bad = |what: &str| anyhow!("{}: {what}", path.display());
        let hlen = map
            .get(..8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize)
            .ok_or_else(|| bad("too small for a safetensors header"))?;
        let header = map
            .get(8..8 + hlen)
            .ok_or_else(|| bad("truncated header"))?;
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(header).map_err(|e| bad(&format!("header: {e}")))?;
        let file = self.maps.len();
        for (name, t) in header {
            if name == "__metadata__" {
                continue;
            }
            let field = |k: &str| t.get(k).ok_or_else(|| bad(&format!("{name}: no {k}")));
            let dtype = field("dtype")?
                .as_str()
                .ok_or_else(|| bad(&format!("{name}: dtype")))?
                .to_string();
            let shape = field("shape")?
                .as_array()
                .and_then(|a| a.iter().map(|d| d.as_u64().map(|d| d as usize)).collect())
                .ok_or_else(|| bad(&format!("{name}: shape")))?;
            let off: Vec<usize> = field("data_offsets")?
                .as_array()
                .and_then(|a| a.iter().map(|d| d.as_u64().map(|d| d as usize)).collect())
                .filter(|o: &Vec<usize>| o.len() == 2 && o[0] <= o[1])
                .ok_or_else(|| bad(&format!("{name}: data_offsets")))?;
            let base = 8 + hlen;
            if base + off[1] > map.len() {
                return Err(bad(&format!("{name}: data past the end of the file")));
            }
            let entry = StEntry {
                file,
                dtype,
                shape,
                start: base + off[0],
                end: base + off[1],
            };
            if self.tensors.insert(name.clone(), entry).is_some() {
                return Err(bad(&format!("{name} is in two shards")));
            }
        }
        self.maps.push((path, map, hlen));
        Ok(())
    }

    /// The tensor's `(dtype, shape)` as stored, without reading it.
    pub fn info(&self, key: &str) -> Option<(&str, &[usize])> {
        self.tensors
            .get(key)
            .map(|e| (e.dtype.as_str(), e.shape.as_slice()))
    }

    fn read(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let e = self
            .tensors
            .get(key)
            .ok_or_else(|| missing_weight(key, &self.keys()))?;
        let raw = &self.maps[e.file].1[e.start..e.end];
        let n: usize = e.shape.iter().product();
        macro_rules! decode {
            ($w:literal, $f:expr) => {{
                let (chunks, _) = raw.as_chunks::<$w>();
                chunks.iter().map($f).collect::<Vec<f32>>()
            }};
        }
        let data = match e.dtype.as_str() {
            "F32" => decode!(4, |c| f32::from_le_bytes(*c)),
            "F16" => decode!(2, |c| half::f16::from_le_bytes(*c).to_f32()),
            "BF16" => decode!(2, |c| half::bf16::from_le_bytes(*c).to_f32()),
            "F64" => decode!(8, |c| f64::from_le_bytes(*c) as f32),
            "I64" => decode!(8, |c| i64::from_le_bytes(*c) as f32),
            "I32" => decode!(4, |c| i32::from_le_bytes(*c) as f32),
            "I16" => decode!(2, |c| i16::from_le_bytes(*c) as f32),
            "I8" => decode!(1, |c| c[0] as i8 as f32),
            "U8" | "BOOL" => decode!(1, |c| c[0] as f32),
            other => bail!("{key}: unsupported safetensors dtype {other}"),
        };
        if data.len() != n {
            bail!(
                "{key}: {} bytes of {} hold {} values, shape {:?} needs {n}",
                raw.len(),
                e.dtype,
                data.len(),
                e.shape
            );
        }
        Ok((data, e.shape.clone()))
    }
}

impl WeightSource for SafetensorsSource {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self.read(key)?;
        if transpose {
            transpose_2d(&data, &shape, key)
        } else {
            Ok((data, shape))
        }
    }
    fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }
    fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.tensors.keys().cloned().collect();
        k.sort();
        k
    }
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        Some(match self.tensors.get(key)?.dtype.as_str() {
            "F32" => rlx_ir::DType::F32,
            "F16" => rlx_ir::DType::F16,
            "BF16" => rlx_ir::DType::BF16,
            _ => return None,
        })
    }
}

/// A GGUF file as a [`WeightSource`]. See the [module docs](self).
pub struct GgufSource {
    file: GgufFile,
    packed: bool,
}

impl GgufSource {
    /// Memory-map `path`. Serves F32 until [`packed`](Self::packed) is set.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file =
            GgufFile::from_path_mmap(path).with_context(|| format!("open {}", path.display()))?;
        Ok(Self {
            file,
            packed: false,
        })
    }

    /// Hand quantized linears (rank 2) and expert banks (rank 3) to the
    /// builder as packed blocks, for every type with an RLX `QuantScheme`.
    /// Float tensors are still served as F32.
    pub fn packed(mut self, on: bool) -> Self {
        self.packed = on;
        self
    }

    /// The parsed file: metadata (`general.architecture`, hyperparameters),
    /// tensor types.
    pub fn file(&self) -> &GgufFile {
        &self.file
    }

    /// Packed bytes, scheme and GGML (innermost-first) shape of `key`, when
    /// packing is on and the type has a scheme.
    fn packed_parts(
        &self,
        key: &str,
        rank: usize,
    ) -> Result<Option<(Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>> {
        if !self.packed {
            return Ok(None);
        }
        let Some(t) = self.file.get(key) else {
            return Ok(None);
        };
        let Some(scheme) = rlx_gguf::quant_scheme(t.dtype) else {
            return Ok(None);
        };
        if t.shape.len() != rank {
            return Ok(None);
        }
        let bytes = self.file.tensor_bytes(t)?.to_vec();
        Ok(Some((bytes, scheme, t.shape.clone())))
    }
}

impl WeightSource for GgufSource {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        if self.file.get(key).is_none() {
            return Err(missing_weight(key, &self.keys()));
        }
        let (data, mut shape) = self.file.dequant_f32(key)?;
        shape.reverse();
        if transpose {
            transpose_2d(&data, &shape, key)
        } else {
            Ok((data, shape))
        }
    }
    fn has(&self, key: &str) -> bool {
        self.file.get(key).is_some()
    }
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        Some(match self.file.get(key)?.dtype {
            rlx_gguf::GgmlType::F32 => rlx_ir::DType::F32,
            rlx_gguf::GgmlType::F16 => rlx_ir::DType::F16,
            rlx_gguf::GgmlType::BF16 => rlx_ir::DType::BF16,
            _ => return None,
        })
    }
    fn has_packed(&self, key: &str) -> bool {
        self.packed
            && self
                .file
                .get(key)
                .is_some_and(|t| rlx_gguf::quant_scheme(t.dtype).is_some())
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        Ok(self
            .packed_parts(key, 2)?
            .map(|(w_q, scheme, s)| GgufPackedLinear {
                w_q,
                scheme,
                in_dim: s[0],
                out_dim: s[1],
                bias: Vec::new(),
            }))
    }
    fn take_packed_bank(&mut self, key: &str) -> Result<Option<GgufPackedBank>> {
        Ok(self
            .packed_parts(key, 3)?
            .map(|(w_q, scheme, s)| GgufPackedBank {
                w_q,
                scheme,
                in_dim: s[0],
                out_dim: s[1],
                num_experts: s[2],
            }))
    }
    fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.file.tensors.keys().cloned().collect();
        k.sort();
        k
    }
}
