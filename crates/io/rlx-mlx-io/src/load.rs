// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Path dispatch + mlx-community directory / safetensors loading.

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use crate::config::{MlxConfig, MlxQuantMode};
use crate::dequant::{QuantizedLayer, dequant_affine_f32, dequant_mxfp4_f32, dequant_mxfp8_f32};
use crate::dtype::decode_f32;
use crate::npz::{load_npy, load_npz};
use rlx_ir::DType;
use rlx_ir::QuantScheme;

/// One tensor from an MLX weight dump (dense f32 and/or raw bytes).
#[derive(Debug, Clone)]
pub struct MlxTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_f32: Option<Vec<f32>>,
    pub data_u8: Option<Vec<u8>>,
    /// True when bytes are MLX packed quant codes (u32/u8), not dense floats.
    pub is_quant_weight: bool,
}

/// Loaded MLX weights + optional quantization config.
#[derive(Debug, Clone)]
pub struct MlxWeights {
    pub tensors: HashMap<String, MlxTensor>,
    pub config: MlxConfig,
    pub source: PathBuf,
}

impl MlxWeights {
    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<_> = self.tensors.keys().cloned().collect();
        n.sort();
        n
    }

    pub fn get(&self, name: &str) -> Option<&MlxTensor> {
        self.tensors.get(name)
    }

    /// Dense f32 map. Quantized Linear/Embedding triples are fused and
    /// dequantized under the logical base name (strip `.weight`).
    ///
    /// Shapes are discarded — prefer [`Self::into_shaped_f32`] for packaging.
    pub fn into_f32_map(self) -> Result<HashMap<String, Vec<f32>>> {
        Ok(self
            .into_shaped_f32()?
            .into_iter()
            .map(|(k, (_shape, data))| (k, data))
            .collect())
    }

    /// Dense f32 map with logical shapes preserved (`[rows, cols]` after dequant).
    pub fn into_shaped_f32(self) -> Result<HashMap<String, crate::ShapedF32>> {
        let config = self.config.clone();
        let mut tensors = self.tensors;
        let mut out = HashMap::new();

        // Collect quantized layer bases: name ending in `.weight` with siblings.
        let weight_keys: Vec<String> = tensors
            .keys()
            .filter(|k| k.ends_with(".weight"))
            .cloned()
            .collect();

        for wk in weight_keys {
            let base = wk.trim_end_matches(".weight").to_string();
            let scales_k = format!("{base}.scales");
            let biases_k = format!("{base}.biases");
            let has_scales = tensors.contains_key(&scales_k);
            if !has_scales {
                continue;
            }
            let w = tensors.remove(&wk).context("weight")?;
            let scales_t = tensors.remove(&scales_k).context("scales")?;
            let biases_t = tensors.remove(&biases_k);

            let qcfg = config.quant_for(&base).unwrap_or_else(|| {
                crate::config::MlxQuantConfig::defaults_for(MlxQuantMode::Affine)
            });

            let rows = scales_t.shape.first().copied().unwrap_or(0);
            let n_groups = scales_t.shape.get(1).copied().unwrap_or(1);
            let cols = n_groups * qcfg.group_size as usize;
            let layer = QuantizedLayer {
                weight: w
                    .data_u8
                    .clone()
                    .or_else(|| {
                        w.data_f32.as_ref().map(|f| {
                            let mut b = Vec::with_capacity(f.len() * 4);
                            for x in f {
                                b.extend_from_slice(&x.to_bits().to_le_bytes());
                            }
                            b
                        })
                    })
                    .unwrap_or_default(),
                weight_shape: w.shape.clone(),
                scales: scales_t.data_f32.clone().unwrap_or_default(),
                scales_shape: scales_t.shape.clone(),
                biases: biases_t.as_ref().and_then(|t| t.data_f32.clone()),
                biases_shape: biases_t.as_ref().map(|t| t.shape.clone()),
                bits: qcfg.bits,
                group_size: qcfg.group_size,
            };

            let dense = match qcfg.mode {
                MlxQuantMode::Affine => {
                    let biases = layer
                        .biases
                        .clone()
                        .unwrap_or_else(|| vec![0.0; rows * n_groups]);
                    let scales = if layer.scales.is_empty() {
                        bail!("affine layer {base}: missing f32 scales");
                    } else {
                        layer.scales.clone()
                    };
                    dequant_affine_f32(
                        &layer.weight,
                        &scales,
                        &biases,
                        layer.bits,
                        layer.group_size,
                        rows,
                        n_groups,
                    )?
                }
                MlxQuantMode::Mxfp4 => {
                    let scales_u8 = scales_t
                        .data_u8
                        .clone()
                        .or_else(|| {
                            scales_t
                                .data_f32
                                .as_ref()
                                .map(|f| f.iter().map(|x| x.to_bits() as u8).collect())
                        })
                        .context("mxfp4 scales")?;
                    dequant_mxfp4_f32(&layer.weight, &scales_u8, qcfg.group_size, rows, n_groups)?
                }
                MlxQuantMode::Mxfp8 => {
                    let scales_u8 = scales_t.data_u8.clone().context("mxfp8 scales")?;
                    dequant_mxfp8_f32(&layer.weight, &scales_u8, qcfg.group_size, rows, n_groups)?
                }
                MlxQuantMode::Nvfp4 => {
                    let scales_u8 = scales_t.data_u8.clone().context("nvfp4 scales")?;
                    dequant_mxfp4_f32(&layer.weight, &scales_u8, qcfg.group_size, rows, n_groups)?
                }
            };
            out.insert(base, (vec![rows, cols], dense));
        }

        for (name, t) in tensors {
            if let Some(f) = t.data_f32 {
                out.insert(name, (t.shape, f));
            } else if t.data_u8.is_some() {
                bail!("tensor {name} has packed bytes but no dequant siblings");
            }
        }
        Ok(out)
    }

    /// Map logical Linear bases to [`QuantScheme`] for packed DequantMatMul graphs.
    pub fn quant_scheme(&self) -> Option<QuantScheme> {
        let q = self.config.quantization.as_ref()?;
        Some(match q.mode {
            MlxQuantMode::Affine => QuantScheme::MlxAffine {
                bits: q.bits as u8,
                group_size: q.group_size,
            },
            MlxQuantMode::Mxfp4 | MlxQuantMode::Nvfp4 => QuantScheme::MlxMxfp4 {
                group_size: q.group_size,
            },
            MlxQuantMode::Mxfp8 => QuantScheme::MlxMxfp8 {
                group_size: q.group_size,
            },
        })
    }

    /// Quant config for a specific layer base — per-module override (mixed
    /// precision, e.g. gpt-oss) or the global config.
    fn qcfg_for(&self, base: &str) -> crate::config::MlxQuantConfig {
        self.config
            .quant_for(base)
            .unwrap_or_else(|| crate::config::MlxQuantConfig::defaults_for(MlxQuantMode::Affine))
    }

    /// [`QuantScheme`] for a specific layer base (per-module aware).
    fn quant_scheme_for(&self, base: &str) -> Option<QuantScheme> {
        let q = self.config.quant_for(base)?;
        Some(match q.mode {
            MlxQuantMode::Affine => QuantScheme::MlxAffine {
                bits: q.bits as u8,
                group_size: q.group_size,
            },
            MlxQuantMode::Mxfp4 | MlxQuantMode::Nvfp4 => QuantScheme::MlxMxfp4 {
                group_size: q.group_size,
            },
            MlxQuantMode::Mxfp8 => QuantScheme::MlxMxfp8 {
                group_size: q.group_size,
            },
        })
    }

    /// Logical (primary) tensor names — everything that is not a
    /// `.scales` / `.biases` sidecar. These are the keys a model builder
    /// addresses (`*.weight`, norm weights, embeddings).
    pub fn logical_keys(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .tensors
            .keys()
            .filter(|k| !k.ends_with(".scales") && !k.ends_with(".biases"))
            .cloned()
            .collect();
        v.sort();
        v
    }

    /// True if `hf_key` names a quantized Linear/Embedding (a `{base}.weight`
    /// with a `{base}.scales` sibling still present).
    pub fn is_quantized_layer(&self, hf_key: &str) -> bool {
        hf_key
            .strip_suffix(".weight")
            .is_some_and(|base| self.tensors.contains_key(&format!("{base}.scales")))
    }

    /// Take one tensor as dense f32, dequantizing in place when `hf_key`
    /// names a quantized layer. Consumed tensors are removed so memory is
    /// freed as the caller drains.
    pub fn take_dense_f32(&mut self, hf_key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        if self.is_quantized_layer(hf_key) {
            let base = hf_key.trim_end_matches(".weight").to_string();
            return self.dequant_layer(&base);
        }
        let t = self
            .tensors
            .remove(hf_key)
            .with_context(|| format!("mlx tensor not found: {hf_key}"))?;
        let data = t
            .data_f32
            .with_context(|| format!("mlx tensor {hf_key} has no f32 data"))?;
        Ok((data, t.shape))
    }

    /// Dequantize quantized layer `{base}` to a dense `[rows, cols]` f32
    /// matrix, consuming `{base}.weight/.scales/.biases`.
    fn dequant_layer(&mut self, base: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let qcfg = self.qcfg_for(base);
        let w = self
            .tensors
            .remove(&format!("{base}.weight"))
            .with_context(|| format!("weight {base}.weight"))?;
        let scales_t = self
            .tensors
            .remove(&format!("{base}.scales"))
            .with_context(|| format!("scales {base}.scales"))?;
        let biases_t = self.tensors.remove(&format!("{base}.biases"));
        let rows = scales_t.shape.first().copied().unwrap_or(0);
        let n_groups = scales_t.shape.get(1).copied().unwrap_or(1);
        let cols = n_groups * qcfg.group_size as usize;
        let w_bytes = w.data_u8.unwrap_or_default();
        let dense = match qcfg.mode {
            MlxQuantMode::Affine => {
                let scales = scales_t
                    .data_f32
                    .with_context(|| format!("affine {base}: f32 scales"))?;
                let biases = biases_t
                    .and_then(|t| t.data_f32)
                    .unwrap_or_else(|| vec![0.0; rows * n_groups]);
                dequant_affine_f32(
                    &w_bytes,
                    &scales,
                    &biases,
                    qcfg.bits,
                    qcfg.group_size,
                    rows,
                    n_groups,
                )?
            }
            MlxQuantMode::Mxfp4 | MlxQuantMode::Nvfp4 => {
                let scales_u8 = mxfp_scale_bytes(&scales_t)?;
                dequant_mxfp4_f32(&w_bytes, &scales_u8, qcfg.group_size, rows, n_groups)?
            }
            MlxQuantMode::Mxfp8 => {
                let scales_u8 = mxfp_scale_bytes(&scales_t)?;
                dequant_mxfp8_f32(&w_bytes, &scales_u8, qcfg.group_size, rows, n_groups)?
            }
        };
        Ok((dense, vec![rows, cols]))
    }

    /// Take a quantized Linear as packed MLX triples for
    /// `Op::DequantMatMul`. Supports [`QuantScheme::MlxAffine`],
    /// [`MlxMxfp4`] (including mlx-lm `nvfp4`, which shares the mxfp4
    /// nibble layout with `group_size` typically 16), and [`MlxMxfp8`].
    /// Consumes `{base}.weight/.scales/.biases`.
    pub fn take_packed_linear(&mut self, hf_key: &str) -> Result<Option<MlxPackedLinear>> {
        if !self.is_quantized_layer(hf_key) {
            return Ok(None);
        }
        let base = hf_key.trim_end_matches(".weight").to_string();
        // Per-module scheme (mixed precision) — NOT the global one, else affine
        // attn/embed tensors get mis-decoded as the global mxfp4 (gpt-oss).
        let Some(scheme) = self.quant_scheme_for(&base) else {
            return Ok(None);
        };
        if !scheme.is_mlx() {
            return Ok(None);
        }
        let qcfg = self.qcfg_for(&base);
        let w = self
            .tensors
            .remove(hf_key)
            .with_context(|| format!("weight {hf_key}"))?;
        let scales_t = self
            .tensors
            .remove(&format!("{base}.scales"))
            .with_context(|| format!("scales {base}.scales"))?;
        let biases_t = self.tensors.remove(&format!("{base}.biases"));
        let rows = scales_t.shape.first().copied().unwrap_or(0);
        let n_groups = scales_t.shape.get(1).copied().unwrap_or(1);
        let cols = n_groups * qcfg.group_size as usize;
        let w_q = w
            .data_u8
            .with_context(|| format!("{hf_key}: no packed bytes"))?;
        let (scales, biases) = match scheme {
            QuantScheme::MlxAffine { .. } => {
                let scales_f = scales_t
                    .data_f32
                    .with_context(|| format!("affine {base}: f32 scales"))?;
                let biases_f = biases_t
                    .and_then(|t| t.data_f32)
                    .unwrap_or_else(|| vec![0.0; rows * n_groups]);
                (f32_le_bytes(&scales_f), f32_le_bytes(&biases_f))
            }
            QuantScheme::MlxMxfp4 { .. } | QuantScheme::MlxMxfp8 { .. } => {
                let scales_u8 = mxfp_scale_bytes(&scales_t)?;
                // Unused zp slot — 4 zero bytes so set_param_typed has a buffer.
                (scales_u8, vec![0u8; 4])
            }
            _ => unreachable!(),
        };
        Ok(Some(MlxPackedLinear {
            w_q,
            scales,
            biases,
            scheme,
            out_shape: vec![rows, cols],
        }))
    }
}

fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Recover raw per-group scale bytes for mxfp modes. `load_safetensors_bytes`
/// may have widened uint8 E8M0/FP8 scale bytes to f32 (byte value); undo that.
fn mxfp_scale_bytes(scales_t: &MlxTensor) -> Result<Vec<u8>> {
    if let Some(u) = scales_t.data_u8.clone() {
        return Ok(u);
    }
    scales_t
        .data_f32
        .as_ref()
        .map(|f| f.iter().map(|x| *x as u8).collect())
        .context("mxfp scales: neither u8 nor f32 present")
}

/// One quantized Linear as packed MLX triples, ready for
/// `Op::DequantMatMul { scheme }` with inputs `[x, w_q, scales, biases]`.
#[derive(Debug, Clone)]
pub struct MlxPackedLinear {
    /// Packed quant codes (raw little-endian bytes; uint32 words for affine).
    pub w_q: Vec<u8>,
    /// Affine: f32 LE bytes. Mxfp: one u8 scale per group (`[n * n_groups]`).
    pub scales: Vec<u8>,
    /// Affine: f32 LE biases. Mxfp: dummy zeros (zp unused).
    pub biases: Vec<u8>,
    pub scheme: QuantScheme,
    /// Logical dense output shape `[out_features, in_features]`.
    pub out_shape: Vec<usize>,
}

impl MlxPackedLinear {
    /// DType for the scale param (F32 affine, U8 mxfp).
    pub fn scale_dtype(&self) -> DType {
        match self.scheme {
            QuantScheme::MlxAffine { .. } => DType::F32,
            QuantScheme::MlxMxfp4 { .. }
            | QuantScheme::MlxMxfp8 { .. }
            | QuantScheme::Nvfp4Block => DType::U8,
            _ => DType::F32,
        }
    }

    /// DType for the bias / zp slot (F32 affine; U8 dummy for mxfp).
    pub fn bias_dtype(&self) -> DType {
        match self.scheme {
            QuantScheme::MlxAffine { .. } => DType::F32,
            _ => DType::U8,
        }
    }

    /// Groups along K (`in_features / group_size`).
    pub fn n_groups(&self) -> usize {
        let gs = self.scheme.mlx_group_size() as usize;
        let k = self.out_shape.get(1).copied().unwrap_or(0);
        k.checked_div(gs).unwrap_or(0)
    }
}

fn load_safetensors_bytes(bytes: &[u8]) -> Result<HashMap<String, MlxTensor>> {
    let st = SafeTensors::deserialize(bytes).context("parse safetensors")?;
    let mut out = HashMap::new();
    for name in st.names() {
        let view = st.tensor(name)?;
        let t = tensor_from_view(name, view.shape().to_vec(), view.dtype(), view.data())?;
        out.insert(name.to_string(), t);
    }
    Ok(out)
}

/// Decode one safetensors tensor (`raw` bytes + dtype + shape) into an
/// [`MlxTensor`] (dense f32, or raw u8 for quant packs). Shared by the eager
/// [`load_safetensors_bytes`] and the lazy [`LazyMlxWeights`] so both classify
/// float-vs-quant identically.
fn tensor_from_view(
    name: &str,
    shape: Vec<usize>,
    dtype: safetensors::Dtype,
    raw: &[u8],
) -> Result<MlxTensor> {
    {
        let (data_f32, data_u8, is_quant_weight) = match dtype {
            safetensors::Dtype::U8 | safetensors::Dtype::I8 => (None, Some(raw.to_vec()), true),
            safetensors::Dtype::U16
            | safetensors::Dtype::I16
            | safetensors::Dtype::U32
            | safetensors::Dtype::I32
            | safetensors::Dtype::U64
            | safetensors::Dtype::I64
                if name.ends_with(".weight")
                    || name.ends_with(".scales")
                    || name.ends_with(".biases") =>
            {
                // Keep raw for quant packs; also try f32 decode for biases/scales when float.
                (None, Some(raw.to_vec()), true)
            }
            safetensors::Dtype::F16
            | safetensors::Dtype::BF16
            | safetensors::Dtype::F32
            | safetensors::Dtype::F64 => (Some(decode_f32(raw, dtype)?), None, false),
            _ => {
                // Prefer f32 decode; fall back to raw bytes.
                match decode_f32(raw, dtype) {
                    Ok(f) => (Some(f), None, false),
                    Err(_) => (None, Some(raw.to_vec()), true),
                }
            }
        };
        // Float scales/biases in mlx affine are usually F16/BF16/F32.
        let (data_f32, data_u8, is_quant_weight) = if (name.ends_with(".scales")
            || name.ends_with(".biases"))
            && data_f32.is_none()
            && data_u8.is_some()
        {
            match decode_f32(raw, dtype) {
                Ok(f) => (Some(f), None, false),
                Err(_) => (data_f32, data_u8, is_quant_weight),
            }
        } else {
            (data_f32, data_u8, is_quant_weight)
        };
        Ok(MlxTensor {
            name: name.to_string(),
            shape,
            data_f32,
            data_u8,
            is_quant_weight,
        })
    }
}

fn load_safetensors_file(path: &Path) -> Result<HashMap<String, MlxTensor>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    load_safetensors_bytes(&bytes)
}

/// Unique, sorted `.safetensors` shard files in an mlx dir (from the
/// `model.safetensors.index.json` weight_map, else a directory glob).
fn gather_shard_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = dir.join("model.safetensors.index.json");
    let mut files: Vec<PathBuf> = Vec::new();
    if index_path.is_file() {
        let idx: serde_json::Value = serde_json::from_slice(&fs::read(&index_path)?)?;
        if let Some(map) = idx.get("weight_map").and_then(|v| v.as_object()) {
            let mut set = std::collections::BTreeSet::new();
            for v in map.values() {
                if let Some(s) = v.as_str() {
                    // A pipeline node holds only ITS layer slice's shards. Skip
                    // shards the index names but that aren't present here, rather
                    // than failing to open the whole checkpoint.
                    let p = dir.join(s);
                    if p.is_file() {
                        set.insert(p);
                    }
                }
            }
            files.extend(set);
        }
    }
    if files.is_empty() {
        for ent in fs::read_dir(dir)? {
            let p = ent?.path();
            if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                files.push(p);
            }
        }
        files.sort();
    }
    if files.is_empty() {
        bail!("no safetensors files in {}", dir.display());
    }
    Ok(files)
}

/// Common read surface shared by the eager [`MlxWeights`] and the lazy
/// [`LazyMlxWeights`] — the four operations rlx-models' `MlxLoader` needs.
pub trait MlxRead: Send {
    fn mlx_config(&self) -> &MlxConfig;
    fn logical_keys(&self) -> Vec<String>;
    fn take_dense_f32(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)>;
    fn take_packed_linear(&mut self, key: &str) -> Result<Option<MlxPackedLinear>>;
    /// `MADV_WILLNEED`-prefetch these tensors so their shard pages read ahead while
    /// the caller compiles its stage (overlap IO with compute). Default no-op
    /// (eager loaders already hold the data); the lazy mmap loader overrides it.
    fn prewarm(&self, _keys: &[&str]) {}
}

impl MlxRead for MlxWeights {
    fn mlx_config(&self) -> &MlxConfig {
        &self.config
    }
    fn logical_keys(&self) -> Vec<String> {
        MlxWeights::logical_keys(self)
    }
    fn take_dense_f32(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        MlxWeights::take_dense_f32(self, key)
    }
    fn take_packed_linear(&mut self, key: &str) -> Result<Option<MlxPackedLinear>> {
        MlxWeights::take_packed_linear(self, key)
    }
}

/// **Lazy, mmap-backed** MLX weights. `open` mmaps each shard and indexes tensor
/// names from the safetensors headers (no data copied); a `take` materializes
/// only the tensor(s) it needs. Peak resident RAM is one tensor (+ any dequant
/// scratch), so a worker can load its shard of a checkpoint far larger than its
/// own RAM — the enabler for running a 96 GB model across small nodes.
/// `MADV_WILLNEED` for the current platform (0 on non-unix, where `advise`
/// ignores it).
#[cfg(unix)]
#[inline]
fn libc_madv_willneed() -> i32 {
    libc::MADV_WILLNEED
}
#[cfg(not(unix))]
#[inline]
fn libc_madv_willneed() -> i32 {
    0
}

/// Cached location of one tensor within its shard mmap — parsed ONCE at open so
/// [`LazyMlxWeights::materialize`] slices directly instead of re-deserializing the
/// (multi-MB) safetensors header on every `take`. `begin`/`end` are byte offsets
/// into the shard mmap; `dtype`/`shape` reproduce the header view.
struct TensorLoc {
    shard: usize,
    begin: usize,
    end: usize,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
}

pub struct LazyMlxWeights {
    shards: Vec<Mmap>,
    locs: HashMap<String, TensorLoc>,
    pub config: MlxConfig,
    pub source: PathBuf,
}

impl LazyMlxWeights {
    fn from_shards(shard_paths: Vec<PathBuf>, config: MlxConfig, source: PathBuf) -> Result<Self> {
        let mut shards = Vec::with_capacity(shard_paths.len());
        let mut locs: HashMap<String, TensorLoc> = HashMap::new();
        for (idx, p) in shard_paths.iter().enumerate() {
            let file = File::open(p).with_context(|| format!("open {}", p.display()))?;
            // SAFETY: read-only mapping of a file we own for the loader's lifetime.
            let mmap =
                unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", p.display()))?;
            {
                let base = mmap.as_ptr() as usize;
                let st = SafeTensors::deserialize(&mmap[..])
                    .with_context(|| format!("safetensors header {}", p.display()))?;
                // Parse the header ONCE: record each tensor's byte range in the mmap
                // so `materialize` never re-deserializes. `view.data()` points into
                // this mmap, so its offset is `ptr - base`.
                for name in st.names() {
                    let view = st.tensor(name)?;
                    let begin = view.data().as_ptr() as usize - base;
                    locs.insert(
                        name.to_string(),
                        TensorLoc {
                            shard: idx,
                            begin,
                            end: begin + view.data().len(),
                            dtype: view.dtype(),
                            shape: view.shape().to_vec(),
                        },
                    );
                }
            } // header borrow released before moving the mmap
            shards.push(mmap);
        }
        Ok(Self {
            shards,
            locs,
            config,
            source,
        })
    }

    /// Copy just this tensor's bytes out of its mmap → dense/quant [`MlxTensor`],
    /// using the cached header offset (no per-take header re-parse).
    fn materialize(&self, name: &str) -> Result<MlxTensor> {
        let loc = self
            .locs
            .get(name)
            .with_context(|| format!("mlx tensor not found: {name}"))?;
        let data = &self.shards[loc.shard][loc.begin..loc.end];
        // Hint the OS to prefetch this tensor's pages (overlap readahead with the
        // copy) just before touching them.
        Self::advise(data, libc_madv_willneed());
        let out = tensor_from_view(name, loc.shape.clone(), loc.dtype, data);
        // Drop this tensor's mmap pages once copied. SAFETY: the shard mmap is
        // read-only and never written, so MADV_DONTNEED (drop the cached pages;
        // re-fault from the file if touched again) cannot lose data. It bounds the
        // page-cache footprint during the build — otherwise materializing a big
        // pipeline stage accumulates the whole shard set (~40GB) in cache ALONGSIDE
        // the arena (~39GB), the OS swaps the arena, and every forward pages it
        // back in (14s compute became 140s wall).
        Self::advise_dontneed(data);
        out
    }

    /// `MADV_WILLNEED`-prefetch a batch of tensors before a stage build so their
    /// pages read ahead while the graph compiles (overlap IO with compute). No-op
    /// on non-unix. Safe to call with names this loader doesn't hold.
    pub fn prewarm(&self, names: &[&str]) {
        for n in names {
            if let Some(loc) = self.locs.get(*n) {
                Self::advise(
                    &self.shards[loc.shard][loc.begin..loc.end],
                    libc_madv_willneed(),
                );
            }
        }
    }

    /// `madvise` a byte slice (page-aligned inward so we never advise a neighbour's
    /// shared page). `WILLNEED` covers the tensor (align start down, end up);
    /// generic helper used for prefetch.
    #[inline]
    fn advise(data: &[u8], advice: i32) {
        #[cfg(unix)]
        unsafe {
            if data.is_empty() {
                return;
            }
            let page = 4096usize;
            let start = data.as_ptr() as usize & !(page - 1); // down
            let end = (data.as_ptr() as usize + data.len() + page - 1) & !(page - 1); // up
            libc::madvise(start as *mut libc::c_void, end - start, advice);
        }
        #[cfg(not(unix))]
        let _ = (data, advice);
    }

    /// `MADV_DONTNEED` a byte slice (page-aligned start UP so we never drop a
    /// neighbour's page that shares the first partial page).
    #[inline]
    fn advise_dontneed(data: &[u8]) {
        #[cfg(unix)]
        unsafe {
            let page = 4096usize;
            let start = data.as_ptr() as usize;
            let aligned = (start + page - 1) & !(page - 1);
            let end = start + data.len();
            if end > aligned {
                libc::madvise(
                    aligned as *mut libc::c_void,
                    end - aligned,
                    libc::MADV_DONTNEED,
                );
            }
        }
        #[cfg(not(unix))]
        let _ = data;
    }

    fn is_quantized_layer(&self, key: &str) -> bool {
        key.strip_suffix(".weight")
            .is_some_and(|b| self.locs.contains_key(&format!("{b}.scales")))
    }

    /// Temp eager [`MlxWeights`] holding just `names` (materialized) so we can
    /// reuse the existing dequant / packed logic verbatim.
    fn temp_with(&self, names: &[&str]) -> Result<MlxWeights> {
        let mut tensors = HashMap::new();
        for n in names {
            if self.locs.contains_key(*n) {
                tensors.insert(n.to_string(), self.materialize(n)?);
            }
        }
        Ok(MlxWeights {
            tensors,
            config: self.config.clone(),
            source: self.source.clone(),
        })
    }

    pub fn logical_keys(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .locs
            .keys()
            .filter(|k| !k.ends_with(".scales") && !k.ends_with(".biases"))
            .cloned()
            .collect();
        v.sort();
        v
    }

    pub fn take_dense_f32(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        if self.is_quantized_layer(key) {
            let base = key.trim_end_matches(".weight");
            let mut temp =
                self.temp_with(&[key, &format!("{base}.scales"), &format!("{base}.biases")])?;
            return temp.take_dense_f32(key);
        }
        let t = self.materialize(key)?;
        let data = t
            .data_f32
            .with_context(|| format!("mlx tensor {key} has no f32 data"))?;
        Ok((data, t.shape))
    }

    pub fn take_packed_linear(&mut self, key: &str) -> Result<Option<MlxPackedLinear>> {
        if !self.is_quantized_layer(key) {
            return Ok(None);
        }
        let base = key.trim_end_matches(".weight");
        let mut temp =
            self.temp_with(&[key, &format!("{base}.scales"), &format!("{base}.biases")])?;
        temp.take_packed_linear(key)
    }
}

impl MlxRead for LazyMlxWeights {
    fn mlx_config(&self) -> &MlxConfig {
        &self.config
    }
    fn logical_keys(&self) -> Vec<String> {
        LazyMlxWeights::logical_keys(self)
    }
    fn take_dense_f32(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        LazyMlxWeights::take_dense_f32(self, key)
    }
    fn take_packed_linear(&mut self, key: &str) -> Result<Option<MlxPackedLinear>> {
        LazyMlxWeights::take_packed_linear(self, key)
    }
    fn prewarm(&self, keys: &[&str]) {
        LazyMlxWeights::prewarm(self, keys)
    }
}

/// Lazily open an mlx-community directory or a single `.safetensors` (mmap, no
/// eager copy). `.npz`/`.npy` aren't supported lazily — use [`load_path`].
pub fn load_path_lazy(path: impl AsRef<Path>) -> Result<LazyMlxWeights> {
    let path = path.as_ref();
    if path.is_dir() {
        let cfg_path = path.join("config.json");
        let config = if cfg_path.is_file() {
            MlxConfig::from_path(&cfg_path)?
        } else {
            MlxConfig::default()
        };
        let files = gather_shard_files(path)?;
        LazyMlxWeights::from_shards(files, config, path.to_path_buf())
    } else if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        let config = path
            .parent()
            .map(|d| d.join("config.json"))
            .filter(|p| p.is_file())
            .map(|p| MlxConfig::from_path(&p))
            .transpose()?
            .unwrap_or_default();
        LazyMlxWeights::from_shards(vec![path.to_path_buf()], config, path.to_path_buf())
    } else {
        bail!(
            "load_path_lazy: expected a dir or .safetensors, got {}",
            path.display()
        );
    }
}

fn load_mlx_dir(dir: &Path) -> Result<MlxWeights> {
    let config_path = dir.join("config.json");
    let config = if config_path.is_file() {
        MlxConfig::from_path(&config_path)?
    } else {
        MlxConfig::default()
    };

    let index_path = dir.join("model.safetensors.index.json");
    let mut files: Vec<PathBuf> = Vec::new();
    if index_path.is_file() {
        let idx: serde_json::Value = serde_json::from_slice(&fs::read(&index_path)?)?;
        if let Some(map) = idx.get("weight_map").and_then(|v| v.as_object()) {
            let mut set = std::collections::BTreeSet::new();
            for v in map.values() {
                if let Some(s) = v.as_str() {
                    let p = dir.join(s);
                    if p.is_file() {
                        set.insert(p);
                    }
                }
            }
            files.extend(set);
        }
    }
    if files.is_empty() {
        for ent in fs::read_dir(dir)? {
            let ent = ent?;
            let p = ent.path();
            if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                files.push(p);
            }
        }
        files.sort();
    }
    if files.is_empty() {
        bail!("no safetensors files in {}", dir.display());
    }

    let mut tensors = HashMap::new();
    for f in files {
        let part = load_safetensors_file(&f).with_context(|| format!("load {}", f.display()))?;
        for (k, v) in part {
            tensors.insert(k, v);
        }
    }
    Ok(MlxWeights {
        tensors,
        config,
        source: dir.to_path_buf(),
    })
}

/// Load MLX weights from a directory, `.safetensors`, `.npz`, or `.npy`.
pub fn load_path(path: impl AsRef<Path>) -> Result<MlxWeights> {
    let path = path.as_ref();
    if path.is_dir() {
        return load_mlx_dir(path);
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "safetensors" => {
            let tensors = load_safetensors_file(path)?;
            let config = {
                let cfg = path.with_file_name("config.json");
                if cfg.is_file() {
                    MlxConfig::from_path(cfg)?
                } else {
                    MlxConfig::default()
                }
            };
            Ok(MlxWeights {
                tensors,
                config,
                source: path.to_path_buf(),
            })
        }
        "npz" => Ok(MlxWeights {
            tensors: load_npz(path)?,
            config: MlxConfig::default(),
            source: path.to_path_buf(),
        }),
        "npy" => Ok(MlxWeights {
            tensors: load_npy(path)?,
            config: MlxConfig::default(),
            source: path.to_path_buf(),
        }),
        other => bail!(
            "unsupported MLX weight path {} (ext {other:?}); expected dir, .safetensors, .npz, or .npy",
            path.display()
        ),
    }
}
