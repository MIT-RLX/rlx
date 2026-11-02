// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`WeightSource`]s over checkpoint files (feature `formats`).
//!
//! - [`SafetensorsSource`]: one `.safetensors` file or a directory of shards,
//!   memory-mapped; F32 / F16 / BF16 / F64, FP8 (`F8_E4M3` / `F8_E5M2`) and
//!   integer tensors read as f32. An FP8 weight comes out with its stored
//!   scale (`<key>_scale_inv` / `<key>_scale`) already applied, and reports
//!   itself as BF16 so a flow under `WeightStorage::Auto` keeps it at 2 B per
//!   weight rather than 4. [`SafetensorsSource::half_bytes`] also serves half
//!   tensors as raw bytes, skipping the f32 host copy, and
//!   [`SafetensorsSource::fp8_packed`] keeps FP8 matmul weights at 1 B each
//!   as `QuantScheme::Fp8E4m3B128` / `Fp8E5m2B128` blocks for `DequantMatMul`.
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

/// A weight [`SafetensorsSource::fp8_packed`] can serve packed: its block-FP8
/// scheme, `[experts, n, k]` (`[out, in]`, `experts = 1` for a matrix) dims and
/// stored scale grid, if any.
struct Fp8Packable {
    scheme: rlx_ir::quant::QuantScheme,
    experts: usize,
    n: usize,
    k: usize,
    grid: Option<(String, rlx_ir::lowp_codec::ScaleGrid)>,
}

/// Safetensors file(s) as a [`WeightSource`]. See the [module docs](self).
pub struct SafetensorsSource {
    maps: Vec<(PathBuf, Mmap, usize)>,
    tensors: HashMap<String, StEntry>,
    fp8_scales: bool,
    fp8_block: Option<[usize; 2]>,
    half_bytes: bool,
    fp8_packed: bool,
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
            fp8_scales: true,
            fp8_block: None,
            half_bytes: false,
            fp8_packed: false,
        };
        for p in files {
            src.add_file(p)?;
        }
        if path.is_dir() {
            src.fp8_block = config_fp8_block(&path.join("config.json"));
        }
        Ok(src)
    }

    /// Apply each FP8 weight's stored scale on read (default on). Off, FP8
    /// tensors come out as their raw decoded codes.
    pub fn fp8_scales(mut self, on: bool) -> Self {
        self.fp8_scales = on;
        self
    }

    /// The `[rows, cols]` block an FP8 scale grid covers. Opening a directory
    /// reads it from `config.json` (`quantization_config.weight_block_size`);
    /// otherwise it is inferred from the shapes (128 when that fits).
    pub fn fp8_block(mut self, block: [usize; 2]) -> Self {
        self.fp8_block = Some(block);
        self
    }

    /// Serve rank-2 F16 / BF16 / FP8 tensors through
    /// [`WeightSource::take_half`] as half bytes (FP8 scaled, then rounded to
    /// bf16), so a flow storing them at that width never builds an f32 copy.
    /// Off by default: whoever reads the flow's `typed_params` must upload
    /// each entry by its dtype (`rlx_core::flow_util::compile_built` does).
    pub fn half_bytes(mut self, on: bool) -> Self {
        self.half_bytes = on;
        self
    }

    /// Hand rank-2 `F8_E4M3` / `F8_E5M2` weights to the builder packed, as
    /// [`QuantScheme::Fp8E4m3B128`](rlx_ir::quant::QuantScheme::Fp8E4m3B128) /
    /// `Fp8E5m2B128` blocks (1 B per weight + one f32 scale per 128) for
    /// `DequantMatMul`,
    /// instead of as f32 / bf16. A weight whose `in` dim is not a multiple of
    /// 128, or whose scale varies inside a 128-element run, is served
    /// unpacked as before. Off by default, like [`GgufSource::packed`].
    pub fn fp8_packed(mut self, on: bool) -> Self {
        self.fp8_packed = on;
        self
    }

    /// Whether `key` packs as block FP8, and if so its scheme, `[out, in]`
    /// dims and scale grid (`None` grid: no stored scale). Metadata only.
    /// `rank` 2 is a linear's `[out, in]`, 3 an expert bank's `[E, out, in]`.
    fn fp8_packable(&self, key: &str, rank: usize) -> Option<Fp8Packable> {
        use rlx_ir::lowp_codec::{ScaleGrid, fp8_b128};
        let e = self.tensors.get(key).filter(|_| self.fp8_packed)?;
        let (experts, n, k) = match (rank, e.shape.as_slice()) {
            (2, &[n, k]) => (1, n, k),
            (3, &[ex, n, k]) => (ex, n, k),
            _ => return None,
        };
        let scheme = match e.dtype.as_str() {
            "F8_E4M3" => rlx_ir::quant::QuantScheme::Fp8E4m3B128,
            "F8_E5M2" => rlx_ir::quant::QuantScheme::Fp8E5m2B128,
            _ => return None,
        };
        if !k.is_multiple_of(fp8_b128::QK) {
            return None;
        }
        let Some(skey) = self.fp8_scale_key(key).filter(|_| self.fp8_scales) else {
            return Some(Fp8Packable {
                scheme,
                experts,
                n,
                k,
                grid: None,
            });
        };
        let g = ScaleGrid::new(&e.shape, &self.tensors[&skey].shape, self.fp8_block).ok()?;
        // One scale per 128-element run along `in`, or the block format can't
        // hold it.
        (g.block[1].is_multiple_of(fp8_b128::QK) || g.grid[1] == 1).then_some(Fp8Packable {
            scheme,
            experts,
            n,
            k,
            grid: Some((skey, g)),
        })
    }

    /// `key` as block-FP8 bytes plus its scheme and `[out, in]` dims, when
    /// [`fp8_packed`](Self::fp8_packed) is on and the layout allows it.
    fn fp8_packed_parts(&self, key: &str, rank: usize) -> Result<Option<(Vec<u8>, Fp8Packable)>> {
        use rlx_ir::lowp_codec::fp8_b128;
        let Some(p) = self.fp8_packable(key, rank) else {
            return Ok(None);
        };
        let scale = match &p.grid {
            None => None,
            Some((skey, _)) => Some(self.read(skey)?),
        };
        let e = &self.tensors[key];
        let codes = &self.maps[e.file].1[e.start..e.end];
        let w_q = fp8_b128::pack_scaled(
            codes,
            &e.shape,
            scale.as_ref().map(|(s, sh)| (s.as_slice(), sh.as_slice())),
            self.fp8_block,
        )
        .map_err(|e| anyhow!("{key}: {e}"))?
        .ok_or_else(|| anyhow!("{key}: FP8 layout changed between check and pack"))?;
        Ok(Some((w_q, p)))
    }

    /// Whether `name` is the stored scale of an FP8 weight.
    fn is_fp8_scale(&self, name: &str) -> bool {
        ["_scale_inv", "_scale"].iter().any(|s| {
            name.strip_suffix(s)
                .and_then(|w| self.tensors.get(w))
                .is_some_and(|w| w.dtype.starts_with("F8_"))
        })
    }

    /// The scale tensor stored for FP8 weight `key`, if any.
    fn fp8_scale_key(&self, key: &str) -> Option<String> {
        ["_scale_inv", "_scale"]
            .iter()
            .map(|s| format!("{key}{s}"))
            .find(|k| self.tensors.contains_key(k))
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
            "F8_E4M3" | "F8_E5M2" => {
                let fmt = if e.dtype == "F8_E4M3" {
                    rlx_ir::quant::ScaledFormat::F8E4M3
                } else {
                    rlx_ir::quant::ScaledFormat::F8E5M2
                };
                let mut out = vec![0.0f32; raw.len()];
                rlx_ir::lowp_codec::decode_slice(fmt, raw, &mut out);
                out
            }
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

/// `quantization_config.weight_block_size` from an HF `config.json`.
fn config_fp8_block(path: &Path) -> Option<[usize; 2]> {
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    let b = cfg
        .get("quantization_config")?
        .get("weight_block_size")?
        .as_array()?;
    match b.as_slice() {
        [r, c] => Some([r.as_u64()? as usize, c.as_u64()? as usize]),
        _ => None,
    }
}

impl WeightSource for SafetensorsSource {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (mut data, shape) = self.read(key)?;
        let fp8 = self.tensors[key].dtype.starts_with("F8_");
        if let Some(skey) = self.fp8_scale_key(key).filter(|_| fp8 && self.fp8_scales) {
            let (scale, sshape) = self.read(&skey)?;
            rlx_ir::lowp_codec::apply_weight_scales(
                &mut data,
                &shape,
                &scale,
                &sshape,
                self.fp8_block,
            )
            .map_err(|e| anyhow!("{key}: {e}"))?;
        }
        if transpose {
            transpose_2d(&data, &shape, key)
        } else {
            Ok((data, shape))
        }
    }
    fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }
    fn has_packed(&self, key: &str) -> bool {
        self.fp8_packable(key, 2).is_some()
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        Ok(self
            .fp8_packed_parts(key, 2)?
            .map(|(w_q, p)| GgufPackedLinear {
                w_q,
                scheme: p.scheme,
                in_dim: p.k,
                out_dim: p.n,
                bias: Vec::new(),
            }))
    }
    /// An `[E, out, in]` FP8 expert stack, packed like
    /// [`take_packed`](WeightSource::take_packed) one expert after another.
    fn take_packed_bank(&mut self, key: &str) -> Result<Option<GgufPackedBank>> {
        Ok(self
            .fp8_packed_parts(key, 3)?
            .map(|(w_q, p)| GgufPackedBank {
                w_q,
                scheme: p.scheme,
                num_experts: p.experts,
                in_dim: p.k,
                out_dim: p.n,
            }))
    }
    /// Every tensor but the scales of FP8 weights while they are applied on
    /// read ([`SafetensorsSource::fp8_scales`]) — those are part of their
    /// weight, not a tensor a builder should load.
    fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self
            .tensors
            .keys()
            .filter(|name| !(self.fp8_scales && self.is_fp8_scale(name)))
            .cloned()
            .collect();
        k.sort();
        k
    }
    /// FP8 reports BF16: the narrowest width a flow stores. The codes alone
    /// are exact in bf16; scaled, they round once (relative error ≤ 2⁻⁹).
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        Some(match self.tensors.get(key)?.dtype.as_str() {
            "F32" => rlx_ir::DType::F32,
            "F16" => rlx_ir::DType::F16,
            "BF16" | "F8_E4M3" | "F8_E5M2" => rlx_ir::DType::BF16,
            _ => return None,
        })
    }
    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let Some(e) = self.tensors.get(key).filter(|_| self.half_bytes) else {
            return Ok(None);
        };
        if e.shape.len() != 2 {
            return Ok(None);
        }
        let (bytes, shape) = match e.dtype.as_str() {
            "F16" | "BF16" => {
                let raw = &self.maps[e.file].1[e.start..e.end];
                let n: usize = e.shape.iter().product();
                if raw.len() != 2 * n {
                    bail!(
                        "{key}: {} bytes of {} for shape {:?}",
                        raw.len(),
                        e.dtype,
                        e.shape
                    );
                }
                (raw.to_vec(), e.shape.clone())
            }
            "F8_E4M3" | "F8_E5M2" => {
                let (data, shape) = self.take(key, false)?;
                (
                    crate::storage::encode_f32(&data, rlx_ir::DType::BF16),
                    shape,
                )
            }
            _ => return Ok(None),
        };
        if !transpose {
            return Ok(Some((bytes, shape)));
        }
        let (rows, cols) = (shape[0], shape[1]);
        let (src, _) = bytes.as_chunks::<2>();
        let mut out = vec![0u8; bytes.len()];
        let (dst, _) = out.as_chunks_mut::<2>();
        for r in 0..rows {
            for c in 0..cols {
                dst[c * rows + r] = src[r * cols + c];
            }
        }
        Ok(Some((out, vec![cols, rows])))
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
