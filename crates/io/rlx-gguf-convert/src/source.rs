// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Source-format readers. Each implementation hands back tensors in
//! `(name, shape, f32 data, native dtype)` form so the rest of the
//! crate doesn't need to know the input format.

use anyhow::Result;

use rlx_gguf::GgmlType;

#[derive(Debug, Clone)]
pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    /// Tensor values converted to f32 (regardless of native dtype).
    pub data: Vec<f32>,
    /// Original storage dtype in the source file — used by the
    /// converter to decide what "keep at native precision" means.
    pub native: GgmlType,
    /// Byte count of this tensor in the source file (e.g. n×2 for
    /// BF16). Used to report a meaningful compression ratio.
    pub source_bytes: usize,
}

/// Source-file reader contract. Implementations live in this module
/// behind cargo features; downstream crates can plug in their own.
pub trait TensorReader {
    fn names(&self) -> Vec<String>;
    fn read_tensor(&self, name: &str) -> Result<NamedTensor>;
}

// ─── safetensors ──────────────────────────────────────────────────

#[cfg(feature = "safetensors")]
pub use safetensors_reader::SafetensorsReader;

#[cfg(feature = "safetensors")]
mod safetensors_reader {
    use std::path::Path;

    use anyhow::{Context, Result, anyhow};
    use safetensors::{SafeTensors, tensor::Dtype as StDtype};

    use super::{NamedTensor, TensorReader};
    use rlx_gguf::GgmlType;

    /// `.safetensors` reader. Slurps the entire file into RAM at open
    /// time — typical model files (≤ a few GB) fit comfortably, and
    /// this avoids the lifetime mess of holding a `SafeTensors<'a>`
    /// alongside its backing buffer.
    pub struct SafetensorsReader {
        data: Vec<u8>,
        /// (name, shape, dtype, start_offset_in_data, len_bytes).
        /// Cached at open so `read_tensor` is just a slice + decode.
        tensors: Vec<TensorMeta>,
    }

    struct TensorMeta {
        name: String,
        shape: Vec<usize>,
        dtype: StDtype,
        start: usize,
        len: usize,
    }

    impl SafetensorsReader {
        pub fn open(path: &Path) -> Result<Self> {
            let data =
                std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            // Parse once to populate the index; we then drop the
            // `SafeTensors` view and just hold raw bytes + metadata.
            let st = SafeTensors::deserialize(&data).context("parse safetensors")?;
            let mut tensors = Vec::with_capacity(st.names().len());
            for name in st.names() {
                let view = st.tensor(name)?;
                let shape: Vec<usize> = view.shape().to_vec();
                let bytes = view.data();
                // The view's bytes are a sub-slice of `data`; recover
                // its position by pointer-diff. SafeTensors guarantees
                // contiguous storage (no re-allocation here).
                let start = (bytes.as_ptr() as usize).saturating_sub(data.as_ptr() as usize);
                let len = bytes.len();
                tensors.push(TensorMeta {
                    name: name.to_string(),
                    shape,
                    dtype: view.dtype(),
                    start,
                    len,
                });
            }
            // Drop the SafeTensors view.
            drop(st);
            Ok(Self { data, tensors })
        }

        fn meta(&self, name: &str) -> Option<&TensorMeta> {
            self.tensors.iter().find(|t| t.name == name)
        }

        fn is_fp8(dtype: StDtype) -> bool {
            matches!(dtype, StDtype::F8_E4M3 | StDtype::F8_E5M2)
        }

        /// The scale stored beside FP8 weight `name` (`_scale_inv` / `_scale`).
        fn fp8_scale(&self, name: &str) -> Option<&TensorMeta> {
            ["_scale_inv", "_scale"]
                .iter()
                .find_map(|s| self.meta(&format!("{name}{s}")))
        }

        /// A scale tensor that belongs to an FP8 weight — folded into the
        /// weight on read, so not a tensor of its own.
        fn is_fp8_scale(&self, name: &str) -> bool {
            ["_scale_inv", "_scale"].iter().any(|s| {
                name.strip_suffix(s)
                    .and_then(|w| self.meta(w))
                    .is_some_and(|w| Self::is_fp8(w.dtype))
            })
        }

        fn convert_to_f32(dtype: StDtype, bytes: &[u8]) -> Result<Vec<f32>> {
            match dtype {
                StDtype::F32 => {
                    if !bytes.len().is_multiple_of(4) {
                        return Err(anyhow!("F32: bad byte count {}", bytes.len()));
                    }
                    Ok(bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect())
                }
                StDtype::F16 => {
                    if !bytes.len().is_multiple_of(2) {
                        return Err(anyhow!("F16: bad byte count {}", bytes.len()));
                    }
                    Ok(bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect())
                }
                StDtype::BF16 => {
                    if !bytes.len().is_multiple_of(2) {
                        return Err(anyhow!("BF16: bad byte count {}", bytes.len()));
                    }
                    Ok(bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect())
                }
                StDtype::F64 => Ok(bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| f64::from_le_bytes(*c) as f32)
                    .collect()),
                StDtype::I32 => Ok(bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_le_bytes(*c) as f32)
                    .collect()),
                StDtype::I64 => Ok(bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| i64::from_le_bytes(*c) as f32)
                    .collect()),
                StDtype::U8 => Ok(bytes.iter().map(|&b| b as f32).collect()),
                StDtype::I8 => Ok(bytes.iter().map(|&b| b as i8 as f32).collect()),
                StDtype::BOOL => Ok(bytes.iter().map(|&b| f32::from(b != 0)).collect()),
                StDtype::F8_E4M3 | StDtype::F8_E5M2 => {
                    let fmt = if dtype == StDtype::F8_E4M3 {
                        rlx_ir::quant::ScaledFormat::F8E4M3
                    } else {
                        rlx_ir::quant::ScaledFormat::F8E5M2
                    };
                    let mut out = vec![0.0f32; bytes.len()];
                    rlx_ir::lowp_codec::decode_slice(fmt, bytes, &mut out);
                    Ok(out)
                }
                other => Err(anyhow!("safetensors dtype {other:?} not supported yet")),
            }
        }

        fn native_for(dtype: StDtype) -> GgmlType {
            match dtype {
                StDtype::F32 => GgmlType::F32,
                StDtype::F16 => GgmlType::F16,
                StDtype::BF16 => GgmlType::BF16,
                _ => GgmlType::F32,
            }
        }
    }

    impl TensorReader for SafetensorsReader {
        fn names(&self) -> Vec<String> {
            self.tensors
                .iter()
                .filter(|t| !self.is_fp8_scale(&t.name))
                .map(|t| t.name.clone())
                .collect()
        }
        fn read_tensor(&self, name: &str) -> Result<NamedTensor> {
            let t = self
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| anyhow!("no tensor {name}"))?;
            let bytes = &self.data[t.start..t.start + t.len];
            let source_bytes = bytes.len();
            let mut data = Self::convert_to_f32(t.dtype, bytes)?;
            if let Some(sc) = self.fp8_scale(name).filter(|_| Self::is_fp8(t.dtype)) {
                let scale =
                    Self::convert_to_f32(sc.dtype, &self.data[sc.start..sc.start + sc.len])?;
                rlx_ir::lowp_codec::apply_weight_scales(
                    &mut data, &t.shape, &scale, &sc.shape, None,
                )
                .map_err(|e| anyhow!("{name}: {e}"))?;
            }
            Ok(NamedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data,
                native: Self::native_for(t.dtype),
                source_bytes,
            })
        }
    }
}

// ─── PyTorch torch.save reader ────────────────────────────────────

#[cfg(feature = "pt")]
pub use pt_reader::PtReader;

#[cfg(feature = "pt")]
mod pt_reader {
    use std::path::Path;

    use anyhow::Result;

    use super::{NamedTensor, TensorReader};
    use rlx_gguf::GgmlType;
    use rlx_nemo::{DType, PtModel};

    /// Reader for a PyTorch `.pt` / `.pth` / `pytorch_model.bin` checkpoint
    /// (a `torch.save` ZIP of a state dict). Backed by `rlx-nemo`'s
    /// non-executing pickle VM + zip reader; every tensor is handed back as
    /// f32, with its on-disk dtype preserved for native-precision passthrough
    /// and an accurate compression ratio.
    pub struct PtReader {
        model: PtModel,
    }

    impl PtReader {
        pub fn open(path: &Path) -> Result<Self> {
            Ok(Self {
                model: PtModel::open(path)?,
            })
        }

        fn native_for(dtype: DType) -> GgmlType {
            match dtype {
                DType::F32 => GgmlType::F32,
                DType::F16 => GgmlType::F16,
                DType::BF16 => GgmlType::BF16,
                // Everything else (int, f64, bool) is materialized as f32.
                _ => GgmlType::F32,
            }
        }
    }

    impl TensorReader for PtReader {
        fn names(&self) -> Vec<String> {
            self.model.names()
        }
        fn read_tensor(&self, name: &str) -> Result<NamedTensor> {
            let t = self.model.tensor(name)?;
            let native = Self::native_for(t.dtype);
            // Report the source footprint (numel × on-disk dtype width) so
            // the converter's compression ratio reflects the real input,
            // not the f32 scratch we decoded into.
            let source_bytes = t.data.len() * t.dtype.size();
            Ok(NamedTensor {
                name: t.name,
                shape: t.shape,
                data: t.data,
                native,
                source_bytes,
            })
        }
    }
}

// ─── ONNX initializer reader ──────────────────────────────────────

#[cfg(feature = "onnx")]
pub use onnx_reader::OnnxReader;

#[cfg(feature = "onnx")]
mod onnx_reader {
    use std::path::Path;

    use anyhow::{Context, Result, anyhow};

    use super::{NamedTensor, TensorReader};
    use rlx_gguf::GgmlType;

    /// Reader for ONNX `.onnx` initializer tensors. Uses the `tract`-free
    /// `rlx_onnx_import` parser. We only pull initializers (model
    /// parameters), not graph inputs/outputs.
    pub struct OnnxReader {
        names: Vec<String>,
        tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
    }

    impl OnnxReader {
        pub fn open(path: &Path) -> Result<Self> {
            let bytes =
                std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            let (_graph, params) =
                rlx_onnx_import::import_onnx_bytes(&bytes).context("rlx-onnx-import parse")?;
            let names: Vec<String> = params.keys().cloned().collect();
            Ok(Self {
                names,
                tensors: params,
            })
        }
    }

    impl TensorReader for OnnxReader {
        fn names(&self) -> Vec<String> {
            self.names.clone()
        }
        fn read_tensor(&self, name: &str) -> Result<NamedTensor> {
            let (data, shape) = self
                .tensors
                .get(name)
                .ok_or_else(|| anyhow!("no tensor {name}"))?
                .clone();
            let source_bytes = data.len() * 4;
            Ok(NamedTensor {
                name: name.to_string(),
                shape,
                data,
                native: GgmlType::F32,
                source_bytes,
            })
        }
    }
}
