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

//! Convert tensors from external formats (safetensors, ONNX) into
//! GGUF with per-tensor quantization. Designed to be called at first
//! inference load: read the source file once, write a GGUF blob with
//! a chosen quant scheme, then on subsequent loads dequant the GGUF
//! directly — cutting both disk footprint and memory at load time
//! for transformer weights (often ≥4× shrink at Q4_K_M).
//!
//! # Quick start
//!
//! ```ignore
//! use rlx_gguf_convert::{Converter, Scheme};
//!
//! let report = Converter::from_safetensors("model.safetensors")?
//!     .default_scheme(Scheme::Q4_K)
//!     .skip_quant_for(|name, shape| {
//!         // Tiny 1-D tensors (norms, biases) stay full-precision.
//!         name.contains("norm") || name.contains("bias") || shape.len() < 2
//!     })
//!     .architecture("llama")
//!     .write_gguf("model.q4_k.gguf")?;
//! println!("wrote {} tensors, {:.2}× smaller",
//!          report.tensors,
//!          report.compression_ratio());
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! # Real-weight benchmarks
//!
//! Validated end-to-end against two production checkpoints (mean
//! cosine is the average of [`Converter::write_gguf`] output
//! dequantized and compared back to the source values for every
//! quantized weight tensor; non-quantized tensors round-trip exactly
//! and aren't included). M2 mini, release build.
//!
//! | Model              | Source size      | Scheme | Output | Shrink | Mean cosine | Wall  |
//! |--------------------|------------------|--------|--------|-------:|------------:|------:|
//! | Bio_ClinicalBERT   | 416 MB F32       | Q8_0   | 113 MB | 3.75×  |    0.999984 | 0.27s |
//! | Bio_ClinicalBERT   | 416 MB F32       | Q6_K   |  86 MB | 4.85×  |    0.999815 | 0.22s |
//! | Bio_ClinicalBERT   | 416 MB F32       | Q4_K   |  59 MB | 7.05×  |    0.996785 | 0.44s |
//! | Bio_ClinicalBERT   | 416 MB F32       | Q4_0   |  59 MB | 7.05×  |    0.996169 | 0.44s |
//! | Qwen3-TTS 0.6B     | 1.7 GB BF16      | Q4_K   | 491 MB | 3.55×  |    0.996712 | 3.7s  |
//!
//! [`ConvertReport::compression_ratio`] reports source-byte shrink
//! (BF16 inputs naturally compress less than F32 inputs because
//! they're already 2× smaller on disk).
//!
//! # Per-tensor schemes
//!
//! Three priority levels, applied in order:
//!
//! 1. Exact-name override — [`Converter::scheme_for_name`].
//! 2. Predicate override — [`Converter::scheme_for`] returning
//!    `Some(scheme)` to override or `None` to fall through.
//! 3. Default — [`Converter::default_scheme`].
//!
//! Tensors whose element count doesn't divide the chosen scheme's
//! block size fall back to F16. Tensors matched by
//! [`Converter::skip_quant_for`] stay at their source dtype
//! (preserved via [`Scheme::F32`] / [`Scheme::F16`] / [`Scheme::BF16`]).
//!
//! # Crate layout
//!
//! * [`Scheme`] / [`Converter`] / [`ConvertReport`] are the public API.
//! * Source readers gate behind features:
//!   * `safetensors` (default) — `.safetensors` files.
//!   * `onnx` — ONNX initializer tensors via `rlx-onnx-import`.
//!   * `pt` — PyTorch `.pt` / `.pth` / `pytorch_model.bin` `torch.save`
//!     checkpoints via `rlx-nemo`.
//! * The encoder side is shared with [`rlx_gguf`], so output
//!   round-trips through [`rlx_gguf::GgufFile::dequant_f32`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

pub use rlx_gguf::{GgmlType, MetaValue};

mod source;
pub use source::{NamedTensor, TensorReader};

/// Quantization scheme to apply to a tensor when converting. Mirrors
/// the [`GgmlType`] variants we have encoders for.
///
/// Variant naming follows the canonical GGUF convention (`Q4_K`,
/// `Q8_0`, …) so it survives copy/paste from llama.cpp docs and CLI
/// flags. Parse a name with [`Scheme::parse`]; map to the underlying
/// [`GgmlType`] with [`Scheme::to_ggml`].
///
/// `Mixed`-style presets (mostly-Q4_K with a few critical projections
/// at Q6_K, llama.cpp's `Q4_K_M`) are not first-class enum variants —
/// express them with a per-tensor override via
/// [`Converter::scheme_for`] / [`Converter::scheme_for_name`].
///
/// # Picking a scheme
///
/// | Scheme  | Bits / elem | When to use |
/// |---------|-------------|-------------|
/// | `F32`   | 32          | "Don't touch" — debugging, gold reference. |
/// | `F16`/`BF16` | 16     | Lossless-ish; default fallback for shape mismatches. |
/// | `Q8_0`  | 8.5         | Highest decode speed; ~0% accuracy loss. |
/// | `Q6_K`  | 6.5         | Near-F16 quality at < 50% size. |
/// | `Q5_K`  | 5.5         | "Best balance" for memory-constrained inference. |
/// | `Q4_K`  | 4.5         | Standard 4-bit; ~7× shrink vs F32 source. |
/// | `Q4_0`  | 4.5         | Legacy; faster decode kernels, slightly worse accuracy. |
/// | `Q3_K`, `Q2_K` | 3.4 / 2.6 | Aggressive shrink; only for tolerant models. |
/// | `IQ4_NL`, `IQ4_XS` | ~4.5 | Non-linear IQ4; GPU dequant on all backends. |
/// | `IQ2_XXS` … `IQ1_M` | 1.6–2.5 | Ultra-low-bit IQ; slower encode, smallest files. |
/// | `TQ1_0`, `TQ2_0` | ~2.5–3.4 | Ternary quants. |
/// | `MXFP4`, `NVFP4` | 4.25 / 4.5 | Microscaling FP4 (OCP / NVIDIA layouts). |
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ4_NL,
    IQ4_XS,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ1_S,
    IQ1_M,
    TQ1_0,
    TQ2_0,
    MXFP4,
    NVFP4,
}

impl Scheme {
    pub fn to_ggml(self) -> GgmlType {
        match self {
            Self::F32 => GgmlType::F32,
            Self::F16 => GgmlType::F16,
            Self::BF16 => GgmlType::BF16,
            Self::Q8_0 => GgmlType::Q8_0,
            Self::Q4_0 => GgmlType::Q4_0,
            Self::Q4_1 => GgmlType::Q4_1,
            Self::Q5_0 => GgmlType::Q5_0,
            Self::Q5_1 => GgmlType::Q5_1,
            Self::Q2_K => GgmlType::Q2K,
            Self::Q3_K => GgmlType::Q3K,
            Self::Q4_K => GgmlType::Q4K,
            Self::Q5_K => GgmlType::Q5K,
            Self::Q6_K => GgmlType::Q6K,
            Self::Q8_K => GgmlType::Q8K,
            Self::IQ4_NL => GgmlType::IQ4NL,
            Self::IQ4_XS => GgmlType::IQ4XS,
            Self::IQ2_XXS => GgmlType::IQ2XXS,
            Self::IQ2_XS => GgmlType::IQ2XS,
            Self::IQ2_S => GgmlType::IQ2S,
            Self::IQ3_XXS => GgmlType::IQ3XXS,
            Self::IQ3_S => GgmlType::IQ3S,
            Self::IQ1_S => GgmlType::IQ1S,
            Self::IQ1_M => GgmlType::IQ1M,
            Self::TQ1_0 => GgmlType::TQ1_0,
            Self::TQ2_0 => GgmlType::TQ2_0,
            Self::MXFP4 => GgmlType::MXFP4,
            Self::NVFP4 => GgmlType::NVFP4,
        }
    }

    /// Parse a scheme name (`"q4_k"`, `"f16"`, …). Case-insensitive.
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_uppercase().as_str() {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "Q8_0" => Self::Q8_0,
            "Q4_0" => Self::Q4_0,
            "Q4_1" => Self::Q4_1,
            "Q5_0" => Self::Q5_0,
            "Q5_1" => Self::Q5_1,
            "Q2_K" => Self::Q2_K,
            "Q3_K" => Self::Q3_K,
            "Q4_K" => Self::Q4_K,
            "Q5_K" => Self::Q5_K,
            "Q6_K" => Self::Q6_K,
            "Q8_K" => Self::Q8_K,
            "IQ4_NL" => Self::IQ4_NL,
            "IQ4_XS" => Self::IQ4_XS,
            "IQ2_XXS" => Self::IQ2_XXS,
            "IQ2_XS" => Self::IQ2_XS,
            "IQ2_S" => Self::IQ2_S,
            "IQ3_XXS" => Self::IQ3_XXS,
            "IQ3_S" => Self::IQ3_S,
            "IQ1_S" => Self::IQ1_S,
            "IQ1_M" => Self::IQ1_M,
            "TQ1_0" => Self::TQ1_0,
            "TQ2_0" => Self::TQ2_0,
            "MXFP4" => Self::MXFP4,
            "NVFP4" => Self::NVFP4,
            other => bail!("unknown scheme {other}"),
        })
    }

    /// The required element-count divisor for this scheme. For example,
    /// Q4_K requires multiples of 256; Q8_0 requires multiples of 32.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q8_0
            | Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::IQ4_NL
            | Self::MXFP4 => 32,
            Self::NVFP4 => 16,
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ4_XS
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ2_S
            | Self::IQ3_XXS
            | Self::IQ3_S
            | Self::IQ1_S
            | Self::IQ1_M
            | Self::TQ1_0
            | Self::TQ2_0 => 256,
        }
    }
}

/// Conversion summary returned by [`Converter::write_gguf`]. Use it
/// to log compression ratios, generate a per-scheme histogram, or
/// drive a re-convert pass with different scheme rules.
///
/// Byte counts are measured against the actual source-file layout
/// (e.g. n×2 for BF16 source tensors), not f32-lifted equivalents —
/// so the ratio matches what a user would see comparing the two
/// files on disk.
#[derive(Debug, Clone)]
pub struct ConvertReport {
    /// Number of tensors written to the output GGUF.
    pub tensors: usize,
    /// Total source-file bytes summed across all converted tensors.
    pub input_bytes: usize,
    /// Total bytes occupied by the encoded tensors (does **not**
    /// include the GGUF header/metadata overhead — that's typically
    /// well under 0.1% of the data segment).
    pub output_bytes: usize,
    /// Per-tensor scheme assignment in the order tensors were written.
    /// Useful for verifying that overrides matched the right tensors.
    pub schemes: Vec<(String, Scheme)>,
    /// Where the GGUF file was written.
    pub output_path: PathBuf,
}

impl ConvertReport {
    /// `input_bytes / output_bytes` — the "shrink factor" most users
    /// expect to see. Returns 0.0 for empty conversions to avoid a
    /// divide-by-zero panic.
    pub fn compression_ratio(&self) -> f64 {
        if self.output_bytes == 0 {
            0.0
        } else {
            self.input_bytes as f64 / self.output_bytes as f64
        }
    }
}

type SchemeFn = Box<dyn Fn(&str, &[usize]) -> Option<Scheme>>;
type SkipFn = Box<dyn Fn(&str, &[usize]) -> bool>;

/// Top-level conversion driver. Build with [`Converter::from_reader`]
/// (or the `from_safetensors` / `from_onnx` convenience constructors
/// behind their feature gates), set a default + per-tensor scheme,
/// then [`Converter::write_gguf`].
///
/// # Builder ordering
///
/// The builder methods are independent; call them in any order. Only
/// the final [`Converter::write_gguf`] call performs I/O. Predicates
/// installed by [`Converter::scheme_for`] / [`Converter::skip_quant_for`]
/// own their captured state (`Fn + 'static`), so the converter is
/// `Send + 'static` itself.
///
/// # Example
///
/// ```ignore
/// use rlx_gguf_convert::{Converter, Scheme};
///
/// let report = Converter::from_safetensors("model.safetensors")?
///     .default_scheme(Scheme::Q4_K)
///     // Promote the embed + output projection — these dominate
///     // quality loss on small models.
///     .scheme_for(|name, _| {
///         if name.contains("embed") || name.ends_with("lm_head.weight") {
///             Some(Scheme::Q6_K)
///         } else {
///             None
///         }
///     })
///     // Keep biases / norms / 1-D tensors at native precision.
///     .skip_quant_for(|name, shape| {
///         shape.len() < 2 || name.contains("norm") || name.contains("bias")
///     })
///     .architecture("llama")
///     .write_gguf("model.q4_k.gguf")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct Converter {
    reader: Box<dyn TensorReader>,
    default_scheme: Scheme,
    per_tensor: HashMap<String, Scheme>,
    scheme_fn: Option<SchemeFn>,
    skip_fn: Option<SkipFn>,
    arch: Option<String>,
    meta: Vec<(String, MetaValue)>,
}

impl Converter {
    /// Build a converter from any [`TensorReader`].
    pub fn from_reader(reader: Box<dyn TensorReader>) -> Self {
        Self {
            reader,
            default_scheme: Scheme::Q4_K,
            per_tensor: HashMap::new(),
            scheme_fn: None,
            skip_fn: None,
            arch: None,
            meta: Vec::new(),
        }
    }

    /// Convenience: open a `.safetensors` file at `path`. Requires the
    /// `safetensors` feature (on by default).
    #[cfg(feature = "safetensors")]
    pub fn from_safetensors(path: impl AsRef<Path>) -> Result<Self> {
        let reader = source::SafetensorsReader::open(path.as_ref())?;
        Ok(Self::from_reader(Box::new(reader)))
    }

    /// Convenience: open a `.onnx` file at `path` and read its
    /// initializer tensors. Requires the `onnx` feature.
    #[cfg(feature = "onnx")]
    pub fn from_onnx(path: impl AsRef<Path>) -> Result<Self> {
        let reader = source::OnnxReader::open(path.as_ref())?;
        Ok(Self::from_reader(Box::new(reader)))
    }

    /// Convenience: open a PyTorch `.pt` / `.pth` / `pytorch_model.bin`
    /// checkpoint (a `torch.save` state dict) at `path`. Requires the `pt`
    /// feature.
    #[cfg(feature = "pt")]
    pub fn from_pt(path: impl AsRef<Path>) -> Result<Self> {
        let reader = source::PtReader::open(path.as_ref())?;
        Ok(Self::from_reader(Box::new(reader)))
    }

    /// Set the default scheme used when no override matches.
    pub fn default_scheme(mut self, scheme: Scheme) -> Self {
        self.default_scheme = scheme;
        self
    }

    /// Override the scheme for a specific tensor name.
    pub fn scheme_for_name(mut self, name: impl Into<String>, scheme: Scheme) -> Self {
        self.per_tensor.insert(name.into(), scheme);
        self
    }

    /// Callback for matching scheme overrides by name + shape. Returns
    /// `Some(scheme)` to set, `None` to fall through to per-name +
    /// default. Use for patterns like "every tensor whose name ends
    /// with `.weight`".
    pub fn scheme_for<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &[usize]) -> Option<Scheme> + 'static,
    {
        self.scheme_fn = Some(Box::new(f));
        self
    }

    /// Callback to skip quantization entirely (leave the tensor at
    /// its native dtype: F32 / F16 / BF16). Common pattern: skip 1-D
    /// tensors (biases, norms) and any tensor whose element count
    /// doesn't divide the chosen scheme's block size.
    pub fn skip_quant_for<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &[usize]) -> bool + 'static,
    {
        self.skip_fn = Some(Box::new(f));
        self
    }

    /// Set `general.architecture` metadata (e.g. `"llama"`, `"qwen3"`).
    pub fn architecture(mut self, arch: impl Into<String>) -> Self {
        self.arch = Some(arch.into());
        self
    }

    /// Add a custom metadata key/value pair.
    pub fn meta(mut self, key: impl Into<String>, value: MetaValue) -> Self {
        self.meta.push((key.into(), value));
        self
    }

    /// Pick the scheme for `(name, shape)` applying overrides in
    /// priority order: `scheme_for_name` → `scheme_for` → default.
    fn resolve_scheme(&self, name: &str, shape: &[usize], native: GgmlType) -> Scheme {
        if let Some(s) = self.per_tensor.get(name) {
            return *s;
        }
        if let Some(f) = self.scheme_fn.as_ref() {
            if let Some(s) = f(name, shape) {
                return s;
            }
        }
        if let Some(f) = self.skip_fn.as_ref() {
            if f(name, shape) {
                return native_to_scheme(native);
            }
        }
        let elems: usize = shape.iter().product();
        // If the tensor's shape doesn't divide the chosen scheme's
        // block size, fall back to F16 — much better than failing the
        // entire convert. (Embeddings often have a head-aligned final
        // dimension but bias rows of 1 element, for example.)
        if !elems.is_multiple_of(self.default_scheme.block_size()) {
            return Scheme::F16;
        }
        self.default_scheme
    }

    /// Run the conversion and write the output GGUF file.
    ///
    /// For each tensor in the source file:
    /// 1. Read into f32 (lifting from whatever native dtype it was).
    /// 2. Resolve a [`Scheme`] via the override stack (name → predicate
    ///    → default), falling back to F16 if the element count doesn't
    ///    divide the chosen scheme's block size.
    /// 3. Encode with [`rlx_gguf::quantize`] and stream into the
    ///    [`rlx_gguf::GgufWriter`].
    ///
    /// On success returns a [`ConvertReport`] describing the per-tensor
    /// scheme assignment, byte counts, and the output path.
    pub fn write_gguf(self, out: impl AsRef<Path>) -> Result<ConvertReport> {
        let out_path = out.as_ref().to_path_buf();
        let names = self.reader.names();
        let mut writer = rlx_gguf::GgufWriter::new();
        if let Some(arch) = &self.arch {
            writer.set_arch(arch);
        }
        for (k, v) in &self.meta {
            writer.set_meta(k.clone(), v.clone());
        }
        let mut input_bytes = 0usize;
        let mut output_bytes = 0usize;
        let mut schemes: Vec<(String, Scheme)> = Vec::with_capacity(names.len());
        for name in names {
            let NamedTensor {
                name,
                shape,
                data,
                native,
                source_bytes,
            } = self
                .reader
                .read_tensor(&name)
                .with_context(|| format!("reading tensor {name}"))?;
            input_bytes += source_bytes;
            let scheme = self.resolve_scheme(&name, &shape, native);
            let dtype = scheme.to_ggml();
            let bytes = rlx_gguf::quantize(&data, dtype)
                .with_context(|| format!("quantize tensor {name} → {scheme:?}"))?;
            output_bytes += bytes.len();
            writer.add_tensor_bytes(name.clone(), shape, dtype, bytes)?;
            schemes.push((name, scheme));
        }
        writer.write_to_path(&out_path)?;
        Ok(ConvertReport {
            tensors: schemes.len(),
            input_bytes,
            output_bytes,
            schemes,
            output_path: out_path,
        })
    }
}

fn native_to_scheme(dtype: GgmlType) -> Scheme {
    match dtype {
        GgmlType::F32 => Scheme::F32,
        GgmlType::F16 => Scheme::F16,
        GgmlType::BF16 => Scheme::BF16,
        // Default for any other native dtype is F16 — we just want to
        // skip lossy quantization, not preserve some exotic input.
        _ => Scheme::F16,
    }
}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    struct StubReader {
        names: Vec<String>,
        tensors: HashMap<String, (Vec<f32>, Vec<usize>, GgmlType)>,
    }

    impl TensorReader for StubReader {
        fn names(&self) -> Vec<String> {
            self.names.clone()
        }
        fn read_tensor(&self, name: &str) -> Result<NamedTensor> {
            let (data, shape, native) = self
                .tensors
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("no tensor {name}"))?
                .clone();
            let source_bytes = data.len() * 4;
            Ok(NamedTensor {
                name: name.to_string(),
                shape,
                data,
                native,
                source_bytes,
            })
        }
    }

    #[test]
    fn convert_stub_to_q4_k_roundtrips() {
        // 2 super-blocks of 256 elements each.
        let n = 512;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.01).collect();
        let mut tensors = HashMap::new();
        tensors.insert("w".to_string(), (data.clone(), vec![2, 256], GgmlType::F32));
        tensors.insert(
            "bias".to_string(),
            (vec![0.5, -0.5], vec![2], GgmlType::F32),
        );
        let reader = StubReader {
            names: vec!["w".into(), "bias".into()],
            tensors,
        };
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let report = Converter::from_reader(Box::new(reader))
            .default_scheme(Scheme::Q4_K)
            .skip_quant_for(|_, shape| shape.len() < 2)
            .architecture("test")
            .write_gguf(tmp.path())
            .unwrap();
        assert_eq!(report.tensors, 2);
        let parsed = rlx_gguf::GgufFile::from_path(tmp.path()).unwrap();
        let (out, shape) = parsed.dequant_f32("w").unwrap();
        assert_eq!(shape, vec![2, 256]);
        let cos: f32 = {
            let dot: f32 = data.iter().zip(&out).map(|(a, b)| a * b).sum();
            let na: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        };
        assert!(cos > 0.99, "Q4_K conversion cosine {cos}");
        // bias was kept at native dtype (F32 here — input was F32 and
        // the skip-quant predicate matched on shape.len() < 2).
        assert_eq!(parsed.get("bias").unwrap().dtype, GgmlType::F32);
    }

    #[test]
    fn scheme_parse_roundtrip() {
        for s in [
            Scheme::F32,
            Scheme::F16,
            Scheme::BF16,
            Scheme::Q8_0,
            Scheme::Q4_K,
            Scheme::Q6_K,
            Scheme::IQ4_NL,
            Scheme::IQ2_XXS,
            Scheme::TQ2_0,
            Scheme::MXFP4,
            Scheme::NVFP4,
        ] {
            let name = format!("{s:?}");
            let parsed = Scheme::parse(&name).unwrap();
            assert_eq!(parsed, s);
        }
    }

    #[test]
    fn iq2_xxs_convert_roundtrips() {
        let n = 512;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.015).sin() * 0.5).collect();
        let mut tensors = HashMap::new();
        tensors.insert("w".to_string(), (data.clone(), vec![2, 256], GgmlType::F32));
        let reader = StubReader {
            names: vec!["w".into()],
            tensors,
        };
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let report = Converter::from_reader(Box::new(reader))
            .default_scheme(Scheme::IQ2_XXS)
            .architecture("test")
            .write_gguf(tmp.path())
            .unwrap();
        assert_eq!(report.tensors, 1);
        let parsed = rlx_gguf::GgufFile::from_path(tmp.path()).unwrap();
        let (out, shape) = parsed.dequant_f32("w").unwrap();
        assert_eq!(shape, vec![2, 256]);
        let cos: f32 = {
            let dot: f32 = data.iter().zip(&out).map(|(a, b)| a * b).sum();
            let na: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        };
        assert!(cos > 0.75, "IQ2_XXS conversion cosine {cos}");
    }
}
