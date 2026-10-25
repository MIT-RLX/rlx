// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weights on disk: GGUF (read, write, quantize), safetensors → GGUF, the MLX /
//! DDUF / NeMo / PyTorch loaders, and `.rlxp` packages.
//!
//! Feature-gated per format family (`gguf-io`, `gguf-convert`, `weights`) so a
//! binding that ships without them pays nothing.

#[cfg(any(feature = "gguf-io", feature = "weights"))]
use serde_json::{Value, json};

#[cfg(any(feature = "gguf-io", feature = "weights"))]
use crate::{ApiError, ApiResult};

#[cfg(any(feature = "gguf-io", feature = "weights"))]
fn io_err(what: String) -> impl FnOnce(anyhow::Error) -> ApiError {
    move |e| ApiError::io(&what, e.as_ref())
}

// ── GGUF ────────────────────────────────────────────────────

/// GGUF read / write / quantize.
#[cfg(feature = "gguf-io")]
pub mod gguf {
    use rlx_gguf::{GgmlType, GgufFile, GgufWriter, MetaValue};

    use super::*;

    /// A GGML tensor type by name: `"Q4_K"`, `"IQ2_XXS"`, `"F16"`, ….
    pub fn ggml_type(name: &str) -> ApiResult<GgmlType> {
        GgmlType::from_name(name).ok_or_else(|| {
            ApiError::type_(format!(
                "unknown GGUF dtype '{name}' — expected e.g. Q4_K, Q6_K, IQ2_XXS, TQ2_0"
            ))
        })
    }

    /// Pack f32 weights into a GGML type.
    pub fn quantize(data: &[f32], dtype: &str) -> ApiResult<Vec<u8>> {
        let ggml = ggml_type(dtype)?;
        rlx_gguf::quantize(data, ggml).map_err(io_err(format!("quantize({dtype})")))
    }

    /// Unpack GGML bytes to f32. `n` (the element count) is inferred from the
    /// byte length when omitted.
    pub fn dequantize(bytes: &[u8], dtype: &str, n: Option<usize>) -> ApiResult<Vec<f32>> {
        let ggml = ggml_type(dtype)?;
        let n = match n {
            Some(n) => n,
            None => (1..=bytes.len().saturating_mul(256))
                .find(|&n| rlx_gguf::bytes_for_public(ggml, n).is_some_and(|b| b == bytes.len()))
                .ok_or_else(|| {
                    ApiError::type_(format!(
                        "dequantize: cannot infer the element count for {dtype} from {} packed \
                         bytes — pass it explicitly",
                        bytes.len()
                    ))
                })?,
        };
        rlx_gguf::dequant_typed(ggml, bytes, n, "dequantize")
            .map_err(io_err(format!("dequantize({dtype}, n={n})")))
    }

    /// One tensor's header.
    #[derive(Clone, Debug)]
    pub struct TensorInfo {
        pub name: String,
        pub shape: Vec<usize>,
        /// The GGML type name (`"Q4_K"`).
        pub dtype: &'static str,
        pub n_elements: usize,
    }

    /// An open (memory-mapped) GGUF file.
    pub struct Gguf {
        pub inner: GgufFile,
    }

    /// Arrays longer than this read back as a placeholder in [`Gguf::metadata`]
    /// — a 150k-entry vocabulary is not "metadata" to print.
    pub const ARRAY_PREVIEW: usize = 64;

    impl Gguf {
        pub fn open(path: &str) -> ApiResult<Self> {
            let inner =
                GgufFile::from_path_mmap(path).map_err(io_err(format!("load_gguf('{path}')")))?;
            Ok(Self { inner })
        }

        pub fn version(&self) -> u32 {
            self.inner.version
        }

        /// Sorted: `HashMap` order is not stable across runs.
        pub fn tensor_names(&self) -> Vec<String> {
            let mut names: Vec<String> = self.inner.keys().map(str::to_owned).collect();
            names.sort();
            names
        }

        pub fn info(&self, name: &str) -> Option<TensorInfo> {
            self.inner.get(name).map(|t| TensorInfo {
                name: t.name.clone(),
                shape: t.shape.clone(),
                dtype: t.dtype.name(),
                n_elements: t.n_elements(),
            })
        }

        pub fn dequant(&self, name: &str) -> ApiResult<Vec<f32>> {
            self.inner
                .dequant_f32(name)
                .map(|(data, _)| data)
                .map_err(io_err(format!("tensor('{name}')")))
        }

        /// The tensor's stored bytes, still in its GGML layout.
        pub fn bytes(&self, name: &str) -> ApiResult<Vec<u8>> {
            let t = self
                .inner
                .get(name)
                .ok_or_else(|| ApiError::type_(format!("tensor not found: {name}")))?;
            self.inner
                .tensor_bytes(t)
                .map(<[u8]>::to_vec)
                .map_err(io_err(format!("tensor_bytes('{name}')")))
        }

        pub fn architecture(&self) -> Option<String> {
            self.inner
                .metadata
                .get("general.architecture")
                .and_then(MetaValue::as_str)
                .map(str::to_owned)
        }

        /// Every metadata key (sorted), arrays past [`ARRAY_PREVIEW`] shortened
        /// to a placeholder unless `full_arrays`.
        pub fn metadata(&self, full_arrays: bool) -> Value {
            let mut keys: Vec<&String> = self.inner.metadata.keys().collect();
            keys.sort();
            let mut map = serde_json::Map::new();
            for k in keys {
                map.insert(k.clone(), meta_json(&self.inner.metadata[k], full_arrays));
            }
            Value::Object(map)
        }

        /// One metadata value in full, or `None`.
        pub fn metadata_value(&self, key: &str) -> Option<Value> {
            self.inner.metadata.get(key).map(|v| meta_json(v, true))
        }
    }

    fn meta_json(v: &MetaValue, full_arrays: bool) -> Value {
        match v {
            MetaValue::U8(x) => json!(x),
            MetaValue::I8(x) => json!(x),
            MetaValue::U16(x) => json!(x),
            MetaValue::I16(x) => json!(x),
            MetaValue::U32(x) => json!(x),
            MetaValue::I32(x) => json!(x),
            MetaValue::U64(x) => json!(x),
            MetaValue::I64(x) => json!(x),
            MetaValue::F32(x) => json!(x),
            MetaValue::F64(x) => json!(x),
            MetaValue::Bool(x) => json!(x),
            MetaValue::String(s) => json!(s),
            MetaValue::Array(items) => {
                if !full_arrays && items.len() > ARRAY_PREVIEW {
                    return json!(format!("[{} items — use metadata_value(key)]", items.len()));
                }
                Value::Array(items.iter().map(|i| meta_json(i, full_arrays)).collect())
            }
        }
    }

    /// What one tensor of a [`write()`] holds.
    pub enum TensorData {
        /// f32 values, quantized to the target type on write.
        F32(Vec<f32>),
        /// Bytes already in the target layout, written through.
        Bytes(Vec<u8>),
    }

    pub struct TensorSpec {
        pub name: String,
        pub shape: Vec<usize>,
        pub dtype: String,
        pub data: TensorData,
    }

    /// Write a GGUF file; returns how many tensors went in. `metadata` is an
    /// object of strings / numbers / booleans.
    pub fn write(
        path: &str,
        mut tensors: Vec<TensorSpec>,
        architecture: Option<&str>,
        metadata: &Value,
    ) -> ApiResult<usize> {
        let mut writer = GgufWriter::new();
        if let Some(arch) = architecture {
            writer.set_arch(arch);
        }
        match metadata {
            Value::Null => {}
            Value::Object(map) => {
                for (key, v) in map {
                    let entry = match v {
                        Value::Bool(b) => MetaValue::Bool(*b),
                        Value::String(s) => MetaValue::String(s.clone()),
                        Value::Number(n) => match (n.as_u64(), n.as_i64()) {
                            (Some(u), _) => MetaValue::U64(u),
                            (None, Some(i)) => MetaValue::I64(i),
                            _ => MetaValue::F64(n.as_f64().unwrap_or(f64::NAN)),
                        },
                        _ => {
                            return Err(ApiError::type_(format!(
                                "metadata['{key}'] must be a string, number or boolean"
                            )));
                        }
                    };
                    writer.set_meta(key.clone(), entry);
                }
            }
            _ => return Err(ApiError::type_("metadata must be an object")),
        }
        // Sorted, so the file is the same whatever order the caller built.
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        let count = tensors.len();
        for t in tensors {
            let ggml = ggml_type(&t.dtype)?;
            let n: usize = t.shape.iter().product();
            let bytes = match t.data {
                TensorData::F32(values) => {
                    if values.len() != n {
                        return Err(ApiError::type_(format!(
                            "tensor '{}': data length {} != shape product {n}",
                            t.name,
                            values.len()
                        )));
                    }
                    if ggml == GgmlType::F32 {
                        values.iter().flat_map(|v| v.to_le_bytes()).collect()
                    } else {
                        rlx_gguf::quantize(&values, ggml)
                            .map_err(io_err(format!("tensor '{}' quantize({})", t.name, t.dtype)))?
                    }
                }
                TensorData::Bytes(bytes) => {
                    if let Some(expect) = rlx_gguf::bytes_for_public(ggml, n)
                        && bytes.len() != expect
                    {
                        return Err(ApiError::type_(format!(
                            "tensor '{}': packed bytes {} != expected {expect} for {}",
                            t.name,
                            bytes.len(),
                            t.dtype
                        )));
                    }
                    bytes
                }
            };
            writer
                .add_tensor_bytes(t.name.clone(), t.shape, ggml, bytes)
                .map_err(io_err(format!("write_gguf tensor '{}'", t.name)))?;
        }
        writer
            .write_to_path(path)
            .map_err(io_err(format!("write_gguf('{path}')")))?;
        Ok(count)
    }
}

/// safetensors (or PyTorch / ONNX, with their features) → GGUF.
#[cfg(feature = "gguf-convert")]
pub mod convert {
    use rlx_gguf_convert::{ConvertReport, Converter, Scheme};

    use super::super::opts::Opts;
    use super::*;

    pub fn scheme(s: &str) -> ApiResult<Scheme> {
        Scheme::parse(s).map_err(|e| ApiError::type_(format!("unknown quant scheme '{s}': {e}")))
    }

    fn open(input: &str) -> ApiResult<Converter> {
        let lower = input.to_ascii_lowercase();
        let what = format!("convert_to_gguf('{input}')");
        if lower.ends_with(".pt") || lower.ends_with(".pth") || lower.ends_with(".bin") {
            #[cfg(feature = "gguf-pt")]
            return Converter::from_pt(input).map_err(io_err(what));
            #[cfg(not(feature = "gguf-pt"))]
            return Err(ApiError::runtime(format!(
                "{input} looks like a PyTorch checkpoint — rebuild with the `gguf-pt` feature"
            )));
        }
        if lower.ends_with(".onnx") {
            #[cfg(feature = "gguf-onnx")]
            return Converter::from_onnx(input).map_err(io_err(what));
            #[cfg(not(feature = "gguf-onnx"))]
            return Err(ApiError::runtime(format!(
                "{input} is ONNX — rebuild with the `gguf-onnx` feature"
            )));
        }
        Converter::from_safetensors(input).map_err(io_err(what))
    }

    /// Convert `input` to a GGUF at `output`, quantizing with `scheme` by
    /// default. Options: `{architecture, overrides: {name: scheme},
    /// skip_norm_bias = true}` — the last keeps 1-D, norm and bias tensors
    /// unquantized.
    pub fn convert_to_gguf(
        input: &str,
        output: &str,
        scheme_name: &str,
        opts: &Value,
    ) -> ApiResult<ConvertReport> {
        let o = Opts::new("conversion", opts)?;
        // Validate the cheap arguments before touching the input file.
        let default_scheme = scheme(scheme_name)?;
        let mut c = open(input)?.default_scheme(default_scheme);
        if let Some(arch) = o.str("architecture")? {
            c = c.architecture(arch);
        }
        let overrides = o.get("overrides").or_else(|| o.get("scheme_overrides"));
        if let Some(Value::Object(map)) = overrides {
            for (name, v) in map {
                let label = v.as_str().ok_or_else(|| {
                    ApiError::type_(format!("overrides['{name}'] must be a scheme name"))
                })?;
                c = c.scheme_for_name(name.clone(), scheme(label)?);
            }
        }
        if o.bool("skip_norm_bias", true)? {
            c = c.skip_quant_for(|name, shape| {
                shape.len() < 2 || name.contains("norm") || name.contains("bias")
            });
        }
        o.finish()?;
        c.write_gguf(output)
            .map_err(io_err("convert_to_gguf write".to_string()))
    }

    /// `{tensors, input_bytes, output_bytes, compression_ratio, output_path, schemes}`.
    pub fn report_json(r: &ConvertReport) -> Value {
        json!({
            "tensors": r.tensors,
            "input_bytes": r.input_bytes,
            "output_bytes": r.output_bytes,
            "compression_ratio": r.compression_ratio(),
            "output_path": r.output_path.to_string_lossy(),
            "schemes": r.schemes.iter().map(|(n, s)| json!([n, format!("{s:?}")])).collect::<Vec<_>>(),
        })
    }
}

/// MLX / DDUF / NeMo / PyTorch weight loaders and `.rlxp` packages.
#[cfg(feature = "weights")]
pub mod weights {
    use std::path::Path;

    use rlx_pkg::{ContainerKind, Package};

    use super::super::opts::Opts;
    use super::*;

    /// Every tensor of a checkpoint as f32, sorted by name. `format` is
    /// `"mlx"` (an MLX dir or one `.safetensors` file; `"safetensors"` is the
    /// same reader), `"dduf"`, `"nemo"` or `"pt"`.
    pub fn load(format: &str, path: &str) -> ApiResult<Vec<(String, Vec<f32>)>> {
        let what = format!("load_{format}('{path}')");
        let mut entries: Vec<(String, Vec<f32>)> = match format {
            "mlx" | "safetensors" => rlx_mlx_io::load_f32_map(path)
                .map_err(io_err(what))?
                .into_iter()
                .collect(),
            "dduf" => rlx_dduf::load_f32_map(path)
                .map_err(io_err(what))?
                .into_iter()
                .collect(),
            "nemo" => {
                let m = rlx_nemo::NemoModel::open(Path::new(path)).map_err(io_err(what))?;
                let mut out = Vec::new();
                for name in m.names() {
                    let t = m
                        .tensor(&name)
                        .map_err(io_err(format!("load_nemo tensor '{name}'")))?;
                    out.push((name, t.data));
                }
                out
            }
            "pt" => {
                let m = rlx_nemo::PtModel::open(Path::new(path)).map_err(io_err(what))?;
                let mut out = Vec::new();
                for name in m.names() {
                    let t = m
                        .tensor(&name)
                        .map_err(io_err(format!("load_pt tensor '{name}'")))?;
                    out.push((name, t.data));
                }
                out
            }
            other => {
                return Err(ApiError::type_(format!(
                    "unknown weight format '{other}' (mlx, safetensors, dduf, nemo, pt)"
                )));
            }
        };
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(entries)
    }

    /// An MLX checkpoint with its quantized linears kept packed: each packed
    /// linear's bindings as raw bytes, every other tensor as f32.
    pub enum MlxEntry {
        F32(Vec<f32>),
        Packed(Vec<u8>),
    }

    pub fn load_mlx_packed(path: &str) -> ApiResult<Vec<(String, MlxEntry)>> {
        let what = format!("load_mlx('{path}')");
        let mut w = rlx_mlx_io::load_path(path).map_err(io_err(what.clone()))?;
        let linears = rlx_mlx_io::collect_packed_linears(&mut w).map_err(io_err(what))?;
        let mut out = Vec::new();
        for b in &linears {
            for (name, bytes, _dt) in rlx_mlx_io::param_bindings_for(b) {
                out.push((name, MlxEntry::Packed(bytes)));
            }
        }
        for name in w.logical_keys() {
            if let Ok((data, _shape)) = w.take_dense_f32(&name) {
                out.push((name, MlxEntry::F32(data)));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    pub fn container(s: &str) -> ApiResult<ContainerKind> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "flat" => ContainerKind::Flat,
            "zip" => ContainerKind::Zip,
            "dir" | "directory" => ContainerKind::Dir,
            other => {
                return Err(ApiError::type_(format!(
                    "unknown container '{other}' (flat, zip, dir)"
                )));
            }
        })
    }

    /// A package's manifest: `{name, format_version, compat_version, features,
    /// has_graph, tensors, sidecars}`.
    pub fn open_rlxp(path: &str) -> ApiResult<Value> {
        let pack = Package::open(path).map_err(io_err(format!("load_rlxp('{path}')")))?;
        let m = pack.manifest();
        let tensors: Vec<String> = pack
            .weights_index()
            .map(|idx| idx.names().map(str::to_string).collect())
            .unwrap_or_default();
        let sidecars: Vec<&str> = m.sidecars.iter().map(|s| s.id.as_str()).collect();
        Ok(json!({
            "name": m.name,
            "format_version": m.format_version,
            "compat_version": m.compat_version,
            "features": m.features,
            "has_graph": pack.has_graph(),
            "tensors": tensors,
            "sidecars": sidecars,
        }))
    }

    /// Check every tensor and sidecar hash: `{ok, tensors_checked, …, failures}`.
    pub fn verify_rlxp(path: &str) -> ApiResult<Value> {
        let what = format!("verify_rlxp('{path}')");
        let pack = Package::open(path).map_err(io_err(what.clone()))?;
        let r = rlx_pkg::verify_package(&pack).map_err(io_err(what))?;
        Ok(json!({
            "ok": r.failures.is_empty(),
            "tensors_checked": r.tensors_checked,
            "tensors_ok": r.tensors_ok,
            "tensors_unchecked": r.tensors_unchecked,
            "tensors_mismatch": r.tensors_mismatch,
            "sidecars_checked": r.sidecars_checked,
            "sidecars_ok": r.sidecars_ok,
            "failures": r.failures,
        }))
    }

    /// Repackage a checkpoint as `.rlxp`. Options: `{container = "flat",
    /// include_graph = false, auto_tier = true, keep_packed = false}`; `from`
    /// is `gguf`, `mlx`, `dduf`, `nemo` or `pt`.
    pub fn to_rlxp(from: &str, src: &str, out: &str, opts: &Value) -> ApiResult<()> {
        let o = Opts::new("rlxp conversion", opts)?;
        let container = container(o.str("container")?.unwrap_or("flat"))?;
        let include_graph = o.bool("include_graph", false)?;
        let auto_tier = o.bool("auto_tier", true)?;
        let keep_packed = o.bool("keep_packed", false)?;
        // Accepted for symmetry with the JS options object, which names it.
        let _ = o.get("from");
        o.finish()?;
        let what = format!("to_rlxp('{src}' -> '{out}')");
        let r = match from.trim().to_ascii_lowercase().as_str() {
            "gguf" => rlx_pkg::gguf_to_rlxp(
                src,
                out,
                &rlx_pkg::GgufImportOptions {
                    container,
                    include_graph,
                    compress_sidecars: true,
                    auto_tier,
                },
            ),
            "mlx" => rlx_pkg::mlx_to_rlxp(
                src,
                out,
                &rlx_pkg::MlxImportOptions {
                    container,
                    include_graph,
                    compress_sidecars: true,
                    auto_tier,
                    dequant_to_f32: !keep_packed,
                    ..Default::default()
                },
            ),
            "dduf" => rlx_pkg::dduf_to_rlxp(
                src,
                out,
                &rlx_pkg::DdufImportOptions {
                    container,
                    include_graph,
                    compress_sidecars: true,
                    auto_tier,
                },
            ),
            "nemo" => rlx_pkg::nemo_to_rlxp(
                src,
                out,
                &rlx_pkg::NemoImportOptions {
                    container,
                    include_graph,
                    compress_sidecars: true,
                    auto_tier,
                },
            ),
            "pt" | "torch" | "pytorch" => rlx_pkg::pt_to_rlxp(
                src,
                out,
                &rlx_pkg::PtImportOptions {
                    container,
                    include_graph,
                    compress_sidecars: true,
                    auto_tier,
                },
            ),
            other => {
                return Err(ApiError::type_(format!(
                    "to_rlxp: unknown source format '{other}' (gguf, mlx, dduf, nemo, pt)"
                )));
            }
        };
        r.map_err(io_err(what))
    }
}
