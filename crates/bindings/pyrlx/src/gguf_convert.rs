// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safetensors → GGUF conversion for Python (`rlx-gguf-convert`).
//!
//! [`convert_to_gguf`] wraps the Rust `Converter` with sensible defaults
//! (skip 1-D / norm / bias tensors unless overridden). Optional cargo
//! features `gguf-onnx` and `gguf-pt` enable ONNX and PyTorch checkpoints.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rlx_bind::runtime::io::convert;
use rlx_gguf_convert::ConvertReport;

use crate::bind::{to_json, to_py};

#[pyclass(name = "GgufConvertReport")]
pub struct PyConvertReport {
    inner: ConvertReport,
}

#[pymethods]
impl PyConvertReport {
    #[getter]
    fn tensors(&self) -> usize {
        self.inner.tensors
    }

    #[getter]
    fn input_bytes(&self) -> usize {
        self.inner.input_bytes
    }

    #[getter]
    fn output_bytes(&self) -> usize {
        self.inner.output_bytes
    }

    #[getter]
    fn output_path(&self) -> String {
        self.inner.output_path.display().to_string()
    }

    fn compression_ratio(&self) -> f64 {
        self.inner.compression_ratio()
    }

    fn schemes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty_bound(py);
        for (name, scheme) in &self.inner.schemes {
            let pair = PyTuple::new_bound(py, [name.clone(), format!("{scheme:?}")]);
            list.append(pair)?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "GgufConvertReport(tensors={}, ratio={:.2}x, path={:?})",
            self.inner.tensors,
            self.compression_ratio(),
            self.inner.output_path.display()
        )
    }
}

/// Convert safetensors (default), ONNX, or PyTorch weights to GGUF.
///
/// ``scheme`` is the default quant scheme (e.g. ``"Q4_K"``, ``"IQ2_XXS"``).
/// ``scheme_overrides`` maps exact tensor names to alternate schemes.
/// When ``skip_norm_bias`` is true (default), 1-D tensors and names
/// containing ``norm`` or ``bias`` stay at native precision.
#[pyfunction]
#[pyo3(name = "convert_to_gguf")]
#[pyo3(signature = (
    input_path,
    output_path,
    scheme,
    *,
    architecture=None,
    skip_norm_bias=true,
    scheme_overrides=None,
))]
pub fn convert_to_gguf(
    input_path: &str,
    output_path: &str,
    scheme: &str,
    architecture: Option<&str>,
    skip_norm_bias: bool,
    scheme_overrides: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyConvertReport> {
    let mut opts = rlx_bind::runtime::serde_json::Map::new();
    if let Some(a) = architecture {
        opts.insert("architecture".into(), a.into());
    }
    opts.insert("skip_norm_bias".into(), skip_norm_bias.into());
    if let Some(o) = scheme_overrides {
        opts.insert("overrides".into(), to_json(o.as_any())?);
    }
    let report = convert::convert_to_gguf(
        input_path,
        output_path,
        scheme,
        &rlx_bind::runtime::Value::Object(opts),
    )
    .map_err(to_py)?;
    Ok(PyConvertReport { inner: report })
}
