// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Safetensors → GGUF conversion for Python (`rlx-gguf-convert`).
//!
//! [`convert_to_gguf`] wraps the Rust `Converter` with sensible defaults
//! (skip 1-D / norm / bias tensors unless overridden). Optional cargo
//! features `gguf-onnx` and `gguf-pt` enable ONNX and PyTorch checkpoints.

use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rlx_gguf_convert::{ConvertReport, Converter, Scheme};

fn map_convert_err(path: &str, e: anyhow::Error) -> PyErr {
    let msg = format!("{e:#}");
    if msg.contains("No such file") || msg.contains("opening") {
        PyFileNotFoundError::new_err(format!("convert_to_gguf({path}): {msg}"))
    } else {
        PyValueError::new_err(format!("convert_to_gguf({path}): {msg}"))
    }
}

fn parse_scheme(s: &str) -> PyResult<Scheme> {
    Scheme::parse(s).map_err(|e| PyValueError::new_err(format!("scheme: {e}")))
}

fn scheme_label(scheme: Scheme) -> &'static str {
    use Scheme::*;
    match scheme {
        F32 => "F32",
        F16 => "F16",
        BF16 => "BF16",
        Q8_0 => "Q8_0",
        Q4_0 => "Q4_0",
        Q4_1 => "Q4_1",
        Q5_0 => "Q5_0",
        Q5_1 => "Q5_1",
        Q2_K => "Q2_K",
        Q3_K => "Q3_K",
        Q4_K => "Q4_K",
        Q5_K => "Q5_K",
        Q6_K => "Q6_K",
        Q8_K => "Q8_K",
        IQ4_NL => "IQ4_NL",
        IQ4_XS => "IQ4_XS",
        IQ2_XXS => "IQ2_XXS",
        IQ2_XS => "IQ2_XS",
        IQ2_S => "IQ2_S",
        IQ3_XXS => "IQ3_XXS",
        IQ3_S => "IQ3_S",
        IQ1_S => "IQ1_S",
        IQ1_M => "IQ1_M",
        TQ1_0 => "TQ1_0",
        TQ2_0 => "TQ2_0",
        MXFP4 => "MXFP4",
        NVFP4 => "NVFP4",
    }
}

fn open_converter(input: &str) -> PyResult<Converter> {
    let lower = input.to_ascii_lowercase();
    if lower.ends_with(".pt") || lower.ends_with(".pth") || lower.ends_with(".bin") {
        #[cfg(feature = "gguf-pt")]
        {
            return Converter::from_pt(input).map_err(|e| map_convert_err(input, e));
        }
        #[cfg(not(feature = "gguf-pt"))]
        {
            return Err(PyValueError::new_err(format!(
                "{input} looks like a PyTorch checkpoint — rebuild pyrlx with gguf-pt feature"
            )));
        }
    }
    if lower.ends_with(".onnx") {
        #[cfg(feature = "gguf-onnx")]
        {
            return Converter::from_onnx(input).map_err(|e| map_convert_err(input, e));
        }
        #[cfg(not(feature = "gguf-onnx"))]
        {
            return Err(PyValueError::new_err(format!(
                "{input} is ONNX — rebuild pyrlx with gguf-onnx feature"
            )));
        }
    }
    Converter::from_safetensors(input).map_err(|e| map_convert_err(input, e))
}

/// Summary returned by [`convert_to_gguf`].
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
            let pair = PyTuple::new_bound(py, [name, scheme_label(*scheme)]);
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
    let default_scheme = parse_scheme(scheme)?;
    let mut converter = open_converter(input_path)?.default_scheme(default_scheme);
    if let Some(arch) = architecture {
        converter = converter.architecture(arch);
    }
    if let Some(overrides) = scheme_overrides {
        for (k, v) in overrides.iter() {
            let name: String = k.extract()?;
            let scheme_str: String = v.extract()?;
            converter = converter.scheme_for_name(name, parse_scheme(&scheme_str)?);
        }
    }
    if skip_norm_bias {
        converter = converter.skip_quant_for(|name, shape| {
            shape.len() < 2 || name.contains("norm") || name.contains("bias")
        });
    }
    let report = converter
        .write_gguf(output_path)
        .map_err(|e| PyValueError::new_err(format!("convert_to_gguf write: {e:#}")))?;
    Ok(PyConvertReport { inner: report })
}
