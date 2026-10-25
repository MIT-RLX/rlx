// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF pack/unpack and v3 file I/O for Python.
//!
//! | API | Role |
//! |-----|------|
//! | [`quantize_gguf`] / [`dequant_gguf`] | Single-tensor encode / decode (aliases: `pyrlx.quantize`, `pyrlx.dequant`) |
//! | [`load_gguf`] / [`write_gguf`] | Read / write GGUF files on disk |
//! | [`PyGgufFile`] | Loaded file: tensor names, metadata, dequant, raw bytes |
//!
//! Shapes follow GGML innermost-first order (same as `rlx_gguf::GgufFile`).
//! For safetensors → GGUF conversion see [`crate::gguf_convert`].

use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rlx_bind::runtime::io::gguf::{self, Gguf, TensorData, TensorSpec};

use crate::bind::{from_json, opt_json, to_py};

/// `quantize_gguf(weights: float32 array, dtype)` → uint8 array of packed blocks.
#[pyfunction]
#[pyo3(signature = (weights, dtype))]
pub fn quantize_gguf<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'py, f32>,
    dtype: &str,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    if !weights.is_contiguous() {
        return Err(PyTypeError::new_err("weights must be C-contiguous f32"));
    }
    let packed = gguf::quantize(weights.as_slice()?, dtype).map_err(to_py)?;
    Ok(PyArray1::from_vec_bound(py, packed))
}

/// `dequant_gguf(packed: uint8 array, dtype, *, num_elements=None)` → float32
/// array. The element count is inferred from the byte length when omitted.
#[pyfunction]
#[pyo3(signature = (packed, dtype, *, num_elements=None))]
pub fn dequant_gguf<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray1<'py, u8>,
    dtype: &str,
    num_elements: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    if !packed.is_contiguous() {
        return Err(PyTypeError::new_err("packed must be C-contiguous u8"));
    }
    let out = gguf::dequantize(packed.as_slice()?, dtype, num_elements).map_err(to_py)?;
    Ok(PyArray1::from_vec_bound(py, out))
}

/// A memory-mapped GGUF file.
#[pyclass(name = "GgufFile")]
pub struct PyGgufFile {
    inner: Gguf,
}

#[pymethods]
impl PyGgufFile {
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Gguf::open(path).map_err(to_py)?,
        })
    }

    fn tensor_names(&self) -> Vec<String> {
        self.inner.tensor_names()
    }

    /// `{name, shape, dtype, n_elements}`; `KeyError` for an unknown name.
    fn tensor_info(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let t = self
            .inner
            .info(name)
            .ok_or_else(|| PyKeyError::new_err(format!("tensor not found: {name}")))?;
        let dict = PyDict::new_bound(py);
        dict.set_item("name", &t.name)?;
        dict.set_item("shape", PyList::new_bound(py, &t.shape))?;
        dict.set_item("dtype", t.dtype)?;
        dict.set_item("n_elements", t.n_elements)?;
        Ok(dict.into())
    }

    /// The tensor dequantized to f32 (flat).
    fn dequant_tensor<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let data = self.inner.dequant(name).map_err(to_py)?;
        Ok(PyArray1::from_vec_bound(py, data))
    }

    /// The tensor's stored bytes, still in its GGML layout.
    fn read_tensor_bytes<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        if self.inner.info(name).is_none() {
            return Err(PyKeyError::new_err(format!("tensor not found: {name}")));
        }
        let bytes = self.inner.bytes(name).map_err(to_py)?;
        Ok(PyArray1::from_vec_bound(py, bytes))
    }

    #[getter]
    fn architecture(&self) -> Option<String> {
        self.inner.architecture()
    }

    /// Every metadata key; arrays past 64 entries (a vocabulary) read as a
    /// placeholder — `metadata_value(key)` returns them in full.
    fn metadata(&self, py: Python<'_>) -> PyObject {
        from_json(py, &self.inner.metadata(false))
    }

    /// One metadata value in full, or `None`.
    fn metadata_value(&self, py: Python<'_>, key: &str) -> PyObject {
        match self.inner.metadata_value(key) {
            Some(v) => from_json(py, &v),
            None => py.None(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "GgufFile(tensors={}, arch={:?})",
            self.inner.inner.tensors.len(),
            self.inner.architecture()
        )
    }
}

#[pyfunction]
#[pyo3(name = "load_gguf")]
pub fn load_gguf(path: &str) -> PyResult<PyGgufFile> {
    PyGgufFile::load(path)
}

/// `{"data": float32 array (quantized to dtype) or uint8 array (already
/// packed), "shape": [...], "dtype": "Q4_K"}`.
fn parse_tensor_spec(name: &str, spec: &Bound<'_, PyAny>) -> PyResult<TensorSpec> {
    let d = spec.downcast::<PyDict>()?;
    let get = |key: &str| {
        d.get_item(key)?
            .ok_or_else(|| PyKeyError::new_err(format!("tensor {name}: missing '{key}'")))
    };
    let data = get("data")?;
    let shape: Vec<usize> = get("shape")?.extract()?;
    let dtype: String = get("dtype")?.extract()?;
    let data = if let Ok(arr) = data.extract::<PyReadonlyArray1<f32>>() {
        if !arr.is_contiguous() {
            return Err(PyTypeError::new_err(format!(
                "tensor {name}: data must be C-contiguous"
            )));
        }
        TensorData::F32(arr.as_slice()?.to_vec())
    } else if let Ok(arr) = data.extract::<PyReadonlyArray1<u8>>() {
        if !arr.is_contiguous() {
            return Err(PyTypeError::new_err(format!(
                "tensor {name}: packed data must be C-contiguous u8"
            )));
        }
        TensorData::Bytes(arr.as_slice()?.to_vec())
    } else {
        return Err(PyTypeError::new_err(format!(
            "tensor {name}: data must be float32 or uint8 ndarray"
        )));
    };
    Ok(TensorSpec {
        name: name.to_string(),
        shape,
        dtype,
        data,
    })
}

/// `write_gguf(path, {name: spec}, *, architecture=None, metadata=None)`.
#[pyfunction]
#[pyo3(signature = (path, tensors, *, architecture=None, metadata=None))]
pub fn write_gguf(
    path: &str,
    tensors: &Bound<'_, PyDict>,
    architecture: Option<&str>,
    metadata: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let mut specs = Vec::with_capacity(tensors.len());
    for (k, v) in tensors.iter() {
        let name: String = k.extract()?;
        specs.push(parse_tensor_spec(&name, &v)?);
    }
    let meta = opt_json(metadata)?;
    gguf::write(path, specs, architecture, &meta).map_err(to_py)?;
    Ok(())
}
