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
use pyo3::exceptions::{PyFileNotFoundError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rlx_gguf::{GgmlType, GgufFile, GgufWriter, MetaValue};

fn parse_ggml_type(name: &str) -> PyResult<GgmlType> {
    use GgmlType::*;
    Ok(match name.to_ascii_uppercase().as_str() {
        "F32" => F32,
        "F16" => F16,
        "BF16" => BF16,
        "Q8_0" => Q8_0,
        "Q4_0" => Q4_0,
        "Q4_1" => Q4_1,
        "Q5_0" => Q5_0,
        "Q5_1" => Q5_1,
        "Q2_K" => Q2K,
        "Q3_K" => Q3K,
        "Q4_K" => Q4K,
        "Q5_K" => Q5K,
        "Q6_K" => Q6K,
        "Q8_K" => Q8K,
        "IQ4_NL" => IQ4NL,
        "IQ4_XS" => IQ4XS,
        "IQ2_XXS" => IQ2XXS,
        "IQ2_XS" => IQ2XS,
        "IQ2_S" => IQ2S,
        "IQ3_XXS" => IQ3XXS,
        "IQ3_S" => IQ3S,
        "IQ1_S" => IQ1S,
        "IQ1_M" => IQ1M,
        "TQ1_0" => TQ1_0,
        "TQ2_0" => TQ2_0,
        "MXFP4" => MXFP4,
        "NVFP4" => NVFP4,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown GGUF dtype {other:?} — expected e.g. Q4_K, IQ2_XXS, TQ2_0"
            )));
        }
    })
}

fn ggml_type_name(ggml: GgmlType) -> &'static str {
    use GgmlType::*;
    match ggml {
        F32 => "F32",
        F16 => "F16",
        BF16 => "BF16",
        Q8_0 => "Q8_0",
        Q4_0 => "Q4_0",
        Q4_1 => "Q4_1",
        Q5_0 => "Q5_0",
        Q5_1 => "Q5_1",
        Q2K => "Q2_K",
        Q3K => "Q3_K",
        Q4K => "Q4_K",
        Q5K => "Q5_K",
        Q6K => "Q6_K",
        Q8K => "Q8_K",
        IQ4NL => "IQ4_NL",
        IQ4XS => "IQ4_XS",
        IQ2XXS => "IQ2_XXS",
        IQ2XS => "IQ2_XS",
        IQ2S => "IQ2_S",
        IQ3XXS => "IQ3_XXS",
        IQ3S => "IQ3_S",
        IQ1S => "IQ1_S",
        IQ1M => "IQ1_M",
        TQ1_0 => "TQ1_0",
        TQ2_0 => "TQ2_0",
        MXFP4 => "MXFP4",
        NVFP4 => "NVFP4",
        I8 => "I8",
        I16 => "I16",
        I32 => "I32",
        I64 => "I64",
        F64 => "F64",
        Q8_1 => "Q8_1",
        Q1_0 => "Q1_0",
        Q2_0 => "Q2_0",
    }
}

fn infer_num_elements(ggml: GgmlType, bytes: &[u8]) -> PyResult<usize> {
    if let Some(n) = (1..=bytes.len().saturating_mul(256))
        .find(|&n| rlx_gguf::bytes_for_public(ggml, n).is_some_and(|b| b == bytes.len()))
    {
        return Ok(n);
    }
    Err(PyValueError::new_err(format!(
        "cannot infer element count for {ggml:?} from {} packed bytes — pass num_elements=",
        bytes.len()
    )))
}

fn dequant_f32(ggml: GgmlType, bytes: &[u8], n: usize) -> PyResult<Vec<f32>> {
    use GgmlType::*;
    let out = match ggml {
        F32 => Ok(bytemuck::cast_slice(bytes).to_vec()),
        F16 | BF16 => Err(anyhow::anyhow!(
            "dequant for F16/BF16 not exposed in pyrlx yet"
        )),
        Q8_0 => rlx_gguf::dequant_q8_0(bytes, n),
        Q4_0 => rlx_gguf::dequant_q4_0(bytes, n),
        Q4_1 => rlx_gguf::dequant_q4_1(bytes, n),
        Q5_0 => rlx_gguf::dequant_q5_0(bytes, n),
        Q5_1 => rlx_gguf::dequant_q5_1(bytes, n),
        Q4K => rlx_gguf::dequant_q4_k(bytes, n),
        Q5K => rlx_gguf::dequant_q5_k(bytes, n),
        Q6K => rlx_gguf::dequant_q6_k(bytes, n),
        Q8K => rlx_gguf::dequant_q8_k(bytes, n),
        Q2K => rlx_gguf::dequant_q2_k(bytes, n),
        Q3K => rlx_gguf::dequant_q3_k(bytes, n),
        TQ1_0 => rlx_gguf::tq_dequant::dequant_tq1_0(bytes, n),
        TQ2_0 => rlx_gguf::tq_dequant::dequant_tq2_0(bytes, n),
        MXFP4 => rlx_gguf::mx_dequant::dequant_mxfp4(bytes, n),
        NVFP4 => rlx_gguf::mx_dequant::dequant_nvfp4(bytes, n),
        IQ4NL => rlx_gguf::iq_dequant::dequant_iq4_nl(bytes, n),
        IQ4XS => rlx_gguf::iq_dequant::dequant_iq4_xs(bytes, n),
        IQ2XXS => rlx_gguf::iq_dequant::dequant_iq2_xxs(bytes, n),
        IQ2XS => rlx_gguf::iq_dequant::dequant_iq2_xs(bytes, n),
        IQ2S => rlx_gguf::iq_dequant::dequant_iq2_s(bytes, n),
        IQ3XXS => rlx_gguf::iq_dequant::dequant_iq3_xxs(bytes, n),
        IQ3S => rlx_gguf::iq_dequant::dequant_iq3_s(bytes, n),
        IQ1S => rlx_gguf::iq_dequant::dequant_iq1_s(bytes, n),
        IQ1M => rlx_gguf::iq_dequant::dequant_iq1_m(bytes, n),
        Q1_0 => rlx_gguf::q1_dequant::dequant_q1_0(bytes, n),
        Q2_0 => rlx_gguf::q2_dequant::dequant_q2_0(bytes, n),
        other => Err(anyhow::anyhow!("dequant for {other:?} not implemented")),
    };
    out.map_err(map_dequant_err(ggml, n))
}

fn map_dequant_err(ggml: GgmlType, n: usize) -> impl FnOnce(anyhow::Error) -> PyErr {
    move |e| PyValueError::new_err(format!("dequant({ggml:?}, n={n}): {e}"))
}

/// Quantize a contiguous f32 weight vector to GGUF-packed bytes.
///
/// ``weights`` must be a 1-D ``float32`` array whose length divides the
/// scheme block size (256 for K/IQ2/IQ3/IQ1/TQ; 32 for Q4_0 / IQ4_NL / MXFP4; …).
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
    let ggml = parse_ggml_type(dtype)?;
    let slice = weights.as_slice()?;
    let packed = rlx_gguf::quantize(slice, ggml)
        .map_err(|e| PyValueError::new_err(format!("quantize({dtype}): {e}")))?;
    Ok(PyArray1::from_vec_bound(py, packed))
}

/// Dequantize GGUF-packed bytes to f32 (for fidelity checks / debugging).
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
    let ggml = parse_ggml_type(dtype)?;
    let bytes = packed.as_slice()?;
    let n = match num_elements {
        Some(n) => n,
        None => infer_num_elements(ggml, bytes)?,
    };
    let out = dequant_f32(ggml, bytes, n)?;
    Ok(PyArray1::from_vec_bound(py, out))
}

fn meta_to_py<'py>(py: Python<'py>, v: &MetaValue) -> Bound<'py, PyAny> {
    match v {
        MetaValue::U8(x) => x.into_py(py),
        MetaValue::I8(x) => (*x as i64).into_py(py),
        MetaValue::U16(x) => x.into_py(py),
        MetaValue::I16(x) => (*x as i64).into_py(py),
        MetaValue::U32(x) => x.into_py(py),
        MetaValue::I32(x) => (*x as i64).into_py(py),
        MetaValue::F32(x) => x.into_py(py),
        MetaValue::Bool(x) => x.into_py(py),
        MetaValue::String(s) => s.into_py(py),
        MetaValue::U64(x) => (*x as i64).into_py(py),
        MetaValue::I64(x) => x.into_py(py),
        MetaValue::F64(x) => x.into_py(py),
        MetaValue::Array { .. } => format!("{v:?}").into_py(py),
    }
    .into_bound(py)
}

fn parse_shape(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let list = obj.downcast::<PyList>()?;
    let mut shape = Vec::with_capacity(list.len());
    for item in list.iter() {
        let d: usize = item
            .extract()
            .map_err(|_| PyTypeError::new_err("shape entries must be non-negative integers"))?;
        shape.push(d);
    }
    if shape.is_empty() {
        return Err(PyValueError::new_err("shape must be non-empty"));
    }
    Ok(shape)
}

fn shape_numel(shape: &[usize]) -> PyResult<usize> {
    shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| PyValueError::new_err("shape product overflow"))
    })
}

fn parse_tensor_spec(
    name: &str,
    spec: &Bound<'_, PyAny>,
) -> PyResult<(Vec<usize>, GgmlType, Vec<u8>)> {
    let d = spec.downcast::<PyDict>()?;
    let data = d
        .get_item("data")?
        .ok_or_else(|| PyKeyError::new_err(format!("tensor {name}: missing 'data'")))?;
    let shape = parse_shape(
        &d.get_item("shape")?
            .ok_or_else(|| PyKeyError::new_err(format!("tensor {name}: missing 'shape'")))?,
    )?;
    let dtype_str: String = d
        .get_item("dtype")?
        .ok_or_else(|| PyKeyError::new_err(format!("tensor {name}: missing 'dtype'")))?
        .extract()?;
    let ggml = parse_ggml_type(&dtype_str)?;
    let n = shape_numel(&shape)?;

    let bytes = if data.is_instance_of::<PyArray1<f32>>() {
        let arr = data.extract::<PyReadonlyArray1<f32>>()?;
        if !arr.is_contiguous() {
            return Err(PyTypeError::new_err(format!(
                "tensor {name}: data must be C-contiguous"
            )));
        }
        let slice = arr.as_slice()?;
        if slice.len() != n {
            return Err(PyValueError::new_err(format!(
                "tensor {name}: data length {} != shape product {n}",
                slice.len()
            )));
        }
        if ggml == GgmlType::F32 {
            bytemuck::cast_slice(slice).to_vec()
        } else {
            rlx_gguf::quantize(slice, ggml).map_err(|e| {
                PyValueError::new_err(format!("tensor {name} quantize({dtype_str}): {e}"))
            })?
        }
    } else if data.is_instance_of::<PyArray1<u8>>() {
        let arr = data.extract::<PyReadonlyArray1<u8>>()?;
        if !arr.is_contiguous() {
            return Err(PyTypeError::new_err(format!(
                "tensor {name}: packed data must be C-contiguous u8"
            )));
        }
        let slice = arr.as_slice()?;
        let expect = rlx_gguf::bytes_for_public(ggml, n).ok_or_else(|| {
            PyValueError::new_err(format!(
                "tensor {name}: element count {n} not aligned to {dtype_str} block size"
            ))
        })?;
        if slice.len() != expect {
            return Err(PyValueError::new_err(format!(
                "tensor {name}: packed bytes {} != expected {expect} for {dtype_str}",
                slice.len()
            )));
        }
        slice.to_vec()
    } else {
        return Err(PyTypeError::new_err(format!(
            "tensor {name}: data must be float32 or uint8 ndarray"
        )));
    };
    Ok((shape, ggml, bytes))
}

/// Loaded GGUF v3 file (tensor metadata + raw data segment).
#[pyclass(name = "GgufFile")]
pub struct PyGgufFile {
    inner: GgufFile,
}

#[pymethods]
impl PyGgufFile {
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = GgufFile::from_path(path).map_err(|e| {
            if e.to_string().contains("No such file") || e.to_string().contains("opening") {
                PyFileNotFoundError::new_err(format!("load_gguf({path}): {e}"))
            } else {
                PyValueError::new_err(format!("load_gguf({path}): {e}"))
            }
        })?;
        Ok(Self { inner })
    }

    fn tensor_names(&self) -> PyResult<Vec<String>> {
        let mut names: Vec<String> = self.inner.keys().map(str::to_owned).collect();
        names.sort();
        Ok(names)
    }

    fn tensor_info(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let t = self
            .inner
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(format!("tensor not found: {name}")))?;
        let dict = PyDict::new_bound(py);
        dict.set_item("name", &t.name)?;
        dict.set_item("shape", PyList::new_bound(py, &t.shape))?;
        dict.set_item("dtype", ggml_type_name(t.dtype))?;
        dict.set_item("n_elements", t.n_elements())?;
        Ok(dict.into())
    }

    fn dequant_tensor<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let (data, _shape) = self
            .inner
            .dequant_f32(name)
            .map_err(|e| PyValueError::new_err(format!("dequant_tensor({name}): {e}")))?;
        Ok(PyArray1::from_vec_bound(py, data))
    }

    fn read_tensor_bytes<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let t = self
            .inner
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(format!("tensor not found: {name}")))?;
        let bytes = self
            .inner
            .tensor_bytes(t)
            .map_err(|e| PyValueError::new_err(format!("read_tensor_bytes({name}): {e}")))?;
        Ok(PyArray1::from_slice_bound(py, bytes))
    }

    #[getter]
    fn architecture(&self) -> Option<String> {
        self.inner
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .map(str::to_owned)
    }

    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        let mut keys: Vec<_> = self.inner.metadata.keys().collect();
        keys.sort();
        for k in keys {
            dict.set_item(k, meta_to_py(py, &self.inner.metadata[k]))?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        let arch = self
            .inner
            .metadata
            .get("general.architecture")
            .and_then(MetaValue::as_str);
        format!(
            "GgufFile(tensors={}, arch={arch:?})",
            self.inner.tensors.len()
        )
    }
}

/// Load a GGUF file from disk.
#[pyfunction]
#[pyo3(name = "load_gguf")]
pub fn load_gguf(path: &str) -> PyResult<PyGgufFile> {
    PyGgufFile::load(path)
}

/// Write a GGUF v3 file from a tensor spec mapping.
///
/// Each entry in ``tensors`` is ``name -> {"data": ndarray, "shape": [..], "dtype": "Q4_K"}``.
/// ``data`` may be f32 (quantized on write) or pre-packed u8.
#[pyfunction]
#[pyo3(signature = (path, tensors, *, architecture=None, metadata=None))]
pub fn write_gguf(
    path: &str,
    tensors: &Bound<'_, PyDict>,
    architecture: Option<&str>,
    metadata: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let mut writer = GgufWriter::new();
    if let Some(arch) = architecture {
        writer.set_arch(arch);
    }
    if let Some(meta) = metadata {
        for (k, v) in meta.iter() {
            let key: String = k.extract()?;
            if let Ok(s) = v.extract::<String>() {
                writer.set_meta(key, MetaValue::String(s));
            } else if let Ok(x) = v.extract::<u64>() {
                writer.set_meta(key, MetaValue::U64(x));
            } else if let Ok(x) = v.extract::<i64>() {
                writer.set_meta(key, MetaValue::I64(x));
            } else if let Ok(x) = v.extract::<f64>() {
                writer.set_meta(key, MetaValue::F64(x));
            } else if let Ok(x) = v.extract::<bool>() {
                writer.set_meta(key, MetaValue::Bool(x));
            } else {
                return Err(PyTypeError::new_err(format!(
                    "metadata[{key}] must be str, int, float, or bool"
                )));
            }
        }
    }
    let mut names: Vec<String> = Vec::new();
    for k in tensors.keys() {
        names.push(k.extract::<String>()?);
    }
    names.sort();
    for name in names {
        let spec = tensors
            .get_item(&name)?
            .ok_or_else(|| PyKeyError::new_err(name.clone()))?;
        let (shape, ggml, bytes) = parse_tensor_spec(&name, &spec)?;
        writer
            .add_tensor_bytes(&name, shape, ggml, bytes)
            .map_err(|e| PyValueError::new_err(format!("write_gguf tensor {name}: {e}")))?;
    }
    writer
        .write_to_path(path)
        .map_err(|e| PyValueError::new_err(format!("write_gguf({path}): {e}")))?;
    Ok(())
}
