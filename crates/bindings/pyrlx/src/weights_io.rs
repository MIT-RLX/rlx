// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Load MLX / DDUF / NeMo / PyTorch weights into Python dicts of float lists
//! (caller can `np.asarray`).

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rlx_bind::runtime::io::weights::{self, MlxEntry};

use crate::bind::to_py;

fn as_dict(py: Python<'_>, entries: Vec<(String, Vec<f32>)>) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    for (k, v) in entries {
        d.set_item(k, v)?;
    }
    Ok(d.into())
}

/// `{name: [f32]}` from an MLX checkpoint; `keep_packed=True` keeps each
/// quantized linear's bindings as raw `bytes`.
#[pyfunction]
#[pyo3(signature = (path, *, keep_packed=false))]
pub fn load_mlx(py: Python<'_>, path: &str, keep_packed: bool) -> PyResult<PyObject> {
    if !keep_packed {
        return as_dict(py, weights::load("mlx", path).map_err(to_py)?);
    }
    let d = PyDict::new_bound(py);
    for (name, entry) in weights::load_mlx_packed(path).map_err(to_py)? {
        match entry {
            MlxEntry::Packed(bytes) => d.set_item(name, PyBytes::new_bound(py, &bytes))?,
            MlxEntry::F32(data) => d.set_item(name, data)?,
        }
    }
    Ok(d.into())
}

/// `{name: [f32]}` from a DDUF archive.
#[pyfunction]
#[pyo3(signature = (path))]
pub fn load_dduf(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    as_dict(py, weights::load("dduf", path).map_err(to_py)?)
}

/// `{name: [f32]}` from a NeMo checkpoint.
#[pyfunction]
#[pyo3(signature = (path))]
pub fn load_nemo(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    as_dict(py, weights::load("nemo", path).map_err(to_py)?)
}

/// `{name: [f32]}` from a PyTorch checkpoint.
#[pyfunction]
#[pyo3(signature = (path))]
pub fn load_pt(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    as_dict(py, weights::load("pt", path).map_err(to_py)?)
}

/// `{name: [f32]}` from a weights file of `format`: `"mlx"`, `"safetensors"`,
/// `"dduf"`, `"nemo"` or `"pt"` — `load_mlx` and the rest with the format
/// fixed.
#[pyfunction]
#[pyo3(signature = (format, path))]
pub fn load_weights(py: Python<'_>, format: &str, path: &str) -> PyResult<PyObject> {
    as_dict(py, weights::load(format, path).map_err(to_py)?)
}
