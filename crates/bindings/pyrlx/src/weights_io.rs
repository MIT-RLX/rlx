// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Load MLX / DDUF / NeMo / PyTorch weights into Python dicts of float lists
//! (caller can `np.asarray`).

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

fn map_err(e: anyhow::Error) -> PyErr {
    PyIOError::new_err(format!("{e:#}"))
}

/// Load MLX weights (dir / safetensors / npz / npy) → `{name: list[float]}`.
///
/// When `keep_packed=True`, quantized Linears stay as
/// `{base}.weight` (bytes), `{base}.scales`, `{base}.biases` instead of
/// dequantizing to dense f32.
#[pyfunction]
#[pyo3(signature = (path, *, keep_packed=false))]
pub fn load_mlx(py: Python<'_>, path: &str, keep_packed: bool) -> PyResult<PyObject> {
    if !keep_packed {
        let map = rlx_mlx_io::load_f32_map(path).map_err(map_err)?;
        let d = PyDict::new_bound(py);
        for (k, v) in map {
            d.set_item(k, v)?;
        }
        return Ok(d.into());
    }
    let mut w = rlx_mlx_io::load_path(path).map_err(map_err)?;
    let linears = rlx_mlx_io::collect_packed_linears(&mut w).map_err(map_err)?;
    let d = PyDict::new_bound(py);
    for b in &linears {
        for (name, bytes, _dt) in rlx_mlx_io::param_bindings_for(b) {
            d.set_item(name, PyBytes::new_bound(py, &bytes))?;
        }
    }
    // Remaining dense tensors as f32 lists.
    for name in w.logical_keys() {
        if let Ok((data, _shape)) = w.take_dense_f32(&name) {
            d.set_item(name, data)?;
        }
    }
    Ok(d.into())
}

/// Load a DDUF (`.dduf`) package → `{component/name: list[float]}`.
#[pyfunction]
#[pyo3(signature = (path))]
pub fn load_dduf(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let map = rlx_dduf::load_f32_map(path).map_err(map_err)?;
    let d = PyDict::new_bound(py);
    for (k, v) in map {
        d.set_item(k, v)?;
    }
    Ok(d.into())
}

/// Load a NeMo (`.nemo`) archive → `{name: list[float]}` (dense f32).
#[pyfunction]
#[pyo3(signature = (path))]
pub fn load_nemo(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let m = rlx_nemo::NemoModel::open(std::path::Path::new(path)).map_err(map_err)?;
    let d = PyDict::new_bound(py);
    for name in m.names() {
        let t = m.tensor(&name).map_err(map_err)?;
        d.set_item(name, t.data)?;
    }
    Ok(d.into())
}

/// Load a PyTorch `.pt` / `.pth` / `pytorch_model.bin` → `{name: list[float]}`.
#[pyfunction]
#[pyo3(signature = (path))]
pub fn load_pt(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let m = rlx_nemo::PtModel::open(std::path::Path::new(path)).map_err(map_err)?;
    let d = PyDict::new_bound(py);
    for name in m.names() {
        let t = m.tensor(&name).map_err(map_err)?;
        d.set_item(name, t.data)?;
    }
    Ok(d.into())
}
