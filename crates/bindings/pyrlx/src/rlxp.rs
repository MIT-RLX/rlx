// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `.rlxp` open / GGUF→RLXP helpers for Python.
//!
//! - [`load_rlxp`] — open a pack; return name, features, tensor names, sidecars
//! - [`verify_rlxp`] — checksum-verify every tensor and sidecar
//! - [`convert_gguf_to_rlxp`] — weight-oriented import (`include_graph` optional)
//!
//! Spec: `docs/rlxp.md`. Full compile/run stays on the Rust
//! `rlx_runtime::pkg` path.

use pyo3::prelude::*;
use rlx_bind::runtime::io::weights;
use rlx_bind::runtime::serde_json::json;

use crate::bind::{from_json, to_py};

/// A package's manifest: `{name, format_version, compat_version, features,
/// has_graph, tensors, sidecars}`.
#[pyfunction]
#[pyo3(signature = (path))]
pub fn load_rlxp(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let m = weights::open_rlxp(path).map_err(to_py)?;
    Ok(from_json(py, &m))
}

/// Check every tensor and sidecar hash: `{ok, tensors_checked, …, failures}`.
#[pyfunction]
pub fn verify_rlxp(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let r = weights::verify_rlxp(path).map_err(to_py)?;
    Ok(from_json(py, &r))
}

fn convert(
    from: &str,
    src: &str,
    out: &str,
    include_graph: bool,
    auto_tier: bool,
    container: &str,
    keep_packed: bool,
) -> PyResult<()> {
    let opts = json!({
        "include_graph": include_graph,
        "auto_tier": auto_tier,
        "container": container,
        "keep_packed": keep_packed,
    });
    weights::to_rlxp(from, src, out, &opts).map_err(to_py)
}

#[pyfunction]
#[pyo3(signature = (gguf_path, out_path, *, include_graph=false, auto_tier=true, container="flat"))]
pub fn convert_gguf_to_rlxp(
    gguf_path: &str,
    out_path: &str,
    include_graph: bool,
    auto_tier: bool,
    container: &str,
) -> PyResult<()> {
    convert(
        "gguf",
        gguf_path,
        out_path,
        include_graph,
        auto_tier,
        container,
        false,
    )
}

#[pyfunction]
#[pyo3(signature = (mlx_path, out_path, *, include_graph=false, auto_tier=true, container="flat", keep_packed=false))]
pub fn convert_mlx_to_rlxp(
    mlx_path: &str,
    out_path: &str,
    include_graph: bool,
    auto_tier: bool,
    container: &str,
    keep_packed: bool,
) -> PyResult<()> {
    convert(
        "mlx",
        mlx_path,
        out_path,
        include_graph,
        auto_tier,
        container,
        keep_packed,
    )
}

#[pyfunction]
#[pyo3(signature = (dduf_path, out_path, *, include_graph=false, auto_tier=true, container="flat"))]
pub fn convert_dduf_to_rlxp(
    dduf_path: &str,
    out_path: &str,
    include_graph: bool,
    auto_tier: bool,
    container: &str,
) -> PyResult<()> {
    convert(
        "dduf",
        dduf_path,
        out_path,
        include_graph,
        auto_tier,
        container,
        false,
    )
}

#[pyfunction]
#[pyo3(signature = (nemo_path, out_path, *, include_graph=false, auto_tier=true, container="flat"))]
pub fn convert_nemo_to_rlxp(
    nemo_path: &str,
    out_path: &str,
    include_graph: bool,
    auto_tier: bool,
    container: &str,
) -> PyResult<()> {
    convert(
        "nemo",
        nemo_path,
        out_path,
        include_graph,
        auto_tier,
        container,
        false,
    )
}

#[pyfunction]
#[pyo3(signature = (pt_path, out_path, *, include_graph=false, auto_tier=true, container="flat"))]
pub fn convert_pt_to_rlxp(
    pt_path: &str,
    out_path: &str,
    include_graph: bool,
    auto_tier: bool,
    container: &str,
) -> PyResult<()> {
    convert(
        "pt",
        pt_path,
        out_path,
        include_graph,
        auto_tier,
        container,
        false,
    )
}

/// Convert a model into an `.rlxp` package: `from_` is `"gguf"`, `"mlx"`,
/// `"dduf"`, `"nemo"` or `"pt"`; `options` is `{include_graph, auto_tier,
/// container, keep_packed}`. The `convert_*_to_rlxp` functions are this with
/// `from_` fixed.
#[pyfunction]
#[pyo3(signature = (from_, src, dst, options=None))]
pub fn to_rlxp(
    from_: &str,
    src: &str,
    dst: &str,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let opts = crate::bind::opt_json(options)?;
    weights::to_rlxp(from_, src, dst, &opts).map_err(to_py)
}
