// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! `.rlxp` open / GGUF→RLXP helpers for Python.
//!
//! - [`load_rlxp`] — open a pack; return name, features, tensor names, sidecars
//! - [`convert_gguf_to_rlxp`] — weight-oriented import (`include_graph` optional)
//!
//! Spec: `docs/rlxp.md`. Full compile/run stays on the Rust
//! `rlx_runtime::pkg` path.

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rlx_pkg::{
    ContainerKind, GgufImportOptions, Package, gguf_to_rlxp,
};

fn map_err(e: anyhow::Error) -> PyErr {
    PyIOError::new_err(format!("{e:#}"))
}

/// Open an `.rlxp` / ZIP / package directory; return a summary dict.
#[pyfunction]
#[pyo3(signature = (path))]
pub fn load_rlxp(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let pack = Package::open(path).map_err(map_err)?;
    let m = pack.manifest();
    let d = PyDict::new_bound(py);
    d.set_item("name", m.name.as_str())?;
    d.set_item("format_version", m.format_version)?;
    d.set_item("compat_version", m.compat_version)?;
    d.set_item("features", m.features.clone())?;
    d.set_item("has_graph", pack.has_graph())?;
    if let Some(idx) = pack.weights_index() {
        let names: Vec<&str> = idx.names().collect();
        d.set_item("tensors", names)?;
    } else {
        d.set_item("tensors", Vec::<String>::new())?;
    }
    let sides: Vec<&str> = m.sidecars.iter().map(|s| s.id.as_str()).collect();
    d.set_item("sidecars", sides)?;
    Ok(d.into())
}

/// Convert a GGUF file to `.rlxp`.
#[pyfunction]
#[pyo3(signature = (gguf_path, out_path, *, include_graph=false, auto_tier=true, container="flat"))]
pub fn convert_gguf_to_rlxp(
    gguf_path: &str,
    out_path: &str,
    include_graph: bool,
    auto_tier: bool,
    container: &str,
) -> PyResult<()> {
    let container = match container {
        "flat" => ContainerKind::Flat,
        "zip" => ContainerKind::Zip,
        "dir" => ContainerKind::Dir,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown container {other:?}; expected flat|zip|dir"
            )));
        }
    };
    let opts = GgufImportOptions {
        container,
        include_graph,
        compress_sidecars: true,
        auto_tier,
    };
    gguf_to_rlxp(gguf_path, out_path, &opts).map_err(map_err)
}
