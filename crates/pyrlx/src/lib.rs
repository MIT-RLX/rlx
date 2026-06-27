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
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// Edition 2024 raised `unsafe_op_in_unsafe_fn` from allow → warn.
// PyO3 0.22's `#[pymethods]` / `#[pyfunction]` macros expand into
// code that doesn't wrap each call in an `unsafe {}` block — the
// expansion is internally sound but the warning lights up dozens of
// times. Silence it crate-wide here.
#![allow(unsafe_op_in_unsafe_fn)]

//! pyrlx — Python bindings for RLX (PyO3 extension `pyrlx._pyrlx`).
//!
//! The user-facing package [`pyrlx`](../../python/pyrlx/__init__.py) re-exports
//! this module and adds a pure-Python DSL (`graph`, `Node`, `set_param`, `run`).
//!
//! # Layers
//!
//! | Layer | Types / functions |
//! |-------|-------------------|
//! | Devices | `available_devices`, `is_available`, `parse_device`, `backends_manifest` |
//! | Build | `Graph` — symbolic IR; shape-inferred where `GraphExt` allows |
//! | Compile | `Session`, `FusionOptions`, `FlexibleSession` |
//! | Execute | `CompiledGraph` — `set_param` / `run` (f32) and `_typed` variants |
//! | Multi-backend | `GraphDevices`, `DeviceRouter`, `DevicePolicy` |
//! | Transforms | `grad`, `jvp`, `hvp`, `vmap`, `nth_order_grad` |
//! | GGUF | `quantize`, `dequant`, `load_gguf`, `write_gguf`, `convert_to_gguf`, `GgufFile` |
//!
//! Graphs are consumed at compile time. Use `pyrlx.set_param` / `pyrlx.run` in
//! Python for dtype-aware NumPy I/O without manual byte packing.
use pyo3::prelude::*;

mod autodiff;
mod device;
mod device_router;
mod dtype;
mod flexible_session;
mod fusion_options;
mod gguf;
#[cfg(feature = "gguf-convert")]
mod gguf_convert;
mod graph;
mod graph_devices;
mod session;

/// Module init — `import pyrlx._pyrlx`.
#[pymodule]
fn _pyrlx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(device::available_devices, m)?)?;
    m.add_function(wrap_pyfunction!(device::is_available, m)?)?;
    m.add_function(wrap_pyfunction!(device::parse_device_py, m)?)?;
    m.add_function(wrap_pyfunction!(device::backends_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(device::fastest_device_for_py, m)?)?;
    m.add_function(wrap_pyfunction!(device::device_report_py, m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::grad, m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::jvp, m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::hvp, m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::nth_order_grad, m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::directional_nth_grad, m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::vmap_py, m)?)?;

    m.add_class::<graph::PyGraph>()?;
    m.add_class::<session::PySession>()?;
    m.add_class::<session::PyCompiled>()?;
    m.add_class::<fusion_options::PyFusionOptions>()?;
    m.add_class::<graph_devices::PyDevicePolicy>()?;
    m.add_class::<graph_devices::PyGraphDevices>()?;
    m.add_class::<graph_devices::PyDeviceCandidate>()?;
    m.add_class::<graph_devices::PyDeviceBenchResult>()?;
    m.add_class::<flexible_session::PyFlexibleSession>()?;
    m.add_class::<device_router::PyDeviceRouter>()?;

    m.add_function(wrap_pyfunction!(gguf::quantize_gguf, m)?)?;
    m.add_function(wrap_pyfunction!(gguf::dequant_gguf, m)?)?;
    m.add_class::<gguf::PyGgufFile>()?;
    m.add_function(wrap_pyfunction!(gguf::load_gguf, m)?)?;
    m.add_function(wrap_pyfunction!(gguf::write_gguf, m)?)?;
    #[cfg(feature = "gguf-convert")]
    {
        m.add_function(wrap_pyfunction!(gguf_convert::convert_to_gguf, m)?)?;
        m.add_class::<gguf_convert::PyConvertReport>()?;
    }

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
