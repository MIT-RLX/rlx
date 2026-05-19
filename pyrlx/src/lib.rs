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

//! pyrlx — Python bindings for RLX.
//!
//! Layered API:
//! * `available_devices()` / `is_available(name)` — query the build's backends.
//! * `Graph` — builder over `rlx_ir::Graph` for hand-rolled test graphs.
//! * `Session(device, precision)` — backend selection at construction.
//! * `CompiledGraph.set_param/run` — the hot-path execution surface.
//! * `Embed` (feature `embed`) — load BERT / NomicBERT / NomicVision and
//!   run on any registered backend.

use pyo3::prelude::*;

mod autodiff;
mod device;
mod dtype;
#[cfg(feature = "embed")]
mod embed;
mod graph;
mod session;

/// Module init — `import pyrlx._pyrlx`.
#[pymodule]
fn _pyrlx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(device::available_devices, m)?)?;
    m.add_function(wrap_pyfunction!(device::is_available, m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::grad, m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::jvp, m)?)?;
    m.add_function(wrap_pyfunction!(autodiff::vmap_py, m)?)?;

    m.add_class::<graph::PyGraph>()?;
    m.add_class::<session::PySession>()?;
    m.add_class::<session::PyCompiled>()?;

    #[cfg(feature = "embed")]
    m.add_class::<embed::PyEmbed>()?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
