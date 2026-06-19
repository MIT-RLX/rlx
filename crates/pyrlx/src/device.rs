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

//! String <-> Device parsing and availability lookup.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rlx_runtime::cost::fastest_device_for_with_policy;
use rlx_runtime::{
    Device, DevicePolicy, device_label as runtime_device_label, device_report, fastest_device_for,
    parse_device as runtime_parse_device,
};

use crate::graph::{PyGraph, graph_ref};
use crate::graph_devices::{PyDeviceCandidate, PyDevicePolicy};

pub(crate) fn parse_device(s: &str) -> PyResult<Device> {
    runtime_parse_device(s).map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(crate) fn device_label(d: Device) -> &'static str {
    runtime_device_label(d)
}

/// `pyrlx.available_devices()` — list of devices that have a backend
/// registered in this build.
#[pyfunction]
pub(crate) fn available_devices() -> Vec<&'static str> {
    rlx_runtime::available_devices()
        .into_iter()
        .map(device_label)
        .collect()
}

/// `pyrlx.is_available("cuda")`
#[pyfunction]
pub(crate) fn is_available(name: &str) -> PyResult<bool> {
    Ok(rlx_runtime::is_available(parse_device(name)?))
}

/// `pyrlx.parse_device("metal")` → `"metal"` (raises on unknown names).
#[pyfunction]
pub(crate) fn parse_device_py(name: &str) -> PyResult<&'static str> {
    Ok(device_label(parse_device(name)?))
}

/// `pyrlx.backends_manifest()` — JSON of Cargo features compiled into this wheel.
#[pyfunction]
pub(crate) fn backends_manifest() -> String {
    rlx_runtime::BackendsManifest::json().to_string()
}

/// Cost-model pick for a graph without building a `GraphDevices` runner.
#[pyfunction]
#[pyo3(signature = (graph, policy=None))]
pub(crate) fn fastest_device_for_py(
    graph: &Bound<'_, PyGraph>,
    policy: Option<PyRef<PyDevicePolicy>>,
) -> PyResult<&'static str> {
    let binding = graph.borrow();
    let g = graph_ref(&binding)?;
    let device = match policy {
        Some(p) => fastest_device_for_with_policy(g, &p.to_policy()),
        None => fastest_device_for(g),
    };
    Ok(device_label(device))
}

/// Per-backend viability report for a graph (blockers + recommended pick).
#[pyfunction]
#[pyo3(signature = (graph, policy=None))]
pub(crate) fn device_report_py(
    graph: &Bound<'_, PyGraph>,
    policy: Option<PyRef<PyDevicePolicy>>,
) -> PyResult<Vec<PyDeviceCandidate>> {
    let binding = graph.borrow();
    let g = graph_ref(&binding)?;
    let policy = match policy {
        Some(p) => p.to_policy(),
        None => DevicePolicy::all(),
    };
    Ok(device_report(g, &policy)
        .into_iter()
        .map(PyDeviceCandidate::from)
        .collect())
}
