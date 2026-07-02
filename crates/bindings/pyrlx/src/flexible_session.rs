// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! `pyrlx.FlexibleSession` — defer backend choice until compile time.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rlx_runtime::FlexibleSession;

use crate::device::parse_device;
use crate::fusion_options::{PyFusionOptions, build_compile_options};
use crate::graph::PyGraph;
use crate::graph_devices::take_graph;
use crate::session::{PyCompiled, shape_to_static_dims};

#[pyclass(name = "FlexibleSession", module = "pyrlx._pyrlx")]
pub(crate) struct PyFlexibleSession {
    inner: FlexibleSession,
}

#[pymethods]
impl PyFlexibleSession {
    #[new]
    #[pyo3(signature = (policy=None))]
    fn new(policy: Option<PyRef<crate::graph_devices::PyDevicePolicy>>) -> Self {
        let inner = match policy {
            Some(p) => FlexibleSession::new().with_device_policy(p.to_policy()),
            None => FlexibleSession::new(),
        };
        Self { inner }
    }

    #[staticmethod]
    fn from_env() -> Self {
        Self {
            inner: FlexibleSession::from_env(),
        }
    }

    #[pyo3(signature = (graph, device=None))]
    fn compile_resolved(
        &self,
        graph: &Bound<'_, PyGraph>,
        device: Option<&str>,
    ) -> PyResult<PyCompiled> {
        let g = take_graph(graph)?;
        let output_shapes: Vec<Vec<usize>> = g
            .outputs
            .iter()
            .map(|id| shape_to_static_dims(g.shape(*id)))
            .collect();
        let hint = match device {
            Some(d) => Some(parse_device(d)?),
            None => None,
        };
        let compiled = self
            .inner
            .compile_resolved(g, hint)
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyCompiled::from_compiled(compiled, output_shapes))
    }

    #[pyo3(signature = (graph, device=None, fusion_options=None, kernel_dispatch=None))]
    fn compile_with_resolved(
        &self,
        graph: &Bound<'_, PyGraph>,
        device: Option<&str>,
        fusion_options: Option<PyRef<PyFusionOptions>>,
        kernel_dispatch: Option<&str>,
    ) -> PyResult<PyCompiled> {
        let g = take_graph(graph)?;
        let output_shapes: Vec<Vec<usize>> = g
            .outputs
            .iter()
            .map(|id| shape_to_static_dims(g.shape(*id)))
            .collect();
        let hint = match device {
            Some(d) => Some(parse_device(d)?),
            None => None,
        };
        let opts = build_compile_options(self.inner.precision(), fusion_options, kernel_dispatch)?;
        let compiled = self
            .inner
            .compile_resolved_with(g, hint, &opts)
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyCompiled::from_compiled(compiled, output_shapes))
    }

    fn __repr__(&self) -> String {
        "<pyrlx.FlexibleSession>".into()
    }
}
