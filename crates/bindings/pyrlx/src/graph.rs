// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `pyrlx.Graph` — Python builder over `rlx_ir::Graph`.
//!
//! # Where the methods come from
//!
//! The builder methods are not written here: they are generated from the op
//! table in `rlx-bind` (one row per op, shared with rlx-js, C and Swift) by
//! [`py_graph_ops!`](crate::py_graph_ops) in [`crate::bind`]. Where the IR
//! infers an output shape the method omits it (`matmul`, `add`, `conv2d`);
//! where it requires one the signature includes it (`matmul_with_shape`,
//! `lora_matmul`). A row's options group becomes trailing
//! keyword-or-positional parameters with defaults
//! (`conv2d(x, w, kernel_size=3, stride=2)`).
//!
//! # Naming
//!
//! Reserved Python keywords get a trailing underscore: `where_`, `eq_`, `lt_`,
//! `le_`, `gt_`, `ge_`, `ne_`. A few rows add a second name for an older
//! spelling (`rope_n`, `fused_residual_ln`).
//!
//! # Lifecycle
//!
//! `Session.compile` moves the inner `Graph` out of `PyGraph`; subsequent calls
//! raise "graph already consumed".

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rlx_ir::Graph;

#[pyclass(name = "Graph", module = "pyrlx._pyrlx")]
pub(crate) struct PyGraph {
    pub(crate) inner: Option<Graph>,
}

pub(crate) fn graph_ref(g: &PyGraph) -> PyResult<&Graph> {
    g.inner
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("graph already consumed"))
}

fn consumed() -> PyErr {
    PyRuntimeError::new_err("this Graph has been consumed by Session.compile() — build a new one")
}

impl PyGraph {
    pub(crate) fn g(&mut self) -> PyResult<&mut Graph> {
        self.inner.as_mut().ok_or_else(consumed)
    }
}

#[pymethods]
impl PyGraph {
    #[new]
    #[pyo3(signature = (name = "graph"))]
    fn new(name: &str) -> Self {
        Self {
            inner: Some(Graph::new(name.to_string())),
        }
    }

    /// A deep copy — compile one, keep building on the other.
    fn clone(&self) -> PyResult<Self> {
        let g = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("graph already consumed"))?;
        Ok(Self {
            inner: Some(g.clone()),
        })
    }

    /// Whether a compile (or a body-graph argument) consumed it.
    fn is_consumed(&self) -> bool {
        self.inner.is_none()
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.as_ref().ok_or_else(consumed)?.len())
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(g) => format!("<pyrlx.Graph nodes={}>", g.len()),
            None => "<pyrlx.Graph (consumed by compile)>".into(),
        }
    }
}
