// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Combining and splitting graphs: `splice`, `parallel`, `sequential`,
//! `extract` and `Partition`.
//!
//! ```python
//! g = pyrlx.Graph("stack")
//! [h] = pyrlx.splice(g, layer, prefix="l0.")
//! [h] = pyrlx.splice(g, layer, prefix="l1.", bind={"x": h})
//! g.set_outputs([h])
//!
//! p = pyrlx.Partition.split_at(g, [cut])
//! parts = [p.graph(i) for i in range(len(p))]
//! ```

use std::collections::HashMap;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rlx_bind::runtime::compose::{self, Partition};
use rlx_bind::runtime::serde_json::{Map, Value};

use crate::bind::to_py;
use crate::graph::PyGraph;

fn borrowed<T>(
    graph: &Bound<'_, PyGraph>,
    what: &str,
    f: impl FnOnce(&rlx_ir::Graph) -> PyResult<T>,
) -> PyResult<T> {
    let g = graph.borrow();
    let inner = g
        .inner
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err(format!("{what}: graph already consumed")))?;
    f(inner)
}

fn wrap(g: rlx_ir::Graph) -> PyGraph {
    PyGraph { inner: Some(g) }
}

/// Copy `source` into `target` and return `source`'s outputs as `target`
/// node ids. `bind` feeds leaves of `source` (inputs, or params) from nodes
/// of `target`; `prefix` keeps `source`'s unbound leaves apart from
/// same-named ones in `target` (by default they are shared — a tied weight);
/// `adapt_shapes` accepts bound nodes of another shape. On error `target` is
/// unchanged.
#[pyfunction]
#[pyo3(signature = (target, source, *, bind = None, prefix = None, adapt_shapes = false))]
pub(crate) fn splice(
    target: &Bound<'_, PyGraph>,
    source: &Bound<'_, PyGraph>,
    bind: Option<HashMap<String, u32>>,
    prefix: Option<String>,
    adapt_shapes: bool,
) -> PyResult<Vec<u32>> {
    let mut opts = Map::new();
    if let Some(bind) = bind {
        let bind: Map<String, Value> = bind.into_iter().map(|(k, v)| (k, v.into())).collect();
        opts.insert("bind".into(), Value::Object(bind));
    }
    if let Some(prefix) = prefix {
        opts.insert("prefix".into(), prefix.into());
    }
    opts.insert("adapt_shapes".into(), adapt_shapes.into());
    // A copy first: `source` may be `target` itself.
    let source = borrowed(source, "splice", |g| Ok(g.clone()))?;
    let mut t = target.borrow_mut();
    let inner = t
        .inner
        .as_mut()
        .ok_or_else(|| PyRuntimeError::new_err("splice: target graph already consumed"))?;
    compose::splice(inner, &source, &Value::Object(opts)).map_err(to_py)
}

/// `a` and `b` side by side: `a`'s outputs, then `b`'s.
#[pyfunction]
#[pyo3(signature = (a, b, name = None))]
pub(crate) fn parallel(
    a: &Bound<'_, PyGraph>,
    b: &Bound<'_, PyGraph>,
    name: Option<&str>,
) -> PyResult<PyGraph> {
    let b = borrowed(b, "parallel", |g| Ok(g.clone()))?;
    borrowed(a, "parallel", |a| {
        compose::parallel(a, &b, name).map(wrap).map_err(to_py)
    })
}

/// `b` after `a`: `b`'s inputs, in creation order, are fed by `a`'s outputs.
#[pyfunction]
#[pyo3(signature = (a, b, name = None))]
pub(crate) fn sequential(
    a: &Bound<'_, PyGraph>,
    b: &Bound<'_, PyGraph>,
    name: Option<&str>,
) -> PyResult<PyGraph> {
    let b = borrowed(b, "sequential", |g| Ok(g.clone()))?;
    borrowed(a, "sequential", |a| {
        compose::sequential(a, &b, name).map(wrap).map_err(to_py)
    })
}

/// The subgraph computing `outputs`, with them as its outputs.
#[pyfunction]
pub(crate) fn extract(graph: &Bound<'_, PyGraph>, outputs: Vec<u32>) -> PyResult<PyGraph> {
    borrowed(graph, "extract", |g| {
        compose::extract(g, outputs).map(wrap).map_err(to_py)
    })
}

/// A graph split into self-contained parts that run in order: part `i`
/// reads its `imports(i)` from earlier parts' `exports` of the same name.
#[pyclass(name = "Partition", module = "pyrlx._pyrlx")]
pub(crate) struct PyPartition {
    inner: Partition,
}

#[pymethods]
impl PyPartition {
    /// Node `i` goes to part `assign[i]` (`node_count()` entries).
    #[new]
    fn new(graph: &Bound<'_, PyGraph>, assign: Vec<u32>) -> PyResult<Self> {
        borrowed(graph, "Partition", |g| {
            let inner = Partition::new(g, &assign).map_err(to_py)?;
            Ok(Self { inner })
        })
    }

    /// Two parts: `cut` and everything it reads, then the rest.
    #[staticmethod]
    fn split_at(graph: &Bound<'_, PyGraph>, cut: Vec<u32>) -> PyResult<Self> {
        borrowed(graph, "Partition.split_at", |g| {
            let inner = Partition::split_at(g, cut).map_err(to_py)?;
            Ok(Self { inner })
        })
    }

    /// One part per independent subgraph; the parts import nothing.
    #[staticmethod]
    fn components(graph: &Bound<'_, PyGraph>) -> PyResult<Self> {
        borrowed(graph, "Partition.components", |g| {
            let inner = Partition::components(g).map_err(to_py)?;
            Ok(Self { inner })
        })
    }

    fn num_parts(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// A copy of part `part`'s graph.
    fn graph(&self, part: usize) -> PyResult<PyGraph> {
        self.inner.graph(part).map(wrap).map_err(to_py)
    }

    fn imports(&self, part: usize) -> PyResult<Vec<String>> {
        self.inner.imports(part).map_err(to_py)
    }

    fn exports(&self, part: usize) -> PyResult<Vec<String>> {
        self.inner.exports(part).map_err(to_py)
    }

    /// Where each source output went: `[(part, output index), …]`.
    fn outputs(&self) -> Vec<(usize, usize)> {
        self.inner.0.outputs.clone()
    }

    /// The parts as one graph again.
    fn merge(&self) -> PyResult<PyGraph> {
        self.inner.merge().map(wrap).map_err(to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyrlx.Partition {} parts={}>",
            self.inner.0.name,
            self.inner.len()
        )
    }
}
