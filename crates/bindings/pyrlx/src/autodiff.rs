// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Python-facing autodiff entry point.
//!
//! Wraps `rlx_opt::autodiff::grad_with_loss` so Python callers can
//! transform a forward graph into one that produces `[loss, grad...]`
//! given a list of NodeIds to differentiate against (typically Param
//! ids returned by `Graph.param(...)`).
//!
//! Caller seeds `d_output` (an Input on the returned graph) with the
//! upstream gradient — typically `1.0` for "differentiate the loss
//! directly." For Hello Resistor we just feed `[1.0]`.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rlx_bind::runtime::autodiff;

use crate::bind::to_py;
use crate::graph::PyGraph;

/// Borrow the forward graph, run a transform, wrap the result.
fn transform(
    graph: &Bound<'_, PyGraph>,
    what: &str,
    f: impl FnOnce(&rlx_ir::Graph) -> rlx_bind::ApiResult<rlx_ir::Graph>,
) -> PyResult<PyGraph> {
    let borrowed = graph.borrow();
    let inner = borrowed.inner.as_ref().ok_or_else(|| {
        PyRuntimeError::new_err(format!("{what}: input Graph has already been consumed"))
    })?;
    Ok(PyGraph {
        inner: Some(f(inner).map_err(to_py)?),
    })
}

/// `grad(graph, wrt)` → a graph whose outputs are `[loss, *d(loss)/d(wrt)]`.
///
/// `wrt` is a list of node ids. The result gains a `d_output` input for the
/// upstream gradient — seed it with `[1.0]` to differentiate the loss itself.
#[pyfunction]
pub(crate) fn grad(graph: &Bound<'_, PyGraph>, wrt: Vec<u32>) -> PyResult<PyGraph> {
    transform(graph, "grad", |g| autodiff::grad(g, wrt))
}

/// `jvp(graph, tangent_for)` → `[*primals, *tangents]`, with one
/// `tangent_<name>` input per seeded node.
#[pyfunction]
pub(crate) fn jvp(graph: &Bound<'_, PyGraph>, tangent_for: Vec<u32>) -> PyResult<PyGraph> {
    transform(graph, "jvp", |g| autodiff::jvp(g, tangent_for))
}

/// Hessian-vector product, forward-over-reverse.
#[pyfunction]
pub(crate) fn hvp(graph: &Bound<'_, PyGraph>, wrt: Vec<u32>) -> PyResult<PyGraph> {
    transform(graph, "hvp", |g| autodiff::hvp(g, wrt))
}

/// The `order`-th derivative w.r.t. the scalar input `wrt_name`.
#[pyfunction]
pub(crate) fn nth_order_grad(
    graph: &Bound<'_, PyGraph>,
    wrt_name: &str,
    order: usize,
) -> PyResult<PyGraph> {
    transform(graph, "nth_order_grad", |g| {
        autodiff::nth_order_grad(g, wrt_name, order)
    })
}

/// Directional higher-order derivative for non-scalar inputs; the result gains
/// inputs `dir_0` … `dir_{order-1}`.
#[pyfunction]
pub(crate) fn directional_nth_grad(
    graph: &Bound<'_, PyGraph>,
    wrt_name: &str,
    order: usize,
) -> PyResult<PyGraph> {
    transform(graph, "directional_nth_grad", |g| {
        autodiff::directional_nth_grad(g, wrt_name, order)
    })
}

/// Batch the named inputs along a new leading axis of `batch_size`.
#[pyfunction(name = "vmap")]
pub(crate) fn vmap_py(
    graph: &Bound<'_, PyGraph>,
    batched_input_names: Vec<String>,
    batch_size: usize,
) -> PyResult<PyGraph> {
    transform(graph, "vmap", |g| {
        autodiff::vmap(g, batched_input_names, batch_size)
    })
}
