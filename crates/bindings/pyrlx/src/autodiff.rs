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
use rlx_ir::NodeId;
use rlx_opt::autodiff;

use crate::graph::PyGraph;

/// `grad(graph, wrt)` — returns a new `Graph` whose outputs are
/// `[loss, dwrt_0, dwrt_1, ...]`. The original graph is borrowed,
/// not consumed — call it again with a different `wrt` if you want
/// gradients w.r.t. a different subset.
///
/// The returned graph has one extra `Input` named `"d_output"` that
/// the caller seeds (shape matches the forward output, typically `[1]`
/// for a scalar loss). The forward graph **must** have exactly one
/// output (the scalar loss).
#[pyfunction]
pub(crate) fn grad(graph: &Bound<'_, PyGraph>, wrt: Vec<u32>) -> PyResult<PyGraph> {
    let borrowed = graph.borrow();
    let inner = borrowed
        .inner
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("grad: input Graph has already been consumed"))?;
    let wrt: Vec<NodeId> = wrt.into_iter().map(NodeId).collect();
    let bwd = autodiff::grad_with_loss(inner, &wrt);
    Ok(PyGraph { inner: Some(bwd) })
}

/// `jvp(graph, tangent_for)` — forward-mode AD. Returns a new graph
/// whose outputs are `[primals..., tangents...]` (the original
/// outputs followed by their tangents in the same order).
///
/// For each `Input`/`Param` listed in `tangent_for`, the returned
/// graph gains a fresh `Input` named `"tangent_<original>"` with the
/// same shape and dtype. Caller seeds these with a perturbation
/// direction; the graph computes `(∂outputs/∂inputs) · tangents`.
///
/// Use this when the input dimension is small and the output
/// dimension is large — e.g., Circulax-style `jacfwd` over a
/// component group's flat parameter vector.
#[pyfunction]
pub(crate) fn jvp(graph: &Bound<'_, PyGraph>, tangent_for: Vec<u32>) -> PyResult<PyGraph> {
    let borrowed = graph.borrow();
    let inner = borrowed
        .inner
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("jvp: input Graph has already been consumed"))?;
    let wrt: Vec<NodeId> = tangent_for.into_iter().map(NodeId).collect();
    let fwd_graph = rlx_opt::autodiff_fwd::jvp(inner, &wrt);
    Ok(PyGraph {
        inner: Some(fwd_graph),
    })
}

/// `hvp(graph, wrt)` — Hessian-vector product via forward-over-reverse AD.
///
/// Returns a graph whose outputs are `[primals..., grads..., tangent_primals...,
/// tangent_grads...]`. Tangent inputs are named `tangent_<original>`.
#[pyfunction]
pub(crate) fn hvp(graph: &Bound<'_, PyGraph>, wrt: Vec<u32>) -> PyResult<PyGraph> {
    let borrowed = graph.borrow();
    let inner = borrowed
        .inner
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("hvp: input Graph has already been consumed"))?;
    let wrt: Vec<NodeId> = wrt.into_iter().map(NodeId).collect();
    let hg = rlx_opt::autodiff_fwd::hvp(inner, &wrt);
    Ok(PyGraph { inner: Some(hg) })
}

/// `nth_order_grad(graph, wrt_name, order)` — stack reverse-mode AD `order` times.
///
/// The forward graph must have a single scalar output. Returns a graph whose
/// sole output is `d^order f / d(wrt)^order`.
#[pyfunction]
pub(crate) fn nth_order_grad(
    graph: &Bound<'_, PyGraph>,
    wrt_name: &str,
    order: usize,
) -> PyResult<PyGraph> {
    let borrowed = graph.borrow();
    let inner = borrowed.inner.as_ref().ok_or_else(|| {
        PyRuntimeError::new_err("nth_order_grad: input Graph has already been consumed")
    })?;
    let hg = rlx_opt::nth_order_grad(inner, wrt_name, order);
    Ok(PyGraph { inner: Some(hg) })
}

/// `directional_nth_grad(graph, wrt_name, order)` — directional higher-order grad.
///
/// Creates `dir_0` … `dir_{order-1}` inputs for per-level contraction.
#[pyfunction]
pub(crate) fn directional_nth_grad(
    graph: &Bound<'_, PyGraph>,
    wrt_name: &str,
    order: usize,
) -> PyResult<PyGraph> {
    let borrowed = graph.borrow();
    let inner = borrowed.inner.as_ref().ok_or_else(|| {
        PyRuntimeError::new_err("directional_nth_grad: input Graph has already been consumed")
    })?;
    let dirs: Vec<&str> = (0..order)
        .map(|i| {
            // Names are ignored by the Rust API — inputs are always dir_<level>.
            let _ = i;
            "v"
        })
        .collect();
    let hg = rlx_opt::directional_nth_grad(inner, wrt_name, &dirs);
    Ok(PyGraph { inner: Some(hg) })
}

/// `vmap(graph, batched_input_names, batch_size)` — vectorise a
/// graph over a leading batch axis.
///
/// `batched_input_names` lists the `Op::Input` names whose leading
/// axis is the batch axis. Inputs/Params not listed are shared
/// across the batch.
///
/// The returned graph has all batched inputs widened with a leading
/// `[batch_size, ...]` dim and every reachable output gets a leading
/// batch axis. Per-op rules cover the elementwise / shape / reduce /
/// matmul / dense-solve / scan / autodiff-backward subset; ops
/// without a rule panic with a clear message.
#[pyfunction(name = "vmap")]
pub(crate) fn vmap_py(
    graph: &Bound<'_, PyGraph>,
    batched_input_names: Vec<String>,
    batch_size: usize,
) -> PyResult<PyGraph> {
    let borrowed = graph.borrow();
    let inner = borrowed
        .inner
        .as_ref()
        .ok_or_else(|| PyRuntimeError::new_err("vmap: input Graph has already been consumed"))?;
    let names: Vec<&str> = batched_input_names.iter().map(|s| s.as_str()).collect();
    let batched = rlx_opt::vmap::vmap(inner, &names, batch_size);
    Ok(PyGraph {
        inner: Some(batched),
    })
}
