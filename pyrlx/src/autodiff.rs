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
