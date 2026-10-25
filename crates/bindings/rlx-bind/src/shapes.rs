// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Output-shape helpers for the IR calls that take an explicit `Shape`.
//!
//! The bindings never ask the caller for a shape the operands determine; the
//! op table computes it with these, or with `rlx_ir::shape::*` directly.

use rlx_ir::{Dim, Graph, NodeId, Shape};

use crate::{ApiError, ApiResult};

/// Static dims of a node, or an error naming the dynamic axis.
pub fn static_dims(g: &Graph, n: NodeId, what: &str) -> ApiResult<Vec<usize>> {
    g.shape(n)
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(v) => Ok(*v),
            Dim::Dynamic(name) => Err(ApiError::type_(format!(
                "{what}: needs a static shape, but axis '{name}' is dynamic"
            ))),
        })
        .collect()
}

/// `(rows, cols)` of a static 2-D matrix.
pub fn matrix_dims(g: &Graph, n: NodeId, what: &str) -> ApiResult<(usize, usize)> {
    let d = static_dims(g, n, what)?;
    match d.as_slice() {
        [m, n] => Ok((*m, *n)),
        _ => Err(ApiError::type_(format!(
            "{what}: expected a 2-D matrix, got rank {}",
            d.len()
        ))),
    }
}

/// `x`'s shape with the last axis replaced by `n` — the output of a matmul
/// whose weight is a packed blob the shape rules cannot read.
pub fn last_axis(g: &Graph, x: NodeId, n: usize) -> Shape {
    let s = g.shape(x);
    let mut dims: Vec<Dim> = s.dims().to_vec();
    if let Some(last) = dims.last_mut() {
        *last = Dim::Static(n);
    }
    Shape::from_dims(&dims, s.dtype())
}

/// `[batch, seq, D·hidden]` for the recurrent ops (`D = 2` when bidirectional).
pub fn recurrent(g: &Graph, x: NodeId, hidden: usize, bidirectional: bool) -> Shape {
    let d = if bidirectional { 2 } else { 1 };
    g.shape(x).clone().with_dim(2, Dim::Static(d * hidden))
}

pub fn matmul(g: &Graph, a: NodeId, b: NodeId, what: &str) -> ApiResult<Shape> {
    rlx_ir::shape::matmul_shape(g.shape(a), g.shape(b))
        .map_err(|e| ApiError::type_(format!("{what} shape: {e}")))
}

/// Output of a one-axis reduction.
pub fn reduced(g: &Graph, x: NodeId, axis: usize, keep_dim: bool) -> ApiResult<Shape> {
    rlx_ir::shape::reduce_shape(g.shape(x), &[axis], keep_dim)
        .map_err(|e| ApiError::type_(format!("reduce shape: {e}")))
}

/// Same shape as `x`.
pub fn same(g: &Graph, x: NodeId) -> Shape {
    rlx_ir::shape::unary_shape(g.shape(x))
}
