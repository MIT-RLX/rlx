// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Graph builder scope.

use rlx_ir::{DType, Dim, Graph, GraphExt, NodeId, Shape};

use crate::Tensor;
use crate::handle::GraphHandle;
use crate::scalar::promote_scalar;

/// Owns a traced graph while you build symbolic tensors.
#[derive(Debug)]
pub struct GraphScope {
    handle: GraphHandle,
    /// Monotonic counter for fresh dynamic-dim symbols.
    next_dyn: u32,
}

impl GraphScope {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            handle: GraphHandle::new(Graph::new(name)),
            next_dyn: 0,
        }
    }

    pub fn graph(&self) -> Graph {
        self.handle.with_graph(|g| g.clone())
    }

    /// Fresh dynamic-dimension symbol (for variable batch / sequence length).
    pub fn fresh_symbol(&mut self) -> u32 {
        let s = self.next_dyn;
        self.next_dyn += 1;
        s
    }

    pub fn input(&mut self, name: impl Into<String>, shape: impl Into<Shape>) -> Tensor {
        let id = self.handle.with_graph(|g| g.input(name, shape.into()));
        Tensor::new(self.handle.clone(), id)
    }

    /// Input with a caller-chosen dynamic symbol on one axis.
    pub fn input_dyn(&mut self, name: impl Into<String>, dims: &[Dim], dtype: DType) -> Tensor {
        let shape = Shape::from_dims(dims, dtype);
        self.input(name, shape)
    }

    pub fn param(&mut self, name: impl Into<String>, shape: impl Into<Shape>) -> Tensor {
        let id = self.handle.with_graph(|g| g.param(name, shape.into()));
        Tensor::new(self.handle.clone(), id)
    }

    /// Rank-0 literal broadcastable in binary ops.
    pub fn scalar(&mut self, value: f64) -> Tensor {
        let id = self
            .handle
            .with_graph(|g| promote_scalar(g, value.into(), DType::F32));
        Tensor::new(self.handle.clone(), id)
    }

    pub fn constant(&mut self, value: f64, dtype: DType) -> Tensor {
        let id = self.handle.with_graph(|g| g.constant(value, dtype));
        Tensor::new(self.handle.clone(), id)
    }

    /// Stack existing tensors along `axis` (zero-copy at IR level).
    pub fn cat(&mut self, tensors: &[&Tensor], axis: usize) -> Tensor {
        let ids: Vec<NodeId> = tensors.iter().map(|t| t.id).collect();
        let id = self.handle.with_graph(|g| g.concat_(ids, axis));
        Tensor::new(self.handle.clone(), id)
    }

    pub fn set_outputs(&mut self, outputs: impl IntoIterator<Item = impl Into<NodeId>>) {
        let ids: Vec<NodeId> = outputs.into_iter().map(Into::into).collect();
        self.handle.with_graph(|g| g.set_outputs(ids));
    }

    pub fn finish(self) -> Graph {
        self.handle.with_graph(|g| g.clone())
    }
}

/// Build a single-output graph from a closure.
///
/// ```rust
/// use rlx_tensor::{graph, shape};
///
/// let g = graph("mlp", |g| {
///     let x = g.input("x", shape![2, 4]);
///     let w = g.param("w", shape![4, 3]);
///     let b = g.param("b", shape![3]);
///     (&x.matmul(&w) + &b).gelu() * 2.0
/// });
/// assert_eq!(g.outputs.len(), 1);
/// ```
pub fn graph<F>(name: impl Into<String>, build: F) -> Graph
where
    F: FnOnce(&mut GraphScope) -> Tensor,
{
    let mut scope = GraphScope::new(name);
    let out = build(&mut scope);
    scope.set_outputs([out]);
    scope.finish()
}

/// Build a graph and return extra metadata from the closure.
pub fn graph_with<F, R>(name: impl Into<String>, build: F) -> (Graph, R)
where
    F: FnOnce(&mut GraphScope) -> R,
{
    let mut scope = GraphScope::new(name);
    let result = build(&mut scope);
    (scope.finish(), result)
}
