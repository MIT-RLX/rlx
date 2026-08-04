// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

    /// Materialize an N-dimensional constant tensor from row-major `data`
    /// (given as `f64`, encoded to `dtype`'s element width). Backs the `rlx!`
    /// array-`const` form (`const mask = [[0,1],[1,0]] : F32;`). `data.len()`
    /// must equal the product of `dims`. Supports `F32`/`F64`/`I32`/`I64`.
    pub fn constant_nd(&mut self, data: Vec<f64>, dims: Vec<usize>, dtype: DType) -> Tensor {
        let n: usize = dims.iter().product();
        assert_eq!(
            data.len(),
            n,
            "constant_nd: {} values for shape {:?} ({n} expected)",
            data.len(),
            dims
        );
        let bytes: Vec<u8> = match dtype {
            DType::F32 => data
                .iter()
                .flat_map(|v| (*v as f32).to_le_bytes())
                .collect(),
            DType::F64 => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            DType::I32 => data
                .iter()
                .flat_map(|v| (*v as i32).to_le_bytes())
                .collect(),
            DType::I64 => data
                .iter()
                .flat_map(|v| (*v as i64).to_le_bytes())
                .collect(),
            DType::U8 => data.iter().map(|v| *v as u8).collect(),
            other => panic!("constant_nd: unsupported dtype {other:?} (use F32/F64/I32/I64/U8)"),
        };
        let shape = Shape::new(&dims, dtype);
        let id = self
            .handle
            .with_graph(|g| g.add_node(rlx_ir::Op::Constant { data: bytes }, vec![], shape));
        Tensor::new(self.handle.clone(), id)
    }

    /// Build a bounded sequential scan (`Op::Scan`) — a single compact loop
    /// node, not an unrolled copy. `init` is the loop carry; `bcasts` are values
    /// held constant across iterations (typically weights); `length` is the
    /// iteration count. `body(carry, bcasts) -> next_carry` is traced ONCE into
    /// a standalone body graph, so the result is `O(1)` IR regardless of
    /// `length` (unlike an unrolled `repeat`). The next carry must match the
    /// carry's shape. Backs the `rlx!` `scan` construct.
    ///
    /// ```rust
    /// use rlx_tensor::{graph, shape};
    /// let g = graph("rnn", |s| {
    ///     let h0 = s.input("h0", shape![1, 8]);
    ///     let w  = s.param("w", shape![8, 8]);
    ///     s.scan_block(&h0, &[&w], 12, |h, bc| (h.matmul(&bc[0])).tanh())
    /// });
    /// assert_eq!(g.outputs.len(), 1);
    /// ```
    pub fn scan_block(
        &mut self,
        init: &Tensor,
        bcasts: &[&Tensor],
        length: u32,
        body: impl FnOnce(&Tensor, &[Tensor]) -> Tensor,
    ) -> Tensor {
        // Ensure carry + bcasts live in this graph (no-op if already here).
        let init_id = self.handle.adopt(&init.handle, init.id);
        let bcast_ids: Vec<NodeId> = bcasts
            .iter()
            .map(|b| self.handle.adopt(&b.handle, b.id))
            .collect();

        // Trace the loop body once into a standalone graph. Its `Op::Input`s, in
        // order, are `[carry, bcast_0, …]` — the order `scan_with_bcasts_and_xs`
        // expects.
        let mut inner = GraphScope::new("scan_body");
        let carry = inner.input("carry", init.shape());
        let bcast_ins: Vec<Tensor> = bcasts
            .iter()
            .enumerate()
            .map(|(i, b)| inner.input(format!("bcast_{i}"), b.shape()))
            .collect();
        let next = body(&carry, &bcast_ins);
        inner.set_outputs([next.id]);
        let body_graph = inner.finish();

        let id = self.handle.with_graph(|g| {
            g.scan_with_bcasts_and_xs(init_id, &bcast_ids, &[], body_graph, length)
        });
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
