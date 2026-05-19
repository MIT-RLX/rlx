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

//! Tracing API — build IR graphs by recording operations on traced tensors.
//!
//! ```rust
//! use rlx_runtime::trace::*;
//! use rlx_ir::{DType, shape::Dim};
//!
//! let graph = trace("model", |t| {
//!     let x = t.input("x", &[4, 384], DType::F32);
//!     let w = t.param("w", &[384, 1536], DType::F32);
//!     let b = t.param("b", &[1536], DType::F32);
//!     let mm = t.matmul(x, w);
//!     let out = (mm + b).gelu();
//!     vec![out]
//! });
//! ```

use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::cell::RefCell;
use std::rc::Rc;

/// A traced tensor — records ops instead of executing them.
#[derive(Clone)]
pub struct TracedTensor {
    pub(crate) id: NodeId,
    graph: Rc<RefCell<Graph>>,
}

/// Records operations into an IR graph.
pub struct Tracer {
    graph: Rc<RefCell<Graph>>,
}

impl Tracer {
    fn new(name: &str) -> Self {
        Self {
            graph: Rc::new(RefCell::new(Graph::new(name))),
        }
    }

    /// Declare a graph input with static dimensions.
    pub fn input(&self, name: &str, dims: &[usize], dtype: DType) -> TracedTensor {
        let id = self.graph.borrow_mut().input(name, Shape::new(dims, dtype));
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    /// Declare a graph input with mixed static/dynamic dimensions.
    pub fn input_dyn(&self, name: &str, dims: &[Dim], dtype: DType) -> TracedTensor {
        let id = self
            .graph
            .borrow_mut()
            .input(name, Shape::from_dims(dims, dtype));
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    /// Declare a parameter (weight) with static dimensions.
    pub fn param(&self, name: &str, dims: &[usize], dtype: DType) -> TracedTensor {
        let id = self.graph.borrow_mut().param(name, Shape::new(dims, dtype));
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    /// Matrix multiply.
    pub fn matmul(&self, lhs: TracedTensor, rhs: TracedTensor) -> TracedTensor {
        let id = self.graph.borrow_mut().mm(lhs.id, rhs.id);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    /// Layer normalization.
    pub fn layer_norm(
        &self,
        x: TracedTensor,
        gamma: TracedTensor,
        beta: TracedTensor,
        eps: f32,
    ) -> TracedTensor {
        let id = self.graph.borrow_mut().ln(x.id, gamma.id, beta.id, eps);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    /// Softmax.
    pub fn softmax(&self, x: TracedTensor, axis: i32) -> TracedTensor {
        let id = self.graph.borrow_mut().sm(x.id, axis);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    /// Gather (embedding lookup).
    pub fn gather(&self, table: TracedTensor, indices: TracedTensor, axis: usize) -> TracedTensor {
        let id = self.graph.borrow_mut().gather_(table.id, indices.id, axis);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }
}

// ── TracedTensor method chaining ────────────────────────────────────────

impl TracedTensor {
    pub fn matmul(self, rhs: TracedTensor) -> TracedTensor {
        let id = self.graph.borrow_mut().mm(self.id, rhs.id);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    pub fn gelu(self) -> TracedTensor {
        let id = self.graph.borrow_mut().gelu(self.id);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    pub fn silu(self) -> TracedTensor {
        let id = self.graph.borrow_mut().silu(self.id);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    pub fn relu(self) -> TracedTensor {
        let id = self.graph.borrow_mut().relu(self.id);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    pub fn layer_norm(self, gamma: TracedTensor, beta: TracedTensor, eps: f32) -> TracedTensor {
        let id = self.graph.borrow_mut().ln(self.id, gamma.id, beta.id, eps);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    pub fn softmax(self, axis: i32) -> TracedTensor {
        let id = self.graph.borrow_mut().sm(self.id, axis);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    pub fn reshape(self, new_shape: &[i64]) -> TracedTensor {
        let id = self
            .graph
            .borrow_mut()
            .reshape_(self.id, new_shape.to_vec());
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    pub fn transpose(self, perm: &[usize]) -> TracedTensor {
        let id = self.graph.borrow_mut().transpose_(self.id, perm.to_vec());
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    pub fn narrow(self, axis: usize, start: usize, len: usize) -> TracedTensor {
        let id = self.graph.borrow_mut().narrow_(self.id, axis, start, len);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    // ── PyTorch-shaped ergonomics (plan #60) ────────────────────────

    /// Number of dimensions. PyTorch's `.dim()`.
    pub fn rank(&self) -> usize {
        self.graph.borrow().shape(self.id).rank()
    }

    /// Output shape — useful for derived computations / asserts.
    pub fn shape(&self) -> rlx_ir::Shape {
        self.graph.borrow().shape(self.id).clone()
    }

    /// 2-D transpose shorthand. `t.t()` swaps the last two axes,
    /// matching PyTorch's `.t()` for matrices.
    pub fn t(&self) -> TracedTensor {
        let rank = self.rank();
        assert!(rank >= 2, ".t() requires rank >= 2");
        let mut perm: Vec<usize> = (0..rank).collect();
        perm.swap(rank - 2, rank - 1);
        let id = self.graph.borrow_mut().transpose_(self.id, perm);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    /// Permute dimensions — alias of [`Self::transpose`] under
    /// PyTorch's name.
    pub fn permute(&self, perm: &[usize]) -> TracedTensor {
        let id = self.graph.borrow_mut().transpose_(self.id, perm.to_vec());
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    /// Insert a length-1 dim at `axis`. Bumps every existing dim
    /// at `>= axis` by one position.
    pub fn unsqueeze(&self, axis: usize) -> TracedTensor {
        let s = self.shape();
        let rank = s.rank();
        assert!(
            axis <= rank,
            "unsqueeze axis {axis} out of range for rank {rank}"
        );
        let mut new_shape: Vec<i64> = (0..rank).map(|i| s.dim(i).unwrap_static() as i64).collect();
        new_shape.insert(axis, 1);
        let id = self.graph.borrow_mut().reshape_(self.id, new_shape);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    /// Drop a length-1 dim at `axis`. Errors at compile-time-of-
    /// graph if the dim isn't 1.
    pub fn squeeze(&self, axis: usize) -> TracedTensor {
        let s = self.shape();
        let rank = s.rank();
        assert!(
            axis < rank,
            "squeeze axis {axis} out of range for rank {rank}"
        );
        assert_eq!(
            s.dim(axis).unwrap_static(),
            1,
            "squeeze axis {axis} has dim {} (must be 1)",
            s.dim(axis).unwrap_static()
        );
        let new_shape: Vec<i64> = (0..rank)
            .filter(|&i| i != axis)
            .map(|i| s.dim(i).unwrap_static() as i64)
            .collect();
        let id = self.graph.borrow_mut().reshape_(self.id, new_shape);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }

    /// Reference-friendly matmul. `a.mm(&b)` doesn't move either.
    pub fn mm(&self, rhs: &TracedTensor) -> TracedTensor {
        let id = self.graph.borrow_mut().mm(self.id, rhs.id);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }
}

// ── Operator overloads ──────────────────────────────────────────────────

impl std::ops::Add for TracedTensor {
    type Output = TracedTensor;
    fn add(self, rhs: TracedTensor) -> TracedTensor {
        let id = self.graph.borrow_mut().add(self.id, rhs.id);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }
}

impl std::ops::Sub for TracedTensor {
    type Output = TracedTensor;
    fn sub(self, rhs: TracedTensor) -> TracedTensor {
        let id = self.graph.borrow_mut().sub(self.id, rhs.id);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }
}

impl std::ops::Mul for TracedTensor {
    type Output = TracedTensor;
    fn mul(self, rhs: TracedTensor) -> TracedTensor {
        let id = self.graph.borrow_mut().mul(self.id, rhs.id);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }
}

impl std::ops::Div for TracedTensor {
    type Output = TracedTensor;
    fn div(self, rhs: TracedTensor) -> TracedTensor {
        let id = self.graph.borrow_mut().div(self.id, rhs.id);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }
}

impl std::ops::Neg for TracedTensor {
    type Output = TracedTensor;
    fn neg(self) -> TracedTensor {
        let id = self.graph.borrow_mut().neg(self.id);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }
}

// ── Reference-based operator overloads (plan #60) ───────────────
//
// Mirror PyTorch's `a + b` behaviour where neither operand is
// consumed. The `&a + &b` form is the cheapest (one
// graph.borrow_mut + one Rc::clone); `a + &b` and `&a + b` cover
// the mixed cases without forcing the caller to add a `.clone()`.

macro_rules! impl_ref_binop {
    ($trait:ident, $method:ident, $graph_method:ident) => {
        // &T op &T
        impl std::ops::$trait<&TracedTensor> for &TracedTensor {
            type Output = TracedTensor;
            fn $method(self, rhs: &TracedTensor) -> TracedTensor {
                let id = self.graph.borrow_mut().$graph_method(self.id, rhs.id);
                TracedTensor {
                    id,
                    graph: self.graph.clone(),
                }
            }
        }
        // T op &T
        impl std::ops::$trait<&TracedTensor> for TracedTensor {
            type Output = TracedTensor;
            fn $method(self, rhs: &TracedTensor) -> TracedTensor {
                (&self).$method(rhs)
            }
        }
        // &T op T
        impl std::ops::$trait<TracedTensor> for &TracedTensor {
            type Output = TracedTensor;
            fn $method(self, rhs: TracedTensor) -> TracedTensor {
                self.$method(&rhs)
            }
        }
    };
}

impl_ref_binop!(Add, add, add);
impl_ref_binop!(Sub, sub, sub);
impl_ref_binop!(Mul, mul, mul);
impl_ref_binop!(Div, div, div);

impl std::ops::Neg for &TracedTensor {
    type Output = TracedTensor;
    fn neg(self) -> TracedTensor {
        let id = self.graph.borrow_mut().neg(self.id);
        TracedTensor {
            id,
            graph: self.graph.clone(),
        }
    }
}

// ── trace() entry point ─────────────────────────────────────────────────

/// Trace a function into an IR graph.
///
/// The closure receives a [`Tracer`] and returns output tensors.
/// All operations are recorded (not executed) into the graph.
pub fn trace<F>(name: &str, f: F) -> Graph
where
    F: FnOnce(&Tracer) -> Vec<TracedTensor>,
{
    let tracer = Tracer::new(name);
    let outputs = f(&tracer);
    let output_ids: Vec<NodeId> = outputs.iter().map(|t| t.id).collect();
    // Drop all TracedTensors (they hold Rc refs to the graph)
    drop(outputs);
    let mut graph = Rc::try_unwrap(tracer.graph)
        .expect("tracer graph still borrowed")
        .into_inner();
    graph.set_outputs(output_ids);
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::Activation;

    #[test]
    fn trace_matmul_bias_gelu() {
        let graph = trace("test", |t| {
            let x = t.input("x", &[4, 15, 384], DType::F32);
            let w = t.param("w", &[384, 1536], DType::F32);
            let b = t.param("b", &[1536], DType::F32);
            let mm = t.matmul(x, w);
            let out = (mm + b).gelu();
            vec![out]
        });

        assert_eq!(graph.len(), 6); // x, w, b, mm, add, gelu
        assert_eq!(
            graph.shape(graph.outputs[0]),
            &Shape::new(&[4, 15, 1536], DType::F32)
        );
        println!("{graph}");
    }

    #[test]
    fn trace_operator_overloads() {
        let graph = trace("ops", |t| {
            let a = t.input("a", &[4, 384], DType::F32);
            let b = t.input("b", &[4, 384], DType::F32);
            let c = a.clone() + b.clone();
            let d = a.clone() * b.clone();
            let e = c - d;
            vec![e]
        });

        assert_eq!(graph.len(), 5); // a, b, add, mul, sub
        assert_eq!(
            graph.shape(graph.outputs[0]),
            &Shape::new(&[4, 384], DType::F32)
        );
    }

    #[test]
    fn trace_method_chaining() {
        let graph = trace("chain", |t| {
            let x = t.input("x", &[4, 15, 384], DType::F32);
            let w = t.param("w", &[384, 1536], DType::F32);
            let out = x.matmul(w).gelu();
            vec![out]
        });

        assert_eq!(graph.len(), 4); // x, w, mm, gelu
        assert_eq!(
            graph.shape(graph.outputs[0]),
            &Shape::new(&[4, 15, 1536], DType::F32)
        );
    }

    #[test]
    fn pytorch_shaped_ergonomics() {
        // Reference-based ops + .t() + .permute + .unsqueeze /
        // .squeeze + .mm — full PyTorch ergonomic surface in one
        // expression.
        let graph = trace("ergonomics", |t| {
            let a = t.input("a", &[4, 8], DType::F32);
            let b = t.param("b", &[8, 4], DType::F32);
            // No clones — &+& and method-style chain.
            let c = a.mm(&b); // [4, 4]
            let d = &c + &c; // [4, 4]
            let e = d.t(); // [4, 4] transposed
            let f = e.unsqueeze(0); // [1, 4, 4]
            let g = f.squeeze(0); // [4, 4]
            let h = g.permute(&[1, 0]); // [4, 4]
            vec![h]
        });
        assert_eq!(
            graph.shape(graph.outputs[0]),
            &Shape::new(&[4, 4], DType::F32)
        );
    }

    #[test]
    fn trace_produces_fuseable_graph() {
        use rlx_opt::fusion::FuseMatMulBiasAct;
        use rlx_opt::pass::Pass;

        let graph = trace("fuseable", |t| {
            let x = t.input("x", &[4, 15, 384], DType::F32);
            let w = t.param("w", &[384, 1536], DType::F32);
            let b = t.param("b", &[1536], DType::F32);
            let mm = t.matmul(x, w);
            let out = (mm + b).gelu();
            vec![out]
        });

        // Before: 6 nodes
        assert_eq!(graph.len(), 6);

        // After fusion: 4 nodes (fused_mm_bias_gelu)
        let fused = FuseMatMulBiasAct.run(graph);
        assert_eq!(fused.len(), 4);

        let out_node = fused.node(fused.outputs[0]);
        assert!(matches!(
            out_node.op,
            Op::FusedMatMulBiasAct {
                activation: Some(Activation::Gelu)
            }
        ));
    }
}
