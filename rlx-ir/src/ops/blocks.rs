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

//! Fusion-friendly block builders — canonical subgraph shapes the
//! optimizer passes in `rlx-opt` already recognize.
//!
//! Model authors should prefer these over hand-wiring `MatMul → Add →
//! Activation` so fusion succeeds regardless of param declaration
//! order elsewhere in the graph.

use crate::infer::GraphExt;
use crate::op::Activation;
use crate::{Graph, NodeId, Op, Shape};

impl Graph {
    /// Dense linear layer: `matmul(input, weight)` with optional rank-1 bias.
    pub fn linear_bias(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
    ) -> NodeId {
        let mm = self.mm(input, weight);
        match bias {
            Some(b) => self.add(mm, b),
            None => mm,
        }
    }

    /// Dense linear with optional bias and epilogue activation.
    pub fn linear_bias_act(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        activation: Option<Activation>,
    ) -> NodeId {
        let x = self.linear_bias(input, weight, bias);
        activation.map_or(x, |act| self.activation_by_kind(x, act))
    }

    /// Emit `Op::FusedMatMulBiasAct` directly — deterministic fusion
    /// without relying on the `FuseMatMulBiasAct` pass.
    pub fn linear_fused(
        &mut self,
        input: NodeId,
        weight: NodeId,
        bias: NodeId,
        activation: Option<Activation>,
        out_shape: Shape,
    ) -> NodeId {
        self.fused_matmul_bias_act(input, weight, bias, activation, out_shape)
    }

    /// Two matmuls sharing the same input — canonical gate+up / QKV
    /// pattern for `FuseSharedInputMatMul`.
    ///
    /// Returns `(first, second)` in declaration order. For SwiGLU,
    /// pass **up** weight first and **gate** weight second so the
    /// post-concat narrow layout matches `FuseSwiGLU` (up @ 0, gate @ N).
    pub fn shared_matmul_pair(
        &mut self,
        input: NodeId,
        w_first: NodeId,
        w_second: NodeId,
    ) -> (NodeId, NodeId) {
        let first = self.mm(input, w_first);
        let second = self.mm(input, w_second);
        (first, second)
    }

    /// SwiGLU FFN block: shared-input gate+up → `silu(gate) * up` → down proj.
    ///
    /// Weight order matches `FuseSwiGLU`'s canonical narrow layout
    /// (up projection first, gate projection second).
    pub fn swiglu_ffn(
        &mut self,
        input: NodeId,
        up_w: NodeId,
        gate_w: NodeId,
        down_w: NodeId,
    ) -> NodeId {
        let (up, gate) = self.shared_matmul_pair(input, up_w, gate_w);
        let gate_silu = self.silu(gate);
        let hidden = self.mul(up, gate_silu);
        self.mm(hidden, down_w)
    }

    /// Fully fused SwiGLU FFN: concat weights → single matmul →
    /// [`Op::FusedSwiGLU`] → down projection. Matches the rewrite
    /// performed by [`FuseSwiGLUDualMatmul`](../../rlx-opt/src/fusion.rs)
    /// without relying on the pass.
    pub fn fused_swiglu_ffn(
        &mut self,
        input: NodeId,
        up_w: NodeId,
        gate_w: NodeId,
        down_w: NodeId,
        out_shape: Shape,
    ) -> NodeId {
        let wu_shape = self.shape(up_w);
        let wg_shape = self.shape(gate_w);
        let k = wu_shape.dim(0).unwrap_static();
        let n_up = wu_shape.dim(1).unwrap_static();
        let n_gate = wg_shape.dim(1).unwrap_static();
        debug_assert_eq!(wu_shape.dim(0), wg_shape.dim(0));

        let concat_shape = Shape::new(&[k, n_up + n_gate], wu_shape.dtype());
        let concat_w = self.concat(vec![up_w, gate_w], 1, concat_shape);

        let input_shape = self.shape(input);
        let out_rank = input_shape.rank();
        let dtype = input_shape.dtype();
        let mut cat_dims: Vec<usize> = (0..out_rank)
            .map(|i| input_shape.dim(i).unwrap_static())
            .collect();
        cat_dims[out_rank - 1] = n_up + n_gate;
        let cat_shape = Shape::new(&cat_dims, dtype);
        let cat_mm = self.matmul(input, concat_w, cat_shape);

        let mut hidden_dims = cat_dims;
        hidden_dims[out_rank - 1] = n_up;
        let hidden_shape = Shape::new(&hidden_dims, dtype);
        let hidden = self.add_node(
            Op::FusedSwiGLU {
                cast_to: None,
                gate_first: false,
            },
            vec![cat_mm],
            hidden_shape,
        );

        let _ = out_shape;
        self.mm(hidden, down_w)
    }

    fn activation_by_kind(&mut self, x: NodeId, act: Activation) -> NodeId {
        match act {
            Activation::Gelu => self.gelu(x),
            Activation::GeluApprox => self.gelu_approx(x),
            Activation::Silu => self.silu(x),
            Activation::Relu => self.relu(x),
            Activation::Exp => self.exp(x),
            Activation::Sqrt => self.sqrt(x),
            Activation::Neg => self.neg(x),
            Activation::Tanh => self.tanh(x),
            Activation::Sigmoid => {
                let s = self.shape(x).clone();
                self.activation(Activation::Sigmoid, x, s)
            }
            Activation::Log => {
                let s = self.shape(x).clone();
                self.activation(Activation::Log, x, s)
            }
            _ => {
                let s = self.shape(x).clone();
                self.activation(act, x, s)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::BinaryOp;
    use crate::{DType, Op};

    fn f32_shape(dims: &[usize]) -> Shape {
        Shape::new(dims, DType::F32)
    }

    #[test]
    fn linear_bias_act_emits_canonical_chain() {
        let mut g = Graph::new("linear");
        let x = g.input("x", f32_shape(&[4, 8]));
        let w = g.param("w", f32_shape(&[8, 16]));
        let b = g.param("b", f32_shape(&[16]));
        let out = g.linear_bias_act(x, w, Some(b), Some(Activation::Silu));
        g.set_outputs(vec![out]);

        let act = g.node(out);
        assert!(matches!(act.op, Op::Activation(Activation::Silu)));
        let add = g.node(act.inputs[0]);
        assert!(matches!(add.op, Op::Binary(BinaryOp::Add)));
        let mm = g.node(add.inputs[0]);
        assert!(matches!(mm.op, Op::MatMul));
    }
}
