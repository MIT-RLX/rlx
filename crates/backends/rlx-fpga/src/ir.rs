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

//! Translation between our `Model` and `rlx_ir::Graph`.
//!
//! The Graph is what the *pass infrastructure* operates on:
//! `rlx_opt::pass::run_passes` for traversal, `rlx_ir::verify::verify`
//! for invariants, fusion-pattern matching for op-sequence rewrites.
//! Per-channel quantization metadata stays out of the IR (the
//! `Op::QConv2d`/`Op::QMatMul` ops only carry a scalar `mult: f32`,
//! and extending them would touch every backend per the workspace
//! rules) — that lives in `Model` / `Hints` and is consumed at codegen
//! time.
//!
//! The `Graph` we produce is therefore a **structural** representation:
//! one node per layer (plus param nodes for weights / biases), with
//! shape inference good enough to satisfy the verifier (which checks
//! input counts and DAG-ness, not full type-checking).
//!
//! NodeId ↔ layer-index mapping is preserved in `IrModel::node_for_layer`.
//! Passes that produce per-layer hints look up by NodeId; codegen looks
//! up by layer index. Both round-trip cleanly.

use rlx_ir::op::{Activation, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::model::{Layer, Model};

/// A `Model` plus the IR `Graph` we built from it. Keeps the layer ↔
/// node mapping so passes can move between the two views.
pub struct IrModel {
    pub graph: Graph,
    /// `node_for_layer[i]` is the IR NodeId for `model.layers[i]`'s
    /// **output** node (the conv/relu/pool/etc. itself, not its
    /// param inputs).
    pub node_for_layer: Vec<NodeId>,
}

/// Build an `rlx_ir::Graph` that mirrors the model's layer sequence.
///
/// What we emit (one entry per layer):
///
/// * `Conv2d { ... }` → `Op::QConv2d` (3 inputs: prev activation, weight param, bias param)
/// * `Relu  { ... }`  → `Op::Activation(Relu)` (1 input)
/// * `MaxPool2d {..}` → `Op::Pool { Max, ... }`
/// * `Dense  { ... }` → `Op::QMatMul` (3 inputs)
/// * `Argmax { ... }` → `Op::TopK { k: 1 }`
///
/// Shapes are NCHW for conv consistency with `Op::QConv2d`'s contract
/// — even though the actual data is NHWC inside our backend. The
/// verifier doesn't care which (it only checks DAG-ness + input
/// counts); shape values are recorded for downstream passes that
/// might want them.
pub fn to_graph(model: &Model) -> IrModel {
    let mut g = Graph::new(model.name.clone());

    let f32_dt = DType::F32;
    let i8_dt = DType::I8;

    // Graph input node: the model input (NHWC i8, but encoded as a 1-D
    // Shape since we don't try to faithfully NCHW-reshape).
    let input_node = g.input("model_input", Shape::new(&[model.input_len], i8_dt));

    let mut prev: NodeId = input_node;
    let mut node_for_layer = Vec::with_capacity(model.layers.len());

    for layer in &model.layers {
        let node = match layer {
            Layer::Conv2d {
                name,
                h_in: _,
                w_in: _,
                c_in,
                c_out,
                kh,
                kw,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                x_zp,
                w_zp,
                out_zp,
                weight_bits: _,
                requant,
                weights,
                bias,
                ..
            } => {
                // Param nodes for weights and bias. Shapes here are
                // synthetic but valid: the verifier doesn't run shape
                // inference, so any plausible Shape works.
                let w_param = g.param(
                    format!("{name}_w"),
                    Shape::new(&[*c_out, *c_in, *kh, *kw], i8_dt),
                );
                let b_shape = match bias {
                    Some(b) => Shape::new(&[b.len()], DType::I32),
                    None => Shape::new(&[*c_out], DType::I32),
                };
                let b_param = g.param(format!("{name}_b"), b_shape);
                let _ = (weights, requant); // codegen consumes these from Model
                // Per-tensor-mult collapsed from the per-channel table:
                // we use the first entry, which is what `Op::QConv2d`'s
                // single `mult` field can carry.
                let scalar_mult = requant
                    .first()
                    .map(|&(m0, sh)| q31_to_f32_mult(m0, sh))
                    .unwrap_or(1.0);
                g.q_conv2d(
                    prev,
                    w_param,
                    b_param,
                    vec![*kh, *kw],
                    vec![*stride_h, *stride_w],
                    vec![*pad_h, *pad_w],
                    vec![1, 1],
                    1,
                    *x_zp,
                    *w_zp,
                    *out_zp,
                    scalar_mult,
                    Shape::new(&[layer.out_len()], i8_dt),
                )
            }
            Layer::Relu { len, .. } => {
                g.activation(Activation::Relu, prev, Shape::new(&[*len], i8_dt))
            }
            Layer::MaxPool2d {
                kh,
                kw,
                stride_h,
                stride_w,
                ..
            } => g.add_node(
                Op::Pool {
                    kind: ReduceOp::Max,
                    kernel_size: vec![*kh, *kw],
                    stride: vec![*stride_h, *stride_w],
                    padding: vec![0, 0],
                },
                vec![prev],
                Shape::new(&[layer.out_len()], i8_dt),
            ),
            Layer::Dense {
                name,
                in_features,
                out_features,
                x_zp,
                w_zp,
                out_zp,
                weight_bits: _,
                requant,
                weights: _,
                bias,
                ..
            } => {
                let w_param = g.param(
                    format!("{name}_w"),
                    Shape::new(&[*in_features, *out_features], i8_dt),
                );
                let b_shape = match bias {
                    Some(b) => Shape::new(&[b.len()], DType::I32),
                    None => Shape::new(&[*out_features], DType::I32),
                };
                let b_param = g.param(format!("{name}_b"), b_shape);
                let scalar_mult = requant
                    .first()
                    .map(|&(m0, sh)| q31_to_f32_mult(m0, sh))
                    .unwrap_or(1.0);
                g.q_matmul(
                    prev,
                    w_param,
                    b_param,
                    *x_zp,
                    *w_zp,
                    *out_zp,
                    scalar_mult,
                    Shape::new(&[*out_features], i8_dt),
                )
            }
            Layer::Argmax { len: _, .. } => {
                g.add_node(
                    Op::TopK { k: 1 },
                    vec![prev],
                    Shape::new(&[1], f32_dt), // TopK returns f32-encoded indices
                )
            }
        };
        node_for_layer.push(node);
        prev = node;
    }

    g.set_outputs(vec![prev]);
    IrModel {
        graph: g,
        node_for_layer,
    }
}

/// Convert a Q0.31 `(M0, shift)` pair back to an approximate f32
/// multiplier. The IR ops only carry per-tensor scalar `mult`; we lose
/// per-channel detail here, but the IR is for analysis (fusion / DCE /
/// memory plan) — the actual requant table the FPGA emits comes from
/// `Layer.requant`, not from this scalar.
fn q31_to_f32_mult(m0: i32, shift: i32) -> f32 {
    let s = m0 as f64 / (1u64 << 31) as f64;
    let scale = 2f64.powi(-shift);
    (s * scale) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tinyconv_mnist_from_cortexm;

    #[test]
    fn graph_has_one_node_per_layer_plus_params_plus_input() {
        let m = tinyconv_mnist_from_cortexm();
        let ir = to_graph(&m);
        // Each layer occupies 1 op node. Conv2d/Dense add 2 param
        // nodes (weight + bias) per layer; Relu/MaxPool/Argmax add 0.
        // Plus 1 input node.
        let mut expected = 1; // input
        for l in &m.layers {
            expected += 1; // op
            if matches!(l, Layer::Conv2d { .. } | Layer::Dense { .. }) {
                expected += 2; // weight + bias params
            }
        }
        assert_eq!(ir.graph.len(), expected);
        assert_eq!(ir.node_for_layer.len(), m.layers.len());
    }

    #[test]
    fn ir_graph_passes_verifier() {
        let m = tinyconv_mnist_from_cortexm();
        let ir = to_graph(&m);
        let errors = rlx_ir::verify::verify(&ir.graph);
        assert!(
            errors.is_empty(),
            "verifier reported errors: {:?}",
            errors.iter().map(|e| e.to_string()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn graph_output_is_argmax() {
        let m = tinyconv_mnist_from_cortexm();
        let ir = to_graph(&m);
        assert_eq!(ir.graph.outputs.len(), 1);
        let out_node = ir.graph.node(ir.graph.outputs[0]);
        assert!(
            matches!(out_node.op, Op::TopK { .. }),
            "expected TopK output, got {:?}",
            out_node.op
        );
    }
}
