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
//! * [`to_graph`] — Model → IR (for pass infrastructure + round-trip).
//! * [`crate::model::Model::from_graph`] — IR → Model (export entry).
//!
//! Weight / bias / per-channel requant tables are baked as named
//! [`Op::Constant`] nodes so [`from_graph`](crate::model::Model::from_graph)
//! can recover a complete FPGA model without side tables. The IR ops
//! themselves still only carry a scalar `mult` (workspace rule: don't
//! extend `QConv2d`/`QMatMul` fields without touching every backend).

use rlx_ir::op::{Activation, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::model::{Layer, Model};

/// A `Model` plus the IR `Graph` we built from it. Keeps the layer ↔
/// node mapping so passes can move between the two views.
pub struct IrModel {
    pub graph: Graph,
    /// `node_for_layer[i]` is the IR NodeId for `model.layers[i]`'s
    /// **output** node (the conv/relu/pool/etc. itself, not its
    /// weight inputs).
    pub node_for_layer: Vec<NodeId>,
}

/// Build an `rlx_ir::Graph` that mirrors the model's layer sequence.
///
/// What we emit (one entry per layer):
///
/// * `Conv2d` → `Op::QConv2d` (activation, weight Constant, bias Constant)
/// * `Relu` → `Op::Activation(Relu)`
/// * `MaxPool2d` → `Op::Pool { Max, ... }`
/// * `Dense` → `Op::QMatMul`
/// * `Argmax` → `Op::TopK { k: 1 }`
///
/// Activation shapes are NCHW for conv consistency with `Op::QConv2d`.
/// A dangling `{name}_requant` Constant stores per-channel `(M0, shift)`
/// pairs for round-trip fidelity.
pub fn to_graph(model: &Model) -> IrModel {
    let mut g = Graph::new(model.name.clone());
    let i8_dt = DType::I8;
    let i32_dt = DType::I32;
    let f32_dt = DType::F32;

    // Infer input NCHW from the first Conv2d when present.
    let (in_c, in_h, in_w) = first_conv_in_shape(model).unwrap_or((1, 1, model.input_len));
    let input_node = g.input("model_input", Shape::new(&[1, in_c, in_h, in_w], i8_dt));

    let mut prev: NodeId = input_node;
    let mut node_for_layer = Vec::with_capacity(model.layers.len());

    for layer in &model.layers {
        let node = match layer {
            Layer::Conv2d {
                name,
                h_in,
                w_in,
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
                weight_encoding,
                requant,
                weights,
                bias,
                ..
            } => {
                let w_id = named_i8_const(
                    &mut g,
                    format!("{name}_w"),
                    weights,
                    Shape::new(&[*c_out, *c_in, *kh, *kw], i8_dt),
                );
                let b_vec = bias.clone().unwrap_or_else(|| vec![0i32; *c_out]);
                let b_id = named_i32_const(
                    &mut g,
                    format!("{name}_b"),
                    &b_vec,
                    Shape::new(&[*c_out], i32_dt),
                );
                bake_requant(&mut g, name, requant);
                bake_weight_encoding(&mut g, name, *weight_encoding);
                let scalar_mult = requant
                    .first()
                    .map(|&(m0, sh)| q31_to_f32_mult(m0, sh))
                    .unwrap_or(1.0);
                let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
                let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
                // Ensure activation rank matches (reshape if prev was flat).
                let act = ensure_nchw(&mut g, prev, 1, *c_in, *h_in, *w_in);
                g.q_conv2d(
                    act,
                    w_id,
                    b_id,
                    vec![*kh, *kw],
                    vec![*stride_h, *stride_w],
                    vec![*pad_h, *pad_w],
                    vec![1, 1],
                    1,
                    *x_zp,
                    *w_zp,
                    *out_zp,
                    scalar_mult,
                    Shape::new(&[1, *c_out, h_out, w_out], i8_dt),
                )
            }
            Layer::Relu { name, len, .. } => {
                let out_shape = match g.shape(prev).rank() {
                    4 => g.shape(prev).clone(),
                    _ => Shape::new(&[*len], i8_dt),
                };
                let id = g.activation(Activation::Relu, prev, out_shape);
                g.node_mut(id).name = Some((*name).to_string());
                id
            }
            Layer::MaxPool2d {
                name,
                h_in,
                w_in,
                c,
                kh,
                kw,
                stride_h,
                stride_w,
            } => {
                let act = ensure_nchw(&mut g, prev, 1, *c, *h_in, *w_in);
                let h_out = (h_in - kh) / stride_h + 1;
                let w_out = (w_in - kw) / stride_w + 1;
                let id = g.add_node(
                    Op::Pool {
                        kind: ReduceOp::Max,
                        kernel_size: vec![*kh, *kw],
                        stride: vec![*stride_h, *stride_w],
                        padding: vec![0, 0],
                    },
                    vec![act],
                    Shape::new(&[1, *c, h_out, w_out], i8_dt),
                );
                g.node_mut(id).name = Some((*name).to_string());
                id
            }
            Layer::Dense {
                name,
                in_features,
                out_features,
                x_zp,
                w_zp,
                out_zp,
                weight_encoding,
                requant,
                weights,
                bias,
                ..
            } => {
                let flat = flatten_to_1d(&mut g, prev, *in_features);
                let w_id = named_i8_const(
                    &mut g,
                    format!("{name}_w"),
                    weights,
                    Shape::new(&[*in_features, *out_features], i8_dt),
                );
                let b_vec = bias.clone().unwrap_or_else(|| vec![0i32; *out_features]);
                let b_id = named_i32_const(
                    &mut g,
                    format!("{name}_b"),
                    &b_vec,
                    Shape::new(&[*out_features], i32_dt),
                );
                bake_requant(&mut g, name, requant);
                bake_weight_encoding(&mut g, name, *weight_encoding);
                let scalar_mult = requant
                    .first()
                    .map(|&(m0, sh)| q31_to_f32_mult(m0, sh))
                    .unwrap_or(1.0);
                g.q_matmul(
                    flat,
                    w_id,
                    b_id,
                    *x_zp,
                    *w_zp,
                    *out_zp,
                    scalar_mult,
                    Shape::new(&[*out_features], i8_dt),
                )
            }
            Layer::Argmax { name, len: _ } => {
                let id = g.add_node(Op::TopK { k: 1 }, vec![prev], Shape::new(&[1], f32_dt));
                g.node_mut(id).name = Some((*name).to_string());
                id
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

fn first_conv_in_shape(model: &Model) -> Option<(usize, usize, usize)> {
    model.layers.iter().find_map(|l| match l {
        Layer::Conv2d {
            c_in, h_in, w_in, ..
        } => Some((*c_in, *h_in, *w_in)),
        _ => None,
    })
}

fn named_i8_const(g: &mut Graph, name: String, data: &[i8], shape: Shape) -> NodeId {
    let bytes: Vec<u8> = data.iter().map(|&x| x as u8).collect();
    let id = g.add_node(Op::Constant { data: bytes }, vec![], shape);
    g.node_mut(id).name = Some(name);
    id
}

fn named_i32_const(g: &mut Graph, name: String, data: &[i32], shape: Shape) -> NodeId {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    let id = g.add_node(Op::Constant { data: bytes }, vec![], shape);
    g.node_mut(id).name = Some(name);
    id
}

fn bake_requant(g: &mut Graph, name: &str, requant: &[(i32, i32)]) {
    // Official side-table layout (prefer over scalar Op::Q* mult):
    //   {name}_requant_m0    — i32[C]
    //   {name}_requant_shift — i32[C]
    // Plus interleaved {name}_requant for older loaders.
    let m0s: Vec<i32> = requant.iter().map(|&(m0, _)| m0).collect();
    let shs: Vec<i32> = requant.iter().map(|&(_, sh)| sh).collect();
    named_i32_const(
        g,
        format!("{name}_requant_m0"),
        &m0s,
        Shape::new(&[m0s.len()], DType::I32),
    );
    named_i32_const(
        g,
        format!("{name}_requant_shift"),
        &shs,
        Shape::new(&[shs.len()], DType::I32),
    );
    let mut bytes = Vec::with_capacity(requant.len() * 8);
    for &(m0, sh) in requant {
        bytes.extend_from_slice(&m0.to_le_bytes());
        bytes.extend_from_slice(&sh.to_le_bytes());
    }
    let id = g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[requant.len() * 2], DType::I32),
    );
    g.node_mut(id).name = Some(format!("{name}_requant"));
}

fn bake_weight_encoding(g: &mut Graph, name: &str, enc: crate::model::WeightEncoding) {
    let code = match enc {
        crate::model::WeightEncoding::SignedInt => 0i32,
        crate::model::WeightEncoding::Fp4E2M1 => 1i32,
    };
    named_i32_const(
        g,
        format!("{name}_wenc"),
        &[code],
        Shape::new(&[1], DType::I32),
    );
}

fn ensure_nchw(g: &mut Graph, prev: NodeId, n: usize, c: usize, h: usize, w: usize) -> NodeId {
    let want = Shape::new(&[n, c, h, w], DType::I8);
    if g.shape(prev).rank() == 4 {
        return prev;
    }
    g.add_node(
        Op::Reshape {
            new_shape: vec![n as i64, c as i64, h as i64, w as i64],
        },
        vec![prev],
        want,
    )
}

fn flatten_to_1d(g: &mut Graph, prev: NodeId, len: usize) -> NodeId {
    if g.shape(prev).rank() == 1 {
        return prev;
    }
    g.add_node(
        Op::Reshape {
            new_shape: vec![len as i64],
        },
        vec![prev],
        Shape::new(&[len], DType::I8),
    )
}

/// Convert a Q0.31 `(M0, shift)` pair back to an approximate f32 multiplier.
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
    fn graph_has_compute_nodes_plus_constants() {
        let m = tinyconv_mnist_from_cortexm();
        let ir = to_graph(&m);
        assert_eq!(ir.node_for_layer.len(), m.layers.len());
        assert!(!ir.graph.nodes().is_empty());
        let errors = rlx_ir::verify::verify(&ir.graph);
        assert!(
            errors.is_empty(),
            "verifier reported errors: {:?}",
            errors.iter().map(|e| e.to_string()).collect::<Vec<_>>()
        );
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

    #[test]
    fn input_is_nchw() {
        let m = tinyconv_mnist_from_cortexm();
        let ir = to_graph(&m);
        let input = ir
            .graph
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Input { .. }))
            .unwrap();
        assert_eq!(input.shape.rank(), 4);
        assert_eq!(input.shape.dim(1).unwrap_static(), 1);
        assert_eq!(input.shape.dim(2).unwrap_static(), 28);
        assert_eq!(input.shape.dim(3).unwrap_static(), 28);
    }
}
