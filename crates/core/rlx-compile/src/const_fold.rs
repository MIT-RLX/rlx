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

//! Constant Folding — evaluate pure-input subgraphs at compile time.
//!
//! A node is foldable when all its inputs are foldable AND the op has
//! a deterministic, pure evaluation (no I/O, no random). We evaluate
//! such subgraphs once at compile time and replace them with `Op::Constant`.
//!
//! Examples that get folded:
//! - `1.0 / sqrt(head_dim)` (attention scale factor)
//! - small reshapes/expands of **F32** constants
//!
//! Non-F32 constants (F16/BF16/I32/…) are left untouched: the fold
//! evaluator works in f32 host buffers and used to mis-decode foreign
//! dtypes as f32 bit patterns, then re-emit `Op::Constant` with the
//! wrong byte width for the declared shape dtype (CUDA AOT runs this
//! pass via `post_fusion_cleanup`; CPU AOT historically skipped it —
//! that asymmetry made F5-TTS text-embed CPU↔CUDA cos collapse to ~0.17).

use rlx_fusion::pass::Pass;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Dim, Graph, NodeId, Op, Shape};
use std::collections::{HashMap, HashSet};

pub struct ConstantFolding;

/// True if this op can be evaluated symbolically with no runtime state.
pub(crate) fn is_pure(op: &Op) -> bool {
    matches!(
        op,
        Op::Activation(_)
            | Op::Binary(_)
            | Op::Compare(_)
            | Op::Reshape { .. }
            | Op::Expand { .. }
            | Op::Cast { .. }
    )
}

/// True if the node's inputs are all known constants (Param, Constant, or
/// previously-folded result).
fn is_foldable(node_id: NodeId, graph: &Graph, folded: &HashSet<NodeId>) -> bool {
    let node = graph.node(node_id);
    if !is_pure(&node.op) {
        return false;
    }
    // Host evaluator emits F32 constant bytes only — refuse to replace a
    // non-F32-typed node (would write 4×N bytes into an F16/I32 shape).
    if node.shape.dtype() != DType::F32 {
        return false;
    }
    node.inputs.iter().all(|i| folded.contains(i))
}

/// Static dims of a shape, or `None` if any dim is dynamic.
pub(crate) fn static_dims(shape: &Shape) -> Option<Vec<usize>> {
    shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Some(*n),
            _ => None,
        })
        .collect()
}

/// NumPy-broadcast a row-major f32 buffer of shape `src_dims` up to `out_dims`.
/// `None` if the shapes are not broadcast-compatible. Handles the same-shape
/// case as an identity copy, so callers can broadcast unconditionally.
fn broadcast_to(src: &[f32], src_dims: &[usize], out_dims: &[usize]) -> Option<Vec<f32>> {
    let rank = out_dims.len();
    if src_dims.len() > rank {
        return None;
    }
    // Right-align src dims into the output rank (prepend 1s).
    let mut sd = vec![1usize; rank];
    sd[rank - src_dims.len()..].copy_from_slice(src_dims);
    for ax in 0..rank {
        if sd[ax] != 1 && sd[ax] != out_dims[ax] {
            return None;
        }
    }
    if src.len() != sd.iter().product::<usize>() {
        return None;
    }
    // Row-major strides over `sd` (same layout as the src buffer).
    let mut strides = vec![1usize; rank];
    for ax in (0..rank.saturating_sub(1)).rev() {
        strides[ax] = strides[ax + 1] * sd[ax + 1];
    }
    let total: usize = out_dims.iter().product();
    let mut out = vec![0f32; total];
    let mut idx = vec![0usize; rank];
    for slot in out.iter_mut() {
        // `idx % sd[ax]` collapses to 0 on broadcast (sd==1) axes.
        let s: usize = (0..rank).map(|ax| (idx[ax] % sd[ax]) * strides[ax]).sum();
        *slot = src[s];
        for ax in (0..rank).rev() {
            idx[ax] += 1;
            if idx[ax] < out_dims[ax] {
                break;
            }
            idx[ax] = 0;
        }
    }
    Some(out)
}

/// Evaluate a foldable node given precomputed input values + their static dims.
/// Returns a flat f32 buffer of the result, or None if not supported.
pub(crate) fn evaluate(
    node: &rlx_ir::Node,
    inputs: &[&Vec<f32>],
    in_dims: &[Vec<usize>],
) -> Option<Vec<f32>> {
    let total = node.shape.num_elements()?;
    let mut out = vec![0f32; total];

    match &node.op {
        Op::Activation(act) => {
            let x = inputs[0];
            for (i, &v) in x.iter().enumerate() {
                out[i] = match act {
                    Activation::Gelu | Activation::GeluApprox => {
                        v * 0.5 * (1.0 + (v * std::f32::consts::FRAC_1_SQRT_2).tanh())
                    }
                    Activation::Silu => v / (1.0 + (-v).exp()),
                    Activation::Relu => v.max(0.0),
                    Activation::Sigmoid => 1.0 / (1.0 + (-v).exp()),
                    Activation::Tanh => v.tanh(),
                    Activation::Exp => v.exp(),
                    Activation::Log => v.ln(),
                    Activation::Sqrt => v.sqrt(),
                    Activation::Rsqrt => 1.0 / v.sqrt(),
                    Activation::Neg => -v,
                    Activation::Abs => v.abs(),
                    Activation::Round => v.round(),
                    Activation::Sin => v.sin(),
                    Activation::Cos => v.cos(),
                    Activation::Tan => v.tan(),
                    Activation::Atan => v.atan(),
                    Activation::Recip => 1.0 / v,
                    Activation::Floor => v.floor(),
                    Activation::Ceil => v.ceil(),
                    Activation::Sign => {
                        if v > 0.0 {
                            1.0
                        } else if v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    }
                    Activation::Softplus => v.max(0.0) + (-(v.abs())).exp().ln_1p(),
                    Activation::Elu => {
                        if v > 0.0 {
                            v
                        } else {
                            v.exp() - 1.0
                        }
                    }
                    Activation::Erf => const_erf(v),
                    Activation::HardSwish => v * (v + 3.0).clamp(0.0, 6.0) / 6.0,
                    Activation::HardSigmoid => (v / 6.0 + 0.5).clamp(0.0, 1.0),
                    Activation::Mish => v * (v.max(0.0) + (-(v.abs())).exp().ln_1p()).tanh(),
                    Activation::Softsign => v / (1.0 + v.abs()),
                    Activation::LogSigmoid => v.min(0.0) - (-(v.abs())).exp().ln_1p(),
                };
            }
            Some(out)
        }
        Op::Binary(op) => {
            // NumPy broadcast both operands to the output shape (folds
            // per-channel scale/bias etc.; a no-op copy when already matching).
            let out_dims = static_dims(&node.shape)?;
            let lhs = broadcast_to(inputs[0], &in_dims[0], &out_dims)?;
            let rhs = broadcast_to(inputs[1], &in_dims[1], &out_dims)?;
            for i in 0..total {
                out[i] = match op {
                    BinaryOp::Add => lhs[i] + rhs[i],
                    BinaryOp::Sub => lhs[i] - rhs[i],
                    BinaryOp::Mul => lhs[i] * rhs[i],
                    BinaryOp::Div => lhs[i] / rhs[i],
                    BinaryOp::Max => lhs[i].max(rhs[i]),
                    BinaryOp::Min => lhs[i].min(rhs[i]),
                    BinaryOp::Pow => lhs[i].powf(rhs[i]),
                    BinaryOp::Mod => lhs[i] % rhs[i],
                    BinaryOp::Atan2 => lhs[i].atan2(rhs[i]),
                    BinaryOp::BitAnd => ((lhs[i] as i64) & (rhs[i] as i64)) as f32,
                    BinaryOp::BitOr => ((lhs[i] as i64) | (rhs[i] as i64)) as f32,
                    BinaryOp::BitXor => ((lhs[i] as i64) ^ (rhs[i] as i64)) as f32,
                    BinaryOp::Shl => ((lhs[i] as i64) << (rhs[i] as i64)) as f32,
                    BinaryOp::Shr => ((lhs[i] as i64) >> (rhs[i] as i64)) as f32,
                };
            }
            Some(out)
        }
        Op::Reshape { .. } => {
            // Reshape preserves element count + row-major order.
            let src = inputs[0];
            if src.len() == total {
                Some(src.clone())
            } else if src.len() == 1 {
                Some(vec![src[0]; total])
            } else {
                None
            }
        }
        Op::Expand { .. } => broadcast_to(inputs[0], &in_dims[0], &static_dims(&node.shape)?),
        // Only identity F32→F32 casts — other casts need real dtype conversion.
        Op::Cast { to } if *to == DType::F32 => {
            let src = inputs[0];
            if src.len() == total {
                Some(src.clone())
            } else if src.len() == 1 {
                Some(vec![src[0]; total])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Encode an f32 buffer as raw bytes for `Op::Constant`.
/// erf via A&S 7.1.26 (matches `rlx_cpu`'s `erf_f32` — same coefficients).
fn const_erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    s * y
}

fn encode_constant(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

impl Pass for ConstantFolding {
    fn name(&self) -> &str {
        "constant_folding"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Walk in topological order, tracking which nodes are foldable
        // and accumulating their evaluated values.
        let mut folded: HashSet<NodeId> = HashSet::new();
        let mut values: HashMap<NodeId, Vec<f32>> = HashMap::new();

        for node in graph.nodes() {
            // Only F32 Constants are foldable seeds. Interpreting F16/I32/…
            // bytes as f32 bit patterns produced wrong values and, when a
            // downstream node was replaced, wrong-sized Constant payloads.
            if let Op::Constant { data } = &node.op {
                if node.shape.dtype() != DType::F32 {
                    continue;
                }
                folded.insert(node.id);
                let f32s: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                values.insert(node.id, f32s);
                continue;
            }
            // Inputs/Params are NOT foldable (their values are runtime).
            if matches!(node.op, Op::Input { .. } | Op::Param { .. }) {
                continue;
            }
            // Try to fold pure ops with all-constant inputs.
            if is_foldable(node.id, &graph, &folded) {
                let inputs: Vec<&Vec<f32>> = node.inputs.iter().map(|i| &values[i]).collect();
                // Static dims of every input; skip fold if any is dynamic.
                let in_dims: Option<Vec<Vec<usize>>> = node
                    .inputs
                    .iter()
                    .map(|i| static_dims(&graph.node(*i).shape))
                    .collect();
                if let Some(in_dims) = in_dims
                    && let Some(result) = evaluate(node, &inputs, &in_dims)
                {
                    folded.insert(node.id);
                    values.insert(node.id, result);
                }
            }
        }

        // Rebuild: replace folded nodes with Op::Constant, rewire others.
        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        for node in graph.nodes() {
            // Foldable downstream nodes get replaced with Constant unless
            // they're terminal Constants/Params themselves.
            if folded.contains(&node.id)
                && !matches!(
                    node.op,
                    Op::Constant { .. } | Op::Param { .. } | Op::Input { .. }
                )
            {
                debug_assert_eq!(
                    node.shape.dtype(),
                    DType::F32,
                    "constant folding must only replace F32 nodes"
                );
                let bytes = encode_constant(&values[&node.id]);
                let new_id =
                    new_graph.add_node(Op::Constant { data: bytes }, vec![], node.shape.clone());
                id_map.insert(node.id, new_id);
                continue;
            }
            // Otherwise copy the node, remapping inputs.
            let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
            let new_id = new_graph.add_node(node.op.clone(), new_inputs, node.shape.clone());
            id_map.insert(node.id, new_id);
        }
        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::*;

    #[test]
    fn folds_constant_arithmetic() {
        // const(2.0) + const(3.0) → const(5.0)
        let mut g = Graph::new("test");
        let a = g.add_node(
            Op::Constant {
                data: 2.0f32.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        let b = g.add_node(
            Op::Constant {
                data: 3.0f32.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        let sum = g.binary(op::BinaryOp::Add, a, b, Shape::new(&[1], DType::F32));
        g.set_outputs(vec![sum]);

        let folded = ConstantFolding.run(g);
        // After folding, the Add node should be a Constant with value 5.0
        let out_node = folded.node(folded.outputs[0]);
        if let Op::Constant { data } = &out_node.op {
            let v = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            assert!((v - 5.0).abs() < 1e-6);
        } else {
            panic!("expected folded Constant, got {:?}", out_node.op);
        }
    }

    #[test]
    fn does_not_fold_input_dependent() {
        let mut g = Graph::new("test");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let c = g.add_node(
            Op::Constant {
                data: vec![0u8; 16],
            },
            vec![],
            Shape::new(&[4], DType::F32),
        );
        let sum = g.binary(op::BinaryOp::Add, x, c, Shape::new(&[4], DType::F32));
        g.set_outputs(vec![sum]);

        let folded = ConstantFolding.run(g);
        // x + c is input-dependent; should NOT be folded.
        assert!(matches!(folded.node(folded.outputs[0]).op, Op::Binary(_)));
    }

    #[test]
    fn leaves_f16_constants_and_casts_alone() {
        // Regression: previously F16 constant bytes were decoded as f32, then
        // a Cast-to-F32 could be replaced with a wrong-sized F32 Constant.
        let mut g = Graph::new("f16_const");
        let c = g.add_node(
            Op::Constant {
                // two f16 values (4 bytes) — must NOT be read as one f32
                data: vec![0x00, 0x3c, 0x00, 0x40], // 1.0f16, 2.0f16
            },
            vec![],
            Shape::new(&[2], DType::F16),
        );
        let cast = g.add_node(
            Op::Cast { to: DType::F32 },
            vec![c],
            Shape::new(&[2], DType::F32),
        );
        g.set_outputs(vec![cast]);

        let folded = ConstantFolding.run(g);
        assert!(
            matches!(folded.node(folded.outputs[0]).op, Op::Cast { .. }),
            "F16→F32 cast must not be constant-folded without a real converter"
        );
        assert!(matches!(
            folded.node(folded.node(folded.outputs[0]).inputs[0]).op,
            Op::Constant { .. }
        ));
        let c_node = folded.node(folded.node(folded.outputs[0]).inputs[0]);
        if let Op::Constant { data } = &c_node.op {
            assert_eq!(data.len(), 4, "F16 constant payload must stay 2 bytes/elem");
            assert_eq!(c_node.shape.dtype(), DType::F16);
        }
    }
}
