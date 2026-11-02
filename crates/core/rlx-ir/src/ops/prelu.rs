// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **PReLU** — `x` where `x ≥ 0`, else `a·x` — as one canonical graph every
//! backend runs, and a matcher a backend uses to run it as one pass.
//!
//! There is no PReLU op: [`Graph::prelu`] builds
//! `max(x, 0) + a·min(x, 0)` from element-wise ops, right for any slope —
//! negative, past one, a learned per-channel vector — where LeakyReLU's
//! `max(x, a·x)` holds only for `0 ≤ a ≤ 1`. [`match_prelu`] recognises that
//! form and the ONNX importer's `relu(x) − a·relu(−x)` (the same values:
//! each side is `x` or `a·x` exactly), so a backend with a one-pass kernel
//! (rlx-wgpu's `rlx_wgpu.prelu`) folds either, and one without runs the
//! element-wise ops.

use crate::op::{Activation, BinaryOp};
use crate::prelude::*;
use crate::{DType, Graph, NodeId, Op, Shape};

impl Graph {
    /// PReLU of `x` with `slope`: a scalar, or one value per channel (dim 1),
    /// shaped to broadcast against `x` (`[1, C, 1, …]`). `max(x, 0) +
    /// slope·min(x, 0)`; see the module docs.
    pub fn prelu(&mut self, x: NodeId, slope: NodeId) -> NodeId {
        let s = self.shape(x).clone();
        let zero = self.add_node(
            Op::Constant {
                data: 0f32.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        let pos = self.binary(BinaryOp::Max, x, zero, s.clone());
        let neg = self.binary(BinaryOp::Min, x, zero, s.clone());
        let neg = self.binary(BinaryOp::Mul, neg, slope, s.clone());
        self.binary(BinaryOp::Add, pos, neg, s)
    }

    /// [`Graph::prelu`] with one slope, a constant.
    pub fn prelu_scalar(&mut self, x: NodeId, slope: f32) -> NodeId {
        let a = self.add_node(
            Op::Constant {
                data: slope.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        self.prelu(x, a)
    }
}

/// A PReLU found by [`match_prelu`].
#[derive(Clone, Debug, PartialEq)]
pub struct PreluMatch {
    /// the input
    pub x: NodeId,
    /// the slope operand: a scalar or per-channel tensor
    pub slope: NodeId,
    /// the nodes between `x` and the root, root excluded — what a fold
    /// removes, when nothing else reads them
    pub interior: Vec<NodeId>,
}

impl PreluMatch {
    /// The slope's value when it is one f32 constant.
    pub fn constant_slope(&self, g: &Graph) -> Option<f32> {
        scalar_f32(g, self.slope)
    }
}

/// A one-element f32 constant's value.
fn scalar_f32(g: &Graph, id: NodeId) -> Option<f32> {
    match &g.node(id).op {
        Op::Constant { data } if data.len() == 4 && g.node(id).shape.dtype() == DType::F32 => {
            Some(f32::from_le_bytes(data[..4].try_into().unwrap()))
        }
        _ => None,
    }
}

fn binary(g: &Graph, id: NodeId, op: BinaryOp) -> Option<[NodeId; 2]> {
    let n = g.node(id);
    match n.op {
        Op::Binary(o) if o == op && n.inputs.len() == 2 => Some([n.inputs[0], n.inputs[1]]),
        _ => None,
    }
}

fn activation(g: &Graph, id: NodeId, act: Activation) -> Option<NodeId> {
    let n = g.node(id);
    match n.op {
        Op::Activation(a) if a == act && n.inputs.len() == 1 => Some(n.inputs[0]),
        _ => None,
    }
}

/// `op(x, 0)` either way round: `x`
fn with_zero(g: &Graph, id: NodeId, op: BinaryOp) -> Option<NodeId> {
    let [a, b] = binary(g, id, op)?;
    match (scalar_f32(g, a), scalar_f32(g, b)) {
        (_, Some(0.0)) => Some(a),
        (Some(0.0), _) => Some(b),
        _ => None,
    }
}

/// `root` as a PReLU, when it is one of
///
/// * `max(x, 0) + a·min(x, 0)` ([`Graph::prelu`]), and
/// * `relu(x) − a·relu(−x)` (the ONNX importer's `PRelu`),
///
/// the commutative operands either way round, every value the root's shape
/// (the slope aside, which broadcasts). Readers of the interior nodes are
/// not checked: a fold removing them must.
pub fn match_prelu(g: &Graph, root: NodeId) -> Option<PreluMatch> {
    let shape = &g.node(root).shape;
    let same = |id: NodeId| &g.node(id).shape == shape;
    // the slope's product with the negative part, `p`: (slope, other factor)
    let product = |p: NodeId| -> Option<[(NodeId, NodeId); 2]> {
        let [a, b] = binary(g, p, BinaryOp::Mul)?;
        Some([(b, a), (a, b)])
    };
    if let Some([l, r]) = binary(g, root, BinaryOp::Add) {
        for (pos, prod) in [(l, r), (r, l)] {
            let Some(x) = with_zero(g, pos, BinaryOp::Max) else {
                continue;
            };
            let Some(pairs) = product(prod) else {
                continue;
            };
            for (slope, neg) in pairs {
                if with_zero(g, neg, BinaryOp::Min) == Some(x)
                    && same(pos)
                    && same(neg)
                    && same(prod)
                    && same(x)
                {
                    return Some(PreluMatch {
                        x,
                        slope,
                        interior: vec![pos, neg, prod],
                    });
                }
            }
        }
    }
    if let Some([pos, prod]) = binary(g, root, BinaryOp::Sub) {
        let x = activation(g, pos, Activation::Relu)?;
        for (slope, rn) in product(prod)? {
            let Some(n) = activation(g, rn, Activation::Relu) else {
                continue;
            };
            if activation(g, n, Activation::Neg) == Some(x)
                && same(pos)
                && same(rn)
                && same(n)
                && same(prod)
                && same(x)
            {
                return Some(PreluMatch {
                    x,
                    slope,
                    interior: vec![pos, n, rn, prod],
                });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn the_builder_is_what_the_matcher_finds() {
        let mut g = Graph::new("prelu");
        let x = g.input("x", sh(&[2, 3, 4]));
        let y = g.prelu_scalar(x, -0.25);
        let m = match_prelu(&g, y).expect("canonical PReLU");
        assert_eq!(m.x, x);
        assert_eq!(m.constant_slope(&g), Some(-0.25));
        assert_eq!(m.interior.len(), 3);

        // a per-channel slope, an input
        let a = g.input("a", sh(&[1, 3, 1]));
        let y = g.prelu(x, a);
        let m = match_prelu(&g, y).expect("per-channel PReLU");
        assert_eq!((m.x, m.slope, m.constant_slope(&g)), (x, a, None));
    }

    #[test]
    fn the_onnx_form_and_swapped_operands_match() {
        let s = sh(&[5]);
        let mut g = Graph::new("onnx");
        let x = g.input("x", s.clone());
        let a = g.input("a", sh(&[1]));
        let pos = g.activation(Activation::Relu, x, s.clone());
        let n = g.activation(Activation::Neg, x, s.clone());
        let rn = g.activation(Activation::Relu, n, s.clone());
        let prod = g.binary(BinaryOp::Mul, a, rn, s.clone());
        let y = g.binary(BinaryOp::Sub, pos, prod, s.clone());
        let m = match_prelu(&g, y).expect("ONNX PRelu");
        assert_eq!((m.x, m.slope), (x, a));

        // `0` first, the product first
        let zero = g.add_node(
            Op::Constant {
                data: 0f32.to_le_bytes().to_vec(),
            },
            vec![],
            sh(&[1]),
        );
        let p = g.binary(BinaryOp::Max, zero, x, s.clone());
        let q = g.binary(BinaryOp::Min, zero, x, s.clone());
        let q = g.binary(BinaryOp::Mul, a, q, s.clone());
        let y = g.binary(BinaryOp::Add, q, p, s.clone());
        assert_eq!(match_prelu(&g, y).map(|m| (m.x, m.slope)), Some((x, a)));
    }

    #[test]
    fn near_misses_do_not_match() {
        let s = sh(&[5]);
        let mut g = Graph::new("miss");
        let x = g.input("x", s.clone());
        let z = g.input("z", s.clone());
        let a = g.input("a", sh(&[1]));
        let one = g.add_node(
            Op::Constant {
                data: 1f32.to_le_bytes().to_vec(),
            },
            vec![],
            sh(&[1]),
        );
        let zero = g.add_node(
            Op::Constant {
                data: 0f32.to_le_bytes().to_vec(),
            },
            vec![],
            sh(&[1]),
        );
        // max against 1, not 0
        let p = g.binary(BinaryOp::Max, x, one, s.clone());
        let q = g.binary(BinaryOp::Min, x, zero, s.clone());
        let q = g.binary(BinaryOp::Mul, q, a, s.clone());
        let y = g.binary(BinaryOp::Add, p, q, s.clone());
        assert_eq!(match_prelu(&g, y), None);
        // the two halves of different tensors
        let p = g.binary(BinaryOp::Max, x, zero, s.clone());
        let q = g.binary(BinaryOp::Min, z, zero, s.clone());
        let q = g.binary(BinaryOp::Mul, q, a, s.clone());
        let y = g.binary(BinaryOp::Add, p, q, s.clone());
        assert_eq!(match_prelu(&g, y), None);
        // LeakyReLU's `max(x, a·x)` is not a PReLU form
        let ax = g.binary(BinaryOp::Mul, x, a, s.clone());
        let y = g.binary(BinaryOp::Max, x, ax, s);
        assert_eq!(match_prelu(&g, y), None);
    }
}
