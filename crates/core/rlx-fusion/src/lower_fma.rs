// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::Fma` to `Mul` + `Add` for backends without a native fused
//! multiply-add.
//!
//! NOTE: this fallback uses TWO roundings, so it does **not** preserve the
//! single-rounding (error-free-transform) precision that `Op::Fma` exists for.
//! Backends that want compensated / double-word arithmetic must implement
//! `Op::Fma` natively (e.g. WGSL/MSL/CUDA `fma()`). This pass only guarantees
//! correctness-of-value on FMA-less backends (e.g. CoreML/ANE) so a graph that
//! uses `Op::Fma` runs everywhere instead of erroring at legalization.

use crate::rewriter::{MatchRewrite, RewriteCtx};
use rlx_ir::op::BinaryOp;
use rlx_ir::*;

pub struct LowerFma;

impl MatchRewrite for LowerFma {
    fn name(&self) -> &str {
        "lower_fma"
    }

    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::Fma]
    }

    fn rewrite(&self, node: &Node, ctx: &mut RewriteCtx) -> Option<NodeId> {
        let (a, b, c) = (ctx.input(0), ctx.input(1), ctx.input(2));
        let prod = ctx.emit(Op::Binary(BinaryOp::Mul), vec![a, b], node.shape.clone());
        Some(ctx.emit(Op::Binary(BinaryOp::Add), vec![prod, c], node.shape.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass::Pass;

    fn f32_shape(dims: &[usize]) -> Shape {
        Shape::new(dims, DType::F32)
    }

    #[test]
    fn lowers_fma_to_mul_add() {
        let mut g = Graph::new("fma_lower");
        let a = g.input("a", f32_shape(&[8]));
        let b = g.input("b", f32_shape(&[8]));
        let c = g.input("c", f32_shape(&[8]));
        let fma = g.add_node(Op::Fma, vec![a, b, c], f32_shape(&[8]));
        g.set_outputs(vec![fma]);

        let out = LowerFma.run(g);

        // No Op::Fma survives; it became one Mul and one Add.
        assert!(
            !out.nodes().iter().any(|n| matches!(n.op, Op::Fma)),
            "LowerFma left an Op::Fma behind"
        );
        let muls = out
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Binary(BinaryOp::Mul)))
            .count();
        let adds = out
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Binary(BinaryOp::Add)))
            .count();
        assert_eq!((muls, adds), (1, 1), "expected exactly one Mul and one Add");

        // The Add must consume the Mul's product and `c` (a*b + c), and the
        // graph's single output is that Add.
        let out_id = out.outputs[0];
        let add = out.node(out_id);
        assert!(matches!(add.op, Op::Binary(BinaryOp::Add)));
        let prod_id = add.inputs[0];
        assert!(matches!(out.node(prod_id).op, Op::Binary(BinaryOp::Mul)));
    }

    #[test]
    fn no_fma_is_a_noop() {
        // A graph with no Op::Fma passes through unchanged (same node count).
        let mut g = Graph::new("no_fma");
        let a = g.input("a", f32_shape(&[4]));
        let b = g.input("b", f32_shape(&[4]));
        let sum = g.add_node(Op::Binary(BinaryOp::Add), vec![a, b], f32_shape(&[4]));
        g.set_outputs(vec![sum]);
        let before = g.nodes().len();
        let out = LowerFma.run(g);
        assert_eq!(out.nodes().len(), before);
    }
}
