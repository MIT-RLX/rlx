// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::Fma` for backends without a native fused multiply-add.
//!
//! **F32 with static shapes: exact.** The single rounding is rebuilt from IEEE f32
//! `mul`/`add`/`compare`/`where` (error-free transforms):
//!
//! 1. `a·b = p + e` exactly (Dekker TwoProduct, Veltkamp split by `2¹²+1`);
//! 2. `(th, tl) = TwoSum(c, p)`, `(v, w) = TwoSum(tl, e)`, `(r, q) = TwoSum(th, v)`;
//! 3. `r` is the correctly rounded `c + a·b` except when `th + v` is exactly a
//!    midpoint (`q ≠ 0` and `r + 2q` representable) and the dropped `w` lies on
//!    `q`'s side — then the answer is `r + 2q`;
//! 4. an exact zero takes `p + c` (IEEE's sign of zero); a non-finite `p` too.
//!
//! Bit-identical to a hardware FMA for operands in the normal range — |a|, |b| <
//! 2¹¹⁵ (the split) and |a·b| ≥ 2⁻¹⁰² (the product error stays normal); checked
//! against `f32::mul_add` on 2.1·10⁸ inputs including 3.6·10⁷ double-rounding
//! ties, every step mutation-checked. Costs ~35 elementwise ops instead of one.
//! The backend must round `Binary` exactly (no contraction of its own) — see
//! `rlx-runtime/tests/elementwise_rounding_probe.rs`.
//!
//! Other dtypes and dynamic shapes keep the two-rounding `mul` + `add`.

use crate::rewriter::{MatchRewrite, RewriteCtx};
use rlx_ir::op::{BinaryOp, CmpOp};
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
        if node.shape.dtype() == DType::F32 && node.shape.dims().iter().all(|d| d.is_static()) {
            return Some(exact_fma_f32(ctx, &node.shape, a, b, c));
        }
        let prod = ctx.emit(Op::Binary(BinaryOp::Mul), vec![a, b], node.shape.clone());
        Some(ctx.emit(Op::Binary(BinaryOp::Add), vec![prod, c], node.shape.clone()))
    }
}

/// `fma(a, b, c)` in f32 with one rounding, from `mul`/`add`/`compare`/`where`.
fn exact_fma_f32(ctx: &mut RewriteCtx, shape: &Shape, a: NodeId, b: NodeId, c: NodeId) -> NodeId {
    let dims: Vec<i64> = shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    let rank = dims.len();
    let bool_shape = Shape::from_dims(shape.dims(), DType::Bool);
    let full = |ctx: &mut RewriteCtx, x: NodeId| -> NodeId {
        if ctx.out.node(x).shape.dims() == shape.dims() {
            x
        } else {
            ctx.emit(
                Op::Expand {
                    target_shape: dims.clone(),
                },
                vec![x],
                shape.clone(),
            )
        }
    };
    let konst = |ctx: &mut RewriteCtx, v: f32| -> NodeId {
        let k = ctx.emit(
            Op::Constant {
                data: v.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&vec![1; rank.max(1)], DType::F32),
        );
        if rank == 0 {
            let s = Shape::new(&[], DType::F32);
            return ctx.emit(Op::Reshape { new_shape: vec![] }, vec![k], s);
        }
        ctx.emit(
            Op::Expand {
                target_shape: dims.clone(),
            },
            vec![k],
            shape.clone(),
        )
    };
    let bin = |ctx: &mut RewriteCtx, op: BinaryOp, x: NodeId, y: NodeId| -> NodeId {
        ctx.emit(Op::Binary(op), vec![x, y], shape.clone())
    };
    let cmp = |ctx: &mut RewriteCtx, op: CmpOp, x: NodeId, y: NodeId| -> NodeId {
        ctx.emit(Op::Compare(op), vec![x, y], bool_shape.clone())
    };
    let sel = |ctx: &mut RewriteCtx, m: NodeId, t: NodeId, f: NodeId| -> NodeId {
        ctx.emit(Op::Where, vec![m, t, f], shape.clone())
    };
    let two_sum = |ctx: &mut RewriteCtx, x: NodeId, y: NodeId| -> (NodeId, NodeId) {
        let s = bin(ctx, BinaryOp::Add, x, y);
        let bb = bin(ctx, BinaryOp::Sub, s, x);
        let sb = bin(ctx, BinaryOp::Sub, s, bb);
        let ea = bin(ctx, BinaryOp::Sub, x, sb);
        let eb = bin(ctx, BinaryOp::Sub, y, bb);
        (s, bin(ctx, BinaryOp::Add, ea, eb))
    };
    let (a, b, c) = (full(ctx, a), full(ctx, b), full(ctx, c));
    let k4097 = konst(ctx, 4097.0);
    let zero = konst(ctx, 0.0);
    let split = |ctx: &mut RewriteCtx, x: NodeId| -> (NodeId, NodeId) {
        let t = bin(ctx, BinaryOp::Mul, k4097, x);
        let d = bin(ctx, BinaryOp::Sub, t, x);
        let hi = bin(ctx, BinaryOp::Sub, t, d);
        (hi, bin(ctx, BinaryOp::Sub, x, hi))
    };
    let p = bin(ctx, BinaryOp::Mul, a, b);
    let (ah, al) = split(ctx, a);
    let (bh, bl) = split(ctx, b);
    // e = ((ah*bh - p) + ah*bl + al*bh) + al*bl
    let hh = bin(ctx, BinaryOp::Mul, ah, bh);
    let e = bin(ctx, BinaryOp::Sub, hh, p);
    let hl = bin(ctx, BinaryOp::Mul, ah, bl);
    let e = bin(ctx, BinaryOp::Add, e, hl);
    let lh = bin(ctx, BinaryOp::Mul, al, bh);
    let e = bin(ctx, BinaryOp::Add, e, lh);
    let ll = bin(ctx, BinaryOp::Mul, al, bl);
    let e = bin(ctx, BinaryOp::Add, e, ll);
    let (th, tl) = two_sum(ctx, c, p);
    let (v, w) = two_sum(ctx, tl, e);
    let (r, q) = two_sum(ctx, th, v);
    // tie: q != 0 and r + 2q representable; then w on q's side → r + 2q
    let q2 = bin(ctx, BinaryOp::Add, q, q);
    let up = bin(ctx, BinaryOp::Add, r, q2);
    let step = bin(ctx, BinaryOp::Sub, up, r);
    let one = konst(ctx, 1.0);
    let neg_one = konst(ctx, -1.0);
    let q_pos = cmp(ctx, CmpOp::Gt, q, zero);
    let sgn = sel(ctx, q_pos, one, neg_one);
    let wq = bin(ctx, BinaryOp::Mul, w, sgn);
    let same = cmp(ctx, CmpOp::Gt, wq, zero);
    let rep = cmp(ctx, CmpOp::Eq, step, q2);
    let qnz = cmp(ctx, CmpOp::Ne, q, zero);
    let r_same = sel(ctx, same, up, r);
    let r_rep = sel(ctx, rep, r_same, r);
    let r = sel(ctx, qnz, r_rep, r);
    // exact zero / non-finite product: p + c
    let pc = bin(ctx, BinaryOp::Add, p, c);
    let r_zero = cmp(ctx, CmpOp::Eq, r, zero);
    let r = sel(ctx, r_zero, pc, r);
    let pp = bin(ctx, BinaryOp::Sub, p, p);
    let finite = cmp(ctx, CmpOp::Eq, pp, zero);
    sel(ctx, finite, r, pc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass::Pass;

    fn f32_shape(dims: &[usize]) -> Shape {
        Shape::new(dims, DType::F32)
    }

    #[test]
    fn lowers_non_f32_fma_to_mul_add() {
        let s = Shape::new(&[8], DType::F64);
        let mut g = Graph::new("fma_lower");
        let a = g.input("a", s.clone());
        let b = g.input("b", s.clone());
        let c = g.input("c", s.clone());
        let fma = g.add_node(Op::Fma, vec![a, b, c], s.clone());
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
    fn lowers_f32_fma_to_the_exact_emulation() {
        let mut g = Graph::new("fma_exact");
        let a = g.input("a", f32_shape(&[8]));
        let b = g.input("b", f32_shape(&[1]));
        let c = g.input("c", f32_shape(&[8]));
        let fma = g.add_node(Op::Fma, vec![a, b, c], f32_shape(&[8]));
        g.set_outputs(vec![fma]);
        let out = LowerFma.run(g);
        assert!(!out.nodes().iter().any(|n| matches!(n.op, Op::Fma)));
        // the tie correction and zero/non-finite fixups end in a Where
        assert!(matches!(out.node(out.outputs[0]).op, Op::Where));
        // a broadcast operand is expanded to the output shape first
        assert!(
            out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Expand { .. }))
        );
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
