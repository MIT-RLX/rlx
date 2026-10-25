// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Affine `Op::Scan` → log-depth prefix scan (`assoc-scan` feature).
//!
//! A scan whose body is a diagonal affine recurrence
//!
//! ```text
//! h_t = a_t ⊙ h_{t-1} + b_t        (a_t, b_t independent of h)
//! ```
//!
//! runs `length` dependent steps — on the CPU an interpreted body per step,
//! on GPU backends a host-driven loop. The step maps compose associatively:
//! applying `(a1, b1)` then `(a2, b2)` is `(a2·a1, a2·b1 + b2)`. So every
//! prefix composition can be computed with a Hillis–Steele scan in
//! `⌈log2 length⌉` rounds of whole-tensor elementwise ops, after which
//! `h_t = A_t ⊙ h_0 + B_t` for all `t` at once.
//!
//! This covers exponential moving averages, first-order IIR filters, diagonal
//! linear RNN / SSM state updates, cumulative sums (`a = 1`) and products
//! (`b = 0`). `a_t` and `b_t` may be any elementwise expression of the per-step
//! inputs, the broadcast inputs and constants; the expression is lifted out of
//! the body and evaluated once over the stacked `[length, ...]` inputs.
//!
//! Not bit-identical to the sequential loop: products and sums associate in a
//! different order. Work is `O(length · log length)` against the loop's
//! `O(length)`, in exchange for depth `O(log length)` and no per-step dispatch.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Node, NodeId, Op, OpKind, Shape};
use std::collections::HashMap;

use crate::pass::Pass;

/// The pass. Scans shorter than `min_length` are left alone (the existing
/// unroll handles short scans well).
pub struct ParallelizeAffineScan {
    pub min_length: u32,
}

impl Default for ParallelizeAffineScan {
    fn default() -> Self {
        Self { min_length: 8 }
    }
}

impl Pass for ParallelizeAffineScan {
    fn name(&self) -> &str {
        "parallelize_affine_scan"
    }

    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::Scan]
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph.nodes().iter().any(|n| self.candidate(n)) {
            return graph;
        }
        let mut out = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::with_capacity(graph.len());
        for node in graph.nodes() {
            let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
            let rewritten = if self.candidate(node) {
                rewrite_scan(&mut out, node, &inputs)
            } else {
                None
            };
            let id = rewritten.unwrap_or_else(|| {
                let id = out.add_node(node.op.clone(), inputs, node.shape.clone());
                let n = out.node_mut(id);
                n.name = node.name.clone();
                n.origin = node.origin.clone();
                id
            });
            id_map.insert(node.id, id);
        }
        out.set_outputs(graph.outputs.iter().map(|o| id_map[o]).collect());
        out
    }
}

impl ParallelizeAffineScan {
    fn candidate(&self, n: &Node) -> bool {
        matches!(
            &n.op,
            Op::Scan { length, num_checkpoints: 0, .. } if *length >= self.min_length
        ) && n.shape.dtype() == DType::F32
    }
}

/// A coefficient of the affine form: known `0`, known `1`, or a lifted
/// `[length, *carry]` node in the outer graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Coef {
    Zero,
    One,
    Node(NodeId),
}

/// Lifts body subexpressions into the outer graph, over the time axis.
struct Lifter<'a> {
    body: &'a Graph,
    out: &'a mut Graph,
    carry: NodeId,
    /// Body input id → outer node already shaped `[length, *carry]`.
    leaves: HashMap<NodeId, NodeId>,
    /// Memo of carry-free body nodes already lifted.
    lifted: HashMap<NodeId, NodeId>,
    /// `[length, *carry]`.
    stacked: Shape,
    carry_dims: Vec<usize>,
}

impl Lifter<'_> {
    fn binary(&mut self, op: BinaryOp, a: NodeId, b: NodeId) -> NodeId {
        self.out
            .add_node(Op::Binary(op), vec![a, b], self.stacked.clone())
    }

    fn stack(&mut self, x: NodeId) -> NodeId {
        stack(self.out, x, &self.carry_dims, &self.stacked)
    }

    /// Evaluate a carry-free body node over all steps. `None` when it uses an
    /// op this lifting does not cover.
    fn lift(&mut self, id: NodeId) -> Option<NodeId> {
        if let Some(&v) = self.lifted.get(&id) {
            return Some(v);
        }
        if let Some(&v) = self.leaves.get(&id) {
            return Some(v);
        }
        let n = self.body.node(id);
        if n.shape.dims().len() != self.carry_dims.len()
            || n.shape.dtype() != DType::F32
            || n.shape
                .dims()
                .iter()
                .zip(&self.carry_dims)
                .any(|(d, &c)| d.unwrap_static() != c)
        {
            return None;
        }
        let v = match &n.op {
            Op::Constant { .. } => {
                let c = self.out.add_node(n.op.clone(), vec![], n.shape.clone());
                self.stack(c)
            }
            Op::Binary(op @ (BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div)) => {
                let a = self.lift(n.inputs[0])?;
                let b = self.lift(n.inputs[1])?;
                self.binary(*op, a, b)
            }
            Op::Activation(act) => {
                let a = self.lift(n.inputs[0])?;
                self.out
                    .add_node(Op::Activation(*act), vec![a], self.stacked.clone())
            }
            _ => return None,
        };
        self.lifted.insert(id, v);
        Some(v)
    }

    fn depends_on_carry(&self, id: NodeId, memo: &mut HashMap<NodeId, bool>) -> bool {
        if id == self.carry {
            return true;
        }
        if let Some(&d) = memo.get(&id) {
            return d;
        }
        let d = self
            .body
            .node(id)
            .inputs
            .iter()
            .any(|&i| self.depends_on_carry(i, memo));
        memo.insert(id, d);
        d
    }

    fn add(&mut self, a: Coef, b: Coef) -> Option<Coef> {
        Some(match (a, b) {
            (Coef::Zero, x) | (x, Coef::Zero) => x,
            (Coef::Node(x), Coef::Node(y)) => Coef::Node(self.binary(BinaryOp::Add, x, y)),
            _ => return None,
        })
    }

    fn sub(&mut self, a: Coef, b: Coef) -> Option<Coef> {
        Some(match (a, b) {
            (x, Coef::Zero) => x,
            (Coef::Zero, Coef::Node(y)) => Coef::Node(self.out.add_node(
                Op::Activation(rlx_ir::op::Activation::Neg),
                vec![y],
                self.stacked.clone(),
            )),
            (Coef::Node(x), Coef::Node(y)) => Coef::Node(self.binary(BinaryOp::Sub, x, y)),
            _ => return None,
        })
    }

    /// `c ⊙ x` for a lifted carry-free `c`.
    fn scale(&mut self, c: NodeId, x: Coef, op: BinaryOp) -> Coef {
        match x {
            Coef::Zero => Coef::Zero,
            Coef::One if op == BinaryOp::Mul => Coef::Node(c),
            Coef::One => unreachable!("1 / c is not formed"),
            Coef::Node(x) => Coef::Node(self.binary(op, x, c)),
        }
    }

    /// `(a, b)` with `body[id] = a ⊙ carry + b`, or `None` if not affine in
    /// the carry (or not expressible with the covered ops).
    fn affine(&mut self, id: NodeId, memo: &mut HashMap<NodeId, bool>) -> Option<(Coef, Coef)> {
        if id == self.carry {
            return Some((Coef::One, Coef::Zero));
        }
        if !self.depends_on_carry(id, memo) {
            return Some((Coef::Zero, Coef::Node(self.lift(id)?)));
        }
        let n = self.body.node(id);
        let (l, r) = match n.inputs.as_slice() {
            [l, r] => (*l, *r),
            _ => return None,
        };
        match &n.op {
            Op::Binary(BinaryOp::Add) => {
                let (al, bl) = self.affine(l, memo)?;
                let (ar, br) = self.affine(r, memo)?;
                Some((self.add(al, ar)?, self.add(bl, br)?))
            }
            Op::Binary(BinaryOp::Sub) => {
                let (al, bl) = self.affine(l, memo)?;
                let (ar, br) = self.affine(r, memo)?;
                Some((self.sub(al, ar)?, self.sub(bl, br)?))
            }
            Op::Binary(BinaryOp::Mul) => {
                // Exactly one side may carry the state.
                let (c, x) = match (
                    self.depends_on_carry(l, memo),
                    self.depends_on_carry(r, memo),
                ) {
                    (true, false) => (r, l),
                    (false, true) => (l, r),
                    _ => return None,
                };
                let c = self.lift(c)?;
                let (a, b) = self.affine(x, memo)?;
                Some((
                    self.scale(c, a, BinaryOp::Mul),
                    self.scale(c, b, BinaryOp::Mul),
                ))
            }
            Op::Binary(BinaryOp::Div) if !self.depends_on_carry(r, memo) => {
                let c = self.lift(r)?;
                let (a, b) = self.affine(l, memo)?;
                if a == Coef::One || b == Coef::One {
                    return None;
                }
                Some((
                    self.scale(c, a, BinaryOp::Div),
                    self.scale(c, b, BinaryOp::Div),
                ))
            }
            _ => None,
        }
    }
}

/// Broadcast an outer `*carry`-shaped node to `[length, *carry]`.
fn stack(out: &mut Graph, x: NodeId, carry_dims: &[usize], stacked: &Shape) -> NodeId {
    let mut one_dims = vec![1usize];
    one_dims.extend(carry_dims);
    let r = out.add_node(
        Op::Reshape {
            new_shape: one_dims.iter().map(|&d| d as i64).collect(),
        },
        vec![x],
        Shape::new(&one_dims, DType::F32),
    );
    let target = stacked.dims().iter().map(|d| d.unwrap_static() as i64);
    out.add_node(
        Op::Expand {
            target_shape: target.collect(),
        },
        vec![r],
        stacked.clone(),
    )
}

/// Shift `x` down by `k` rows along axis 0, filling with `fill`.
fn shift(out: &mut Graph, x: NodeId, k: usize, len: usize, fill: f32, stacked: &Shape) -> NodeId {
    // Built from `Narrow` / `Expand` / `Concat` rather than `Pad`: every GPU
    // backend runs those on device, while `Pad` is host-staged on some (Metal),
    // which turned each of the 2·log2(length) shifts into a device sync.
    let dims: Vec<usize> = stacked.dims().iter().map(|d| d.unwrap_static()).collect();
    let with_rows = |rows: usize| {
        let mut d = dims.clone();
        d[0] = rows;
        Shape::new(&d, DType::F32)
    };
    let head = out.add_node(
        Op::Narrow {
            axis: 0,
            start: 0,
            len: len - k,
        },
        vec![x],
        with_rows(len - k),
    );
    let per_row: usize = dims[1..].iter().product();
    let fill_row = out.add_node(
        Op::Constant {
            data: fill.to_le_bytes().repeat(per_row),
        },
        vec![],
        with_rows(1),
    );
    let mut target: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
    target[0] = k as i64;
    let block = out.add_node(
        Op::Expand {
            target_shape: target,
        },
        vec![fill_row],
        with_rows(k),
    );
    out.add_node(Op::Concat { axis: 0 }, vec![block, head], stacked.clone())
}

/// Emit the prefix-scan replacement for one `Op::Scan`, or `None` to keep it.
fn rewrite_scan(out: &mut Graph, node: &Node, inputs: &[NodeId]) -> Option<NodeId> {
    let Op::Scan {
        body,
        length,
        save_trajectory,
        num_bcast,
        num_xs,
        ..
    } = &node.op
    else {
        return None;
    };
    let (len, nb, nx) = (*length as usize, *num_bcast as usize, *num_xs as usize);
    let [result] = body.outputs.as_slice() else {
        return None;
    };
    let body_inputs: Vec<NodeId> = body
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Input { .. }))
        .map(|n| n.id)
        .collect();
    if body_inputs.len() != 1 + nb + nx || inputs.len() != 1 + nb + nx {
        return None;
    }
    let carry = body_inputs[0];
    let carry_shape = body.node(carry).shape.clone();
    if carry_shape.dtype() != DType::F32 || carry_shape.dims().iter().any(|d| !d.is_static()) {
        return None;
    }
    let carry_dims: Vec<usize> = carry_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let mut stacked_dims = vec![len];
    stacked_dims.extend(&carry_dims);
    let stacked = Shape::new(&stacked_dims, DType::F32);

    // Per-step inputs must already be `[length, *carry]`; bcasts `*carry`.
    let same = |s: &Shape, want: &[usize]| {
        s.dtype() == DType::F32
            && s.dims().len() == want.len()
            && s.dims()
                .iter()
                .zip(want)
                .all(|(d, &w)| d.is_static() && d.unwrap_static() == w)
    };
    if !same(&out.node(inputs[0]).shape, &carry_dims) {
        return None;
    }
    for &b in &inputs[1..1 + nb] {
        if !same(&out.node(b).shape, &carry_dims) {
            return None;
        }
    }
    for &x in &inputs[1 + nb..] {
        if !same(&out.node(x).shape, &stacked_dims) {
            return None;
        }
    }

    // Analyse into a scratch graph first so a decline leaves `out` untouched.
    let mut scratch = out.clone();
    let mut lifter = Lifter {
        body,
        out: &mut scratch,
        carry,
        leaves: HashMap::new(),
        lifted: HashMap::new(),
        stacked: stacked.clone(),
        carry_dims: carry_dims.clone(),
    };
    for (i, &b) in inputs[1..1 + nb].iter().enumerate() {
        let s = lifter.stack(b);
        lifter.leaves.insert(body_inputs[1 + i], s);
    }
    for (i, &x) in inputs[1 + nb..].iter().enumerate() {
        lifter.leaves.insert(body_inputs[1 + nb + i], x);
    }
    let mut memo = HashMap::new();
    let (mut a, mut b) = lifter.affine(*result, &mut memo)?;
    if a == Coef::Zero {
        // `h_t = b_t`: no recurrence left; the rewrite would be correct but the
        // body is better served by the ordinary path.
        return None;
    }

    // Hillis–Steele: after the round with offset `k`, row `t` holds the
    // composition of steps `max(0, t-2k+1) ..= t`.
    let mut k = 1;
    while k < len {
        if let Coef::Node(bn) = b {
            let b_prev = shift(&mut scratch, bn, k, len, 0.0, &stacked);
            b = Coef::Node(match a {
                Coef::Node(an) => {
                    let ab = scratch.add_node(
                        Op::Binary(BinaryOp::Mul),
                        vec![an, b_prev],
                        stacked.clone(),
                    );
                    scratch.add_node(Op::Binary(BinaryOp::Add), vec![ab, bn], stacked.clone())
                }
                _ => scratch.add_node(Op::Binary(BinaryOp::Add), vec![b_prev, bn], stacked.clone()),
            });
        }
        if let Coef::Node(an) = a {
            let a_prev = shift(&mut scratch, an, k, len, 1.0, &stacked);
            a = Coef::Node(scratch.add_node(
                Op::Binary(BinaryOp::Mul),
                vec![an, a_prev],
                stacked.clone(),
            ));
        }
        k *= 2;
    }

    // h_t = A_t ⊙ h_0 + B_t.
    let h0 = stack(&mut scratch, inputs[0], &carry_dims, &stacked);
    let ah0 = match a {
        Coef::One => h0,
        Coef::Node(an) => {
            scratch.add_node(Op::Binary(BinaryOp::Mul), vec![an, h0], stacked.clone())
        }
        Coef::Zero => unreachable!(),
    };
    let traj = match b {
        Coef::Zero => ah0,
        Coef::Node(bn) => {
            scratch.add_node(Op::Binary(BinaryOp::Add), vec![ah0, bn], stacked.clone())
        }
        Coef::One => return None,
    };

    let result = if *save_trajectory {
        traj
    } else {
        let mut row_dims = vec![1usize];
        row_dims.extend(&carry_dims);
        let row = scratch.add_node(
            Op::Narrow {
                axis: 0,
                start: len - 1,
                len: 1,
            },
            vec![traj],
            Shape::new(&row_dims, DType::F32),
        );
        scratch.add_node(
            Op::Reshape {
                new_shape: carry_dims.iter().map(|&d| d as i64).collect(),
            },
            vec![row],
            carry_shape,
        )
    };
    if scratch.node(result).shape != node.shape {
        return None;
    }
    *out = scratch;
    Some(result)
}
