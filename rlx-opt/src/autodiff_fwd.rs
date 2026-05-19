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

//! Forward-mode AD (JVP transform).
//!
//! Companion to `autodiff::grad_with_loss` (reverse-mode). Where the
//! reverse pass walks the graph backward to accumulate gradients into
//! a small set of parameters, this pass walks **forward** to push a
//! handful of input perturbations through to all outputs. It's the
//! right tool when:
//!
//! * the input dimension is small and the output dimension is large
//!   (Jacobian column at a time → forward-mode wins);
//! * the workload is Newton-style iterations on a small parameter
//!   vector where `jacfwd` over the flat parameter vector is the
//!   right shape for the Jacobian.
//!
//! ## Semantics
//!
//! `jvp(forward, tangent_for)` emits a new `Graph` that takes:
//!
//! * every original `Input` / `Param` node, and
//! * one extra `Input` per node in `tangent_for`, named
//!   `"tangent_<original_name>"`, with the same shape and dtype.
//!
//! It produces:
//!
//! * the original forward outputs (primals), unchanged, and
//! * one tangent per primal output, in the same order.
//!
//! Outputs of the returned graph are therefore
//! `[primal_0, …, primal_{k-1}, tangent_0, …, tangent_{k-1}]`.
//!
//! ## Implementation
//!
//! Standard pushforward: each forward node `f(x_0, ..., x_n)` gets a
//! tangent `t_y = Σ_i (∂f/∂x_i) · t_{x_i}` where `t_{x_i}` is the
//! tangent of input `x_i` (or symbolic zero when none of the
//! function's parameters reach that input). Symbolic zeros are tracked
//! as `None` so we don't bloat the graph with multiplications by 0
//! constants.
//!
//! For ops that don't yet have a JVP rule we panic — same policy as
//! the reverse pass — so a silent miscompute is impossible.

use rlx_ir::op::*;
use rlx_ir::shape::Dim;
use rlx_ir::*;
use std::collections::HashMap;

/// Compute the JVP graph for `forward`, perturbing each `Input` /
/// `Param` named in `tangent_for`. Returns a new graph whose outputs
/// are `[primals..., tangents...]`, in the order forward listed them.
///
/// # Limitations
/// * Forward must have at least one output (typical: 1 for Hello
///   Resistor). All forward outputs get tangents in the result.
/// * `tangent_for` must list `Input` / `Param` nodes only. Tangent
///   inputs for intermediate nodes don't make sense in this API.
/// * Hitting an op without a JVP rule is a panic, not a silent
///   miscompute.
pub fn jvp(forward: &Graph, tangent_for: &[NodeId]) -> Graph {
    // Pre-passes that the reverse pass also runs — keeps the JVP rule
    // table small (no need to handle fused-attention etc. directly).
    use crate::pass::Pass as _;
    let forward_owned = crate::fusion::UnfuseElementwiseRegions.run(forward.clone());
    let forward_owned = crate::control_flow::inline_if(forward_owned);
    let forward_owned = crate::control_flow::unroll_while(forward_owned);
    let forward = &forward_owned;

    let mut bwd = Graph::new(format!("{}_jvp", forward.name));

    // Mirror every forward node — the tangent rules need access to the
    // primal values (`a` and `b` for MatMul JVP, `x` for DenseSolve JVP).
    let mut fwd_to_bwd: HashMap<NodeId, NodeId> = HashMap::new();
    for node in forward.nodes() {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| fwd_to_bwd[i]).collect();
        let new_id = bwd.add_node(node.op.clone(), inputs, node.shape.clone());
        fwd_to_bwd.insert(node.id, new_id);
    }

    // Build tangents for the seeded inputs — a fresh Input named
    // "tangent_<original>" with the same shape.
    let mut tangents: HashMap<NodeId, NodeId> = HashMap::new();
    for &id in tangent_for {
        let original = forward.node(id);
        let name = match &original.op {
            Op::Input { name } | Op::Param { name } => name.clone(),
            other => panic!("jvp: tangent_for[{id}] must be Input/Param, got {other:?}"),
        };
        let tangent = bwd.input(format!("tangent_{name}"), original.shape.clone());
        tangents.insert(id, tangent);
    }

    // Walk forward in topological order; emit tangents node-by-node.
    for fwd_node in forward.nodes() {
        // If this node is itself a seeded tangent, the tangent is
        // already in the map — skip the rule.
        if tangents.contains_key(&fwd_node.id) {
            continue;
        }
        let in_tangents: Vec<Option<NodeId>> = fwd_node
            .inputs
            .iter()
            .map(|id| tangents.get(id).copied())
            .collect();
        if in_tangents.iter().all(Option::is_none) {
            // No upstream tangent reaches this node — symbolic zero,
            // don't emit anything. Downstream JVP rules will see None
            // for this slot.
            continue;
        }
        if let Some(t_out) = jvp_rule(fwd_node, &in_tangents, &fwd_to_bwd, &mut bwd) {
            tangents.insert(fwd_node.id, t_out);
        }
    }

    // Outputs: [primals..., tangents-of-primals...]
    let mut outs = Vec::with_capacity(2 * forward.outputs.len());
    for &out in &forward.outputs {
        outs.push(fwd_to_bwd[&out]);
    }
    for &out in &forward.outputs {
        let t = match tangents.get(&out) {
            Some(&t) => t,
            None => zero_like(fwd_to_bwd[&out], &mut bwd),
        };
        outs.push(t);
    }
    bwd.set_outputs(outs);
    bwd
}

/// Hessian-vector product via forward-over-reverse.
///
/// Composes [`crate::autodiff::grad_with_loss`] (reverse-mode) with
/// [`jvp`] (forward-mode): the JVP through the gradient function is
/// `H·v` where H is the Hessian of the loss w.r.t. the seeded inputs.
///
/// The resulting graph has inputs:
/// * Every `Op::Input` / `Op::Param` from the forward graph (unchanged
///   names + shapes).
/// * `"d_output"` — upstream loss gradient (typically `[1.0]`).
/// * `"tangent_<name>"` per entry in `wrt` — the v vector(s).
///
/// And outputs (in this order):
/// * `[primal_loss, grad_0, …, grad_{k-1}]` — the reverse-mode
///   pass's outputs unchanged.
/// * `[tangent_loss, H·v_0, …, H·v_{k-1}]` — JVP of each. The first
///   entry is `<grad, v>` (a scalar, sometimes useful for stoppage
///   tests); the rest are the Hessian-vector products.
pub fn hvp(forward: &Graph, wrt: &[NodeId]) -> Graph {
    let bwd = crate::autodiff::grad_with_loss(forward, wrt);
    // Re-find each `wrt` input by name in the backward graph
    // (grad_with_loss preserves Input/Param names but reassigns NodeIds).
    let names: Vec<String> = wrt
        .iter()
        .map(|&id| match &forward.node(id).op {
            Op::Input { name } | Op::Param { name } => name.clone(),
            other => panic!("hvp: wrt[{id}] must be Input/Param, got {other:?}"),
        })
        .collect();
    let bwd_ids: Vec<NodeId> = names
        .iter()
        .map(|name| {
            bwd.nodes()
                .iter()
                .find(|n| match &n.op {
                    Op::Input { name: n_name } | Op::Param { name: n_name } => n_name == name,
                    _ => false,
                })
                .map(|n| n.id)
                .unwrap_or_else(|| panic!("hvp: input '{name}' missing in backward graph"))
        })
        .collect();
    jvp(&bwd, &bwd_ids)
}

/// Build a constant zero with the same shape/dtype as `like`.
fn zero_like(like: NodeId, bwd: &mut Graph) -> NodeId {
    let shape = bwd.node(like).shape.clone();
    let n_bytes = shape.size_bytes().unwrap_or(0);
    let data = vec![0u8; n_bytes];
    bwd.add_node(Op::Constant { data }, vec![], shape)
}

/// Per-op JVP rule. Returns the tangent node for this forward node,
/// or `None` if the result is symbolically zero (caller treats that
/// as "no tangent").
///
/// `t_inputs[i]` is `Some(node)` when input `i` has a non-zero
/// tangent, `None` for symbolic zero. The rule should short-circuit
/// when convenient (e.g., `Add` with one operand zero just returns
/// the other tangent).
fn jvp_rule(
    node: &Node,
    t_inputs: &[Option<NodeId>],
    fwd_map: &HashMap<NodeId, NodeId>,
    bwd: &mut Graph,
) -> Option<NodeId> {
    match &node.op {
        // Leaves: handled by the seed phase. If we reach a leaf here
        // without a seeded tangent, the tangent is zero.
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => None,

        Op::Binary(op) => {
            let a_p = fwd_map[&node.inputs[0]];
            let b_p = fwd_map[&node.inputs[1]];
            let out_shape = node.shape.clone();
            match op {
                BinaryOp::Add => {
                    // t_y = t_a + t_b, with zeros short-circuited.
                    match (t_inputs[0], t_inputs[1]) {
                        (Some(ta), Some(tb)) => Some(bwd.binary(BinaryOp::Add, ta, tb, out_shape)),
                        (Some(ta), None) => Some(ta),
                        (None, Some(tb)) => Some(tb),
                        (None, None) => None,
                    }
                }
                BinaryOp::Sub => {
                    // t_y = t_a − t_b
                    match (t_inputs[0], t_inputs[1]) {
                        (Some(ta), Some(tb)) => Some(bwd.binary(BinaryOp::Sub, ta, tb, out_shape)),
                        (Some(ta), None) => Some(ta),
                        (None, Some(tb)) => {
                            let s = bwd.node(tb).shape.clone();
                            Some(bwd.activation(Activation::Neg, tb, s))
                        }
                        (None, None) => None,
                    }
                }
                BinaryOp::Mul => {
                    // t_y = t_a · b + a · t_b
                    let ta_b =
                        t_inputs[0].map(|ta| bwd.binary(BinaryOp::Mul, ta, b_p, out_shape.clone()));
                    let a_tb =
                        t_inputs[1].map(|tb| bwd.binary(BinaryOp::Mul, a_p, tb, out_shape.clone()));
                    match (ta_b, a_tb) {
                        (Some(x), Some(y)) => Some(bwd.binary(BinaryOp::Add, x, y, out_shape)),
                        (Some(x), None) | (None, Some(x)) => Some(x),
                        (None, None) => None,
                    }
                }
                BinaryOp::Div => {
                    // t_y = (t_a · b − a · t_b) / b²
                    // Lazy implementation: build via primitives.
                    let ta_b =
                        t_inputs[0].map(|ta| bwd.binary(BinaryOp::Mul, ta, b_p, out_shape.clone()));
                    let a_tb =
                        t_inputs[1].map(|tb| bwd.binary(BinaryOp::Mul, a_p, tb, out_shape.clone()));
                    let numer = match (ta_b, a_tb) {
                        (Some(x), Some(y)) => {
                            Some(bwd.binary(BinaryOp::Sub, x, y, out_shape.clone()))
                        }
                        (Some(x), None) => Some(x),
                        (None, Some(y)) => {
                            let s = bwd.node(y).shape.clone();
                            Some(bwd.activation(Activation::Neg, y, s))
                        }
                        (None, None) => None,
                    };
                    numer.map(|n| {
                        let bb = bwd.binary(BinaryOp::Mul, b_p, b_p, out_shape.clone());
                        bwd.binary(BinaryOp::Div, n, bb, out_shape)
                    })
                }
                BinaryOp::Min => {
                    // y = min(a, b); ∂y/∂a = 1 if a<b else 0,
                    // ∂y/∂b = 1 if b<a else 0. → Where(a<b, t_a, t_b).
                    let zero = scalar_const(0.0, &out_shape, bwd);
                    let cond =
                        bwd.add_node(Op::Compare(CmpOp::Lt), vec![a_p, b_p], out_shape.clone());
                    let ta = t_inputs[0].unwrap_or(zero);
                    let tb = t_inputs[1].unwrap_or(zero);
                    Some(bwd.add_node(Op::Where, vec![cond, ta, tb], out_shape))
                }
                BinaryOp::Max => {
                    // y = max(a, b); pick t_a when a>b → Lt(b, a) is the
                    // canonical encoding (no Gt method on builder, but
                    // reading the args swapped is identical).
                    let zero = scalar_const(0.0, &out_shape, bwd);
                    let cond =
                        bwd.add_node(Op::Compare(CmpOp::Lt), vec![b_p, a_p], out_shape.clone());
                    let ta = t_inputs[0].unwrap_or(zero);
                    let tb = t_inputs[1].unwrap_or(zero);
                    Some(bwd.add_node(Op::Where, vec![cond, ta, tb], out_shape))
                }
                BinaryOp::Pow => {
                    panic!("jvp: rule for Binary(Pow) not implemented yet")
                }
            }
        }

        Op::Activation(kind) => {
            // t_y = act'(x) · t_x — composed from primitives so we
            // don't need a new "ActivationDerivative" op.
            let t_x = t_inputs[0]?;
            let x = fwd_map[&node.inputs[0]];
            let s = node.shape.clone();
            let deriv = match kind {
                Activation::Neg => {
                    // act' ≡ −1 → t_y = −t_x.
                    return Some(bwd.activation(Activation::Neg, t_x, s));
                }
                Activation::Exp => {
                    // act'(x) = exp(x). Since y = exp(x) is already
                    // mirrored in bwd, reuse it (`fwd_map[node.id]`).
                    fwd_map[&node.id]
                }
                Activation::Log => {
                    // act'(x) = 1/x. Build as Div by x.
                    let one = scalar_const(1.0, &s, bwd);
                    bwd.binary(BinaryOp::Div, one, x, s.clone())
                }
                Activation::Sqrt => {
                    // act'(x) = 0.5 / sqrt(x). y = sqrt(x) already in bwd.
                    let half = scalar_const(0.5, &s, bwd);
                    let y = fwd_map[&node.id];
                    bwd.binary(BinaryOp::Div, half, y, s.clone())
                }
                Activation::Rsqrt => {
                    // act'(x) = −0.5 · x^(−3/2) = −0.5 · y³ where y = rsqrt(x).
                    let y = fwd_map[&node.id];
                    let y2 = bwd.binary(BinaryOp::Mul, y, y, s.clone());
                    let y3 = bwd.binary(BinaryOp::Mul, y2, y, s.clone());
                    let neg_half = scalar_const(-0.5, &s, bwd);
                    bwd.binary(BinaryOp::Mul, neg_half, y3, s.clone())
                }
                Activation::Tanh => {
                    // act'(x) = 1 − tanh(x)² = 1 − y².
                    let y = fwd_map[&node.id];
                    let y2 = bwd.binary(BinaryOp::Mul, y, y, s.clone());
                    let one = scalar_const(1.0, &s, bwd);
                    bwd.binary(BinaryOp::Sub, one, y2, s.clone())
                }
                Activation::Sigmoid => {
                    // act'(x) = y · (1 − y).
                    let y = fwd_map[&node.id];
                    let one = scalar_const(1.0, &s, bwd);
                    let one_minus_y = bwd.binary(BinaryOp::Sub, one, y, s.clone());
                    bwd.binary(BinaryOp::Mul, y, one_minus_y, s.clone())
                }
                Activation::Relu => {
                    // act'(x) = step(x). Compose: where(x > 0, t_x, 0).
                    let zero = scalar_const(0.0, &s, bwd);
                    let mask = bwd.add_node(
                        Op::Compare(CmpOp::Gt),
                        vec![x, zero],
                        Shape::from_dims(s.dims(), DType::Bool),
                    );
                    let zero2 = scalar_const(0.0, &s, bwd);
                    return Some(bwd.add_node(Op::Where, vec![mask, t_x, zero2], s));
                }
                Activation::Sin => {
                    // act'(x) = cos(x).
                    bwd.activation(Activation::Cos, x, s.clone())
                }
                Activation::Cos => {
                    // act'(x) = −sin(x).
                    let sx = bwd.activation(Activation::Sin, x, s.clone());
                    bwd.activation(Activation::Neg, sx, s.clone())
                }
                Activation::Tan => {
                    // act'(x) = 1 + tan²(x) = 1 + y²
                    let y = fwd_map[&node.id];
                    let y2 = bwd.binary(BinaryOp::Mul, y, y, s.clone());
                    let one = scalar_const(1.0, &s, bwd);
                    bwd.binary(BinaryOp::Add, one, y2, s.clone())
                }
                Activation::Atan => {
                    // act'(x) = 1 / (1 + x²)
                    let x2 = bwd.binary(BinaryOp::Mul, x, x, s.clone());
                    let one = scalar_const(1.0, &s, bwd);
                    let denom = bwd.binary(BinaryOp::Add, one, x2, s.clone());
                    let one2 = scalar_const(1.0, &s, bwd);
                    bwd.binary(BinaryOp::Div, one2, denom, s.clone())
                }
                Activation::Abs
                | Activation::Round
                | Activation::Gelu
                | Activation::GeluApprox
                | Activation::Silu => {
                    panic!("jvp: rule for Activation({kind:?}) not implemented yet")
                }
            };
            // Default chain rule path: t_y = deriv · t_x.
            Some(bwd.binary(BinaryOp::Mul, deriv, t_x, node.shape.clone()))
        }

        Op::MatMul => {
            // y = a @ b   ⇒   t_y = t_a @ b + a @ t_b
            let a_p = fwd_map[&node.inputs[0]];
            let b_p = fwd_map[&node.inputs[1]];
            let out_shape = node.shape.clone();
            let ta_b = t_inputs[0].map(|ta| bwd.matmul(ta, b_p, out_shape.clone()));
            let a_tb = t_inputs[1].map(|tb| bwd.matmul(a_p, tb, out_shape.clone()));
            match (ta_b, a_tb) {
                (Some(x), Some(y)) => Some(bwd.binary(BinaryOp::Add, x, y, out_shape)),
                (Some(x), None) | (None, Some(x)) => Some(x),
                (None, None) => None,
            }
        }

        Op::DenseSolve => {
            // X = solve(A, B). Differentiate A·X = B:
            //   t_A · X + A · t_X = t_B
            //   ⇒ t_X = solve(A, t_B − t_A · X)
            //
            // Rank-1 (b: [N]) needs reshape-to-column to feed t_A · b
            // through matmul (no vector·matrix op). Rank-2 (B: [N, K])
            // is direct matmul: t_A: [N,N] @ X: [N,K] = [N,K].
            let a_p = fwd_map[&node.inputs[0]];
            let x = fwd_map[&node.id];
            let x_shape = node.shape.clone();
            let dtype = x_shape.dtype();

            // Build the matmul `t_A · X` (or its rank-1 vector cousin).
            let make_ta_x = |t_a: NodeId, bwd: &mut Graph| -> NodeId {
                match x_shape.rank() {
                    1 => {
                        let n = match x_shape.dim(0) {
                            Dim::Static(n) => n,
                            Dim::Dynamic(_) => panic!("jvp: DenseSolve dynamic N not supported"),
                        };
                        let x_col_shape =
                            Shape::from_dims(&[Dim::Static(n), Dim::Static(1)], dtype);
                        let x_col = bwd.add_node(
                            Op::Reshape {
                                new_shape: vec![n as i64, 1],
                            },
                            vec![x],
                            x_col_shape.clone(),
                        );
                        let prod_col = bwd.matmul(t_a, x_col, x_col_shape);
                        bwd.add_node(
                            Op::Reshape {
                                new_shape: vec![n as i64],
                            },
                            vec![prod_col],
                            x_shape.clone(),
                        )
                    }
                    2 => {
                        // Direct matmul: [N, N] @ [N, K] = [N, K].
                        bwd.matmul(t_a, x, x_shape.clone())
                    }
                    r => panic!("jvp: DenseSolve B must be rank 1 or 2, got rank {r}"),
                }
            };

            let rhs = match (t_inputs[0], t_inputs[1]) {
                (Some(t_a), Some(t_b)) => {
                    let prod = make_ta_x(t_a, bwd);
                    bwd.binary(BinaryOp::Sub, t_b, prod, x_shape.clone())
                }
                (Some(t_a), None) => {
                    let prod = make_ta_x(t_a, bwd);
                    bwd.activation(Activation::Neg, prod, x_shape.clone())
                }
                (None, Some(t_b)) => t_b,
                (None, None) => return None,
            };
            Some(bwd.dense_solve(a_p, rhs, x_shape))
        }

        Op::Reshape { new_shape } => {
            let t_x = t_inputs[0]?;
            Some(bwd.add_node(
                Op::Reshape {
                    new_shape: new_shape.clone(),
                },
                vec![t_x],
                node.shape.clone(),
            ))
        }

        Op::Transpose { perm } => {
            let t_x = t_inputs[0]?;
            Some(bwd.add_node(
                Op::Transpose { perm: perm.clone() },
                vec![t_x],
                node.shape.clone(),
            ))
        }

        Op::Expand { target_shape } => {
            let t_x = t_inputs[0]?;
            Some(bwd.add_node(
                Op::Expand {
                    target_shape: target_shape.clone(),
                },
                vec![t_x],
                node.shape.clone(),
            ))
        }

        Op::Narrow { axis, start, len } => {
            let t_x = t_inputs[0]?;
            Some(bwd.add_node(
                Op::Narrow {
                    axis: *axis,
                    start: *start,
                    len: *len,
                },
                vec![t_x],
                node.shape.clone(),
            ))
        }

        // FFT is linear over the 2N-real-block layout: JVP just pushes
        // the tangent through the same op with the same direction.
        // Mirrors the reverse-mode rule (VJP(fft)=ifft, VJP(ifft)=fft)
        // but without the flag flip — forward-mode propagates tangents
        // along the same linear map, not its transpose.
        Op::Fft { inverse } => {
            let t_x = t_inputs[0]?;
            Some(bwd.fft(t_x, *inverse))
        }

        // Complex conjugate is R-linear (not C-linear), but under the
        // JAX-style cotangent convention the JVP and VJP rules coincide
        // in form: tangent of conj(z) is conj(tangent_of_z).
        Op::Conjugate => {
            let t_x = t_inputs[0]?;
            Some(bwd.conjugate(t_x))
        }

        Op::Concat { axis } => {
            // Linear: tangent of concat is concat of tangents. Fill
            // missing-tangent slots with zero constants of the right
            // per-input shape (Concat needs all inputs present).
            if t_inputs.iter().all(Option::is_none) {
                return None;
            }
            let mut t_ins: Vec<NodeId> = Vec::with_capacity(t_inputs.len());
            for (i, t) in t_inputs.iter().enumerate() {
                match t {
                    Some(node_id) => t_ins.push(*node_id),
                    None => {
                        let primal_in = fwd_map[&node.inputs[i]];
                        t_ins.push(zero_like(primal_in, bwd));
                    }
                }
            }
            Some(bwd.add_node(Op::Concat { axis: *axis }, t_ins, node.shape.clone()))
        }

        Op::Reduce { op, axes, keep_dim } => {
            // Linear reductions (Sum, Mean) commute with the tangent.
            let t_x = t_inputs[0]?;
            match op {
                ReduceOp::Sum | ReduceOp::Mean => {
                    Some(bwd.reduce(t_x, *op, axes.clone(), *keep_dim, node.shape.clone()))
                }
                ReduceOp::Min | ReduceOp::Max | ReduceOp::Prod => {
                    panic!("jvp: rule for Reduce::{op:?} not implemented yet")
                }
            }
        }

        Op::Where => {
            // y = where(cond, a, b). Cond is non-differentiable;
            // tangent flows through the chosen branch.
            let cond_p = fwd_map[&node.inputs[0]];
            let s = node.shape.clone();
            match (t_inputs[1], t_inputs[2]) {
                (Some(ta), Some(tb)) => Some(bwd.add_node(Op::Where, vec![cond_p, ta, tb], s)),
                (Some(ta), None) => {
                    let zero = zero_like(ta, bwd);
                    Some(bwd.add_node(Op::Where, vec![cond_p, ta, zero], s))
                }
                (None, Some(tb)) => {
                    let zero = zero_like(tb, bwd);
                    Some(bwd.add_node(Op::Where, vec![cond_p, zero, tb], s))
                }
                (None, None) => None,
            }
        }

        Op::Compare(_) => None, // discrete output, zero tangent
        Op::Cast { to } => {
            let t_x = t_inputs[0]?;
            Some(bwd.add_node(Op::Cast { to: *to }, vec![t_x], node.shape.clone()))
        }

        // User-defined sub-graph (Op::CustomFn) with override JVP. The
        // jvp_body's primal Inputs map to the outer node's primals (via
        // `fwd_map`); each "tangent_i" Input maps to the corresponding
        // entry in `t_inputs` (zero tangent → use a fresh zero
        // Constant). Returns the body's single output as the new
        // tangent NodeId.
        Op::CustomFn {
            jvp_body: Some(jvp_body),
            num_inputs,
            ..
        } => {
            let mut sub_to_bwd: HashMap<NodeId, NodeId> = HashMap::new();

            for sub_node in jvp_body.nodes() {
                let new_id = match &sub_node.op {
                    Op::Input { name } if name == "primal_output" => fwd_map[&node.id],
                    Op::Input { name } if name.starts_with("tangent_") => {
                        let idx: usize = name["tangent_".len()..].parse().expect(
                            "custom_fn jvp_body: tangent name must be \
                                     'tangent_<i>' where i is a usize",
                        );
                        assert!(idx < *num_inputs as usize);
                        match t_inputs[idx] {
                            Some(t) => t,
                            None => {
                                // Zero tangent for this input.
                                scalar_const(0.0, &sub_node.shape, bwd)
                            }
                        }
                    }
                    Op::Input { .. } => {
                        // Primal input — match by NodeId order against
                        // node.inputs (excluding the special-named slots).
                        let mut primal_input_ids: Vec<NodeId> = jvp_body
                            .nodes()
                            .iter()
                            .filter_map(|n| match &n.op {
                                Op::Input { name }
                                    if !name.starts_with("tangent_") && name != "primal_output" =>
                                {
                                    Some(n.id)
                                }
                                _ => None,
                            })
                            .collect();
                        primal_input_ids.sort();
                        let idx = primal_input_ids
                            .iter()
                            .position(|&id| id == sub_node.id)
                            .expect("custom_fn jvp_body: primal Input not found");
                        fwd_map[&node.inputs[idx]]
                    }
                    _ => {
                        let new_inputs: Vec<NodeId> =
                            sub_node.inputs.iter().map(|i| sub_to_bwd[i]).collect();
                        bwd.add_node(sub_node.op.clone(), new_inputs, sub_node.shape.clone())
                    }
                };
                sub_to_bwd.insert(sub_node.id, new_id);
            }

            Some(sub_to_bwd[&jvp_body.outputs[0]])
        }

        // CustomFn without a jvp_body: not yet supported (would
        // require inlining fwd_body and recursively differentiating).
        Op::CustomFn { jvp_body: None, .. } => {
            panic!(
                "jvp: Op::CustomFn has no jvp_body. Either supply \
                    one to Graph::custom_fn(...), or inline the forward \
                    body into the parent graph before differentiating."
            )
        }

        Op::Custom { name, .. } => {
            // Dispatch through the OpExtension registry. The op's
            // `jvp` method receives the per-input tangents and emits
            // the tangent subgraph itself.
            let ext = rlx_ir::lookup_op(name)
                .unwrap_or_else(|| panic!("jvp: Op::Custom('{name}') not registered"));
            let mut ctx = rlx_ir::JvpContext {
                tangents: t_inputs,
                fwd_map,
                bwd,
            };
            ext.jvp(node, &mut ctx)
        }

        other => panic!("jvp: no rule for op {other:?}"),
    }
}

/// Build a scalar constant matching `shape`'s dtype, broadcastable
/// (shape `[1]`) so it can participate in element-wise ops with the
/// chain rule. Supports F32 and F64 — the dtypes the IR and CPU
/// backend currently exercise. Other dtypes panic — add when needed.
fn scalar_const(value: f64, shape: &Shape, bwd: &mut Graph) -> NodeId {
    let bytes = match shape.dtype() {
        DType::F32 => (value as f32).to_le_bytes().to_vec(),
        DType::F64 => value.to_le_bytes().to_vec(),
        other => panic!("scalar_const: dtype {other:?} not supported"),
    };
    let scalar_shape = Shape::from_dims(&[Dim::Static(1)], shape.dtype());
    bwd.add_node(Op::Constant { data: bytes }, vec![], scalar_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// JVP of x = solve(A, b) wrt b only. Forward output is a vector.
    /// Should produce a graph that solves A · t_x = t_b.
    #[test]
    fn jvp_dense_solve_b_only() {
        let mut g = Graph::new("jvp_db");
        let a = g.input("A", Shape::new(&[2, 2], DType::F64));
        let b = g.input("b", Shape::new(&[2], DType::F64));
        let x = g.dense_solve(a, b, Shape::new(&[2], DType::F64));
        g.set_outputs(vec![x]);

        let jg = jvp(&g, &[b]);
        // Outputs: [primal_x, tangent_x]
        assert_eq!(jg.outputs.len(), 2);
        // Tangent path: must contain at least one DenseSolve other
        // than the forward mirror (1 forward + 1 tangent ≥ 2 total).
        let n_solves = jg
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::DenseSolve))
            .count();
        assert!(
            n_solves >= 2,
            "tangent path should add a DenseSolve, got\n{jg}"
        );
    }

    /// JVP of x = solve(A, b) wrt A. Tangent must build t_A · x and
    /// negate it before the second solve.
    #[test]
    fn jvp_dense_solve_a_only() {
        let mut g = Graph::new("jvp_da");
        let a = g.input("A", Shape::new(&[2, 2], DType::F64));
        let b = g.input("b", Shape::new(&[2], DType::F64));
        let x = g.dense_solve(a, b, Shape::new(&[2], DType::F64));
        g.set_outputs(vec![x]);

        let jg = jvp(&g, &[a]);
        let n_solves = jg
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::DenseSolve))
            .count();
        let n_neg = jg
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Activation(Activation::Neg)))
            .count();
        let n_mm = jg
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::MatMul))
            .count();
        assert!(n_solves >= 2, "expected ≥2 DenseSolve, got\n{jg}");
        assert!(n_neg >= 1, "expected a Neg for −t_A·x, got\n{jg}");
        assert!(n_mm >= 1, "expected a MatMul for t_A·x, got\n{jg}");
    }

    /// Identity check: with t_b = 0 and t_A = 0, the tangent output
    /// should be zero (or a constant zero node).
    #[test]
    fn jvp_with_no_seeded_tangents_produces_zero_output() {
        let mut g = Graph::new("jvp_no_seed");
        let a = g.input("A", Shape::new(&[2, 2], DType::F64));
        let b = g.input("b", Shape::new(&[2], DType::F64));
        let x = g.dense_solve(a, b, Shape::new(&[2], DType::F64));
        g.set_outputs(vec![x]);

        let jg = jvp(&g, &[]); // no seeds — full derivative is zero
        assert_eq!(jg.outputs.len(), 2);
        // Tangent output must be a Constant (the zero we synthesize).
        let tangent_out = jg.node(jg.outputs[1]);
        assert!(
            matches!(tangent_out.op, Op::Constant { .. }),
            "expected zero Constant for tangent, got {:?}",
            tangent_out.op
        );
    }
}
