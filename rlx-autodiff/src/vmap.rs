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

//! Batched function transformation (vmap).
//!
//! Lifts a graph that operates on shape `[*]` to operate on shape
//! `[B, *]`, threading a leading batch axis through every op. Mirror
//! of JAX's `vmap` with MVP constraints:
//!
//! * Leading-axis batching only — `in_axes` is a list of input names
//!   to batch on axis 0; everything else is shared across the batch.
//! * Outputs always land with the batch axis at 0.
//! * Per-op rules cover the elementwise / shape / reduce / matmul
//!   subset. Ops without a rule panic, mirroring the autodiff pass's
//!   policy of "no silent miscompute."
//!
//! ## Use case
//!
//! Parameter sweeps: build a graph parameterised by a small input
//! vector, `vmap` over the batched parameter values to evaluate every
//! variant in one shot, take a gradient w.r.t. the parameter vector.
//! Pairs naturally with `Op::BatchedDenseSolve` for batched implicit
//! solves.

use rlx_ir::shape::Dim;
use rlx_ir::*;
use std::collections::{HashMap, HashSet};

/// Vectorize `forward` over a leading batch axis.
///
/// `batched_input_names` lists the `Op::Input` names whose leading
/// axis is the batch axis after vmap. Inputs/Params not in the list
/// are shared across the batch (they get broadcast on demand by ops
/// that consume them alongside batched values).
///
/// The returned graph:
/// * Has the same input names as `forward`. Batched inputs gain a
///   leading `[batch_size, ...]` dim.
/// * Has the same output count. Every output gains a leading batch
///   axis (out_axes = 0 implicit).
/// * Has the same set of `Op::Param` slots — params are always shared.
///
/// # Panics
/// Panics on any op without a vmap rule. Add rules incrementally.
pub fn vmap(forward: &Graph, batched_input_names: &[&str], batch_size: usize) -> Graph {
    let batched_set: HashSet<&str> = batched_input_names.iter().copied().collect();
    let mut out = Graph::new(format!("{}_vmap", forward.name));
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    // Set of node IDs (in the OUTPUT graph) that carry a leading batch
    // axis. `lift_to_batched` reads this to decide whether a value
    // needs broadcasting before being combined with a batched value.
    let mut batched: HashSet<NodeId> = HashSet::new();

    for node in forward.nodes() {
        let new_id = match &node.op {
            Op::Input { name } => {
                if batched_set.contains(name.as_str()) {
                    let mut dims: Vec<Dim> = vec![Dim::Static(batch_size)];
                    dims.extend(node.shape.dims().iter().copied());
                    let s = Shape::from_dims(&dims, node.shape.dtype());
                    let id = out.input(name.clone(), s);
                    batched.insert(id);
                    id
                } else {
                    out.input(name.clone(), node.shape.clone())
                }
            }
            Op::Param { name } => {
                // Params are always shared in the MVP. Convert to
                // Input if you need batched params.
                out.param(name.clone(), node.shape.clone())
            }
            Op::Constant { data } => out.add_node(
                Op::Constant { data: data.clone() },
                vec![],
                node.shape.clone(),
            ),
            _ => {
                let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                let any_batched = new_inputs.iter().any(|i| batched.contains(i));
                if !any_batched {
                    // No batched input reaches this node — the original
                    // op shape applies and the node is shared.
                    out.add_node(node.op.clone(), new_inputs, node.shape.clone())
                } else {
                    let id = vmap_op(node, &new_inputs, &mut out, &mut batched, batch_size);
                    batched.insert(id);
                    id
                }
            }
        };
        id_map.insert(node.id, new_id);
    }

    let new_outputs: Vec<NodeId> = forward.outputs.iter().map(|o| id_map[o]).collect();
    out.set_outputs(new_outputs);
    out
}

/// Apply the per-op vmap rule. At least one input is batched.
fn vmap_op(
    node: &Node,
    new_inputs: &[NodeId],
    out: &mut Graph,
    batched: &mut HashSet<NodeId>,
    batch_size: usize,
) -> NodeId {
    let orig_shape = &node.shape;
    let dtype = orig_shape.dtype();

    // Output shape with leading batch axis.
    let batched_shape = || -> Shape {
        let mut dims: Vec<Dim> = vec![Dim::Static(batch_size)];
        dims.extend(orig_shape.dims().iter().copied());
        Shape::from_dims(&dims, dtype)
    };

    match &node.op {
        // ── Pure elementwise — broadcast unbatched inputs, apply op ──
        Op::Binary(_) | Op::Activation(_) | Op::Where | Op::Compare(_) | Op::Cast { .. } => {
            let lifted: Vec<NodeId> = new_inputs
                .iter()
                .map(|&id| lift_to_batched(out, id, batched, batch_size))
                .collect();
            for &id in &lifted {
                batched.insert(id);
            }
            out.add_node(node.op.clone(), lifted, batched_shape())
        }

        // ── Reshape: prepend batch dim ──
        Op::Reshape { new_shape } => {
            let lifted = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(lifted);
            let mut bsh: Vec<i64> = vec![batch_size as i64];
            bsh.extend(new_shape.iter().copied());
            out.add_node(
                Op::Reshape { new_shape: bsh },
                vec![lifted],
                batched_shape(),
            )
        }

        // ── Transpose: shift perm by 1, prepend 0 ──
        Op::Transpose { perm } => {
            let lifted = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(lifted);
            let mut new_perm: Vec<usize> = vec![0];
            new_perm.extend(perm.iter().map(|p| p + 1));
            out.add_node(
                Op::Transpose { perm: new_perm },
                vec![lifted],
                batched_shape(),
            )
        }

        // ── Expand: prepend batch dim to target_shape ──
        Op::Expand { target_shape } => {
            let lifted = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(lifted);
            let mut bsh: Vec<i64> = vec![batch_size as i64];
            bsh.extend(target_shape.iter().copied());
            out.add_node(
                Op::Expand { target_shape: bsh },
                vec![lifted],
                batched_shape(),
            )
        }

        // ── Reduce: shift axes by 1 (don't reduce batch axis) ──
        Op::Reduce { op, axes, keep_dim } => {
            let lifted = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(lifted);
            let new_axes: Vec<usize> = axes.iter().map(|a| a + 1).collect();
            out.add_node(
                Op::Reduce {
                    op: *op,
                    axes: new_axes,
                    keep_dim: *keep_dim,
                },
                vec![lifted],
                batched_shape(),
            )
        }

        // ── MatMul: rely on built-in batch broadcasting ──
        // Per Op::MatMul docs: "Batch dimensions are broadcast." So
        // [B, M, K] @ [B, K, N] → [B, M, N], and [B, M, K] @ [K, N]
        // also works via broadcasting.
        Op::MatMul => {
            let a = lift_to_batched(out, new_inputs[0], batched, batch_size);
            let b = lift_to_batched(out, new_inputs[1], batched, batch_size);
            batched.insert(a);
            batched.insert(b);
            out.matmul(a, b, batched_shape())
        }

        // ── DenseSolve: emit BatchedDenseSolve ──
        // A becomes [B, N, N], b becomes [B, N] or [B, N, K].
        Op::DenseSolve => {
            let a = lift_to_batched(out, new_inputs[0], batched, batch_size);
            let b = lift_to_batched(out, new_inputs[1], batched, batch_size);
            batched.insert(a);
            batched.insert(b);
            out.batched_dense_solve(a, b, batched_shape())
        }

        // ── Scan: recursively vmap the body ──
        //
        // Forward Op::Scan iterates `length` times, with carry shape
        // `*carry` and per-step xs shape `*per_step_i`. After vmap:
        //   * init becomes `[B, *carry]`
        //   * each xs_i becomes `[B, length, *per_step_i]`; we
        //     transpose to `[length, B, *per_step_i]` so the Scan
        //     reads per-step slices of shape `[B, *per_step_i]`.
        //   * the body is recursively vmap'd — its inputs gain a
        //     leading B and its computations become batched.
        //   * the inner Scan output is `[B, *carry]` (final-only)
        //     or `[length, B, *carry]` (trajectory). For trajectory,
        //     we add a final transpose to put batch at axis 0:
        //     `[B, length, *carry]`.
        Op::Scan {
            body,
            length,
            save_trajectory,
            num_xs,
            num_checkpoints: _,
            num_bcast,
        } => {
            // Lift init to [B, *carry].
            let init_b = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(init_b);

            // Bcasts are lifted to [B, *bcast]; the body sees them
            // un-transposed (no per-step axis). Same handling as the
            // carry — each iteration the body reads the lifted slot.
            let mut bcasts_b: Vec<NodeId> = Vec::with_capacity(*num_bcast as usize);
            for i in 0..*num_bcast as usize {
                let bcast_in = new_inputs[1 + i];
                let lifted = lift_to_batched(out, bcast_in, batched, batch_size);
                batched.insert(lifted);
                bcasts_b.push(lifted);
            }

            // For each xs: lift to [B, length, *per_step], then
            // transpose first two axes → [length, B, *per_step].
            let xs_base = 1 + *num_bcast as usize;
            let mut xs_t: Vec<NodeId> = Vec::with_capacity(*num_xs as usize);
            for i in 0..*num_xs as usize {
                let xs_in = new_inputs[xs_base + i];
                let lifted = lift_to_batched(out, xs_in, batched, batch_size);
                batched.insert(lifted);
                let xs_shape = out.node(lifted).shape.clone();
                let r = xs_shape.rank();
                let mut perm: Vec<usize> = vec![1, 0];
                for k in 2..r {
                    perm.push(k);
                }
                let mut new_dims: Vec<Dim> = xs_shape.dims().to_vec();
                new_dims.swap(0, 1);
                let new_shape = Shape::from_dims(&new_dims, xs_shape.dtype());
                let transposed = out.add_node(Op::Transpose { perm }, vec![lifted], new_shape);
                batched.insert(transposed);
                xs_t.push(transposed);
            }

            // Recursively vmap the body. All body inputs are batched
            // — carry comes in as `[B, *carry]`, each x_t comes in as
            // `[B, *per_step_i]`. Collect names and dispatch.
            let body_input_names_owned: Vec<String> = body
                .nodes()
                .iter()
                .filter_map(|n| match &n.op {
                    Op::Input { name } => Some(name.clone()),
                    _ => None,
                })
                .collect();
            let body_input_names: Vec<&str> =
                body_input_names_owned.iter().map(|s| s.as_str()).collect();
            let body_b = vmap(body, &body_input_names, batch_size);

            // Compute the inner Scan's natural output shape.
            //   final-only:   [B, *carry]
            //   trajectory:   [length, B, *carry]
            let dtype = orig_shape.dtype();
            // `orig_shape` was either `*carry` (final-only) or
            // `[length, *carry]` (trajectory).
            let inner_out_shape: Shape = if *save_trajectory {
                // Original was [length, *carry]; new is [length, B, *carry].
                let mut dims: Vec<Dim> = vec![orig_shape.dim(0)];
                dims.push(Dim::Static(batch_size));
                for i in 1..orig_shape.rank() {
                    dims.push(orig_shape.dim(i));
                }
                Shape::from_dims(&dims, dtype)
            } else {
                // Original was *carry; new is [B, *carry].
                let mut dims: Vec<Dim> = vec![Dim::Static(batch_size)];
                for i in 0..orig_shape.rank() {
                    dims.push(orig_shape.dim(i));
                }
                Shape::from_dims(&dims, dtype)
            };

            // Build inputs: init + lifted bcasts + transposed xs.
            let mut inner_inputs = vec![init_b];
            inner_inputs.extend_from_slice(&bcasts_b);
            inner_inputs.extend_from_slice(&xs_t);

            let inner_id = out.add_node(
                Op::Scan {
                    body: Box::new(body_b),
                    length: *length,
                    save_trajectory: *save_trajectory,
                    num_xs: *num_xs,
                    num_checkpoints: 0,
                    num_bcast: *num_bcast,
                },
                inner_inputs,
                inner_out_shape,
            );

            if *save_trajectory {
                // Trajectory: transpose [length, B, *carry] → [B, length, *carry].
                let r = orig_shape.rank() + 1; // includes leading length + B
                let mut perm: Vec<usize> = vec![1, 0];
                for k in 2..r {
                    perm.push(k);
                }
                out.add_node(Op::Transpose { perm }, vec![inner_id], batched_shape())
            } else {
                inner_id
            }
        }

        // ── Narrow: shift axis by 1 ──
        Op::Narrow { axis, start, len } => {
            let lifted = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(lifted);
            out.add_node(
                Op::Narrow {
                    axis: axis + 1,
                    start: *start,
                    len: *len,
                },
                vec![lifted],
                batched_shape(),
            )
        }

        // ── Concat: shift axis by 1 ──
        Op::Concat { axis } => {
            let lifted: Vec<NodeId> = new_inputs
                .iter()
                .map(|&id| lift_to_batched(out, id, batched, batch_size))
                .collect();
            for &id in &lifted {
                batched.insert(id);
            }
            out.add_node(Op::Concat { axis: axis + 1 }, lifted, batched_shape())
        }

        // ── Softmax: shift axis by 1 ──
        Op::Softmax { axis } => {
            let lifted = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(lifted);
            out.add_node(
                Op::Softmax { axis: *axis + 1 },
                vec![lifted],
                batched_shape(),
            )
        }

        // ── Cumsum: shift axis by 1 ──
        Op::Cumsum { axis, exclusive } => {
            let lifted = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(lifted);
            out.add_node(
                Op::Cumsum {
                    axis: *axis + 1,
                    exclusive: *exclusive,
                },
                vec![lifted],
                batched_shape(),
            )
        }

        // ── LayerNorm: shift axis by 1 ──
        // Inputs: [x, gamma, beta]. gamma/beta apply on the feature
        // axis only — they stay shared across batch (lift_to_batched
        // is a no-op if they're already batched, otherwise they're
        // broadcast on demand by the kernel).
        Op::LayerNorm { axis, eps } => {
            let x = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(x);
            out.add_node(
                Op::LayerNorm {
                    axis: *axis + 1,
                    eps: *eps,
                },
                vec![x, new_inputs[1], new_inputs[2]],
                batched_shape(),
            )
        }

        // ── RmsNorm: shift axis by 1 ──
        Op::RmsNorm { axis, eps } => {
            let x = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(x);
            out.add_node(
                Op::RmsNorm {
                    axis: *axis + 1,
                    eps: *eps,
                },
                vec![x, new_inputs[1], new_inputs[2]],
                batched_shape(),
            )
        }

        // ── Gather: shift axis by 1; both table and indices lift ──
        // table[indices] selects along the original axis, with B
        // prepended both inputs index per-batch.
        Op::Gather { axis } => {
            let table = lift_to_batched(out, new_inputs[0], batched, batch_size);
            let indices = lift_to_batched(out, new_inputs[1], batched, batch_size);
            batched.insert(table);
            batched.insert(indices);
            out.add_node(
                Op::Gather { axis: axis + 1 },
                vec![table, indices],
                batched_shape(),
            )
        }

        // ── ScatterAdd: lift updates and indices; output gains B axis ──
        // Forward: output[indices[i]] += updates[i]. After vmap each
        // batch's scatter is independent; the existing kernel iterates
        // a flat updates list, so as long as updates and indices are
        // batched on axis 0 and the output's leading dim is B, the
        // executor handles per-batch slicing via the scatter indices.
        Op::ScatterAdd => {
            let updates = lift_to_batched(out, new_inputs[0], batched, batch_size);
            let indices = lift_to_batched(out, new_inputs[1], batched, batch_size);
            batched.insert(updates);
            batched.insert(indices);
            out.add_node(Op::ScatterAdd, vec![updates, indices], batched_shape())
        }

        // ── ElementwiseRegion: same policy as plain elementwise ──
        // The chain operates on shape `[*]` per element; lifting all
        // inputs to `[B, *]` and letting the chain run with the wider
        // shape gives the right per-batch result. The fused kernel's
        // `input_modulus` machinery already handles broadcast inputs
        // — but for true unbatched-into-batched broadcast we'd need
        // to update those moduli. For MVP: lift everything to
        // batched (so all inputs share `[B, *]`), keep the chain.
        Op::ElementwiseRegion { .. } => {
            let lifted: Vec<NodeId> = new_inputs
                .iter()
                .map(|&id| lift_to_batched(out, id, batched, batch_size))
                .collect();
            for &id in &lifted {
                batched.insert(id);
            }
            out.add_node(node.op.clone(), lifted, batched_shape())
        }

        // ── DotGeneral: shift contracting + batch dim indices by 1 ──
        Op::DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            rhs_batch,
        } => {
            let lhs = lift_to_batched(out, new_inputs[0], batched, batch_size);
            let rhs = lift_to_batched(out, new_inputs[1], batched, batch_size);
            batched.insert(lhs);
            batched.insert(rhs);
            // Every dim index shifts by 1; axis 0 (the new batch axis)
            // joins lhs_batch and rhs_batch since both operands are
            // now batched on it.
            let mut new_lhs_b: Vec<usize> = vec![0];
            new_lhs_b.extend(lhs_batch.iter().map(|i| i + 1));
            let mut new_rhs_b: Vec<usize> = vec![0];
            new_rhs_b.extend(rhs_batch.iter().map(|i| i + 1));
            out.add_node(
                Op::DotGeneral {
                    lhs_contracting: lhs_contracting.iter().map(|i| i + 1).collect(),
                    rhs_contracting: rhs_contracting.iter().map(|i| i + 1).collect(),
                    lhs_batch: new_lhs_b,
                    rhs_batch: new_rhs_b,
                },
                vec![lhs, rhs],
                batched_shape(),
            )
        }

        // ── Backward ops emitted by autodiff: same shape lift as
        // elementwise. ReluBackward and ActivationBackward read
        // (x, dy) and write dx — same shape across all three. Lift
        // and keep the op kind unchanged.
        Op::ReluBackward | Op::ActivationBackward { .. } => {
            let lifted: Vec<NodeId> = new_inputs
                .iter()
                .map(|&id| lift_to_batched(out, id, batched, batch_size))
                .collect();
            for &id in &lifted {
                batched.insert(id);
            }
            out.add_node(node.op.clone(), lifted, batched_shape())
        }

        // ── ScanBackward: recursive AD-loop vmap ──
        // Same shape-juggling as Op::Scan's vmap rule, plus the
        // extra `upstream` input and a body_vjp instead of a body.
        Op::ScanBackward {
            body_vjp,
            length,
            save_trajectory,
            num_xs,
            num_checkpoints: _,
            forward_body: _,
        } => {
            // init [B, *carry]
            let init_b = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(init_b);

            // trajectory after lift is [B, length, *carry]; ScanBackward's
            // executor reads it row-by-row indexed by t along axis 0,
            // so transpose to [length, B, *carry].
            let traj_lifted = lift_to_batched(out, new_inputs[1], batched, batch_size);
            batched.insert(traj_lifted);
            let traj_t = transpose_swap_01(out, traj_lifted);
            batched.insert(traj_t);

            // upstream layout depends on save_trajectory:
            //   save_trajectory=true:  same shape as trajectory →
            //     transpose [B, length, *carry] → [length, B, *carry]
            //   save_trajectory=false: [B, *carry] (carry shape; no
            //     length axis) → no transpose, but lift if needed.
            let up_lifted = lift_to_batched(out, new_inputs[2], batched, batch_size);
            batched.insert(up_lifted);
            let up_t = if *save_trajectory {
                let id = transpose_swap_01(out, up_lifted);
                batched.insert(id);
                id
            } else {
                up_lifted
            };

            // Per-xs: lift to [B, length, *per_step], transpose to [length, B, *per_step].
            let mut xs_t: Vec<NodeId> = Vec::with_capacity(*num_xs as usize);
            for i in 0..*num_xs as usize {
                let xs_in = new_inputs[3 + i];
                let lifted = lift_to_batched(out, xs_in, batched, batch_size);
                batched.insert(lifted);
                let t = transpose_swap_01(out, lifted);
                batched.insert(t);
                xs_t.push(t);
            }

            // Recursively vmap body_vjp. All its Op::Input nodes are
            // marked batched (carry, every x_t_i, AND "d_output").
            let body_input_names_owned: Vec<String> = body_vjp
                .nodes()
                .iter()
                .filter_map(|n| match &n.op {
                    Op::Input { name } => Some(name.clone()),
                    _ => None,
                })
                .collect();
            let body_input_names: Vec<&str> =
                body_input_names_owned.iter().map(|s| s.as_str()).collect();
            let body_vjp_b = vmap(body_vjp, &body_input_names, batch_size);

            // dinit shape: [B, *carry] (orig_shape was *carry).
            let mut dinit_dims: Vec<Dim> = vec![Dim::Static(batch_size)];
            for i in 0..orig_shape.rank() {
                dinit_dims.push(orig_shape.dim(i));
            }
            let dinit_shape = Shape::from_dims(&dinit_dims, dtype);

            let mut inner_inputs = vec![init_b, traj_t, up_t];
            inner_inputs.extend_from_slice(&xs_t);

            out.scan_backward(
                init_b,
                traj_t,
                up_t,
                &xs_t,
                body_vjp_b,
                *length,
                *save_trajectory,
                dinit_shape,
            )
        }

        // ── ScanBackwardXs: like ScanBackward but output is per-step
        // dxs_i. Inner output is [length, B, *per_step]; transpose
        // back to [B, length, *per_step] so batch ends up at axis 0.
        Op::ScanBackwardXs {
            body_vjp,
            length,
            save_trajectory,
            num_xs,
            xs_idx,
            num_checkpoints: _,
            forward_body: _,
        } => {
            let init_b = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(init_b);
            let traj_lifted = lift_to_batched(out, new_inputs[1], batched, batch_size);
            batched.insert(traj_lifted);
            let traj_t = transpose_swap_01(out, traj_lifted);
            batched.insert(traj_t);
            let up_lifted = lift_to_batched(out, new_inputs[2], batched, batch_size);
            batched.insert(up_lifted);
            let up_t = if *save_trajectory {
                let id = transpose_swap_01(out, up_lifted);
                batched.insert(id);
                id
            } else {
                up_lifted
            };

            let mut xs_t: Vec<NodeId> = Vec::with_capacity(*num_xs as usize);
            for i in 0..*num_xs as usize {
                let xs_in = new_inputs[3 + i];
                let lifted = lift_to_batched(out, xs_in, batched, batch_size);
                batched.insert(lifted);
                let t = transpose_swap_01(out, lifted);
                batched.insert(t);
                xs_t.push(t);
            }

            let body_input_names_owned: Vec<String> = body_vjp
                .nodes()
                .iter()
                .filter_map(|n| match &n.op {
                    Op::Input { name } => Some(name.clone()),
                    _ => None,
                })
                .collect();
            let body_input_names: Vec<&str> =
                body_input_names_owned.iter().map(|s| s.as_str()).collect();
            let body_vjp_b = vmap(body_vjp, &body_input_names, batch_size);

            // Inner output natural shape is [length, B, *per_step]
            // (orig_shape is [length, *per_step]).
            let mut inner_dims: Vec<Dim> = vec![orig_shape.dim(0)];
            inner_dims.push(Dim::Static(batch_size));
            for i in 1..orig_shape.rank() {
                inner_dims.push(orig_shape.dim(i));
            }
            let inner_shape = Shape::from_dims(&inner_dims, dtype);

            let inner_id = out.scan_backward_xs(
                init_b,
                traj_t,
                up_t,
                &xs_t,
                body_vjp_b,
                *length,
                *save_trajectory,
                *xs_idx,
                inner_shape,
            );

            // Final transpose [length, B, *per_step] → [B, length, *per_step].
            transpose_swap_01(out, inner_id)
        }

        // ── Quantize / Dequantize: per-channel; chan_axis +1 if Some ──
        Op::Quantize {
            axis,
            scales,
            zero_points,
        } => {
            let lifted = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(lifted);
            let new_axis = axis.map(|a| a + 1);
            out.add_node(
                Op::Quantize {
                    axis: new_axis,
                    scales: scales.clone(),
                    zero_points: zero_points.clone(),
                },
                vec![lifted],
                batched_shape(),
            )
        }
        Op::Dequantize {
            axis,
            scales,
            zero_points,
        } => {
            let lifted = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(lifted);
            let new_axis = axis.map(|a| a + 1);
            out.add_node(
                Op::Dequantize {
                    axis: new_axis,
                    scales: scales.clone(),
                    zero_points: zero_points.clone(),
                },
                vec![lifted],
                batched_shape(),
            )
        }
        Op::FakeQuantize {
            bits,
            axis,
            ste,
            scale_mode,
        } => {
            let lifted: Vec<NodeId> = new_inputs
                .iter()
                .map(|&id| lift_to_batched(out, id, batched, batch_size))
                .collect();
            for &id in &lifted {
                batched.insert(id);
            }
            let new_axis = axis.map(|a| a + 1);
            out.add_node(
                Op::FakeQuantize {
                    bits: *bits,
                    axis: new_axis,
                    ste: *ste,
                    scale_mode: *scale_mode,
                },
                lifted,
                batched_shape(),
            )
        }
        Op::FakeQuantizeBackward { bits, axis, ste } => {
            let lifted: Vec<NodeId> = new_inputs
                .iter()
                .map(|&id| lift_to_batched(out, id, batched, batch_size))
                .collect();
            for &id in &lifted {
                batched.insert(id);
            }
            let new_axis = axis.map(|a| a + 1);
            out.add_node(
                Op::FakeQuantizeBackward {
                    bits: *bits,
                    axis: new_axis,
                    ste: *ste,
                },
                lifted,
                batched_shape(),
            )
        }
        Op::FakeQuantizeLSQ { bits, axis } => {
            let lifted: Vec<NodeId> = new_inputs
                .iter()
                .map(|&id| lift_to_batched(out, id, batched, batch_size))
                .collect();
            for &id in &lifted {
                batched.insert(id);
            }
            out.add_node(
                Op::FakeQuantizeLSQ {
                    bits: *bits,
                    axis: axis.map(|a| a + 1),
                },
                lifted,
                batched_shape(),
            )
        }
        Op::FakeQuantizeLSQBackwardX { bits, axis } => {
            let lifted: Vec<NodeId> = new_inputs
                .iter()
                .map(|&id| lift_to_batched(out, id, batched, batch_size))
                .collect();
            for &id in &lifted {
                batched.insert(id);
            }
            out.add_node(
                Op::FakeQuantizeLSQBackwardX {
                    bits: *bits,
                    axis: axis.map(|a| a + 1),
                },
                lifted,
                batched_shape(),
            )
        }
        Op::FakeQuantizeLSQBackwardScale { bits, axis } => {
            let lifted: Vec<NodeId> = new_inputs
                .iter()
                .map(|&id| lift_to_batched(out, id, batched, batch_size))
                .collect();
            for &id in &lifted {
                batched.insert(id);
            }
            out.add_node(
                Op::FakeQuantizeLSQBackwardScale {
                    bits: *bits,
                    axis: axis.map(|a| a + 1),
                },
                lifted,
                batched_shape(),
            )
        }

        // ── LayerNorm/RmsNorm backward: axis +1, lift inputs ──
        Op::LayerNormBackwardInput { axis, eps } => {
            let lifted: Vec<NodeId> = new_inputs
                .iter()
                .map(|&id| lift_to_batched(out, id, batched, batch_size))
                .collect();
            for &id in &lifted {
                batched.insert(id);
            }
            out.add_node(
                Op::LayerNormBackwardInput {
                    axis: axis + 1,
                    eps: *eps,
                },
                lifted,
                batched_shape(),
            )
        }
        Op::LayerNormBackwardGamma { axis, eps } => {
            let lifted: Vec<NodeId> = new_inputs
                .iter()
                .map(|&id| lift_to_batched(out, id, batched, batch_size))
                .collect();
            for &id in &lifted {
                batched.insert(id);
            }
            out.add_node(
                Op::LayerNormBackwardGamma {
                    axis: axis + 1,
                    eps: *eps,
                },
                lifted,
                batched_shape(),
            )
        }

        // ── TopK / Sample: operate on the last axis (logits). After
        // vmap they still operate on the last axis — just lift inputs.
        Op::TopK { k } => {
            let lifted = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(lifted);
            out.add_node(Op::TopK { k: *k }, vec![lifted], batched_shape())
        }
        Op::Sample {
            top_k,
            top_p,
            temperature,
            seed,
        } => {
            let lifted = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(lifted);
            out.add_node(
                Op::Sample {
                    top_k: *top_k,
                    top_p: *top_p,
                    temperature: *temperature,
                    seed: *seed,
                },
                vec![lifted],
                batched_shape(),
            )
        }

        // ── LoraMatMul: lift x/w/a/b, output [B, *] ──
        Op::LoraMatMul { scale } => {
            let lifted: Vec<NodeId> = new_inputs
                .iter()
                .map(|&id| lift_to_batched(out, id, batched, batch_size))
                .collect();
            for &id in &lifted {
                batched.insert(id);
            }
            out.add_node(Op::LoraMatMul { scale: *scale }, lifted, batched_shape())
        }

        // ── Conv / Pool / Attention / Rope: reshape-trick ──
        // The kernel expects a specific input rank. We fold the new
        // batch axis into the existing leading axis (N for Conv/Pool,
        // batch for Attention) via Reshape, run the op, then reshape
        // back to expose the vmap batch axis.
        Op::Conv {
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        } => {
            let x = lift_to_batched(out, new_inputs[0], batched, batch_size);
            let w = new_inputs[1]; // weights stay shared
            batched.insert(x);
            // x is [B, N, C_in, H, W]. Flatten B*N → 4-D, run conv,
            // reshape back.
            let x_shape = out.node(x).shape.clone();
            let r = x_shape.rank();
            assert!(r == 5, "vmap Conv: expected 5-D after lift, got {r}");
            let n_orig = match x_shape.dim(1) {
                Dim::Static(n) => n,
                _ => panic!("dynamic N"),
            };
            let bn = batch_size * n_orig;
            let inner_dims_static: Vec<i64> = (2..r)
                .map(|i| match x_shape.dim(i) {
                    Dim::Static(d) => d as i64,
                    _ => -1,
                })
                .collect();
            let mut flat_dims = vec![bn as i64];
            flat_dims.extend(inner_dims_static.iter().copied());
            let mut flat_dim_objs = vec![Dim::Static(bn)];
            for i in 2..r {
                flat_dim_objs.push(x_shape.dim(i));
            }
            let flat_shape = Shape::from_dims(&flat_dim_objs, x_shape.dtype());
            let x_flat = out.add_node(
                Op::Reshape {
                    new_shape: flat_dims,
                },
                vec![x],
                flat_shape,
            );
            // Conv output: [B*N, C_out, H_out, W_out] in flat form.
            let mut conv_out_dims = vec![Dim::Static(bn)];
            for i in 1..orig_shape.rank() {
                conv_out_dims.push(orig_shape.dim(i));
            }
            let conv_out_shape = Shape::from_dims(&conv_out_dims, dtype);
            let conv_out = out.add_node(
                Op::Conv {
                    kernel_size: kernel_size.clone(),
                    stride: stride.clone(),
                    padding: padding.clone(),
                    dilation: dilation.clone(),
                    groups: *groups,
                },
                vec![x_flat, w],
                conv_out_shape,
            );
            // Reshape back to [B, N, C_out, H_out, W_out].
            let mut final_dims_static: Vec<i64> = vec![batch_size as i64];
            for i in 0..orig_shape.rank() {
                final_dims_static.push(match orig_shape.dim(i) {
                    Dim::Static(d) => d as i64,
                    _ => -1,
                });
            }
            out.add_node(
                Op::Reshape {
                    new_shape: final_dims_static,
                },
                vec![conv_out],
                batched_shape(),
            )
        }
        Op::Pool {
            kind,
            kernel_size,
            stride,
            padding,
        } => {
            let x = lift_to_batched(out, new_inputs[0], batched, batch_size);
            batched.insert(x);
            let x_shape = out.node(x).shape.clone();
            let r = x_shape.rank();
            assert!(r == 5, "vmap Pool: expected 5-D after lift, got {r}");
            let n_orig = match x_shape.dim(1) {
                Dim::Static(n) => n,
                _ => panic!("dynamic N"),
            };
            let bn = batch_size * n_orig;
            let mut flat_dims = vec![bn as i64];
            for i in 2..r {
                flat_dims.push(match x_shape.dim(i) {
                    Dim::Static(d) => d as i64,
                    _ => -1,
                });
            }
            let mut flat_dim_objs = vec![Dim::Static(bn)];
            for i in 2..r {
                flat_dim_objs.push(x_shape.dim(i));
            }
            let flat_shape = Shape::from_dims(&flat_dim_objs, x_shape.dtype());
            let x_flat = out.add_node(
                Op::Reshape {
                    new_shape: flat_dims,
                },
                vec![x],
                flat_shape,
            );
            let mut pool_dims = vec![Dim::Static(bn)];
            for i in 1..orig_shape.rank() {
                pool_dims.push(orig_shape.dim(i));
            }
            let pool_out_shape = Shape::from_dims(&pool_dims, dtype);
            let pool_out = out.add_node(
                Op::Pool {
                    kind: *kind,
                    kernel_size: kernel_size.clone(),
                    stride: stride.clone(),
                    padding: padding.clone(),
                },
                vec![x_flat],
                pool_out_shape,
            );
            let mut final_dims_static: Vec<i64> = vec![batch_size as i64];
            for i in 0..orig_shape.rank() {
                final_dims_static.push(match orig_shape.dim(i) {
                    Dim::Static(d) => d as i64,
                    _ => -1,
                });
            }
            out.add_node(
                Op::Reshape {
                    new_shape: final_dims_static,
                },
                vec![pool_out],
                batched_shape(),
            )
        }

        // ── Ops with hard kernel-shape requirements that need real
        // engineering before vmap can support them. Panic with a
        // pointer to the right follow-up rather than silently lifting
        // and producing wrong shapes.
        Op::Attention { .. }
        | Op::FusedAttentionBlock { .. }
        | Op::FusedTransformerLayer { .. }
        | Op::Rope { .. } => panic!(
            "vmap: {:?} kernels expect a fixed input rank — extra batch \
             axis would need either decomposition (use rlx-opt unfuse \
             passes first) or a kernel rewrite. Skipped in MVP.",
            node.op,
        ),

        // ── Conv2dBackwardInput: reshape-trick around the kernel ──
        // Inputs: [dy, w]. dy is [N, C_out, H_out, W_out]; w is
        // [C_out, C_in/g, kH, kW]. Output: [N, C_in, H, W].
        // After vmap with N batched: dy [B, N, C_out, H_out, W_out],
        // weights stay shared. Fold B into N, run Conv2dBackwardInput,
        // fold back.
        Op::Conv2dBackwardInput {
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
        } => {
            let dy = lift_to_batched(out, new_inputs[0], batched, batch_size);
            let w = new_inputs[1];
            batched.insert(dy);
            let dy_shape = out.node(dy).shape.clone();
            assert_eq!(
                dy_shape.rank(),
                5,
                "vmap Conv2dBackwardInput: expected 5-D dy"
            );
            let n_orig = match dy_shape.dim(1) {
                Dim::Static(n) => n,
                _ => panic!("dynamic N"),
            };
            let bn = batch_size * n_orig;
            let mut flat_dims_static: Vec<i64> = vec![bn as i64];
            for i in 2..dy_shape.rank() {
                flat_dims_static.push(match dy_shape.dim(i) {
                    Dim::Static(d) => d as i64,
                    _ => -1,
                });
            }
            let mut flat_dim_objs = vec![Dim::Static(bn)];
            for i in 2..dy_shape.rank() {
                flat_dim_objs.push(dy_shape.dim(i));
            }
            let dy_flat = out.add_node(
                Op::Reshape {
                    new_shape: flat_dims_static,
                },
                vec![dy],
                Shape::from_dims(&flat_dim_objs, dy_shape.dtype()),
            );
            // Output flat shape: [B*N, C_in, H, W].
            let mut out_flat_dim_objs = vec![Dim::Static(bn)];
            for i in 1..orig_shape.rank() {
                out_flat_dim_objs.push(orig_shape.dim(i));
            }
            let out_flat_shape = Shape::from_dims(&out_flat_dim_objs, dtype);
            let out_flat = out.add_node(
                Op::Conv2dBackwardInput {
                    kernel_size: kernel_size.clone(),
                    stride: stride.clone(),
                    padding: padding.clone(),
                    dilation: dilation.clone(),
                    groups: *groups,
                },
                vec![dy_flat, w],
                out_flat_shape,
            );
            // Reshape back to [B, N, C_in, H, W].
            let mut final_dims: Vec<i64> = vec![batch_size as i64];
            for i in 0..orig_shape.rank() {
                final_dims.push(match orig_shape.dim(i) {
                    Dim::Static(d) => d as i64,
                    _ => -1,
                });
            }
            out.add_node(
                Op::Reshape {
                    new_shape: final_dims,
                },
                vec![out_flat],
                batched_shape(),
            )
        }

        // ── MaxPool2dBackward: reshape-trick like Conv ──
        // Inputs: [x, dy]. Both 4-D NCHW. After vmap: [B, N, C, H, W].
        // Fold B into N for both, run, fold back.
        Op::MaxPool2dBackward {
            kernel_size,
            stride,
            padding,
        } => {
            let x = lift_to_batched(out, new_inputs[0], batched, batch_size);
            let dy = lift_to_batched(out, new_inputs[1], batched, batch_size);
            batched.insert(x);
            batched.insert(dy);
            let x_shape = out.node(x).shape.clone();
            assert_eq!(x_shape.rank(), 5, "vmap MaxPool2dBackward: expected 5-D x");
            let n_orig = match x_shape.dim(1) {
                Dim::Static(n) => n,
                _ => panic!("dynamic N"),
            };
            let bn = batch_size * n_orig;
            let flatten = |out: &mut Graph, id: NodeId| -> NodeId {
                let s = out.node(id).shape.clone();
                let mut flat_objs = vec![Dim::Static(bn)];
                for i in 2..s.rank() {
                    flat_objs.push(s.dim(i));
                }
                let flat_shape = Shape::from_dims(&flat_objs, s.dtype());
                let mut flat_static: Vec<i64> = vec![bn as i64];
                for i in 2..s.rank() {
                    flat_static.push(match s.dim(i) {
                        Dim::Static(d) => d as i64,
                        _ => -1,
                    });
                }
                out.add_node(
                    Op::Reshape {
                        new_shape: flat_static,
                    },
                    vec![id],
                    flat_shape,
                )
            };
            let x_flat = flatten(out, x);
            let dy_flat = flatten(out, dy);
            let mut out_flat_objs = vec![Dim::Static(bn)];
            for i in 1..orig_shape.rank() {
                out_flat_objs.push(orig_shape.dim(i));
            }
            let out_flat_shape = Shape::from_dims(&out_flat_objs, dtype);
            let pool_out = out.add_node(
                Op::MaxPool2dBackward {
                    kernel_size: kernel_size.clone(),
                    stride: stride.clone(),
                    padding: padding.clone(),
                },
                vec![x_flat, dy_flat],
                out_flat_shape,
            );
            let mut final_dims: Vec<i64> = vec![batch_size as i64];
            for i in 0..orig_shape.rank() {
                final_dims.push(match orig_shape.dim(i) {
                    Dim::Static(d) => d as i64,
                    _ => -1,
                });
            }
            out.add_node(
                Op::Reshape {
                    new_shape: final_dims,
                },
                vec![pool_out],
                batched_shape(),
            )
        }

        Op::Conv2dBackwardWeight { .. } => panic!(
            "vmap: Conv2dBackwardWeight: weight gradient is summed across \
             samples — vmap-batching gives a B-stack of independent dWs. \
             Reshape-trick doesn't apply since the output isn't naturally \
             N-leading. Add a per-batch dW pattern when needed.",
        ),

        Op::SelectiveScan { .. }
        | Op::GroupedMatMul
        | Op::QMatMul { .. }
        | Op::QConv2d { .. }
        | Op::DequantMatMul { .. } => panic!(
            "vmap: {:?} has its own internal batch handling; \
             the right rule depends on whether the user wants \
             nested batching or to fold into the existing batch \
             dim. Add a rule when a real workload demands it.",
            node.op,
        ),

        // ── DequantGroupedMatMul: shared expert weights, batched tokens ──
        Op::DequantGroupedMatMul { scheme } => {
            let x = lift_to_batched(out, new_inputs[0], batched, batch_size);
            let idx = lift_to_batched(out, new_inputs[2], batched, batch_size);
            let w = new_inputs[1];
            batched.insert(x);
            batched.insert(idx);
            let x_shape = out.node(x).shape.clone();
            assert_eq!(x_shape.rank(), 3, "vmap DequantGroupedMatMul: expected 3-D x");
            let m_orig = match x_shape.dim(1) {
                Dim::Static(v) => v,
                _ => panic!("dynamic M"),
            };
            let k = match x_shape.dim(2) {
                Dim::Static(v) => v as i64,
                _ => -1,
            };
            let bm = batch_size * m_orig;
            let n = match orig_shape.dim(orig_shape.rank() - 1) {
                Dim::Static(v) => v as i64,
                _ => -1,
            };
            let x_flat = out.add_node(
                Op::Reshape {
                    new_shape: vec![bm as i64, k],
                },
                vec![x],
                Shape::from_dims(
                    &[Dim::Static(bm), x_shape.dim(2)],
                    orig_shape.dtype(),
                ),
            );
            let idx_flat = out.add_node(
                Op::Reshape {
                    new_shape: vec![bm as i64],
                },
                vec![idx],
                Shape::from_dims(&[Dim::Static(bm)], orig_shape.dtype()),
            );
            let y_flat = out.add_node(
                Op::DequantGroupedMatMul { scheme: *scheme },
                vec![x_flat, w, idx_flat],
                Shape::from_dims(
                    &[Dim::Static(bm), orig_shape.dim(orig_shape.rank() - 1)],
                    orig_shape.dtype(),
                ),
            );
            let mut final_dims: Vec<i64> = vec![batch_size as i64, m_orig as i64];
            final_dims.push(n);
            out.add_node(
                Op::Reshape { new_shape: final_dims },
                vec![y_flat],
                batched_shape(),
            )
        }

        Op::DequantMoEWeights { .. } => panic!(
            "vmap: DequantMoEWeights is a weight materialization helper; \
             vmap the downstream GroupedMatMul / DequantGroupedMatMul instead.",
        ),

        Op::FusedSwiGLU { .. } | Op::FusedMatMulBiasAct { .. } | Op::FusedResidualLN { .. }
        | Op::FusedResidualRmsNorm { .. } => {
            panic!(
                "vmap: {:?} is fused — decompose first via \
             `rlx_fusion::UnfuseElementwiseRegions` (or \
             `rlx_fusion::unfuse_fused_for_autodiff`) so the simpler \
             ops get vmap'd individually.",
                node.op,
            )
        }

        Op::SoftmaxCrossEntropyWithLogits | Op::SoftmaxCrossEntropyBackward => panic!(
            "vmap: SoftmaxCrossEntropy* expect 2-D logits; lifting to \
             3-D would need a kernel change. Workaround: reshape \
             logits to 2-D before the op and back after.",
        ),

        Op::Custom { name, .. } => {
            // Dispatch through the OpExtension registry. The op's
            // `vmap` impl receives the already-lifted inputs and
            // returns the lifted output. Default impl returns None,
            // which we surface as a clear panic.
            let ext = rlx_ir::lookup_op(name)
                .unwrap_or_else(|| panic!("vmap: Op::Custom('{name}') not registered"));
            let is_batched: Vec<bool> = new_inputs.iter().map(|i| batched.contains(i)).collect();
            let mut ctx = rlx_ir::VmapContext {
                lifted_inputs: new_inputs,
                is_batched: &is_batched,
                batch_size,
                out,
            };
            match ext.vmap(node, &mut ctx) {
                Some(id) => id,
                None => panic!(
                    "vmap: Op::Custom('{name}') has no vmap rule registered. \
                     Override `OpExtension::vmap` on the impl to add one."
                ),
            }
        }

        // CustomFn: recursively vmap each body (fwd / vjp / jvp). All
        // Inputs in each body are treated as batched — primals become
        // [B, *primal] (matching the lifted outer inputs), and the
        // AD-special-named Inputs ("primal_output", "d_output",
        // "tangent_*") are likewise batched since the outer graph
        // wires them to batched producers post-vmap.
        Op::CustomFn {
            fwd_body,
            vjp_body,
            jvp_body,
            num_inputs,
        } => {
            // Lift each primal input to [B, *primal].
            let mut lifted_inputs: Vec<NodeId> = Vec::with_capacity(*num_inputs as usize);
            for &raw in new_inputs.iter() {
                let lifted = lift_to_batched(out, raw, batched, batch_size);
                batched.insert(lifted);
                lifted_inputs.push(lifted);
            }

            let vmap_body = |body: &Graph| -> Graph {
                let names_owned: Vec<String> = body
                    .nodes()
                    .iter()
                    .filter_map(|n| match &n.op {
                        Op::Input { name } => Some(name.clone()),
                        _ => None,
                    })
                    .collect();
                let names: Vec<&str> = names_owned.iter().map(|s| s.as_str()).collect();
                vmap(body, &names, batch_size)
            };

            let fwd_b = vmap_body(fwd_body);
            let vjp_b = vjp_body.as_ref().map(|g| vmap_body(g));
            let jvp_b = jvp_body.as_ref().map(|g| vmap_body(g));

            // Output shape: [B, *orig_output].
            let mut out_dims: Vec<Dim> = vec![Dim::Static(batch_size)];
            for i in 0..orig_shape.rank() {
                out_dims.push(orig_shape.dim(i));
            }
            let out_shape = Shape::from_dims(&out_dims, orig_shape.dtype());

            let id = out.add_node(
                Op::CustomFn {
                    fwd_body: Box::new(fwd_b),
                    vjp_body: vjp_b.map(Box::new),
                    jvp_body: jvp_b.map(Box::new),
                    num_inputs: *num_inputs,
                },
                lifted_inputs,
                out_shape,
            );
            batched.insert(id);
            id
        }

        other => panic!(
            "vmap: no rule for op {:?}. Add a per-op rule in vmap.rs.",
            other,
        ),
    }
}

/// Swap the first two axes of a tensor (perm = [1, 0, 2, 3, ...]).
/// Used by the Scan / ScanBackward / ScanBackwardXs vmap rules to
/// move the batch axis between the natural-after-vmap leading
/// position and the position the inner Scan-family op expects
/// (`length` first, batch second per row).
fn transpose_swap_01(out: &mut Graph, id: NodeId) -> NodeId {
    let s = out.node(id).shape.clone();
    let r = s.rank();
    debug_assert!(r >= 2, "transpose_swap_01 needs rank ≥ 2");
    let mut perm: Vec<usize> = vec![1, 0];
    for i in 2..r {
        perm.push(i);
    }
    let mut new_dims: Vec<Dim> = s.dims().to_vec();
    new_dims.swap(0, 1);
    let new_shape = Shape::from_dims(&new_dims, s.dtype());
    out.add_node(Op::Transpose { perm }, vec![id], new_shape)
}

/// Make sure `id` carries a leading batch axis. If it already does,
/// return it unchanged. Otherwise emit `Reshape([1, *])` followed by
/// `Expand([B, *])`.
fn lift_to_batched(
    out: &mut Graph,
    id: NodeId,
    batched: &HashSet<NodeId>,
    batch_size: usize,
) -> NodeId {
    if batched.contains(&id) {
        return id;
    }
    let orig_shape = out.node(id).shape.clone();
    let dtype = orig_shape.dtype();

    // Reshape [orig...] → [1, orig...].
    let mut dims_with_1: Vec<Dim> = vec![Dim::Static(1)];
    dims_with_1.extend(orig_shape.dims().iter().copied());
    let with1_shape = Shape::from_dims(&dims_with_1, dtype);
    let reshape_dims: Vec<i64> = dims_with_1
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n as i64,
            Dim::Dynamic(_) => -1,
        })
        .collect();
    let with1 = out.add_node(
        Op::Reshape {
            new_shape: reshape_dims,
        },
        vec![id],
        with1_shape,
    );

    // Expand [1, orig...] → [B, orig...].
    let mut target_dims: Vec<i64> = vec![batch_size as i64];
    for d in orig_shape.dims().iter() {
        target_dims.push(match d {
            Dim::Static(n) => *n as i64,
            Dim::Dynamic(_) => -1,
        });
    }
    let mut target_shape_dims: Vec<Dim> = vec![Dim::Static(batch_size)];
    target_shape_dims.extend(orig_shape.dims().iter().copied());
    let target_shape = Shape::from_dims(&target_shape_dims, dtype);
    out.add_node(
        Op::Expand {
            target_shape: target_dims,
        },
        vec![with1],
        target_shape,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::{BinaryOp, ReduceOp};

    /// Smallest possible vmap: elementwise scaling. f(x) = 2·x.
    /// vmap(f) over batch=4 should produce a graph with:
    ///   Input "x" : [4, 3] f64
    ///   Constant 2: [3] f64 (shared, lifted via Reshape+Expand at use site)
    ///   Mul        : [4, 3] f64
    /// Asserts the structural shape and that the batched output node
    /// is recorded as such.
    #[test]
    fn vmap_elementwise_scaling_lifts_to_batched_shape() {
        let n = 3usize;
        let batch = 4usize;
        let mut g = Graph::new("scale");
        let x = g.input("x", Shape::new(&[n], DType::F64));
        let two_bytes: Vec<u8> = (0..n).flat_map(|_| 2.0_f64.to_le_bytes()).collect();
        let two = g.add_node(
            Op::Constant { data: two_bytes },
            vec![],
            Shape::new(&[n], DType::F64),
        );
        let y = g.binary(BinaryOp::Mul, x, two, Shape::new(&[n], DType::F64));
        g.set_outputs(vec![y]);

        let bg = vmap(&g, &["x"], batch);
        // Output should be [batch, n].
        let out_id = bg.outputs[0];
        let out_shape = &bg.node(out_id).shape;
        assert_eq!(out_shape.dims().len(), 2);
        assert_eq!(out_shape.dim(0), Dim::Static(batch));
        assert_eq!(out_shape.dim(1), Dim::Static(n));
    }

    /// vmap of a matmul: f(x) = MatMul(x, w). x is `[m, k]`, batched
    /// to `[B, m, k]`. w stays `[k, n]`. Output: `[B, m, n]`. Built-in
    /// MatMul batch broadcasting handles it; vmap just lifts x.
    #[test]
    fn vmap_matmul_with_shared_weight() {
        let m = 2usize;
        let k = 3usize;
        let n = 4usize;
        let batch = 5usize;
        let mut g = Graph::new("mm");
        let x = g.input("x", Shape::new(&[m, k], DType::F64));
        let w = g.input("w", Shape::new(&[k, n], DType::F64));
        let y = g.matmul(x, w, Shape::new(&[m, n], DType::F64));
        g.set_outputs(vec![y]);

        let bg = vmap(&g, &["x"], batch);
        let out_id = bg.outputs[0];
        let out_shape = &bg.node(out_id).shape;
        assert_eq!(out_shape.dims().len(), 3);
        assert_eq!(out_shape.dim(0), Dim::Static(batch));
        assert_eq!(out_shape.dim(1), Dim::Static(m));
        assert_eq!(out_shape.dim(2), Dim::Static(n));
    }

    /// basic test for the recently-added rules: build a graph that
    /// exercises Gather, ElementwiseRegion fallback, ReluBackward,
    /// and ActivationBackward — vmap and assert it completes without
    /// hitting the catch-all panic.
    #[test]
    fn vmap_extended_op_set_lifts_without_panic() {
        // Gather pattern: table[indices] with table batched.
        let mut g = Graph::new("gather_check");
        let table = g.input("table", Shape::new(&[5, 4], DType::F32));
        let idx = g.input("idx", Shape::new(&[3], DType::F32));
        let out_node = g.add_node(
            Op::Gather { axis: 0 },
            vec![table, idx],
            Shape::new(&[3, 4], DType::F32),
        );
        g.set_outputs(vec![out_node]);
        let bg = vmap(&g, &["table"], 2);
        // Output: [B, 3, 4].
        let s = &bg.node(bg.outputs[0]).shape;
        assert_eq!(s.rank(), 3);
        assert_eq!(s.dim(0), Dim::Static(2));

        // ReluBackward check — inputs (x, dy), output dx.
        let mut g = Graph::new("relu_bwd_check");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let dy = g.input("dy", Shape::new(&[4], DType::F32));
        let dx = g.add_node(Op::ReluBackward, vec![x, dy], Shape::new(&[4], DType::F32));
        g.set_outputs(vec![dx]);
        let bg = vmap(&g, &["x"], 3);
        let s = &bg.node(bg.outputs[0]).shape;
        assert_eq!(s.rank(), 2);
        assert_eq!(s.dim(0), Dim::Static(3));
    }

    /// vmap composition test: f(x) = sum(x · w + b) → loss per-batch.
    /// Asserts the output is `[batch]` (sum over axis 1 of [batch, n]).
    #[test]
    fn vmap_combined_matmul_add_reduce() {
        let n = 3usize;
        let batch = 4usize;
        let mut g = Graph::new("combined");
        let x = g.input("x", Shape::new(&[n], DType::F64));
        let w = g.input("w", Shape::new(&[n, n], DType::F64));
        let b = g.input("b", Shape::new(&[n], DType::F64));
        // Reshape x to [1, n] so MatMul works on [1, n] @ [n, n] = [1, n]
        let x_row = g.add_node(
            Op::Reshape {
                new_shape: vec![1, n as i64],
            },
            vec![x],
            Shape::new(&[1, n], DType::F64),
        );
        let mm = g.matmul(x_row, w, Shape::new(&[1, n], DType::F64));
        let mm_flat = g.add_node(
            Op::Reshape {
                new_shape: vec![n as i64],
            },
            vec![mm],
            Shape::new(&[n], DType::F64),
        );
        let yv = g.binary(BinaryOp::Add, mm_flat, b, Shape::new(&[n], DType::F64));
        let loss = g.reduce(
            yv,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F64),
        );
        g.set_outputs(vec![loss]);

        let bg = vmap(&g, &["x"], batch);
        let out = bg.node(bg.outputs[0]);
        // After Reduce::Sum on shifted axis (1), keep_dim=false → shape [B, 1].
        // (Reduce shifts axis 0 → 1; the original [1] output becomes [B, 1].)
        assert_eq!(out.shape.dim(0), Dim::Static(batch));
        assert_eq!(out.shape.rank(), 2);
    }
}
