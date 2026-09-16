// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Primitive composition for [`Op::GatedDeltaNetBackward`].
//!
//! Three backends have the fused backward kernel (CPU, Metal, MLX); eight run
//! [`Op::GatedDeltaNet`] forward. Without this the other five — CUDA, ROCm,
//! wgpu, TPU, CoreML — hit `backend doesn't claim support` the moment a
//! gated-delta-net block is differentiated, and the only way out was setting
//! `RLX_GDN_UNFUSE_FOR_AD=1` before the gradient was taken. That is a global
//! env flag deciding a per-backend question, and it has to be set *upstream* of
//! AD, so a caller who reaches the error has already built the wrong graph.
//!
//! The decomposition does not re-derive the reverse scan by hand. It rebuilds
//! the forward from the backward node's own inputs, unrolls it, and
//! differentiates *that* — so it is the unrolled path, reconstructed at the
//! backend boundary instead of chosen globally beforehand. Two consequences
//! worth stating:
//!
//! * It agrees with the fused kernel by construction, not by coincidence —
//!   the unrolled path is what
//!   `rlx-autodiff/tests/gated_delta_net_fused_backward.rs` already checks the
//!   kernel against (worst 4.5e-8), and is itself finite-difference-verified.
//! * It is the slow path, deliberately. Unrolling costs ~32× because every
//!   timestep materializes a fresh `[B·H, N, N]` state in SSA form. A backend
//!   that wants the 32× writes a kernel and lists the op in `supported_ops`,
//!   at which point `preserved` keeps this composition out of the way.

use std::collections::HashMap;

use rlx_ir::{Graph, NodeId, Op, Shape};

use crate::autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};

/// Leaf names for the scratch forward, in `Op::GatedDeltaNet` input order —
/// which is also the order [`rlx_ir::GdnBackwardLayout::slices`] packs.
const FWD_LEAVES: [&str; 6] = ["q", "k", "v", "g", "beta", "state"];

/// The cotangent seed `grad_with_loss_wrt` adds to every backward graph.
const D_OUTPUT: &str = "d_output";

/// Emit the packed gradient bundle of a `GatedDeltaNetBackward` node into `out`
/// as primitives. `inputs` are already mapped into `out`'s id space and follow
/// the op's contract: `[q, k, v, g, beta]`, `+ [state]` when `carry_state`,
/// then `dy` last.
pub fn compose_gated_delta_net_backward(
    out: &mut Graph,
    inputs: &[NodeId],
    state_size: usize,
    carry_state: bool,
    gate_per_channel: bool,
    packed_shape: &Shape,
) -> NodeId {
    let n_fwd = if carry_state { 6 } else { 5 };
    assert_eq!(
        inputs.len(),
        n_fwd + 1,
        "GatedDeltaNetBackward expects [q, k, v, g, beta{}, dy]",
        if carry_state { ", state" } else { "" }
    );
    let dy = inputs[n_fwd];

    // 1. Rebuild the forward on a scratch graph — one `Op::Input` per operand,
    //    named so step 3 can designate them after prepare renumbers everything.
    let mut fwd = Graph::new("gdn_backward_decompose");
    let y_shape = out.node(inputs[0]).shape.clone();
    let leaves: Vec<NodeId> = (0..n_fwd)
        .map(|i| fwd.input(FWD_LEAVES[i], out.node(inputs[i]).shape.clone()))
        .collect();
    let y = fwd.add_node(
        Op::GatedDeltaNet {
            state_size,
            carry_state,
            gate_per_channel,
        },
        leaves,
        y_shape,
    );
    fwd.set_outputs(vec![y]);

    // 2. Unroll the time loop unconditionally. Routing this through
    //    `unfuse_fused_for_autodiff` would be a no-op under the default flag —
    //    it keeps GDN fused precisely so the fused kernel gets used — and the
    //    node would then re-emit the op being decomposed.
    let fwd = rlx_fusion::unfuse_gated_delta_net_always(fwd);

    // 3. Differentiate. `outputs[0]` is `y` rather than a scalar loss, so
    //    `d_output` takes `y`'s shape and the result is a plain VJP seeded by
    //    `dy`. Leaf *names* are the only designators that survive
    //    `prepare_graph_for_ad`, hence `Wrt::Leaf`.
    let wrt: Vec<Wrt> = FWD_LEAVES[..n_fwd]
        .iter()
        .map(|n| Wrt::Leaf((*n).to_string()))
        .collect();
    let bwd = grad_with_loss_wrt(&fwd, &wrt, GradWithLossOptions::STRICT.with_aux(false));
    // Its VJP rules can emit first-order `*Backward` ops of their own
    // (RmsNorm/Activation in the unrolled body); fold those before splicing so
    // the caller's fixpoint doesn't have to re-walk what we just inlined.
    let bwd = crate::decompose_backward::decompose_backward_ops(bwd);

    // 4. Splice into `out`, wiring leaves BY NAME. Position-based wiring
    //    (`inline_subgraph_into`) would silently depend on where AD happened to
    //    place `d_output` among the inputs.
    let mut leaf_for: HashMap<&str, NodeId> = HashMap::new();
    for (i, name) in FWD_LEAVES[..n_fwd].iter().enumerate() {
        leaf_for.insert(name, inputs[i]);
    }
    leaf_for.insert(D_OUTPUT, dy);

    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in bwd.nodes() {
        let new_id = match &node.op {
            Op::Input { name } => *leaf_for.get(name.as_str()).unwrap_or_else(|| {
                panic!(
                    "GatedDeltaNetBackward decompose: gradient graph has an \
                     unexpected leaf {name:?} — expected only {FWD_LEAVES:?} \
                     and {D_OUTPUT:?}"
                )
            }),
            _ => {
                let ins: Vec<NodeId> = node.inputs.iter().map(|i| map[i]).collect();
                out.add_node(node.op.clone(), ins, node.shape.clone())
            }
        };
        map.insert(node.id, new_id);
    }

    // 5. Pack. `bwd.outputs` is `[y, dq, dk, dv, dg, dbeta, (dstate)]` — aux
    //    mirroring is off and the forward has one output, so everything past
    //    index 0 is a gradient, in `wrt` order == packing order.
    let dtype = packed_shape.dtype();
    let flat: Vec<NodeId> = bwd.outputs[1..]
        .iter()
        .map(|&gid| {
            let src = map[&gid];
            let n = out
                .node(src)
                .shape
                .num_elements()
                .expect("GatedDeltaNetBackward decompose: gradient must be statically shaped");
            out.reshape(src, vec![n as i64], Shape::new(&[n], dtype))
        })
        .collect();
    debug_assert_eq!(
        flat.len(),
        n_fwd,
        "one gradient per forward input must reach the packing step"
    );
    out.add_node(Op::Concat { axis: 0 }, flat, packed_shape.clone())
}
