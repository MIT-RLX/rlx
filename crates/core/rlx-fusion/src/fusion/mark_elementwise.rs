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

//! `mark_elementwise` — extracted from the `fusion` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

// ── Helper: graph rewriter ──────────────────────────────────────────────

use crate::graph_rewrite::Rewriter;

// ── Pass 1: MatMul + Bias + Activation → FusedMatMulBiasAct ─────────────

use super::*;

pub struct MarkElementwiseRegions;


impl Pass for MarkElementwiseRegions {
    fn name(&self) -> &str {
        "mark_elementwise_regions"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Tally consumer counts for every node id.
        let mut consumers: HashMap<NodeId, usize> = HashMap::new();
        for node in graph.nodes() {
            for &input in &node.inputs {
                *consumers.entry(input).or_insert(0) += 1;
            }
        }
        for &out in &graph.outputs {
            *consumers.entry(out).or_insert(0) += 1;
        }

        // Predicate: does this op qualify for chain inclusion?
        let chain_eligible = |op: &Op| -> bool {
            matches!(
                op,
                Op::Activation(_) | Op::Cast { .. } | Op::Binary(_) | Op::Compare(_) | Op::Where
            )
        };

        // Per-node refinement: a `Cast { to }` only qualifies when the
        // destination dtype matches the operand's dtype. The chain
        // kernel runs entirely in f32 register scratch and writes the
        // tail back to the output node's arena slot — which is sized
        // for the tail dtype. A cross-dtype Cast inside the chain would
        // lose precision (no actual conversion happens in scratch) AND
        // mis-size the final write (an F16 output slot is half the
        // bytes of f32). Same-dtype Casts are trivially propagated.
        let chain_step_safe = |graph: &Graph, node: &rlx_ir::Node| -> bool {
            match &node.op {
                Op::Cast { to } => {
                    let in_dt = graph.shape(node.inputs[0]).dtype();
                    *to == in_dt
                }
                _ => true,
            }
        };

        // For each node, compute which "chain root" it belongs to.
        // A chain consists of a sequence of single-consumer chain-eligible
        // nodes leading to a chain "tail" (last node before a multi-consumer
        // or non-eligible boundary). We assign each node a `region_id`
        // (= the tail's NodeId) iff it's part of a region with ≥2 ops.
        // Walk in topological (forward) order; for each chain-eligible
        // node whose every input is either non-region OR a single-consumer
        // region member, extend its parent chain.
        let mut region_of: HashMap<NodeId, NodeId> = HashMap::new();
        let mut chain_step_idx: HashMap<NodeId, u32> = HashMap::new();

        for node in graph.nodes() {
            if !chain_eligible(&node.op) {
                continue;
            }
            if !chain_step_safe(&graph, node) {
                continue;
            }
            // Each input must either match the output element count
            // exactly OR be a trailing-shape broadcast (its element
            // count divides the output's). The kernel reads
            // `arena[input_offs[i] + (gid % input_modulus[i])]` for
            // broadcast inputs; non-broadcast inputs leave the modulus
            // at 0 to skip the modulo.
            let out_shape = &node.shape;
            let out_elems = out_shape.num_elements();
            let shape_ok = node.inputs.iter().all(|id| {
                let in_elems = graph.shape(*id).num_elements();
                match (in_elems, out_elems) {
                    (Some(i), Some(o)) if i == o => true,
                    (Some(i), Some(o)) if i > 0 && o % i == 0 => true,
                    _ => false,
                }
            });
            if !shape_ok {
                continue;
            }
            // A chain extends an input's chain when the input is itself
            // chain-eligible AND has exactly one consumer (= this node).
            // If multiple inputs satisfy this, the chains must be the same
            // (= they share a chain root); pick that root.
            let mut parent_root: Option<NodeId> = None;
            let mut all_inputs_single_consumer = true;
            for &input in &node.inputs {
                // BLAS / splat render ops are explicit fusion boundaries.
                if graph.node(input).op.is_fusion_boundary() {
                    parent_root = None;
                    all_inputs_single_consumer = false;
                    break;
                }
                if let Some(&root) = region_of.get(&input) {
                    if consumers.get(&input).copied() != Some(1) {
                        all_inputs_single_consumer = false;
                        break;
                    }
                    match parent_root {
                        None => parent_root = Some(root),
                        Some(r) if r == root => {}
                        Some(_) => {
                            parent_root = None;
                            all_inputs_single_consumer = false;
                            break;
                        }
                    }
                }
            }
            if !all_inputs_single_consumer {
                // Start a fresh chain rooted at this node.
                region_of.insert(node.id, node.id);
                chain_step_idx.insert(node.id, 0);
                continue;
            }
            let root = parent_root.unwrap_or(node.id);
            // step idx = max(parents' idx in same chain) + 1
            let next_idx = node
                .inputs
                .iter()
                .filter_map(|id| {
                    if region_of.get(id) == Some(&root) {
                        chain_step_idx.get(id).copied()
                    } else {
                        None
                    }
                })
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);
            let limits = crate::limits::active_fusion_limits();
            if next_idx >= limits.max_elementwise_steps {
                region_of.insert(node.id, node.id);
                chain_step_idx.insert(node.id, 0);
                continue;
            }
            region_of.insert(node.id, root);
            chain_step_idx.insert(node.id, next_idx);
        }

        // Group nodes by region_id; only regions with ≥2 nodes are worth fusing.
        // The "region tail" (= last node) becomes the new ElementwiseRegion node.
        let mut by_region: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for node in graph.nodes() {
            if let Some(&root) = region_of.get(&node.id) {
                by_region.entry(root).or_default().push(node.id);
            }
        }

        // Each region's "tail" is the node with the highest chain_step_idx.
        // For correctness, the tail must be the only node in the region with
        // a non-region or multi-consumer outflow — otherwise the region would
        // span past it. Skip regions where the tail isn't unique (= chain
        // forks internally).
        let mut tail_of_region: HashMap<NodeId, NodeId> = HashMap::new();
        for (root, members) in &by_region {
            if members.len() < 2 {
                continue;
            }
            let max_idx = members.iter().map(|id| chain_step_idx[id]).max().unwrap();
            let tails: Vec<_> = members
                .iter()
                .filter(|id| chain_step_idx[id] == max_idx)
                .collect();
            if tails.len() != 1 {
                continue;
            }
            tail_of_region.insert(*root, *tails[0]);
        }

        // Drop "regions" that aren't worth fusing (size < 2 or non-unique tail).
        let by_region: HashMap<NodeId, Vec<NodeId>> = by_region
            .into_iter()
            .filter(|(root, _)| tail_of_region.contains_key(root))
            .collect();

        if by_region.is_empty() {
            return graph;
        }

        // Rewrite the graph: copy non-region nodes verbatim; for each region,
        // emit a single ElementwiseRegion at the tail's position (in topo order)
        // and replace each region member's NodeId in the id map with that.
        let mut rw = Rewriter::new(&graph.name);
        // Track region nodes already emitted (we emit at tail's topo position).
        let mut emitted_region: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            if let Some(&root) = region_of.get(&node.id)
                && let Some(&tail) = tail_of_region.get(&root)
            {
                if emitted_region.contains_key(&root) {
                    // Member but tail already emitted (or not tail). Map to
                    // either the new region node (if tail) or to a sentinel
                    // we never look up directly. Internal members are not
                    // referenced after fusion (single-consumer guarantee),
                    // so we map them to the region node id for safety.
                    let region_new = emitted_region[&root];
                    rw.replace(node.id, region_new);
                    continue;
                }
                if node.id == tail {
                    // Sort region members in topological (= chain step) order.
                    let members = &by_region[&root];
                    let mut ordered: Vec<NodeId> = members.clone();
                    ordered.sort_by_key(|id| chain_step_idx[id]);

                    // Collect external inputs (chain inputs that aren't members).
                    // SSA: each chain step refers to either an external input
                    // or a previous step. Build the chain.
                    let mut external_inputs: Vec<NodeId> = Vec::new();
                    let mut input_idx_of: HashMap<NodeId, u32> = HashMap::new();
                    let mut step_idx_of: HashMap<NodeId, u32> = HashMap::new();
                    for (i, member_id) in ordered.iter().enumerate() {
                        step_idx_of.insert(*member_id, i as u32);
                        let n = graph.node(*member_id);
                        for &inp in &n.inputs {
                            if !step_idx_of.contains_key(&inp) && !input_idx_of.contains_key(&inp) {
                                let idx = external_inputs.len() as u32;
                                input_idx_of.insert(inp, idx);
                                external_inputs.push(inp);
                            }
                        }
                    }

                    let limits = crate::limits::active_fusion_limits();
                    if external_inputs.len() as u32 > limits.max_elementwise_inputs
                        || ordered.len() as u32 > limits.max_elementwise_steps
                    {
                        for &mid in &ordered {
                            rw.copy_node(graph.node(mid));
                        }
                        continue;
                    }

                    let resolve = |id: NodeId| -> ChainOperand {
                        if let Some(&i) = input_idx_of.get(&id) {
                            ChainOperand::Input(i)
                        } else {
                            ChainOperand::Step(step_idx_of[&id])
                        }
                    };
                    let mut chain: Vec<ChainStep> = Vec::with_capacity(ordered.len());
                    for member_id in &ordered {
                        let n = graph.node(*member_id);
                        let step = match &n.op {
                            Op::Activation(a) => ChainStep::Activation(*a, resolve(n.inputs[0])),
                            Op::Cast { to } => ChainStep::Cast(*to, resolve(n.inputs[0])),
                            Op::Binary(op) => {
                                ChainStep::Binary(*op, resolve(n.inputs[0]), resolve(n.inputs[1]))
                            }
                            Op::Compare(op) => {
                                ChainStep::Compare(*op, resolve(n.inputs[0]), resolve(n.inputs[1]))
                            }
                            Op::Where => ChainStep::Where(
                                resolve(n.inputs[0]),
                                resolve(n.inputs[1]),
                                resolve(n.inputs[2]),
                            ),
                            _ => unreachable!("non-chain-eligible op in region"),
                        };
                        chain.push(step);
                    }

                    // PLAN L2 quality: per-input broadcast metadata.
                    // `scalar_input_mask` is the fast-path bitfield
                    // (bit `i` set ⇒ input `i` is a single-element
                    // scalar). `input_modulus[i]` is the per-input
                    // element count: 0 means "no broadcast" (kernel
                    // reads gid directly), >0 means tile by modulo.
                    // Encoder enforces `out_elems % in_elems == 0`
                    // upstream so the modulo divides cleanly.
                    let mut scalar_input_mask: u32 = 0;
                    let mut input_modulus = [0u32; 16];
                    let region_shape_elems = graph.node(tail).shape.num_elements();
                    for (i, &ext) in external_inputs.iter().enumerate() {
                        if i >= 16 {
                            break;
                        }
                        let in_elems = graph.shape(ext).num_elements();
                        match (in_elems, region_shape_elems) {
                            (Some(1), Some(o)) if o != 1 => {
                                scalar_input_mask |= 1u32 << i;
                                input_modulus[i] = 1;
                            }
                            (Some(i_n), Some(o)) if i_n != o && i_n > 0 => {
                                input_modulus[i] = i_n as u32;
                            }
                            _ => { /* no broadcast: leave modulus 0 */ }
                        }
                    }
                    let region_new = rw.add_fused(
                        Op::ElementwiseRegion {
                            chain,
                            num_inputs: external_inputs.len() as u32,
                            scalar_input_mask,
                            input_modulus,
                            prologue: RegionPrologue::None,
                            prologue_input: 0,
                        },
                        &external_inputs,
                        graph.node(tail).shape.clone(),
                    );
                    emitted_region.insert(root, region_new);
                    rw.replace(node.id, region_new);
                    continue;
                } else {
                    // Region member but not tail; skip (will be replaced
                    // when the tail is processed).
                    rw.replace(node.id, NodeId(u32::MAX)); // sentinel
                    continue;
                }
            }
            rw.copy_node(node);
        }

        // Final cleanup pass: any sentinel id_map entries get rewired to
        // their region's emitted node now that emission is done.
        // (Actually the order above means tails are processed in topo
        // order and members appear before tails in topo order, so by the
        // time a member's consumer is rewritten its id_map points to the
        // sentinel. Fix-up: walk again, rewrite sentinels.)
        // Simpler approach: process region members in second pass.
        // The current order processes tail last per region, so non-tail
        // members get sentinels. Their consumers are either other region
        // members (which we don't directly use the input from) or the
        // tail itself. Since the tail builds its own chain via members
        // directly from the original graph, the rewriter's id_map for
        // non-tail members is only consulted for the tail's input list —
        // which we resolve via `external_inputs` (already correctly
        // mapped via add_fused → map_inputs). So sentinels are safe.

        rw.finish(&graph.outputs)
    }
}

// ── PLAN L2 fallback: UnfuseElementwiseRegions ───────────────────────
//
// Decompose `Op::ElementwiseRegion` back into its constituent atomic
// ops (Activation / Cast / Binary / Compare). The output of the
// region is replaced with the result of the chain's last step;
// internal step results become individual nodes wired into the rest
// of the graph. Used by backends that don't have a native region
// kernel — they get the *correctness* of L2's IR-level fusion (no op
// missing) without needing to implement region codegen. Run BEFORE
// the backend's own lowering. No-op when the graph contains no
// ElementwiseRegion nodes.

