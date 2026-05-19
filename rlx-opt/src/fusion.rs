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

//! Fusion passes — pattern-match and replace subgraphs with fused ops.
//!
//! Each pass scans the graph in reverse topological order, looking for
//! specific multi-node patterns and replacing them with single fused nodes.
//! These are the same fusions we hand-coded in burnembed's ndarray_fused.rs.

use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

// ── Helper: graph rewriter ──────────────────────────────────────────────

/// Maps old NodeIds to new NodeIds during graph rewriting.
struct Rewriter {
    new_graph: Graph,
    id_map: HashMap<NodeId, NodeId>,
}

impl Rewriter {
    fn new(name: &str) -> Self {
        Self {
            new_graph: Graph::new(name),
            id_map: HashMap::new(),
        }
    }

    /// Map an old NodeId to its new equivalent.
    fn map(&self, old: NodeId) -> NodeId {
        self.id_map[&old]
    }

    /// Map a list of old NodeIds.
    fn map_inputs(&self, old_inputs: &[NodeId]) -> Vec<NodeId> {
        old_inputs.iter().map(|id| self.map(*id)).collect()
    }

    /// True iff every old NodeId in `ids` has already been mapped — used by fusion
    /// patterns to gate a rewrite on its inputs being live in the new graph.
    fn all_mapped(&self, ids: &[NodeId]) -> bool {
        ids.iter().all(|id| self.id_map.contains_key(id))
    }

    /// Copy a node from the old graph, remapping inputs.
    fn copy_node(&mut self, node: &Node) -> NodeId {
        let new_inputs = self.map_inputs(&node.inputs);
        let new_id = self
            .new_graph
            .add_node(node.op.clone(), new_inputs, node.shape.clone());
        self.id_map.insert(node.id, new_id);
        new_id
    }

    /// Add a new fused node (not from the old graph).
    fn add_fused(&mut self, op: Op, old_inputs: &[NodeId], shape: Shape) -> NodeId {
        let new_inputs: Vec<NodeId> = old_inputs.iter().map(|id| self.map(*id)).collect();
        self.new_graph.add_node(op, new_inputs, shape)
    }

    /// Mark an old node as replaced by a new node.
    fn replace(&mut self, old_id: NodeId, new_id: NodeId) {
        self.id_map.insert(old_id, new_id);
    }

    fn finish(mut self, old_outputs: &[NodeId]) -> Graph {
        let new_outputs = old_outputs.iter().map(|id| self.map(*id)).collect();
        self.new_graph.set_outputs(new_outputs);
        self.new_graph
    }
}

// ── Pass 1: MatMul + Bias + Activation → FusedMatMulBiasAct ─────────────

/// Fuses `matmul → add(bias) → activation` into a single FusedMatMulBiasAct.
///
/// This is the single most impactful fusion — it eliminates two intermediate
/// tensors and three memory passes (matmul write, bias read+write, act read+write)
/// down to one (matmul write with inline bias+activation).
///
/// Also fuses `matmul → add(bias)` without activation.
pub struct FuseMatMulBiasAct;

impl Pass for FuseMatMulBiasAct {
    fn name(&self) -> &str {
        "fuse_matmul_bias_act"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut rw = Rewriter::new(&graph.name);
        // Track which nodes are consumed by fusion (skip them in copy)
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();

        // Forward pass: copy nodes, detect patterns
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            // Pattern: MatMul → Add(bias) → Activation
            // or:      MatMul → Add(bias)
            if matches!(node.op, Op::MatMul) {
                let mm_id = node.id;
                let mm_users: Vec<_> = graph.users(mm_id);

                // Check for single-use Add(bias) consumer
                if mm_users.len() == 1 {
                    let add_node = graph.node(mm_users[0]);
                    if let Op::Binary(BinaryOp::Add) = &add_node.op {
                        // Determine which input is the bias (the non-matmul one)
                        let (bias_id, _mm_input) = if add_node.inputs[0] == mm_id {
                            (add_node.inputs[1], add_node.inputs[0])
                        } else {
                            (add_node.inputs[0], add_node.inputs[1])
                        };

                        // Check if bias is a param/const with broadcastable shape
                        let bias_shape = graph.shape(bias_id);
                        if bias_shape.rank() <= 1 {
                            let add_id = add_node.id;
                            let add_users = graph.users(add_id);

                            // Check for activation consumer
                            let mut activation = None;
                            let mut act_id = None;
                            if add_users.len() == 1 {
                                let act_node = graph.node(add_users[0]);
                                if let Op::Activation(a) = &act_node.op {
                                    activation = Some(*a);
                                    act_id = Some(act_node.id);
                                }
                            }

                            // Emit fused node
                            let out_shape = if let Some(aid) = act_id {
                                graph.shape(aid).clone()
                            } else {
                                add_node.shape.clone()
                            };

                            let fused_id = rw.add_fused(
                                Op::FusedMatMulBiasAct { activation },
                                &[node.inputs[0], node.inputs[1], bias_id],
                                out_shape,
                            );

                            // Map old nodes to the fused result
                            rw.replace(mm_id, fused_id);
                            rw.replace(add_id, fused_id);
                            fused_away.insert(add_id, ());
                            if let Some(aid) = act_id {
                                rw.replace(aid, fused_id);
                                fused_away.insert(aid, ());
                            }
                            continue;
                        }
                    }
                }
            }

            // No fusion — copy as-is
            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

// ── Pass 2: Add(residual) + LayerNorm → FusedResidualLN ─────────────────

/// Fuses `add(x, residual) → layer_norm` into FusedResidualLN.
///
/// Also detects `add(x, residual) → add(bias) → layer_norm` for the
/// bias variant (used in BERT's output projection).
pub struct FuseResidualLN;

impl Pass for FuseResidualLN {
    fn name(&self) -> &str {
        "fuse_residual_ln"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Graph outputs hold implicit references to their producing
        // nodes that don't show up in any node's `inputs` (use_count
        // walks node inputs only). Treat being-a-graph-output as a
        // use so we don't fuse-away an intermediate the caller still
        // wants to read — this used to silently corrupt multi-block
        // encoders (e.g. SAM 2 stage outputs) by collapsing the
        // residual add of block N into block N+1's LN.
        let mut is_output: HashMap<NodeId, ()> = HashMap::new();
        for &oid in &graph.outputs {
            is_output.insert(oid, ());
        }
        // Pre-scan: find all Add nodes consumed by LayerNorm
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for node in graph.nodes() {
            if let Op::LayerNorm { .. } = &node.op {
                let ln_input_id = node.inputs[0];
                let ln_input = graph.node(ln_input_id);
                if matches!(ln_input.op, Op::Binary(BinaryOp::Add))
                    && graph.use_count(ln_input_id) == 1
                    && !is_output.contains_key(&ln_input_id)
                {
                    fused_away.insert(ln_input_id, ());
                }
            }
        }

        let mut rw = Rewriter::new(&graph.name);

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if let Op::LayerNorm { eps, .. } = &node.op {
                let ln_input_id = node.inputs[0];
                let ln_input = graph.node(ln_input_id);

                if matches!(ln_input.op, Op::Binary(BinaryOp::Add))
                    && fused_away.contains_key(&ln_input_id)
                {
                    let (x_id, residual_id) = (ln_input.inputs[0], ln_input.inputs[1]);
                    let gamma_id = node.inputs[1];
                    let beta_id = node.inputs[2];

                    let fused_id = rw.add_fused(
                        Op::FusedResidualLN {
                            has_bias: false,
                            eps: *eps,
                        },
                        &[x_id, residual_id, gamma_id, beta_id],
                        node.shape.clone(),
                    );

                    rw.replace(ln_input_id, fused_id);
                    rw.replace(node.id, fused_id);
                    continue;
                }
            }

            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

// ── Pass 3: Shared-input MatMul concat (QKV, SwiGLU fc11+fc12) ──────────

/// Detects two MatMul nodes with the same input and concatenates their
/// weight matrices into a single larger MatMul.
///
/// Pattern:
///   %a = matmul(%x, %w1)
///   %b = matmul(%x, %w2)
/// Becomes:
///   %ab = matmul(%x, concat(%w1, %w2))
///   %a = narrow(%ab, ..., 0, n1)
///   %b = narrow(%ab, ..., n1, n2)
///
/// This saves one full input read (the shared input is read once instead
/// of twice). Critical for SwiGLU (fc11+fc12) and QKV fusion.
pub struct FuseSharedInputMatMul;

impl Pass for FuseSharedInputMatMul {
    fn name(&self) -> &str {
        "fuse_shared_input_matmul"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Find pairs of MatMul/FusedMatMulBiasAct nodes sharing the same first input
        let mut input_to_matmuls: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for node in graph.nodes() {
            match &node.op {
                Op::MatMul | Op::FusedMatMulBiasAct { .. } => {
                    input_to_matmuls
                        .entry(node.inputs[0])
                        .or_default()
                        .push(node.id);
                }
                _ => {}
            }
        }

        // Find fuseable pairs (adjacent matmuls, same input, compatible shapes)
        let mut fuse_pairs: Vec<(NodeId, NodeId)> = Vec::new();
        for matmuls in input_to_matmuls.values() {
            if matmuls.len() == 2 {
                let a = graph.node(matmuls[0]);
                let b = graph.node(matmuls[1]);
                // Both must be plain MatMul (not already fused with bias+act)
                if matches!(a.op, Op::MatMul) && matches!(b.op, Op::MatMul) {
                    // Weight shapes must be compatible: [K, N1] and [K, N2] with same K
                    let w1_shape = graph.shape(a.inputs[1]);
                    let w2_shape = graph.shape(b.inputs[1]);
                    if w1_shape.rank() == 2
                        && w2_shape.rank() == 2
                        && w1_shape.dim(0) == w2_shape.dim(0)
                    {
                        fuse_pairs.push((matmuls[0], matmuls[1]));
                    }
                }
            }
        }

        if fuse_pairs.is_empty() {
            return graph; // no changes
        }

        // Rebuild graph with fused matmuls
        let mut rw = Rewriter::new(&graph.name);
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            // Check if this node is the first of a fuse pair
            let mut found_pair = None;
            for &(a, b) in &fuse_pairs {
                if node.id == a {
                    found_pair = Some((a, b));
                    break;
                }
            }

            if let Some((a_id, b_id)) = found_pair {
                let a = graph.node(a_id);
                let b = graph.node(b_id);

                let input_id = a.inputs[0];
                let w1_id = a.inputs[1];
                let w2_id = b.inputs[1];

                // Topology guard. `node` is the EARLIER matmul `a`;
                // `b`'s weight `w2_id` may not be visited yet if it
                // sits between `a` and `b` in topo order — common in
                // grad graphs where the backward op for `matmul(A,B)`
                // emits `d_A = d_out · Bᵀ` and `d_B = Aᵀ · d_out`
                // and the transposes / d_out producers are interleaved
                // with the two matmuls. Without this guard the
                // `add_fused(Concat, [w1, w2], ..)` below panics in
                // `Rewriter::map` with "no entry found for key".
                // Graceful fallback: skip the fusion for this pair —
                // `a` gets copied normally below, and `b` (not in
                // `fused_away`) gets copied when iteration reaches it.
                if !rw.all_mapped(&[input_id, w1_id, w2_id]) {
                    rw.copy_node(node);
                    continue;
                }

                let w1_shape = graph.shape(w1_id);
                let w2_shape = graph.shape(w2_id);
                let k = w1_shape.dim(0).unwrap_static();
                let n1 = w1_shape.dim(1).unwrap_static();
                let n2 = w2_shape.dim(1).unwrap_static();
                let combined_n = n1 + n2;

                // Concat weights
                let concat_shape = Shape::new(&[k, combined_n], w1_shape.dtype());
                let concat_id = rw.add_fused(Op::Concat { axis: 1 }, &[w1_id, w2_id], concat_shape);

                // Combined matmul
                let out_rank = a.shape.rank();
                let mut mm_dims: Vec<usize> = (0..out_rank)
                    .map(|i| a.shape.dim(i).unwrap_static())
                    .collect();
                mm_dims[out_rank - 1] = combined_n;
                let mm_shape = Shape::new(&mm_dims, a.shape.dtype());
                let mm_id = rw.new_graph.add_node(
                    Op::MatMul,
                    vec![rw.map(input_id), concat_id],
                    mm_shape.clone(),
                );

                // Narrow to split outputs
                let narrow_a = rw.new_graph.add_node(
                    Op::Narrow {
                        axis: out_rank - 1,
                        start: 0,
                        len: n1,
                    },
                    vec![mm_id],
                    a.shape.clone(),
                );
                let narrow_b = rw.new_graph.add_node(
                    Op::Narrow {
                        axis: out_rank - 1,
                        start: n1,
                        len: n2,
                    },
                    vec![mm_id],
                    b.shape.clone(),
                );

                rw.replace(a_id, narrow_a);
                rw.replace(b_id, narrow_b);
                fused_away.insert(b_id, ());
                continue;
            }

            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

// ── Pass 4: Detect SwiGLU pattern → FusedSwiGLU ────────────────────────

/// Detects the post-`FuseSharedInputMatMul` SwiGLU pattern and replaces it
/// with a single `Op::FusedSwiGLU` node consuming the concatenated matmul.
///
/// Pattern (after `FuseSharedInputMatMul` has fused fc11+fc12 into one mm):
///   %cat   = matmul(%x, concat(%fc11_w, %fc12_w))   ; shape [..., 2N]
///   %up    = narrow(%cat, axis=-1, 0, N)            ; shape [..., N]
///   %gate  = narrow(%cat, axis=-1, N, N)            ; shape [..., N]
///   %silu  = silu(%gate)
///   %out   = mul(%up, %silu)
///
/// Becomes:
///   %out   = fused_swiglu(%cat)
///
/// Saves three kernel launches (two narrows + silu + mul → one kernel) and
/// keeps up/gate resident in registers.
///
/// Single-use guard: only fuses when each intermediate (narrow, narrow, silu)
/// has exactly one consumer. The mul may have any number of consumers.
pub struct FuseSwiGLU;

impl Pass for FuseSwiGLU {
    fn name(&self) -> &str {
        "fuse_swiglu"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Scan for Mul nodes whose two inputs match the SwiGLU pattern.
        // Collect rewrites first, then rebuild.
        // up_narrow_id / silu_id / gate_narrow_id are kept for pattern-shape
        // self-documentation even though only the rewrite path reads
        // mul_id / cat_id / out_n.
        #[allow(dead_code)]
        struct Match {
            mul_id: NodeId, // root of the pattern (output)
            up_narrow_id: NodeId,
            silu_id: NodeId,
            gate_narrow_id: NodeId,
            cat_id: NodeId, // shared source — input to FusedSwiGLU
            out_n: usize,   // half of the concat dim
        }

        let mut matches: Vec<Match> = Vec::new();
        let mut consumed: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            // Looking for: mul(narrow(cat, 0, n), silu(narrow(cat, n, n)))
            //   — or symmetrically with up/gate swapped.
            if !matches!(node.op, Op::Binary(BinaryOp::Mul)) {
                continue;
            }
            let lhs_id = node.inputs[0];
            let rhs_id = node.inputs[1];
            let lhs = graph.node(lhs_id);
            let rhs = graph.node(rhs_id);

            // Decide which side is silu(gate) — the silu branch.
            let (up_narrow, silu_id, silu_node) =
                if matches!(rhs.op, Op::Activation(Activation::Silu)) {
                    (lhs, rhs_id, rhs)
                } else if matches!(lhs.op, Op::Activation(Activation::Silu)) {
                    (rhs, lhs_id, lhs)
                } else {
                    continue;
                };

            // up side must be a Narrow.
            let (up_axis, up_start, up_len) = match &up_narrow.op {
                Op::Narrow { axis, start, len } => (*axis, *start, *len),
                _ => continue,
            };
            // silu input must be a Narrow.
            let gate_narrow_id = silu_node.inputs[0];
            let gate_narrow = graph.node(gate_narrow_id);
            let (g_axis, g_start, g_len) = match &gate_narrow.op {
                Op::Narrow { axis, start, len } => (*axis, *start, *len),
                _ => continue,
            };

            // Both narrows must come from the same source on the same axis,
            // covering the two halves: (0..N) and (N..2N).
            if up_narrow.inputs[0] != gate_narrow.inputs[0] {
                continue;
            }
            if up_axis != g_axis {
                continue;
            }
            if up_len != g_len {
                continue;
            }
            // Either ordering: (up @ 0, gate @ N) or (up @ N, gate @ 0)
            let n = up_len;
            let valid = (up_start == 0 && g_start == n) || (up_start == n && g_start == 0);
            if !valid {
                continue;
            }
            // For now we only support the canonical ordering up=lo, gate=hi.
            // Skip the swapped variant — it would require swizzling the
            // concat or the kernel; not worth complicating until a model
            // produces it.
            if !(up_start == 0 && g_start == n) {
                continue;
            }

            // Single-use checks: narrows feed only into silu+mul, silu feeds
            // only into mul. The cat itself can have arbitrary other users.
            if graph.use_count(up_narrow.id) != 1 {
                continue;
            }
            if graph.use_count(gate_narrow_id) != 1 {
                continue;
            }
            if graph.use_count(silu_id) != 1 {
                continue;
            }

            matches.push(Match {
                mul_id: node.id,
                up_narrow_id: up_narrow.id,
                silu_id,
                gate_narrow_id,
                cat_id: up_narrow.inputs[0],
                out_n: n,
            });
            consumed.insert(up_narrow.id, ());
            consumed.insert(gate_narrow_id, ());
            consumed.insert(silu_id, ());
        }

        if matches.is_empty() {
            return graph;
        }

        // Rebuild graph, replacing matched mul nodes with FusedSwiGLU.
        let mut rw = Rewriter::new(&graph.name);
        let match_by_mul: HashMap<NodeId, &Match> = matches.iter().map(|m| (m.mul_id, m)).collect();

        for node in graph.nodes() {
            if consumed.contains_key(&node.id) {
                continue;
            }

            if let Some(m) = match_by_mul.get(&node.id) {
                // Output shape = mul's output shape (= [..., N]).
                let out_shape = node.shape.clone();
                debug_assert_eq!(
                    out_shape.dim(out_shape.rank() - 1).unwrap_static(),
                    m.out_n,
                    "FuseSwiGLU: output last dim should be N"
                );
                let fused_id =
                    rw.add_fused(Op::FusedSwiGLU { cast_to: None }, &[m.cat_id], out_shape);
                rw.replace(node.id, fused_id);
                continue;
            }

            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

// ── Pass 5: Fuse Attention Block (QKV → SDPA → OutProj) ────────────────

/// Fuses `matmul(QKV) → narrow(Q,K,V) → [rope] → attention → matmul(out)`
/// into a single FusedAttentionBlock when batch*seq is small.
///
/// The optimizer auto-detects batch size from graph input shapes. For small
/// inputs (batch*seq ≤ 64), intermediate tensors fit in L1 cache, making a
/// monolithic kernel faster than separate BLAS calls.
///
/// Threshold is configurable via `RLX_FUSE_ATTN_THRESHOLD` (default: 64).
pub struct FuseAttentionBlock;

impl FuseAttentionBlock {
    /// Check if the graph has small enough inputs to benefit from fusion.
    /// Currently unused — `Pass::run` is a no-op since attention fusion
    /// happens at thunk-compile time, not graph-rewrite time. Kept here
    /// for the planned graph-level rewrite path.
    #[allow(dead_code)]
    fn should_fuse(graph: &Graph) -> bool {
        let threshold: usize = std::env::var("RLX_FUSE_ATTN_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        for node in graph.nodes() {
            if let Op::Input { .. } = &node.op
                && node.shape.rank() >= 2
            {
                let d0 = node.shape.dim(0);
                let d1 = node.shape.dim(1);
                if d0.is_static() && d1.is_static() {
                    let b = d0.unwrap_static();
                    let s = d1.unwrap_static();
                    if b * s <= threshold {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl Pass for FuseAttentionBlock {
    fn name(&self) -> &str {
        "fuse_attention_block"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Attention block fusion is done at the thunk level (compile_thunks)
        // instead of the graph level, to avoid complex Rewriter issues.
        // This pass is a no-op; the thunk compiler handles it directly.
        graph
    }
}

// ── PLAN L2: MarkElementwiseRegions ─────────────────────────────────────
//
// Walk the graph and collapse maximal chains of element-wise ops
// (Activation / Cast / Binary / Compare) into a single
// `Op::ElementwiseRegion`. Conditions for inclusion in a chain:
//   - Op is element-wise per `is_elementwise()` (excluding Where which
//     has a 3-input mask semantic that doesn't compose into a single
//     scalar register chain cleanly — keep as separate op for now).
//   - Output shape exactly equals every input shape (no broadcast —
//     broadcast scalar/vector adds register-pattern complexity, defer).
//   - Every intermediate (chain-internal) value has exactly one
//     consumer in the *whole* graph. Multi-consumer values must
//     materialize.
// The chain start can read graph-level inputs / params / earlier-fused
// nodes; the chain end is the last single-consumer or terminal node.
// This is the simplest correct cut — N-ary chain fusion replaces the
// pairwise `fuse_elementwise_chains` pattern in each backend with one
// IR-level pass + a single backend kernel. See PLAN L2.

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
            region_of.insert(node.id, root);
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

pub struct UnfuseElementwiseRegions;

impl Pass for UnfuseElementwiseRegions {
    fn name(&self) -> &str {
        "unfuse_elementwise_regions"
    }

    fn run(&self, graph: Graph) -> Graph {
        let any_region = graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ElementwiseRegion { .. }));
        if !any_region {
            return graph;
        }

        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if let Op::ElementwiseRegion {
                chain,
                num_inputs: _,
                scalar_input_mask: _,
                input_modulus: _,
            } = &node.op
            {
                // Region inputs (in the new graph) — the rewriter has
                // already mapped each old input id.
                let region_inputs: Vec<NodeId> = node.inputs.iter().map(|id| rw.map(*id)).collect();
                let mut step_ids: Vec<NodeId> = Vec::with_capacity(chain.len());
                let region_shape = node.shape.clone();
                let region_dims: Vec<_> = region_shape.dims().to_vec();
                // Per-step result dtype, indexed by step position.
                // The chain may pass through Cast steps that change the
                // dtype mid-chain; using `region_shape.dtype()` blindly
                // would mis-tag intermediate Activation/Binary/Where
                // shapes. Track the dtype propagated through each step.
                let mut step_dtypes: Vec<rlx_ir::DType> = Vec::with_capacity(chain.len());
                let region_dtype = region_shape.dtype();
                let dtype_of = |op: &ChainOperand,
                                ins: &[NodeId],
                                step_dt: &[rlx_ir::DType],
                                rw: &Rewriter|
                 -> rlx_ir::DType {
                    match *op {
                        ChainOperand::Input(i) => rw.new_graph.node(ins[i as usize]).shape.dtype(),
                        ChainOperand::Step(i) => step_dt[i as usize],
                    }
                };
                // Shape of an operand in the rewritten graph. Critical
                // for broadcast inputs: a region whose final shape is
                // `[8, 1]` can still have a scalar operand at some
                // step; tagging that step with region_dims would lie
                // about its element count and trip the binary/activation
                // kernels (which size their reads/writes off the IR
                // shape, not the broadcast-aware semantics the L2
                // region kernel would have used). Use the actual node
                // shape so the unfused pipeline matches what each op
                // semantically produces.
                let shape_of = |op: &ChainOperand,
                                ins: &[NodeId],
                                step_ids: &[NodeId],
                                rw: &Rewriter|
                 -> Shape {
                    match *op {
                        ChainOperand::Input(i) => rw.new_graph.node(ins[i as usize]).shape.clone(),
                        ChainOperand::Step(i) => {
                            rw.new_graph.node(step_ids[i as usize]).shape.clone()
                        }
                    }
                };
                for step in chain {
                    let resolve = |op: &ChainOperand| -> NodeId {
                        match *op {
                            ChainOperand::Input(i) => region_inputs[i as usize],
                            ChainOperand::Step(i) => step_ids[i as usize],
                        }
                    };
                    let (new_id, dt) = match step {
                        ChainStep::Activation(a, src) => {
                            let s = resolve(src);
                            let dt = dtype_of(src, &region_inputs, &step_dtypes, &rw);
                            // Activation is element-wise: output shape
                            // == input shape (preserve broadcast-source
                            // shapes; do NOT promote to region_dims).
                            let src_shape = shape_of(src, &region_inputs, &step_ids, &rw);
                            let dims: Vec<_> = src_shape.dims().to_vec();
                            let shape = Shape::from_dims(&dims, dt);
                            (
                                rw.new_graph.add_node(Op::Activation(*a), vec![s], shape),
                                dt,
                            )
                        }
                        ChainStep::Cast(to, src) => {
                            let s = resolve(src);
                            let src_shape = shape_of(src, &region_inputs, &step_ids, &rw);
                            let dims: Vec<_> = src_shape.dims().to_vec();
                            let shape = Shape::from_dims(&dims, *to);
                            (
                                rw.new_graph.add_node(Op::Cast { to: *to }, vec![s], shape),
                                *to,
                            )
                        }
                        ChainStep::Binary(op, lhs, rhs) => {
                            let l = resolve(lhs);
                            let r = resolve(rhs);
                            let dt = dtype_of(lhs, &region_inputs, &step_dtypes, &rw);
                            // Binary: NumPy-style broadcast of operands.
                            let lhs_shape = shape_of(lhs, &region_inputs, &step_ids, &rw);
                            let rhs_shape = shape_of(rhs, &region_inputs, &step_ids, &rw);
                            let bcast = rlx_ir::shape::broadcast(&lhs_shape, &rhs_shape)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "unfuse_elementwise_regions: cannot broadcast \
                                         {lhs_shape:?} ⊗ {rhs_shape:?} for Binary({op:?}): {e}"
                                    )
                                });
                            let dims: Vec<_> = bcast.dims().to_vec();
                            let shape = Shape::from_dims(&dims, dt);
                            (
                                rw.new_graph.add_node(Op::Binary(*op), vec![l, r], shape),
                                dt,
                            )
                        }
                        ChainStep::Compare(op, lhs, rhs) => {
                            let l = resolve(lhs);
                            let r = resolve(rhs);
                            let lhs_shape = shape_of(lhs, &region_inputs, &step_ids, &rw);
                            let rhs_shape = shape_of(rhs, &region_inputs, &step_ids, &rw);
                            let bcast = rlx_ir::shape::broadcast(&lhs_shape, &rhs_shape)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "unfuse_elementwise_regions: cannot broadcast \
                                         {lhs_shape:?} ⊗ {rhs_shape:?} for Compare({op:?}): {e}"
                                    )
                                });
                            let dims: Vec<_> = bcast.dims().to_vec();
                            let shape = Shape::from_dims(&dims, rlx_ir::DType::Bool);
                            (
                                rw.new_graph.add_node(Op::Compare(*op), vec![l, r], shape),
                                rlx_ir::DType::Bool,
                            )
                        }
                        ChainStep::Where(c, x, y) => {
                            let cn = resolve(c);
                            let xn = resolve(x);
                            let yn = resolve(y);
                            let dt = dtype_of(x, &region_inputs, &step_dtypes, &rw);
                            // Where: broadcast across (cond, then, else).
                            let c_shape = shape_of(c, &region_inputs, &step_ids, &rw);
                            let x_shape = shape_of(x, &region_inputs, &step_ids, &rw);
                            let y_shape = shape_of(y, &region_inputs, &step_ids, &rw);
                            let bcast_xy = rlx_ir::shape::broadcast(&x_shape, &y_shape)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "unfuse_elementwise_regions: cannot broadcast \
                                         then/else {x_shape:?} ⊗ {y_shape:?} for Where: {e}"
                                    )
                                });
                            let bcast = rlx_ir::shape::broadcast(&c_shape, &bcast_xy)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "unfuse_elementwise_regions: cannot broadcast cond \
                                         {c_shape:?} ⊗ {bcast_xy:?} for Where: {e}"
                                    )
                                });
                            let dims: Vec<_> = bcast.dims().to_vec();
                            let shape = Shape::from_dims(&dims, dt);
                            (
                                rw.new_graph.add_node(Op::Where, vec![cn, xn, yn], shape),
                                dt,
                            )
                        }
                    };
                    step_ids.push(new_id);
                    step_dtypes.push(dt);
                }
                let _ = region_dtype;
                let _ = region_dims;
                // The region's "output" (= last step) replaces the original
                // ElementwiseRegion node id.
                let last = *step_ids.last().expect("chain non-empty per pass invariant");
                rw.replace(node.id, last);
                continue;
            }
            rw.copy_node(node);
        }
        rw.finish(&graph.outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass::run_passes;

    fn f32_shape(dims: &[usize]) -> Shape {
        Shape::new(dims, DType::F32)
    }

    #[test]
    fn fuse_matmul_bias_gelu() {
        let mut g = Graph::new("test");
        let x = g.input("x", f32_shape(&[4, 15, 384]));
        let w = g.param("w", f32_shape(&[384, 1536]));
        let b = g.param("b", f32_shape(&[1536]));
        let mm = g.matmul(x, w, f32_shape(&[4, 15, 1536]));
        let add = g.binary(BinaryOp::Add, mm, b, f32_shape(&[4, 15, 1536]));
        let out = g.activation(Activation::Gelu, add, f32_shape(&[4, 15, 1536]));
        g.set_outputs(vec![out]);

        assert_eq!(g.len(), 6); // input, w, b, mm, add, gelu

        let fused = FuseMatMulBiasAct.run(g);
        println!("{fused}");

        // Should be: input, w, b, fused_mm_bias_gelu
        assert_eq!(fused.len(), 4);
        let out_node = fused.node(fused.outputs[0]);
        assert!(matches!(
            out_node.op,
            Op::FusedMatMulBiasAct {
                activation: Some(Activation::Gelu)
            }
        ));
    }

    #[test]
    fn fuse_matmul_bias_no_act() {
        let mut g = Graph::new("test");
        let x = g.input("x", f32_shape(&[4, 15, 384]));
        let w = g.param("w", f32_shape(&[384, 384]));
        let b = g.param("b", f32_shape(&[384]));
        let mm = g.matmul(x, w, f32_shape(&[4, 15, 384]));
        let add = g.binary(BinaryOp::Add, mm, b, f32_shape(&[4, 15, 384]));
        g.set_outputs(vec![add]);

        let fused = FuseMatMulBiasAct.run(g);
        assert_eq!(fused.len(), 4);
        let out_node = fused.node(fused.outputs[0]);
        assert!(matches!(
            out_node.op,
            Op::FusedMatMulBiasAct { activation: None }
        ));
    }

    #[test]
    fn fuse_residual_layer_norm() {
        let mut g = Graph::new("test");
        let x = g.input("x", f32_shape(&[4, 15, 384]));
        let residual = g.input("residual", f32_shape(&[4, 15, 384]));
        let gamma = g.param("gamma", f32_shape(&[384]));
        let beta = g.param("beta", f32_shape(&[384]));
        let add = g.binary(BinaryOp::Add, x, residual, f32_shape(&[4, 15, 384]));
        let ln = g.layer_norm(add, gamma, beta, -1, 1e-12, f32_shape(&[4, 15, 384]));
        g.set_outputs(vec![ln]);

        assert_eq!(g.len(), 6); // x, residual, gamma, beta, add, ln

        let fused = FuseResidualLN.run(g);
        println!("{fused}");

        // Should be: x, residual, gamma, beta, fused_residual_ln
        assert_eq!(fused.len(), 5);
        let out_node = fused.node(fused.outputs[0]);
        assert!(matches!(
            out_node.op,
            Op::FusedResidualLN {
                has_bias: false,
                ..
            }
        ));
    }

    #[test]
    fn fuse_shared_input_matmul() {
        let mut g = Graph::new("swiglu");
        let x = g.input("x", f32_shape(&[60, 768]));
        let w1 = g.param("fc11", f32_shape(&[768, 2048]));
        let w2 = g.param("fc12", f32_shape(&[768, 2048]));
        let mm1 = g.matmul(x, w1, f32_shape(&[60, 2048]));
        let mm2 = g.matmul(x, w2, f32_shape(&[60, 2048]));
        g.set_outputs(vec![mm1, mm2]);

        assert_eq!(g.len(), 5); // x, w1, w2, mm1, mm2

        let fused = FuseSharedInputMatMul.run(g);
        println!("{fused}");

        // Should have: x, w1, w2, concat(w1,w2), combined_mm, narrow1, narrow2
        assert!(fused.len() <= 7);
        // Both outputs should be Narrow ops
        for &out in &fused.outputs {
            assert!(matches!(fused.node(out).op, Op::Narrow { .. }));
        }
    }

    /// Regression: `FuseSharedInputMatMul` used to panic in
    /// `Rewriter::map` with "no entry found for key" when the second
    /// matmul of a share-pair had its weight (`w2`) inserted into the
    /// graph AFTER the first matmul (`mm1`) — the topology produced
    /// by `grad_with_loss` on any matmul-bearing forward graph (the
    /// matmul VJP rule emits `d_A = d_out·Bᵀ` and `d_B = Aᵀ·d_out`,
    /// which share `d_out` with intermediate transposes interleaved).
    /// The fix gracefully skips fusion when any input is still
    /// unmapped — `mm1` is copied as-is, `mm2` likewise when reached.
    #[test]
    fn fuse_shared_input_matmul_skips_when_w2_comes_after_mm1() {
        let mut g = Graph::new("late_w2");
        let x = g.input("x", f32_shape(&[8, 16]));
        let w1 = g.param("w1", f32_shape(&[16, 8]));
        let mm1 = g.matmul(x, w1, f32_shape(&[8, 8]));
        // w2 inserted AFTER mm1 — mirrors the grad-graph topology.
        let w2 = g.param("w2", f32_shape(&[16, 8]));
        let mm2 = g.matmul(x, w2, f32_shape(&[8, 8]));
        g.set_outputs(vec![mm1, mm2]);

        // Must NOT panic.
        let fused = FuseSharedInputMatMul.run(g);

        // Both outputs stay as plain MatMul (unfused) because the
        // pass bailed on the topology.
        for &out in &fused.outputs {
            assert!(
                matches!(fused.node(out).op, Op::MatMul),
                "expected unfused MatMul, got {:?}",
                fused.node(out).op
            );
        }
    }

    /// Full pipeline: build a BERT FFN subgraph and run all fusion passes.
    #[test]
    fn full_bert_ffn_fusion() {
        let mut g = Graph::new("bert_ffn");
        let f = DType::F32;

        let x = g.input("hidden", Shape::new(&[4, 15, 384], f));
        let residual = g.input("residual", Shape::new(&[4, 15, 384], f));

        // Output projection result + residual + LN
        let out_w = g.param("out.w", Shape::new(&[384, 384], f));
        let out_b = g.param("out.b", Shape::new(&[384], f));
        let out_mm = g.matmul(x, out_w, Shape::new(&[4, 15, 384], f));
        let out_add = g.binary(BinaryOp::Add, out_mm, out_b, Shape::new(&[4, 15, 384], f));
        let res_add = g.binary(
            BinaryOp::Add,
            out_add,
            residual,
            Shape::new(&[4, 15, 384], f),
        );
        let gamma = g.param("ln.g", Shape::new(&[384], f));
        let beta = g.param("ln.b", Shape::new(&[384], f));
        let ln = g.layer_norm(
            res_add,
            gamma,
            beta,
            -1,
            1e-12,
            Shape::new(&[4, 15, 384], f),
        );

        // FFN intermediate: matmul + bias + gelu
        let int_w = g.param("int.w", Shape::new(&[384, 1536], f));
        let int_b = g.param("int.b", Shape::new(&[1536], f));
        let int_mm = g.matmul(ln, int_w, Shape::new(&[4, 15, 1536], f));
        let int_add = g.binary(BinaryOp::Add, int_mm, int_b, Shape::new(&[4, 15, 1536], f));
        let gelu = g.activation(Activation::Gelu, int_add, Shape::new(&[4, 15, 1536], f));

        // FFN output: matmul + bias
        let out2_w = g.param("out2.w", Shape::new(&[1536, 384], f));
        let out2_b = g.param("out2.b", Shape::new(&[384], f));
        let out2_mm = g.matmul(gelu, out2_w, Shape::new(&[4, 15, 384], f));
        let out2_add = g.binary(BinaryOp::Add, out2_mm, out2_b, Shape::new(&[4, 15, 384], f));

        g.set_outputs(vec![out2_add]);

        let before = g.len();
        println!("=== BEFORE fusion ({before} nodes) ===\n{g}");

        // Run all passes
        let passes: Vec<&dyn Pass> = vec![&FuseMatMulBiasAct, &FuseResidualLN];
        let optimized = run_passes(g, &passes, false);
        let after = optimized.len();
        println!("=== AFTER fusion ({after} nodes) ===\n{optimized}");

        // Should have eliminated:
        // - 2 Add + 1 Gelu from matmul_bias_gelu fusion (×2 matmuls)
        // - 1 Add from residual_ln fusion
        assert!(
            after < before,
            "fusion should reduce node count: {before} → {after}"
        );

        // Check that fused ops exist
        let ops: Vec<String> = optimized
            .nodes()
            .iter()
            .map(|n| format!("{}", n.op))
            .collect();
        let has_fused_mm = ops.iter().any(|s| s.contains("fused_mm_bias"));
        assert!(has_fused_mm, "should have fused_mm_bias_act: {ops:?}");
    }

    /// FuseSwiGLU fires on the canonical Nomic-style pattern produced by
    /// `FuseSharedInputMatMul` (concat'd matmul → narrow×2 → silu → mul).
    #[test]
    fn fuse_swiglu_canonical() {
        let mut g = Graph::new("nomic_ffn");
        let f = DType::F32;
        // After FuseSharedInputMatMul: cat = mm(x, concat(fc11, fc12)) → [60, 4096]
        let cat = g.input("cat", Shape::new(&[60, 4096], f));
        let up = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 0,
                len: 2048,
            },
            vec![cat],
            Shape::new(&[60, 2048], f),
        );
        let gate = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 2048,
                len: 2048,
            },
            vec![cat],
            Shape::new(&[60, 2048], f),
        );
        let silu = g.activation(Activation::Silu, gate, Shape::new(&[60, 2048], f));
        let out = g.binary(BinaryOp::Mul, up, silu, Shape::new(&[60, 2048], f));
        g.set_outputs(vec![out]);

        let before = g.len();
        let fused = FuseSwiGLU.run(g);
        let after = fused.len();
        // Removed: up, gate, silu, mul → replaced by FusedSwiGLU.
        // Net: -3 nodes (4 removed, 1 added).
        assert_eq!(
            after,
            before - 3,
            "should remove narrows+silu+mul, add FusedSwiGLU"
        );
        let out_node = fused.node(fused.outputs[0]);
        assert!(
            matches!(out_node.op, Op::FusedSwiGLU { cast_to: None }),
            "output should be FusedSwiGLU, got {}",
            out_node.op
        );
        // FusedSwiGLU's input is the cat tensor.
        let in_id = out_node.inputs[0];
        assert!(matches!(fused.node(in_id).op, Op::Input { .. }));
    }

    /// FuseSwiGLU does NOT fire when narrows are shared with another consumer
    /// (would corrupt the second consumer's view of the data).
    #[test]
    fn fuse_swiglu_skips_when_narrow_has_extra_user() {
        let mut g = Graph::new("contended");
        let f = DType::F32;
        let cat = g.input("cat", Shape::new(&[60, 4096], f));
        let up = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 0,
                len: 2048,
            },
            vec![cat],
            Shape::new(&[60, 2048], f),
        );
        let gate = g.add_node(
            Op::Narrow {
                axis: 1,
                start: 2048,
                len: 2048,
            },
            vec![cat],
            Shape::new(&[60, 2048], f),
        );
        let silu = g.activation(Activation::Silu, gate, Shape::new(&[60, 2048], f));
        let out = g.binary(BinaryOp::Mul, up, silu, Shape::new(&[60, 2048], f));
        // Extra user of `up` — this should block fusion.
        let extra = g.activation(Activation::Relu, up, Shape::new(&[60, 2048], f));
        g.set_outputs(vec![out, extra]);

        let before = g.len();
        let fused = FuseSwiGLU.run(g);
        // Pass should be a no-op when fusion is unsafe.
        assert_eq!(fused.len(), before);
        // No FusedSwiGLU node anywhere.
        let any_fused = fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::FusedSwiGLU { .. }));
        assert!(!any_fused, "should not fuse when narrow has extra user");
    }

    // ── MarkElementwiseRegions (PLAN L2) ────────────────────────────

    #[test]
    fn region_collapses_add_mul_relu_chain() {
        // Build: out = relu(add(a, b) * c). All same shape, single consumer
        // chain. Should fuse into one ElementwiseRegion.
        let f = DType::F32;
        let mut g = Graph::new("ew");
        let a = g.input("a", Shape::new(&[8], f));
        let b = g.input("b", Shape::new(&[8], f));
        let c = g.input("c", Shape::new(&[8], f));
        let s = Shape::new(&[8], f);
        let add = g.binary(BinaryOp::Add, a, b, s.clone());
        let mul = g.binary(BinaryOp::Mul, add, c, s.clone());
        let relu = g.activation(Activation::Relu, mul, s.clone());
        g.set_outputs(vec![relu]);

        let before = g.len();
        let fused = MarkElementwiseRegions.run(g);

        // Three element-wise ops collapsed into one region node.
        let regions: Vec<_> = fused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
            .collect();
        assert_eq!(regions.len(), 1, "expected one ElementwiseRegion");
        let region = regions[0];
        assert_eq!(
            region.inputs.len(),
            3,
            "region has 3 external inputs (a, b, c)"
        );
        if let Op::ElementwiseRegion {
            chain, num_inputs, ..
        } = &region.op
        {
            assert_eq!(*num_inputs, 3);
            assert_eq!(chain.len(), 3);
            // Step 0: Add(Input(0), Input(1))
            match &chain[0] {
                ChainStep::Binary(
                    BinaryOp::Add,
                    ChainOperand::Input(0),
                    ChainOperand::Input(1),
                ) => {}
                other => panic!("step 0 unexpected: {other:?}"),
            }
            // Step 1: Mul(Step(0), Input(2))
            match &chain[1] {
                ChainStep::Binary(BinaryOp::Mul, ChainOperand::Step(0), ChainOperand::Input(2)) => {
                }
                other => panic!("step 1 unexpected: {other:?}"),
            }
            // Step 2: Activation(Relu, Step(1))
            match &chain[2] {
                ChainStep::Activation(Activation::Relu, ChainOperand::Step(1)) => {}
                other => panic!("step 2 unexpected: {other:?}"),
            }
        } else {
            unreachable!();
        }
        // Original chain (3 ops) replaced by 1 region; net node count is
        // (inputs 3) + (region 1) = 4 (vs 3 + 3 = 6 before).
        assert!(fused.len() < before);
    }

    #[test]
    fn region_does_not_fuse_when_intermediate_has_multiple_consumers() {
        // out1 = add(a, b); out2 = relu(out1). out1 also fed to out_extra.
        // Multi-consumer on out1 forbids fusion.
        let f = DType::F32;
        let mut g = Graph::new("ew");
        let a = g.input("a", Shape::new(&[4], f));
        let b = g.input("b", Shape::new(&[4], f));
        let s = Shape::new(&[4], f);
        let add = g.binary(BinaryOp::Add, a, b, s.clone());
        let relu = g.activation(Activation::Relu, add, s.clone());
        let extra = g.activation(Activation::Sigmoid, add, s.clone());
        g.set_outputs(vec![relu, extra]);

        let before = g.len();
        let fused = MarkElementwiseRegions.run(g);
        // No region: add has two consumers (relu and extra), so the chain
        // can't extend through it. Each downstream activation is alone in
        // its region (size 1, doesn't fuse).
        let regions: Vec<_> = fused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
            .collect();
        assert_eq!(regions.len(), 0);
        assert_eq!(fused.len(), before);
    }

    #[test]
    fn region_skips_chains_of_length_one() {
        // Single relu — no fusion (size 1 = degenerate).
        let f = DType::F32;
        let mut g = Graph::new("ew");
        let a = g.input("a", Shape::new(&[4], f));
        let r = g.activation(Activation::Relu, a, Shape::new(&[4], f));
        g.set_outputs(vec![r]);

        let fused = MarkElementwiseRegions.run(g);
        let any_region = fused
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ElementwiseRegion { .. }));
        assert!(!any_region);
    }

    #[test]
    fn unfuse_decomposes_region_back_to_atomic_ops() {
        // Build the same chain, fuse it, then unfuse — expect the
        // original atomic ops back (Add, Mul, Relu).
        let f = DType::F32;
        let mut g = Graph::new("ew_unfuse");
        let a = g.input("a", Shape::new(&[8], f));
        let b = g.input("b", Shape::new(&[8], f));
        let c = g.input("c", Shape::new(&[8], f));
        let s = Shape::new(&[8], f);
        let add = g.binary(BinaryOp::Add, a, b, s.clone());
        let mul = g.binary(BinaryOp::Mul, add, c, s.clone());
        let relu = g.activation(Activation::Relu, mul, s);
        g.set_outputs(vec![relu]);

        let fused = MarkElementwiseRegions.run(g);
        // Sanity: fusion happened.
        assert!(
            fused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
        );

        let unfused = UnfuseElementwiseRegions.run(fused);
        // No region nodes left.
        assert!(
            !unfused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
        );
        // Original atomic ops are back: Add, Mul, Relu.
        let bin_count = unfused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Binary(_)))
            .count();
        let act_count = unfused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Activation(_)))
            .count();
        assert_eq!(bin_count, 2, "Add + Mul restored");
        assert_eq!(act_count, 1, "Relu restored");
    }

    #[test]
    fn unfuse_is_noop_when_no_region_present() {
        let f = DType::F32;
        let mut g = Graph::new("noop");
        let a = g.input("a", Shape::new(&[4], f));
        let r = g.activation(Activation::Relu, a, Shape::new(&[4], f));
        g.set_outputs(vec![r]);
        let n_before = g.len();
        let result = UnfuseElementwiseRegions.run(g);
        // Pass returns unchanged graph (early return on no-region check).
        assert_eq!(result.len(), n_before);
    }

    #[test]
    fn region_includes_where_step() {
        // Build: cmp = a > b; sel = where(cmp, a, b); out = sel + a
        // The compare → where → add chain is fully element-wise; the
        // Where step lands inside the region thanks to the L2-quality
        // extension that adds `Op::Where` to the chain-eligible set.
        let f = DType::F32;
        let mut g = Graph::new("region_where");
        let a = g.input("a", Shape::new(&[4], f));
        let b = g.input("b", Shape::new(&[4], f));
        let s = Shape::new(&[4], f);
        let cmp = g.add_node(Op::Compare(CmpOp::Gt), vec![a, b], s.clone());
        let sel = g.add_node(Op::Where, vec![cmp, a, b], s.clone());
        let add = g.binary(BinaryOp::Add, sel, a, s.clone());
        g.set_outputs(vec![add]);

        let fused = MarkElementwiseRegions.run(g);
        let regions: Vec<_> = fused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::ElementwiseRegion { .. }))
            .collect();
        assert_eq!(regions.len(), 1, "expected one ElementwiseRegion");
        if let Op::ElementwiseRegion { chain, .. } = &regions[0].op {
            // 3 steps: Compare a > b, Where, Add
            assert_eq!(chain.len(), 3);
            assert!(
                matches!(chain[1], ChainStep::Where(_, _, _)),
                "step 1 should be Where, got {:?}",
                chain[1]
            );
        } else {
            unreachable!();
        }
    }

    #[test]
    fn unfuse_decomposes_where_step_back_to_op_where() {
        // Round-trip: build a region with a Where step, decompose it,
        // verify the resulting graph contains an Op::Where node.
        let f = DType::F32;
        let mut g = Graph::new("unfuse_where");
        let a = g.input("a", Shape::new(&[4], f));
        let b = g.input("b", Shape::new(&[4], f));
        let s = Shape::new(&[4], f);
        let cmp = g.add_node(Op::Compare(CmpOp::Gt), vec![a, b], s.clone());
        let sel = g.add_node(Op::Where, vec![cmp, a, b], s.clone());
        let add = g.binary(BinaryOp::Add, sel, a, s.clone());
        g.set_outputs(vec![add]);
        let fused = MarkElementwiseRegions.run(g);
        let unfused = UnfuseElementwiseRegions.run(fused);
        let where_count = unfused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Where))
            .count();
        assert_eq!(
            where_count, 1,
            "decomposer should re-emit one Op::Where for the chain step"
        );
    }
}
