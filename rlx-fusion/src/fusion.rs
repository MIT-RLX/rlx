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
    #[allow(dead_code)]
    fn all_mapped(&self, ids: &[NodeId]) -> bool {
        ids.iter().all(|id| self.id_map.contains_key(id))
    }

    /// Copy any not-yet-mapped nodes from `old` so fusion rewrites can
    /// reference operands declared later in the source graph (e.g. a bias
    /// param appended after its matmul consumer, or a reshape input that
    /// has not been reached in the linear rewrite walk yet).
    fn ensure_mapped(&mut self, old: &Graph, ids: &[NodeId]) {
        for &id in ids {
            if self.id_map.contains_key(&id) {
                continue;
            }
            let node = old.node(id);
            if !node.inputs.is_empty() {
                self.ensure_mapped(old, &node.inputs);
            }
            self.copy_node(node);
        }
    }

    /// Copy a node from the old graph, remapping inputs.
    fn copy_node(&mut self, node: &Node) -> NodeId {
        let new_inputs = self.map_inputs(&node.inputs);
        let new_id = self
            .new_graph
            .add_node(node.op.clone(), new_inputs, node.shape.clone());
        let new_node = self.new_graph.node_mut(new_id);
        new_node.name = node.name.clone();
        new_node.origin = node.origin.clone();
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
///
/// Epilogue activations are fused only when every backend can apply them
/// inline with the matmul (today: Gelu and Silu). Other activations — e.g.
/// Exp in qwen35 softplus — stay as separate ops so Metal does not silently
/// drop the epilogue.
pub struct FuseMatMulBiasAct;

/// Activations that may be folded into `FusedMatMulBiasAct` epilogues.
fn fusible_mm_bias_epilogue_activation(act: Activation) -> bool {
    matches!(act, Activation::Gelu | Activation::Silu)
}

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
                                if let Op::Activation(a) = &act_node.op
                                    && fusible_mm_bias_epilogue_activation(*a)
                                {
                                    activation = Some(*a);
                                    act_id = Some(act_node.id);
                                }
                            }

                            // Emit fused node. Bias may be declared after
                            // the matmul in the source graph — copy it early
                            // instead of requiring builders to order params first.
                            let out_shape = if let Some(aid) = act_id {
                                graph.shape(aid).clone()
                            } else {
                                add_node.shape.clone()
                            };

                            rw.ensure_mapped(&graph, &[node.inputs[0], node.inputs[1], bias_id]);
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

// ── Pass 2b: Add(residual) + RmsNorm → FusedResidualRmsNorm ─────────────

/// Fuses `add(x, residual) → rms_norm` into [`Op::FusedResidualRmsNorm`].
pub struct FuseResidualRmsNorm;

impl Pass for FuseResidualRmsNorm {
    fn name(&self) -> &str {
        "fuse_residual_rms_norm"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut is_output: HashMap<NodeId, ()> = HashMap::new();
        for &oid in &graph.outputs {
            is_output.insert(oid, ());
        }
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for node in graph.nodes() {
            if let Op::RmsNorm { .. } = &node.op {
                let rn_input_id = node.inputs[0];
                let rn_input = graph.node(rn_input_id);
                if matches!(rn_input.op, Op::Binary(BinaryOp::Add))
                    && graph.use_count(rn_input_id) == 1
                    && !is_output.contains_key(&rn_input_id)
                {
                    fused_away.insert(rn_input_id, ());
                }
            }
        }

        let mut rw = Rewriter::new(&graph.name);

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if let Op::RmsNorm { eps, .. } = &node.op {
                let rn_input_id = node.inputs[0];
                let rn_input = graph.node(rn_input_id);

                if matches!(rn_input.op, Op::Binary(BinaryOp::Add))
                    && fused_away.contains_key(&rn_input_id)
                {
                    let (x_id, residual_id) = (rn_input.inputs[0], rn_input.inputs[1]);
                    let gamma_id = node.inputs[1];
                    let beta_id = node.inputs[2];

                    let fused_id = rw.add_fused(
                        Op::FusedResidualRmsNorm {
                            has_bias: false,
                            eps: *eps,
                        },
                        &[x_id, residual_id, gamma_id, beta_id],
                        node.shape.clone(),
                    );

                    rw.replace(rn_input_id, fused_id);
                    rw.replace(node.id, fused_id);
                    continue;
                }
            }

            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

// ── Pass 2c: RmsNorm → Reshape(leading flatten) ─────────────────────────

/// Fuses `rms_norm([…, H]) → reshape([∏leading, H])` into a single
/// `RmsNorm` with the flattened output shape, eliminating a memcpy.
///
/// Matches the Qwen3.5 pre-norm pattern where normalized activations
/// are immediately reshaped to 2-D for matmul.
pub struct FuseRmsNormReshape;

fn leading_flatten_shape(in_shape: &Shape, new_shape: &[i64]) -> Option<Shape> {
    rlx_ir::shape::leading_flatten_shape(in_shape, new_shape)
}

fn sole_consumer(graph: &Graph, id: NodeId) -> Option<NodeId> {
    graph
        .nodes()
        .iter()
        .find(|n| n.inputs.contains(&id))
        .map(|n| n.id)
}

impl Pass for FuseRmsNormReshape {
    fn name(&self) -> &str {
        "fuse_rms_norm_reshape"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut is_output: HashMap<NodeId, ()> = HashMap::new();
        for &oid in &graph.outputs {
            is_output.insert(oid, ());
        }

        let mut flat_shape: HashMap<NodeId, Shape> = HashMap::new();
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for node in graph.nodes() {
            if let Op::RmsNorm { .. } = &node.op {
                if graph.use_count(node.id) != 1 || is_output.contains_key(&node.id) {
                    continue;
                }
                let Some(reshape_id) = sole_consumer(&graph, node.id) else {
                    continue;
                };
                if is_output.contains_key(&reshape_id) {
                    continue;
                }
                let reshape = graph.node(reshape_id);
                if let Op::Reshape { new_shape } = &reshape.op {
                    if let Some(flat) = leading_flatten_shape(&node.shape, new_shape) {
                        flat_shape.insert(node.id, flat);
                        fused_away.insert(reshape_id, ());
                    }
                }
            }
        }

        let mut rw = Rewriter::new(&graph.name);

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if let Op::RmsNorm { axis, eps, .. } = &node.op {
                if let Some(flat) = flat_shape.get(&node.id) {
                    let Some(reshape_id) = sole_consumer(&graph, node.id) else {
                        rw.copy_node(node);
                        continue;
                    };
                    let fused_id = rw.add_fused(
                        Op::RmsNorm {
                            axis: *axis,
                            eps: *eps,
                        },
                        &node.inputs,
                        flat.clone(),
                    );
                    rw.replace(node.id, fused_id);
                    rw.replace(reshape_id, fused_id);
                    continue;
                }
            }

            rw.copy_node(node);
        }

        rw.finish(&graph.outputs)
    }
}

// ── Pass 3b: Dual MatMul SwiGLU (gate+up before shared-input concat) ─────

/// Fuses the common LLM FFN pattern in one rewrite:
///   gate = matmul(x, wg); up = matmul(x, wu); out = mul(silu(gate), up)
///
/// Becomes:
///   cat = matmul(x, concat(wu, wg))   // up weights first for kernel layout
///   out = fused_swiglu(cat)
///
/// Eliminates two `[..., N]` matmul outputs plus a silu buffer — the
/// largest memory win on transformer FFN blocks.
pub struct FuseSwiGLUDualMatmul;

impl FuseSwiGLUDualMatmul {
    fn match_dual_swiglu(
        graph: &Graph,
        mul_node: &Node,
    ) -> Option<(NodeId, NodeId, NodeId, NodeId, NodeId)> {
        if !matches!(mul_node.op, Op::Binary(BinaryOp::Mul)) {
            return None;
        }
        let lhs = graph.node(mul_node.inputs[0]);
        let rhs = graph.node(mul_node.inputs[1]);
        let (up_mm, silu_id, silu_node) = if matches!(rhs.op, Op::Activation(Activation::Silu)) {
            (lhs, mul_node.inputs[1], rhs)
        } else if matches!(lhs.op, Op::Activation(Activation::Silu)) {
            (rhs, mul_node.inputs[0], lhs)
        } else {
            return None;
        };
        if !matches!(up_mm.op, Op::MatMul) {
            return None;
        }
        let gate_mm = graph.node(silu_node.inputs[0]);
        if !matches!(gate_mm.op, Op::MatMul) {
            return None;
        }
        if up_mm.inputs[0] != gate_mm.inputs[0] {
            return None;
        }
        if graph.use_count(silu_id) != 1 {
            return None;
        }
        Some((mul_node.id, gate_mm.id, up_mm.id, up_mm.inputs[0], silu_id))
    }
}

impl Pass for FuseSwiGLUDualMatmul {
    fn name(&self) -> &str {
        "fuse_swiglu_dual_matmul"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut matches: Vec<(NodeId, NodeId, NodeId, NodeId, NodeId)> = Vec::new();
        let mut consumed: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            if let Some((mul_id, gate_mm, up_mm, _, silu_id)) =
                Self::match_dual_swiglu(&graph, node)
            {
                matches.push((mul_id, gate_mm, up_mm, graph.node(up_mm).inputs[0], silu_id));
                consumed.insert(gate_mm, ());
                consumed.insert(up_mm, ());
                consumed.insert(silu_id, ());
            }
        }

        if matches.is_empty() {
            return graph;
        }

        let match_by_mul: HashMap<NodeId, (NodeId, NodeId, NodeId)> = matches
            .into_iter()
            .map(|(mul, gate, up, input, _silu)| (mul, (gate, up, input)))
            .collect();

        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if consumed.contains_key(&node.id) {
                continue;
            }
            if let Some(&(gate_mm, up_mm, input_id)) = match_by_mul.get(&node.id) {
                let gate = graph.node(gate_mm);
                let up = graph.node(up_mm);
                let wg = gate.inputs[1];
                let wu = up.inputs[1];
                rw.ensure_mapped(&graph, &[input_id, wg, wu]);

                let wu_shape = graph.shape(wu);
                let wg_shape = graph.shape(wg);
                let k = wu_shape.dim(0).unwrap_static();
                let n_up = wu_shape.dim(1).unwrap_static();
                let n_gate = wg_shape.dim(1).unwrap_static();
                debug_assert_eq!(wu_shape.dim(0), wg_shape.dim(0));

                // Up weights first → canonical FusedSwiGLU layout (gate_first=false).
                let concat_shape = Shape::new(&[k, n_up + n_gate], wu_shape.dtype());
                let concat_w = rw.add_fused(Op::Concat { axis: 1 }, &[wu, wg], concat_shape);

                let out_rank = up.shape.rank();
                let mut mm_dims: Vec<Dim> = (0..out_rank).map(|i| up.shape.dim(i)).collect();
                mm_dims[out_rank - 1] = Dim::Static(n_up + n_gate);
                let cat_shape = Shape::from_dims(&mm_dims, up.shape.dtype());
                let cat_id =
                    rw.new_graph
                        .add_node(Op::MatMul, vec![rw.map(input_id), concat_w], cat_shape);

                let fused_id = rw.new_graph.add_node(
                    Op::FusedSwiGLU {
                        cast_to: None,
                        gate_first: false,
                    },
                    vec![cat_id],
                    node.shape.clone(),
                );
                rw.replace(node.id, fused_id);
                continue;
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
        struct FuseGroup {
            input_id: NodeId,
            matmul_ids: Vec<NodeId>,
        }

        let mut input_to_matmuls: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for node in graph.nodes() {
            if matches!(node.op, Op::MatMul) {
                input_to_matmuls
                    .entry(node.inputs[0])
                    .or_default()
                    .push(node.id);
            }
        }

        let mut groups: Vec<FuseGroup> = Vec::new();
        for (input_id, matmul_ids) in input_to_matmuls {
            if matmul_ids.len() < 2 {
                continue;
            }
            let first = graph.node(matmul_ids[0]);
            let w0 = graph.shape(first.inputs[1]);
            if w0.rank() != 2 {
                continue;
            }
            let compatible = matmul_ids.iter().all(|&id| {
                let m = graph.node(id);
                matches!(m.op, Op::MatMul)
                    && graph.shape(m.inputs[1]).rank() == 2
                    && graph.shape(m.inputs[1]).dim(0) == w0.dim(0)
            });
            if compatible {
                groups.push(FuseGroup {
                    input_id,
                    matmul_ids,
                });
            }
        }

        if groups.is_empty() {
            return graph;
        }

        let group_by_first: HashMap<NodeId, &FuseGroup> =
            groups.iter().map(|g| (g.matmul_ids[0], g)).collect();

        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for g in &groups {
            for &id in &g.matmul_ids[1..] {
                fused_away.insert(id, ());
            }
        }

        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if let Some(group) = group_by_first.get(&node.id) {
                let matmuls: Vec<_> = group.matmul_ids.iter().map(|&id| graph.node(id)).collect();
                let weight_ids: Vec<NodeId> = matmuls.iter().map(|m| m.inputs[1]).collect();
                rw.ensure_mapped(&graph, std::slice::from_ref(&group.input_id));
                rw.ensure_mapped(&graph, &weight_ids);

                let w0_shape = graph.shape(weight_ids[0]);
                let k = w0_shape.dim(0).unwrap_static();
                let ns: Vec<usize> = weight_ids
                    .iter()
                    .map(|&w| graph.shape(w).dim(1).unwrap_static())
                    .collect();
                let combined_n: usize = ns.iter().sum();

                let concat_shape = Shape::new(&[k, combined_n], w0_shape.dtype());
                let concat_id = rw.add_fused(Op::Concat { axis: 1 }, &weight_ids, concat_shape);

                let out_rank = matmuls[0].shape.rank();
                let mut mm_dims: Vec<Dim> =
                    (0..out_rank).map(|i| matmuls[0].shape.dim(i)).collect();
                mm_dims[out_rank - 1] = Dim::Static(combined_n);
                let mm_shape = Shape::from_dims(&mm_dims, matmuls[0].shape.dtype());
                let mm_id = rw.new_graph.add_node(
                    Op::MatMul,
                    vec![rw.map(group.input_id), concat_id],
                    mm_shape,
                );

                let mut start = 0usize;
                for (mm, &n) in matmuls.iter().zip(&ns) {
                    let narrow = rw.new_graph.add_node(
                        Op::Narrow {
                            axis: out_rank - 1,
                            start,
                            len: n,
                        },
                        vec![mm_id],
                        mm.shape.clone(),
                    );
                    rw.replace(mm.id, narrow);
                    start += n;
                }
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
            mul_id: NodeId,
            up_narrow_id: NodeId,
            silu_id: NodeId,
            gate_narrow_id: NodeId,
            cat_id: NodeId,
            out_n: usize,
            gate_first: bool,
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
            let n = up_len;
            // Canonical: up @ 0, gate @ N. Swapped (gate-first builders): gate @ 0, up @ N.
            let gate_first = up_start == n && g_start == 0;
            if !(gate_first || (up_start == 0 && g_start == n)) {
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
                gate_first,
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
                let fused_id = rw.add_fused(
                    Op::FusedSwiGLU {
                        cast_to: None,
                        gate_first: m.gate_first,
                    },
                    &[m.cat_id],
                    out_shape,
                );
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
        let threshold: usize = rlx_ir::env::var("RLX_FUSE_ATTN_THRESHOLD")
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
//
// Fusion boundaries: chains do not extend across inputs whose producer
// satisfies [`rlx_ir::Op::is_fusion_boundary`] (BLAS, Gaussian splat, …).

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

/// Unfuse only `ElementwiseRegion` nodes that exceed [`crate::limits::FusionLimits`].
///
/// Run after [`MarkElementwiseRegions`] when marking may still produce
/// oversized chains (e.g. limits tightened per backend).
pub fn clip_elementwise_regions(graph: Graph, limits: crate::limits::FusionLimits) -> Graph {
    let oversize = |n: &rlx_ir::Node| -> bool {
        matches!(
            &n.op,
            Op::ElementwiseRegion {
                chain,
                num_inputs,
                ..
            } if *num_inputs > limits.max_elementwise_inputs
                || chain.len() as u32 > limits.max_elementwise_steps
        )
    };
    if !graph.nodes().iter().any(oversize) {
        return graph;
    }

    let mut rw = Rewriter::new(&graph.name);
    for node in graph.nodes() {
        if !oversize(node) {
            rw.copy_node(node);
            continue;
        }

        let Op::ElementwiseRegion {
            chain,
            num_inputs: _,
            scalar_input_mask: _,
            input_modulus: _,
        } = &node.op
        else {
            unreachable!();
        };

        let region_inputs: Vec<NodeId> = node.inputs.iter().map(|id| rw.map(*id)).collect();
        let mut step_ids: Vec<NodeId> = Vec::with_capacity(chain.len());
        let region_shape = node.shape.clone();
        let region_dims: Vec<_> = region_shape.dims().to_vec();
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
        let shape_of =
            |op: &ChainOperand, ins: &[NodeId], step_ids: &[NodeId], rw: &Rewriter| -> Shape {
                match *op {
                    ChainOperand::Input(i) => rw.new_graph.node(ins[i as usize]).shape.clone(),
                    ChainOperand::Step(i) => rw.new_graph.node(step_ids[i as usize]).shape.clone(),
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
                    let l_shape = shape_of(lhs, &region_inputs, &step_ids, &rw);
                    let r_shape = shape_of(rhs, &region_inputs, &step_ids, &rw);
                    let bcast = l_shape
                        .broadcast_with(&r_shape)
                        .unwrap_or_else(|e| panic!("clip_elementwise_regions: {e}"));
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
                    let l_shape = shape_of(lhs, &region_inputs, &step_ids, &rw);
                    let r_shape = shape_of(rhs, &region_inputs, &step_ids, &rw);
                    let bcast = l_shape
                        .broadcast_with(&r_shape)
                        .unwrap_or_else(|e| panic!("clip_elementwise_regions: {e}"));
                    let dims: Vec<_> = bcast.dims().to_vec();
                    let shape = Shape::from_dims(&dims, rlx_ir::DType::U8);
                    (
                        rw.new_graph.add_node(Op::Compare(*op), vec![l, r], shape),
                        rlx_ir::DType::U8,
                    )
                }
                ChainStep::Where(cond, x, y) => {
                    let cn = resolve(cond);
                    let xn = resolve(x);
                    let yn = resolve(y);
                    let dt = dtype_of(x, &region_inputs, &step_dtypes, &rw);
                    let x_shape = shape_of(x, &region_inputs, &step_ids, &rw);
                    let y_shape = shape_of(y, &region_inputs, &step_ids, &rw);
                    let c_shape = shape_of(cond, &region_inputs, &step_ids, &rw);
                    let bcast_xy = x_shape
                        .broadcast_with(&y_shape)
                        .unwrap_or_else(|e| panic!("clip_elementwise_regions: {e}"));
                    let bcast = c_shape.broadcast_with(&bcast_xy).unwrap_or_else(|e| {
                        panic!("clip_elementwise_regions: cannot broadcast cond {c_shape:?} ⊗ {bcast_xy:?} for Where: {e}")
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
        let _ = (region_dtype, region_dims);
        let last = *step_ids
            .last()
            .expect("oversize region has non-empty chain");
        rw.replace(node.id, last);
    }
    rw.finish(&graph.outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::FusionLimits;
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
    fn fuse_matmul_bias_skips_unsupported_activation_epilogue() {
        let mut g = Graph::new("test");
        let x = g.input("x", f32_shape(&[8, 1024]));
        let w = g.param("w", f32_shape(&[1024, 16]));
        let b = g.param("b", f32_shape(&[16]));
        let mm = g.matmul(x, w, f32_shape(&[8, 16]));
        let add = g.binary(BinaryOp::Add, mm, b, f32_shape(&[8, 16]));
        let exp = g.activation(Activation::Exp, add, f32_shape(&[8, 16]));
        g.set_outputs(vec![exp]);

        let fused = FuseMatMulBiasAct.run(g);
        // mm + bias fuse; Exp stays separate (qwen35 softplus pattern).
        assert_eq!(fused.len(), 5);
        let out_node = fused.node(fused.outputs[0]);
        assert!(matches!(out_node.op, Op::Activation(Activation::Exp)));
        let add_node = fused.node(out_node.inputs[0]);
        assert!(matches!(
            add_node.op,
            Op::FusedMatMulBiasAct { activation: None }
        ));
    }

    #[test]
    fn fuse_matmul_bias_act_with_late_bias_param() {
        use rlx_ir::infer::GraphExt;

        let mut g = Graph::new("late_bias");
        let x = g.input("x", f32_shape(&[8, 16]));
        let w = g.param("w", f32_shape(&[16, 32]));
        let out = {
            let mm = g.mm(x, w);
            let b = g.param("b", f32_shape(&[32]));
            let biased = g.add(mm, b);
            g.gelu(biased)
        };
        g.set_outputs(vec![out]);

        let fused = FuseMatMulBiasAct.run(g);
        assert!(
            fused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedMatMulBiasAct { .. })),
            "bias param declared after matmul must still fuse:\n{fused}"
        );
    }

    #[test]
    fn swiglu_ffn_builder_fuses_end_to_end() {
        let mut g = Graph::new("swiglu_block");
        let x = g.input("x", f32_shape(&[4, 768]));
        let up_w = g.param("up", f32_shape(&[768, 2048]));
        let gate_w = g.param("gate", f32_shape(&[768, 2048]));
        let down_w = g.param("down", f32_shape(&[2048, 768]));
        let out = g.swiglu_ffn(x, up_w, gate_w, down_w);
        g.set_outputs(vec![out]);

        let g = FuseSharedInputMatMul.run(g);
        let g = FuseSwiGLU.run(g);
        assert!(
            g.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedSwiGLU { .. })),
            "swiglu_ffn builder should match FuseSwiGLU:\n{g}"
        );
    }

    #[test]
    fn fuse_swiglu_dual_matmul_gate_first() {
        use rlx_ir::infer::GraphExt;

        let mut g = Graph::new("qwen3_ffn");
        let x = g.input("x", f32_shape(&[4, 768]));
        let gate_w = g.param("gate", f32_shape(&[768, 2048]));
        let up_w = g.param("up", f32_shape(&[768, 2048]));
        let gate = g.mm(x, gate_w);
        let up = g.mm(x, up_w);
        let gate_act = g.silu(gate);
        let out = g.mul(gate_act, up);
        g.set_outputs(vec![out]);

        let fused = FuseSwiGLUDualMatmul.run(g);
        assert!(
            fused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedSwiGLU { .. })),
            "gate-first dual matmul should fuse:\n{fused}"
        );
        assert!(
            fused.len() <= 6,
            "dual fusion should collapse to x + weights + concat + mm + fused_swiglu, got {} nodes",
            fused.len()
        );
    }

    #[test]
    fn fuse_shared_input_matmul_three_way_qkv() {
        let mut g = Graph::new("qkv");
        let x = g.input("x", f32_shape(&[8, 512]));
        let wq = g.param("wq", f32_shape(&[512, 128]));
        let wk = g.param("wk", f32_shape(&[512, 128]));
        let wv = g.param("wv", f32_shape(&[512, 128]));
        let q = g.matmul(x, wq, f32_shape(&[8, 128]));
        let k = g.matmul(x, wk, f32_shape(&[8, 128]));
        let v = g.matmul(x, wv, f32_shape(&[8, 128]));
        g.set_outputs(vec![q, k, v]);

        let fused = FuseSharedInputMatMul.run(g);
        assert_eq!(
            fused.len(),
            9,
            "x + 3 weights + concat + mm + 3 narrows = 9"
        );
        for &out in &fused.outputs {
            assert!(matches!(fused.node(out).op, Op::Narrow { .. }));
        }
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
    fn fuse_residual_rms_norm() {
        let mut g = Graph::new("test");
        let x = g.input("x", f32_shape(&[4, 15, 384]));
        let residual = g.input("residual", f32_shape(&[4, 15, 384]));
        let gamma = g.param("gamma", f32_shape(&[384]));
        let beta = g.param("beta", f32_shape(&[384]));
        let add = g.binary(BinaryOp::Add, x, residual, f32_shape(&[4, 15, 384]));
        let rn = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![add, gamma, beta],
            f32_shape(&[4, 15, 384]),
        );
        g.set_outputs(vec![rn]);

        assert_eq!(g.len(), 6);

        let fused = FuseResidualRmsNorm.run(g);
        assert_eq!(fused.len(), 5);
        let out_node = fused.node(fused.outputs[0]);
        assert!(matches!(
            out_node.op,
            Op::FusedResidualRmsNorm {
                has_bias: false,
                ..
            }
        ));
    }

    #[test]
    fn fuse_rms_norm_reshape() {
        let mut g = Graph::new("test");
        let x = g.input("x", f32_shape(&[1, 8, 512]));
        let gamma = g.param("gamma", f32_shape(&[512]));
        let beta = g.param("beta", f32_shape(&[512]));
        let rn = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![x, gamma, beta],
            f32_shape(&[1, 8, 512]),
        );
        let flat = g.add_node(
            Op::Reshape {
                new_shape: vec![8, 512],
            },
            vec![rn],
            f32_shape(&[8, 512]),
        );
        let w = g.param("w", f32_shape(&[512, 128]));
        let mm = g.matmul(flat, w, f32_shape(&[8, 128]));
        g.set_outputs(vec![mm]);

        let fused = FuseRmsNormReshape.run(g);
        // x, gamma, beta, rms_norm(2d), w, matmul — no separate reshape
        assert_eq!(fused.len(), 6);
        let rn_node = fused.node(fused.node(fused.outputs[0]).inputs[0]);
        assert!(matches!(rn_node.op, Op::RmsNorm { .. }));
        assert_eq!(rn_node.shape.dim(0).unwrap_static(), 8);
        assert_eq!(rn_node.shape.dim(1).unwrap_static(), 512);
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

    /// Regression: `FuseSharedInputMatMul` used to panic when `w2` is
    /// declared after `mm1`. `ensure_mapped` now copies late operands.
    #[test]
    fn fuse_shared_input_matmul_with_late_w2_param() {
        let mut g = Graph::new("late_w2");
        let x = g.input("x", f32_shape(&[8, 16]));
        let w1 = g.param("w1", f32_shape(&[16, 8]));
        let mm1 = g.matmul(x, w1, f32_shape(&[8, 8]));
        let w2 = g.param("w2", f32_shape(&[16, 8]));
        let mm2 = g.matmul(x, w2, f32_shape(&[8, 8]));
        g.set_outputs(vec![mm1, mm2]);

        let fused = FuseSharedInputMatMul.run(g);
        for &out in &fused.outputs {
            assert!(
                matches!(fused.node(out).op, Op::Narrow { .. }),
                "late w2 should still fuse via ensure_mapped, got {:?}",
                fused.node(out).op
            );
        }
    }

    /// Regression: qwen35moe FFN declares router / shared-expert matmuls on the
    /// same flattened hidden state with weights scattered through the block.
    #[test]
    fn fuse_shared_input_matmul_moe_ffn_pattern() {
        let mut g = Graph::new("moe_ffn");
        let rows = 4usize;
        let n_embd = 16usize;
        let n_expert = 4usize;
        let n_ff = 16usize;

        let h_in = g.input("h", f32_shape(&[1, rows, n_embd]));
        let h_2d = g.reshape_(h_in, vec![rows as i64, n_embd as i64]);

        let router_w = g.param("router_w", f32_shape(&[n_embd, n_expert]));
        let router_logits = g.matmul(h_2d, router_w, f32_shape(&[rows, n_expert]));

        // MoE body omitted — only the shared-expert tail matters for fusion order.
        let shared_router_w = g.param("shared_router_w", f32_shape(&[n_embd, 1]));
        let shared_logits = g.matmul(h_2d, shared_router_w, f32_shape(&[rows, 1]));
        let shared_gate = g.activation(Activation::Sigmoid, shared_logits, f32_shape(&[rows, 1]));

        let s_gate_w = g.param("s_gate_w", f32_shape(&[n_embd, n_ff]));
        let s_up_w = g.param("s_up_w", f32_shape(&[n_embd, n_ff]));
        let s_gate = g.matmul(h_2d, s_gate_w, f32_shape(&[rows, n_ff]));
        let s_up = g.matmul(h_2d, s_up_w, f32_shape(&[rows, n_ff]));
        let s_gate_silu = g.silu(s_gate);
        let s_swiglu = g.mul(s_gate_silu, s_up);

        g.set_outputs(vec![router_logits, shared_gate, s_swiglu]);

        let fused = FuseSharedInputMatMul.run(g);
        let narrow_count = fused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Narrow { .. }))
            .count();
        assert!(
            narrow_count >= 4,
            "expected four narrow slices from fused h_2d matmuls, got {narrow_count}"
        );
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
            matches!(
                out_node.op,
                Op::FusedSwiGLU {
                    cast_to: None,
                    gate_first: false
                }
            ),
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
    fn clip_unfuses_region_over_step_cap() {
        use rlx_ir::op::{Activation, ChainOperand, ChainStep};

        let mut g = Graph::new("clip");
        let x = g.input("x", f32_shape(&[4]));
        let mut chain: Vec<ChainStep> = Vec::new();
        let mut prev = ChainOperand::Input(0);
        for _ in 0..40 {
            chain.push(ChainStep::Activation(Activation::Relu, prev));
            prev = ChainOperand::Step(chain.len() as u32 - 1);
        }
        let y = g.add_node(
            Op::ElementwiseRegion {
                chain,
                num_inputs: 1,
                scalar_input_mask: 0,
                input_modulus: [0; 16],
            },
            vec![x],
            f32_shape(&[4]),
        );
        g.set_outputs(vec![y]);

        let clipped = clip_elementwise_regions(g, FusionLimits::GPU_NATIVE);
        assert!(
            !clipped
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::ElementwiseRegion { .. })),
            "oversized region should be decomposed"
        );
        assert!(clipped.len() > 5);
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
