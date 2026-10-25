// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Run a graph as if every value were stored in a narrower float format.
//!
//! [`emulate_float_format`] inserts `Op::RoundToFormat` after the weights and
//! after every compute op that produces a float tensor, so every value a
//! consumer reads is representable in the format. Arithmetic inside an op still
//! accumulates in `f32`, as FP8/FP4 tensor cores do.
//!
//! Skipped on purpose:
//!
//! - graph inputs and constants (unless [`EmulateOptions::round_inputs`] /
//!   [`EmulateOptions::round_constants`]): they are host data crossing the
//!   boundary — attention masks holding `-inf` / `-1e9`, KV caches, positions.
//!   Rounded, a block-scaled `-inf` mask becomes `-6` in `f4` and attention
//!   leaks; values *computed* from weights (embeddings, K/V) are still rounded;
//! - pure data movement (reshape, transpose, slicing, concat, gather …): its
//!   values were already rounded where they were produced;
//! - non-float tensors (indices, masks);
//! - graph outputs, which keep the unrounded `f32` value so callers (an LM's
//!   sampler, a forecaster's quantile map) see full precision;
//! - for formats of 8 bits or fewer, params with fewer than two dimensions
//!   (norm scales, biases, scalars; [`rounds_param_shape`]). Low-bit
//!   deployments quantize the matrices and keep these in high precision: they
//!   are a negligible share of the bytes, and per-channel gains spanning more
//!   binades than a 6-bit block covers flush to zero (Gemma 3's norm weights
//!   took its f6 perplexity from 15 to 500). Values computed from params and
//!   constants alone, like Gemma's `1 + w` norm gain, count as params here;
//! - position tables: params read only as a `Rope` op's cos/sin tables
//!   ([`position_table_params`]). They are precomputed from positions, not
//!   learned, and rounded to a few bits they scramble every position;
//! - anything matched by [`EmulateOptions::keep_ops`] / [`EmulateOptions::keep_params`].
//!
//! The inserted op has a straight-through gradient, so the same rewrite turns a
//! forward graph into a quantization-aware-training graph.

use std::collections::HashMap;

use rlx_ir::op::{Activation, ReduceOp};
use rlx_ir::*;

/// What [`emulate_float_format`] leaves alone.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmulateOptions {
    /// Also round graph inputs (off by default; see the module docs).
    pub round_inputs: bool,
    /// Also round `Op::Constant`s (off by default; see the module docs).
    pub round_constants: bool,
    /// Also round values consumed only by a softmax (attention scores and the
    /// mask added to them). Off by default: low-precision attention kernels
    /// keep `Q·Kᵀ`, the mask and the softmax in f32, and rounding them is
    /// destructive under block scaling — a block holding a `-1e9` mask entry
    /// gets a scale near `1e9`, which flushes every real score in it to zero.
    pub round_softmax_inputs: bool,
    /// Round weights (`Op::Param`) inside the graph on every run. Off by
    /// default: the runtime rounds them once as they are uploaded
    /// (`CompiledGraph::set_param`), which is exact and does not re-read every
    /// weight per call. Training graphs whose weights never pass through the
    /// host (an in-graph optimizer) need it on.
    pub params_in_graph: bool,
    /// Weights (`Op::Param`) are already rounded on the host; don't round
    /// them again on every run. Inference callers that round once at load
    /// should set this — rounding in-graph re-reads every weight per call.
    pub params_prerounded: bool,
    /// `Op::Param` names whose values stay in `f32` (and whose consumers'
    /// outputs are still rounded). Matched by substring.
    pub keep_params: Vec<String>,
    /// Op kinds whose outputs are not rounded.
    pub keep_ops: Vec<OpKind>,
    /// Round the residual stream: values read by a normalization (`RmsNorm`,
    /// `LayerNorm`, …), which in a pre-norm network are the running sum every
    /// block adds into. `None` rounds it for formats wider than 8 bits and
    /// keeps it in f32 for narrower ones, as FP8/FP4 inference keeps the
    /// residual in high precision and quantizes what the matmuls read:
    /// rounded to 4 bits after every block add, the error compounds through
    /// the depth of the network.
    pub round_residual: Option<bool>,
}

impl EmulateOptions {
    fn keeps_param(&self, name: &str) -> bool {
        self.keep_params.iter().any(|k| name.contains(k.as_str()))
    }
}

/// Ops that only move or select values already in the format.
fn is_data_movement(op: &Op) -> bool {
    matches!(
        op.kind(),
        OpKind::Reshape
            | OpKind::Transpose
            | OpKind::Narrow
            | OpKind::Concat
            | OpKind::Gather
            | OpKind::Expand
            | OpKind::StopGradient
            | OpKind::RoundToFormat
    )
}

/// Insert `RoundToFormat` after every float-producing node of `graph`.
pub fn emulate_float_format(
    graph: Graph,
    format: FloatFormat,
    scaling: FormatScaling,
    opts: &EmulateOptions,
) -> Graph {
    let f = Some((format, scaling));
    emulate_number_format(graph, f, f, opts)
}

/// [`emulate_float_format`] with weights and intermediates in their own
/// formats (`NumberFormat::Mixed`); `None` leaves that side in f32. Weights
/// here means params rounded in-graph (`params_in_graph`) and values computed
/// from params alone; params rounded on upload never reach this pass.
pub fn emulate_number_format(
    graph: Graph,
    weights: Option<(FloatFormat, FormatScaling)>,
    activations: Option<(FloatFormat, FormatScaling)>,
    opts: &EmulateOptions,
) -> Graph {
    let softmax_feed = if opts.round_softmax_inputs {
        std::collections::HashSet::new()
    } else {
        softmax_feeds(&graph)
    };
    let tables = position_table_params(&graph);
    // Computed from params and constants alone: a weight in all but name.
    let mut weight_derived: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    for n in graph.nodes() {
        let derived = match &n.op {
            Op::Param { .. } | Op::Constant { .. } => true,
            Op::Input { .. } => false,
            _ => !n.inputs.is_empty() && n.inputs.iter().all(|i| weight_derived.contains(i)),
        };
        if derived {
            weight_derived.insert(n.id);
        }
    }
    let narrow = activations.is_some_and(|(f, _)| 1 + f.exp_bits as u32 + f.man_bits as u32 <= 8);
    let residual = if opts.round_residual.unwrap_or(!narrow) {
        std::collections::HashSet::new()
    } else {
        norm_inputs(&graph)
    };
    // A weight of `shape` rounded under `weights`, if it is.
    let weight_fmt = |shape: &Shape| weights.filter(|&(f, _)| rounds_param_shape(f, shape));
    let mut out = Graph::new(&graph.name);
    // Old id → (id in `out`, the format consumers read it rounded to).
    let mut map: HashMap<NodeId, (NodeId, Option<(FloatFormat, FormatScaling)>)> = HashMap::new();
    // Rounded copies, made on first use so a value only the caller reads
    // (a graph output) never gets one.
    let mut rounded: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|i| match map[i] {
                (raw, Some((f, sc))) => *rounded
                    .entry(raw)
                    .or_insert_with(|| round_node(&mut out, raw, f, sc)),
                (raw, None) => raw,
            })
            .collect();
        let raw = out.add_node(node.op.clone(), inputs, node.shape.clone());
        let fmt = if !node.shape.dtype().is_float()
            || is_data_movement(&node.op)
            || softmax_feed.contains(&node.id)
            || residual.contains(&node.id)
            || opts.keep_ops.contains(&node.op.kind())
        {
            None
        } else {
            match &node.op {
                Op::Param { name } => {
                    if !opts.params_in_graph
                        || opts.params_prerounded
                        || opts.keeps_param(name)
                        || tables.contains(name)
                    {
                        None
                    } else {
                        weight_fmt(&node.shape)
                    }
                }
                Op::Input { .. } => activations.filter(|_| opts.round_inputs),
                Op::Constant { .. } => activations.filter(|_| opts.round_constants),
                _ if weight_derived.contains(&node.id) => weight_fmt(&node.shape),
                _ => activations,
            }
        };
        map.insert(node.id, (raw, fmt));
    }

    let outputs: Vec<NodeId> = graph.outputs.iter().map(|o| map[o].0).collect();
    out.set_outputs(outputs);
    out
}

/// Nodes whose every consumer is a `Softmax`, or another such node reached
/// through an elementwise or movement op (`scores · scale + mask → softmax`).
fn softmax_feeds(graph: &Graph) -> std::collections::HashSet<NodeId> {
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for n in graph.nodes() {
        for &i in &n.inputs {
            consumers.entry(i).or_default().push(n.id);
        }
    }
    let outputs: std::collections::HashSet<NodeId> = graph.outputs.iter().copied().collect();
    let mut feeds = std::collections::HashSet::new();
    for n in graph.nodes().iter().rev() {
        let Some(users) = consumers.get(&n.id) else {
            continue;
        };
        if outputs.contains(&n.id) {
            continue;
        }
        let all_feed = users.iter().all(|&u| {
            let op = &graph.node(u).op;
            matches!(op, Op::Softmax { .. })
                || (feeds.contains(&u)
                    && (matches!(op, Op::Binary(_) | Op::Activation(_)) || is_data_movement(op)))
        });
        if all_feed {
            feeds.insert(n.id);
        }
    }
    feeds
}

/// Whether a weight of `shape` is rounded under `format`: always for formats
/// wider than 8 bits, otherwise only matrices and higher-rank tensors (see the
/// module docs).
pub fn rounds_param_shape(format: FloatFormat, shape: &Shape) -> bool {
    1 + format.exp_bits as u32 + format.man_bits as u32 > 8 || shape.rank() >= 2
}

/// Names of params read only as a `Rope` op's cos/sin tables, directly or
/// through data movement (a decode step gathers its row; a prefill narrows the
/// table to its length). They are position encodings, not weights.
pub fn position_table_params(graph: &Graph) -> std::collections::HashSet<String> {
    let mut consumers: HashMap<NodeId, Vec<(NodeId, usize)>> = HashMap::new();
    for n in graph.nodes() {
        for (k, &i) in n.inputs.iter().enumerate() {
            consumers.entry(i).or_default().push((n.id, k));
        }
    }
    let outputs: std::collections::HashSet<NodeId> = graph.outputs.iter().copied().collect();
    // Whether every use of `id` ends as a Rope table operand.
    fn only_tables(
        graph: &Graph,
        consumers: &HashMap<NodeId, Vec<(NodeId, usize)>>,
        outputs: &std::collections::HashSet<NodeId>,
        id: NodeId,
        depth: usize,
    ) -> bool {
        let Some(users) = consumers.get(&id) else {
            return false;
        };
        if outputs.contains(&id) || depth > 8 {
            return false;
        }
        users.iter().all(|&(u, k)| match &graph.node(u).op {
            Op::Rope { .. } => k >= 1,
            // A gathered row of the table, not the indices.
            Op::Gather { .. } if k != 0 => false,
            op if is_data_movement(op) => only_tables(graph, consumers, outputs, u, depth + 1),
            _ => false,
        })
    }
    graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } if only_tables(graph, &consumers, &outputs, n.id, 0) => {
                Some(name.clone())
            }
            _ => None,
        })
        .collect()
}

/// Nodes read by a normalization op (looking through data movement): the
/// residual stream of a pre-norm network.
fn norm_inputs(graph: &Graph) -> std::collections::HashSet<NodeId> {
    graph
        .nodes()
        .iter()
        .filter(|n| {
            matches!(
                n.op.kind(),
                OpKind::RmsNorm
                    | OpKind::LayerNorm
                    | OpKind::LayerNorm2d
                    | OpKind::GroupNorm
                    | OpKind::BatchNormInference
            )
        })
        .filter_map(|n| {
            // Through reshapes and the like to the op that computed the value.
            let mut x = *n.inputs.first()?;
            while is_data_movement(&graph.node(x).op) {
                x = *graph.node(x).inputs.first()?;
            }
            Some(x)
        })
        .collect()
}

/// `RoundToFormat(x)`, adding the tensor-wide max magnitude for per-tensor scaling.
pub fn round_node(g: &mut Graph, x: NodeId, format: FloatFormat, scaling: FormatScaling) -> NodeId {
    let shape = g.shape(x).clone();
    let mut inputs = vec![x];
    if scaling == FormatScaling::PerTensor {
        let abs = g.activation(Activation::Abs, x, shape.clone());
        let axes: Vec<usize> = (0..shape.rank()).collect();
        let rshape = shape::reduce_shape(&shape, &axes, false).expect("reduce shape");
        inputs.push(g.reduce(abs, ReduceOp::Max, axes, false, rshape));
    }
    g.add_node(Op::RoundToFormat { format, scaling }, inputs, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_graph() -> Graph {
        let mut g = Graph::new("t");
        let x = g.add_node(
            Op::Input { name: "x".into() },
            vec![],
            Shape::new(&[2, 4], DType::F32),
        );
        let w = g.add_node(
            Op::Param { name: "w".into() },
            vec![],
            Shape::new(&[4, 3], DType::F32),
        );
        let y = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[2, 3], DType::F32));
        let t = g.add_node(
            Op::Transpose { perm: vec![1, 0] },
            vec![y],
            Shape::new(&[3, 2], DType::F32),
        );
        let z = g.activation(Activation::Silu, t, Shape::new(&[3, 2], DType::F32));
        g.set_outputs(vec![z]);
        g
    }

    fn count_rounds(g: &Graph) -> usize {
        g.nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::RoundToFormat { .. }))
            .count()
    }

    #[test]
    fn rounds_producers_not_movement_or_outputs() {
        let g = emulate_float_format(
            tiny_graph(),
            FloatFormat::F16,
            FormatScaling::None,
            &EmulateOptions::default(),
        );
        // Only the matmul: weights are rounded on upload by default, inputs
        // stay f32, the transpose is movement, the silu is the output.
        assert_eq!(count_rounds(&g), 1);
        let with_inputs = emulate_float_format(
            tiny_graph(),
            FloatFormat::F16,
            FormatScaling::None,
            &EmulateOptions {
                round_inputs: true,
                params_in_graph: true,
                ..Default::default()
            },
        );
        assert_eq!(count_rounds(&with_inputs), 3);
        let out = g.outputs[0];
        assert!(matches!(g.node(out).op, Op::Activation(Activation::Silu)));
    }

    #[test]
    fn softmax_inputs_stay_f32() {
        let mut g = Graph::new("attn");
        let s = Shape::new(&[2, 4], DType::F32);
        let x = g.input("x", s.clone());
        let mask = g.input("mask", s.clone());
        let w = g.add_node(
            Op::Param { name: "w".into() },
            vec![],
            Shape::new(&[4, 4], DType::F32),
        );
        let scores = g.add_node(Op::MatMul, vec![x, w], s.clone());
        let masked = g.binary(rlx_ir::op::BinaryOp::Add, scores, mask, s.clone());
        let p = g.add_node(Op::Softmax { axis: -1 }, vec![masked], s.clone());
        let y = g.add_node(Op::MatMul, vec![p, w], s.clone());
        g.set_outputs(vec![y]);
        let out = emulate_float_format(
            g.clone(),
            FloatFormat::F16,
            FormatScaling::None,
            &EmulateOptions::default(),
        );
        // Only the softmax output is rounded: scores and the masked scores feed it.
        assert_eq!(count_rounds(&out), 1);
        let all = emulate_float_format(
            g,
            FloatFormat::F16,
            FormatScaling::None,
            &EmulateOptions {
                round_softmax_inputs: true,
                ..Default::default()
            },
        );
        assert_eq!(count_rounds(&all), 3);
    }

    #[test]
    fn prerounded_params_and_per_tensor_scale() {
        let opts = EmulateOptions {
            params_prerounded: true,
            ..Default::default()
        };
        let f4 = FloatFormat::parse("f4").unwrap();
        let g = emulate_float_format(tiny_graph(), f4, FormatScaling::PerTensor, &opts);
        assert_eq!(count_rounds(&g), 1);
        for n in g.nodes() {
            if let Op::RoundToFormat { .. } = n.op {
                assert_eq!(n.inputs.len(), 2, "per-tensor scaling carries amax");
            }
        }
    }

    /// A pre-norm block: `h = x·w1; r = x + h; n = rms_norm(r, g, b);
    /// y = rope(n·w2, cos, sin)`, with `g' = g + 1` computed in the graph.
    fn pre_norm_block() -> Graph {
        let f = DType::F32;
        let mut g = Graph::new("block");
        let s = Shape::new(&[2, 4], f);
        let v = Shape::new(&[4], f);
        let x = g.input("x", s.clone());
        let w1 = g.param("w1", Shape::new(&[4, 4], f));
        let w2 = g.param("w2", Shape::new(&[4, 4], f));
        let gamma = g.param("norm.weight", v.clone());
        let one = g.add_node(
            Op::Constant {
                data: 1f32.to_le_bytes().repeat(4),
            },
            vec![],
            v.clone(),
        );
        let gain = g.binary(rlx_ir::op::BinaryOp::Add, gamma, one, v.clone());
        let beta = g.param("norm.bias", v.clone());
        let cos = g.param("rope.cos", Shape::new(&[2, 2], f));
        let sin = g.param("rope.sin", Shape::new(&[2, 2], f));
        let h = g.add_node(Op::MatMul, vec![x, w1], s.clone());
        let r = g.binary(rlx_ir::op::BinaryOp::Add, x, h, s.clone());
        let n = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![r, gain, beta],
            s.clone(),
        );
        let q = g.add_node(Op::MatMul, vec![n, w2], s.clone());
        let y = g.add_node(
            Op::Rope {
                head_dim: 4,
                n_rot: 4,
                style: Default::default(),
            },
            vec![q, cos, sin],
            s.clone(),
        );
        g.set_outputs(vec![y]);
        g
    }

    fn rounded_inputs_of(g: &Graph, kind: OpKind) -> Vec<bool> {
        let n = g.nodes().iter().find(|n| n.op.kind() == kind).unwrap();
        n.inputs
            .iter()
            .map(|&i| matches!(g.node(i).op, Op::RoundToFormat { .. }))
            .collect()
    }

    #[test]
    fn narrow_formats_keep_residual_vectors_and_position_tables() {
        let f8 = FloatFormat::parse("f8").unwrap();
        let opts = EmulateOptions {
            params_in_graph: true,
            ..Default::default()
        };
        let g = emulate_float_format(pre_norm_block(), f8, FormatScaling::Block(32), &opts);
        // The residual sum, the computed gain and the bias reach the norm unrounded.
        assert_eq!(
            rounded_inputs_of(&g, OpKind::RmsNorm),
            vec![false, false, false]
        );
        // The matmul reads the rounded norm output and weight; RoPE reads its
        // tables as they are.
        assert_eq!(
            rounded_inputs_of(&g, OpKind::Rope),
            vec![true, false, false]
        );
        assert_eq!(
            position_table_params(&pre_norm_block()),
            ["rope.cos", "rope.sin"].map(String::from).into()
        );
        // f16 rounds the residual and the vectors (not the tables).
        let g16 = emulate_float_format(
            pre_norm_block(),
            FloatFormat::F16,
            FormatScaling::None,
            &opts,
        );
        assert_eq!(
            rounded_inputs_of(&g16, OpKind::RmsNorm),
            vec![true, true, true]
        );
        assert_eq!(
            rounded_inputs_of(&g16, OpKind::Rope),
            vec![true, false, false]
        );
        // Forcing the residual on under f8.
        let forced = EmulateOptions {
            round_residual: Some(true),
            ..opts
        };
        let g = emulate_float_format(pre_norm_block(), f8, FormatScaling::Block(32), &forced);
        assert_eq!(
            rounded_inputs_of(&g, OpKind::RmsNorm),
            vec![true, false, false]
        );
    }
}
