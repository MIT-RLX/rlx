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

//! Fusion diagnostics — what fused, what missed, and why.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{Graph, NodeId, Op, node_label};
use std::fmt;

/// Why a recognizable fusion pattern was not collapsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissReason {
    MultiConsumer,
    NonAddBiasConsumer,
    BiasRankTooHigh { rank: usize },
    UnsupportedEpilogueActivation(Activation),
    SharedMatmulCount { count: usize },
    SwigluGateBeforeUp,
    SwigluNotSharedInput,
    NotFused,
}

/// A single fusion opportunity that remains in the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissedFusion {
    pub pattern: &'static str,
    pub node: NodeId,
    pub reason: MissReason,
    /// HIR label / node name when available.
    pub context: Option<String>,
    /// Actionable fix hint.
    pub hint: Option<String>,
}

/// Before/after fusion statistics and missed-pattern tally.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FusionReport {
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub matmul_before: usize,
    pub attention: usize,
    pub rope: usize,
    pub narrow: usize,
    pub matmul_after: usize,
    pub silu: usize,
    pub mul: usize,
    pub fused_matmul_bias_act: usize,
    pub fused_swiglu: usize,
    pub fused_residual_ln: usize,
    pub fused_residual_rms_norm: usize,
    pub fused_attention_block: usize,
    pub fused_transformer_layer: usize,
    pub elementwise_region: usize,
    pub missed: Vec<MissedFusion>,
}

impl FusionReport {
    /// Compare an unfused graph with the post-pass result.
    pub fn analyze(before: &Graph, after: &Graph) -> Self {
        let before_stats = count_ops(before);
        let after_stats = count_ops(after);
        let missed = scan_misses(after);
        Self {
            nodes_before: before.len(),
            nodes_after: after.len(),
            matmul_before: before_stats.matmul,
            attention: after_stats.attention,
            rope: after_stats.rope,
            narrow: after_stats.narrow,
            matmul_after: after_stats.matmul,
            silu: after_stats.silu,
            mul: after_stats.mul,
            fused_matmul_bias_act: after_stats.fused_matmul_bias_act,
            fused_swiglu: after_stats.fused_swiglu,
            fused_residual_ln: after_stats.fused_residual_ln,
            fused_residual_rms_norm: after_stats.fused_residual_rms_norm,
            fused_attention_block: after_stats.fused_attention_block,
            fused_transformer_layer: after_stats.fused_transformer_layer,
            elementwise_region: after_stats.elementwise_region,
            missed,
        }
    }

    /// Scan a graph (typically post-fusion) for patterns that should
    /// have collapsed but did not.
    pub fn scan(graph: &Graph) -> Self {
        let stats = count_ops(graph);
        let missed = scan_misses(graph);
        Self {
            nodes_before: graph.len(),
            nodes_after: graph.len(),
            matmul_before: stats.matmul,
            matmul_after: stats.matmul,
            attention: stats.attention,
            rope: stats.rope,
            narrow: stats.narrow,
            silu: stats.silu,
            mul: stats.mul,
            fused_matmul_bias_act: stats.fused_matmul_bias_act,
            fused_swiglu: stats.fused_swiglu,
            fused_residual_ln: stats.fused_residual_ln,
            fused_residual_rms_norm: stats.fused_residual_rms_norm,
            fused_attention_block: stats.fused_attention_block,
            fused_transformer_layer: stats.fused_transformer_layer,
            elementwise_region: stats.elementwise_region,
            missed,
        }
    }

    pub fn missed_matmul_bias_act(&self) -> usize {
        self.missed
            .iter()
            .filter(|m| m.pattern == "matmul_bias_act")
            .count()
    }

    pub fn missed_swiglu(&self) -> usize {
        self.missed.iter().filter(|m| m.pattern == "swiglu").count()
    }

    pub fn missed_shared_matmul(&self) -> usize {
        self.missed
            .iter()
            .filter(|m| m.pattern == "shared_input_matmul")
            .count()
    }

    /// One-line summary suitable for logs and CSV benches.
    pub fn summary_line(&self) -> String {
        format!(
            "nodes={}→{} matmul={}→{} fused_mm_act={} fused_swiglu={} \
             elementwise_region={} missed_mm_act={} missed_swiglu={} missed_shared_mm={}",
            self.nodes_before,
            self.nodes_after,
            self.matmul_before,
            self.matmul_after,
            self.fused_matmul_bias_act,
            self.fused_swiglu,
            self.elementwise_region,
            self.missed_matmul_bias_act(),
            self.missed_swiglu(),
            self.missed_shared_matmul(),
        )
    }
}

impl fmt::Display for FusionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "fusion report:")?;
        writeln!(f, "  {}", self.summary_line())?;
        if !self.missed.is_empty() {
            writeln!(f, "  missed patterns:")?;
            for m in &self.missed {
                write!(f, "    {} @ {}", m.pattern, m.node)?;
                if let Some(ref c) = m.context {
                    write!(f, " ({c})")?;
                }
                write!(f, " — {:?}", m.reason)?;
                if let Some(ref h) = m.hint {
                    write!(f, " → {h}")?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct OpCounts {
    matmul: usize,
    attention: usize,
    rope: usize,
    narrow: usize,
    silu: usize,
    mul: usize,
    fused_matmul_bias_act: usize,
    fused_swiglu: usize,
    fused_residual_ln: usize,
    fused_residual_rms_norm: usize,
    fused_attention_block: usize,
    fused_transformer_layer: usize,
    elementwise_region: usize,
}

fn count_ops(graph: &Graph) -> OpCounts {
    let mut s = OpCounts::default();
    for node in graph.nodes() {
        match &node.op {
            Op::Attention { .. } => s.attention += 1,
            Op::Rope { .. } => s.rope += 1,
            Op::Narrow { .. } => s.narrow += 1,
            Op::MatMul => s.matmul += 1,
            Op::Activation(Activation::Silu) => s.silu += 1,
            Op::Binary(BinaryOp::Mul) => s.mul += 1,
            Op::FusedMatMulBiasAct { .. } => s.fused_matmul_bias_act += 1,
            Op::FusedSwiGLU { .. } => s.fused_swiglu += 1,
            Op::FusedResidualLN { .. } => s.fused_residual_ln += 1,
            Op::FusedResidualRmsNorm { .. } => s.fused_residual_rms_norm += 1,
            Op::FusedAttentionBlock { .. } => s.fused_attention_block += 1,
            Op::FusedTransformerLayer { .. } => s.fused_transformer_layer += 1,
            Op::ElementwiseRegion { .. } => s.elementwise_region += 1,
            _ => {}
        }
    }
    s
}

fn missed_entry(
    graph: &Graph,
    pattern: &'static str,
    node: NodeId,
    reason: MissReason,
) -> MissedFusion {
    MissedFusion {
        pattern,
        node,
        context: Some(node_label(graph, node)),
        hint: Some(fusion_hint(&reason)),
        reason,
    }
}

fn fusion_hint(reason: &MissReason) -> String {
    match reason {
        MissReason::MultiConsumer => {
            "single-consumer chain required — clone input or use HirOp::LinearFused".into()
        }
        MissReason::NonAddBiasConsumer => "use linear+bias or HirModule::linear_fused".into(),
        MissReason::BiasRankTooHigh { .. } => "bias must be rank-1".into(),
        MissReason::UnsupportedEpilogueActivation(_) => {
            "FuseMatMulBiasAct supports Gelu/Silu only".into()
        }
        MissReason::SharedMatmulCount { .. } => "use shared_linear_pair or HirOp::SwiGLU".into(),
        MissReason::SwigluGateBeforeUp => "pass up_w before gate_w in swiglu_ffn".into(),
        MissReason::SwigluNotSharedInput => "gate and up must share the same input".into(),
        MissReason::NotFused => "check inspect_pipeline / RLX_FUSION_REPORT=1".into(),
    }
}

fn scan_misses(graph: &Graph) -> Vec<MissedFusion> {
    let mut missed = Vec::new();
    missed.extend(scan_missed_matmul_bias_act(graph));
    missed.extend(scan_missed_shared_matmul(graph));
    missed.extend(scan_missed_swiglu(graph));
    missed
}

fn scan_missed_matmul_bias_act(graph: &Graph) -> Vec<MissedFusion> {
    let mut out = Vec::new();
    for node in graph.nodes() {
        if !matches!(node.op, Op::MatMul) {
            continue;
        }
        let mm_id = node.id;
        let users = graph.users(mm_id);
        if users.len() != 1 {
            if users.len() > 1 {
                out.push(missed_entry(
                    graph,
                    "matmul_bias_act",
                    mm_id,
                    MissReason::MultiConsumer,
                ));
            }
            continue;
        }
        let add_node = graph.node(users[0]);
        let Op::Binary(BinaryOp::Add) = &add_node.op else {
            out.push(missed_entry(
                graph,
                "matmul_bias_act",
                mm_id,
                MissReason::NonAddBiasConsumer,
            ));
            continue;
        };
        let bias_id = if add_node.inputs[0] == mm_id {
            add_node.inputs[1]
        } else {
            add_node.inputs[0]
        };
        let bias_rank = graph.shape(bias_id).rank();
        if bias_rank > 1 {
            out.push(missed_entry(
                graph,
                "matmul_bias_act",
                mm_id,
                MissReason::BiasRankTooHigh { rank: bias_rank },
            ));
            continue;
        }
        let add_users = graph.users(add_node.id);
        if add_users.len() == 1 {
            if let Op::Activation(act) = &graph.node(add_users[0]).op
                && !fusible_mm_bias_epilogue(*act)
            {
                out.push(missed_entry(
                    graph,
                    "matmul_bias_act",
                    mm_id,
                    MissReason::UnsupportedEpilogueActivation(*act),
                ));
            }
        }
    }
    out
}

fn fusible_mm_bias_epilogue(act: Activation) -> bool {
    matches!(act, Activation::Gelu | Activation::Silu)
}

fn scan_missed_shared_matmul(graph: &Graph) -> Vec<MissedFusion> {
    let mut input_to_matmuls: std::collections::HashMap<NodeId, Vec<NodeId>> =
        std::collections::HashMap::new();
    for node in graph.nodes() {
        if matches!(node.op, Op::MatMul) {
            input_to_matmuls
                .entry(node.inputs[0])
                .or_default()
                .push(node.id);
        }
    }
    let mut out = Vec::new();
    for matmuls in input_to_matmuls.values() {
        if matmuls.len() == 2 {
            let a = graph.node(matmuls[0]);
            let b = graph.node(matmuls[1]);
            let w1 = graph.shape(a.inputs[1]);
            let w2 = graph.shape(b.inputs[1]);
            if w1.rank() == 2 && w2.rank() == 2 && w1.dim(0) == w2.dim(0) {
                out.push(missed_entry(
                    graph,
                    "shared_input_matmul",
                    matmuls[0],
                    MissReason::NotFused,
                ));
            }
        } else if matmuls.len() > 2 {
            out.push(missed_entry(
                graph,
                "shared_input_matmul",
                matmuls[0],
                MissReason::SharedMatmulCount {
                    count: matmuls.len(),
                },
            ));
        }
    }
    out
}

fn scan_missed_swiglu(graph: &Graph) -> Vec<MissedFusion> {
    let mut out = Vec::new();
    for node in graph.nodes() {
        if !matches!(node.op, Op::Binary(BinaryOp::Mul)) {
            continue;
        }
        let lhs = graph.node(node.inputs[0]);
        let rhs = graph.node(node.inputs[1]);
        let (up_side, silu_side) = if matches!(rhs.op, Op::Activation(Activation::Silu)) {
            (lhs, rhs)
        } else if matches!(lhs.op, Op::Activation(Activation::Silu)) {
            (rhs, lhs)
        } else {
            continue;
        };
        if !matches!(up_side.op, Op::MatMul) {
            continue;
        }
        let gate_mm = graph.node(silu_side.inputs[0]);
        if !matches!(gate_mm.op, Op::MatMul) {
            continue;
        }
        if up_side.inputs[0] != gate_mm.inputs[0] {
            out.push(missed_entry(
                graph,
                "swiglu",
                node.id,
                MissReason::SwigluNotSharedInput,
            ));
            continue;
        }
        // Gate-before-up declaration order prevents FuseSwiGLU after shared-input concat.
        if graph
            .nodes()
            .iter()
            .position(|n| n.id == up_side.id)
            .zip(graph.nodes().iter().position(|n| n.id == gate_mm.id))
            .is_some_and(|(up_idx, gate_idx)| gate_idx < up_idx)
        {
            out.push(missed_entry(
                graph,
                "swiglu",
                node.id,
                MissReason::SwigluGateBeforeUp,
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;
    use rlx_ir::Shape;
    use rlx_ir::infer::GraphExt;

    fn f32_shape(dims: &[usize]) -> Shape {
        Shape::new(dims, DType::F32)
    }

    #[test]
    fn report_counts_fused_ops() {
        use crate::fusion::{FuseSharedInputMatMul, FuseSwiGLU};
        use crate::pass::Pass;

        let mut g = Graph::new("report");
        let x = g.input("x", f32_shape(&[4, 768]));
        let up_w = g.param("up", f32_shape(&[768, 128]));
        let gate_w = g.param("gate", f32_shape(&[768, 128]));
        let down_w = g.param("down", f32_shape(&[128, 768]));
        let out = g.swiglu_ffn(x, up_w, gate_w, down_w);
        g.set_outputs(vec![out]);
        let before = g.clone();

        g = FuseSharedInputMatMul.run(g);
        g = FuseSwiGLU.run(g);

        let report = FusionReport::analyze(&before, &g);
        assert_eq!(report.fused_swiglu, 1);
        assert!(report.nodes_after < report.nodes_before);
    }

    #[test]
    fn report_flags_gate_before_up() {
        let mut g = Graph::new("gate_first");
        let x = g.input("x", f32_shape(&[4, 8]));
        let gate_w = g.param("gate", f32_shape(&[8, 16]));
        let up_w = g.param("up", f32_shape(&[8, 16]));
        let gate = g.mm(x, gate_w);
        let up = g.mm(x, up_w);
        let gate_silu = g.silu(gate);
        let out = g.mul(gate_silu, up);
        g.set_outputs(vec![out]);

        let report = FusionReport::scan(&g);
        assert!(report.missed_swiglu() >= 1);
        assert!(
            report
                .missed
                .iter()
                .any(|m| m.reason == MissReason::SwigluGateBeforeUp)
        );
    }
}
