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

//! Text exporters for inspecting HIR / MIR / LIR during lowering.
//!
//! Use [`inspect_hir`], [`inspect_mir`], and [`inspect_lir`] to dump
//! each pipeline stage as human-readable text (similar to LLVM `-print-*`
//! flags). [`inspect_graph`] is the MIR body formatter shared by MIR and
//! LIR dumps.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::hir::{HirModule, HirNode, HirOp};
use crate::lir::{LirBufferPlan, LirModule, LirViewAlias};
use crate::mir::MirModule;
use crate::phase::Phase;
use crate::pretty::{header_line, op_kinds_line, pretty_print};
use crate::{Graph, NodeId};

/// Annotated HIR module dump.
pub fn inspect_hir(hir: &HirModule) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "hir @{} ({} nodes, {} outputs, fusion={:?})",
        hir.name,
        hir.len(),
        hir.outputs.len(),
        hir.fusion_policy,
    )
    .unwrap();
    writeln!(out, "{}", hir_op_kinds_line(hir)).unwrap();
    writeln!(out).unwrap();

    let mut tag_w = 0usize;
    for node in hir.nodes() {
        let t = hir_node_tag(node);
        tag_w = tag_w.max(t.len());
    }

    for node in hir.nodes() {
        let tag = hir_node_tag(node);
        write!(out, "  {tag:<width$} = ", width = tag_w).unwrap();
        write!(out, "{}", format_hir_op(&node.op)).unwrap();
        if !node.inputs.is_empty() {
            write!(out, "(").unwrap();
            for (i, inp) in node.inputs.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ").unwrap();
                }
                write!(out, "{inp}").unwrap();
            }
            write!(out, ")").unwrap();
        }
        write!(out, " : {}", node.shape).unwrap();
        if hir.outputs.contains(&node.id) {
            write!(out, "  ← output").unwrap();
        }
        writeln!(out).unwrap();
    }
    if !hir.outputs.is_empty() {
        write!(out, "  return ").unwrap();
        for (i, o) in hir.outputs.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "{o}").unwrap();
        }
        writeln!(out).unwrap();
    }
    out
}

/// Annotated MIR module dump (optimized tensor DAG).
pub fn inspect_mir(mir: &MirModule) -> String {
    inspect_mir_with_diff(mir, None)
}

/// MIR dump with optional fusion diff against a pre-optimize snapshot.
pub fn inspect_mir_with_diff(mir: &MirModule, before: Option<&MirModule>) -> String {
    let g = mir.as_graph();
    let mut out = String::new();
    writeln!(out, "mir @{} {{", mir.name()).unwrap();
    if let Some(b) = before {
        writeln!(out).unwrap();
        out.push_str(&inspect_graph_diff(b.as_graph(), g));
        writeln!(out).unwrap();
        writeln!(out, "--- graph ---").unwrap();
    }
    writeln!(out).unwrap();
    out.push_str(&pretty_print(g));
    if !out.ends_with('\n') {
        out.push('\n');
    }
    write!(out, "}}").unwrap();
    out
}

/// Diff two MIR snapshots (typically pre/post fusion).
pub fn inspect_mir_diff(before: &MirModule, after: &MirModule) -> String {
    inspect_graph_diff(before.as_graph(), after.as_graph())
}

/// Summarize graph changes between pipeline stages.
pub fn inspect_graph_diff(before: &Graph, after: &Graph) -> String {
    use std::collections::BTreeMap;

    let mut out = String::new();
    writeln!(
        out,
        "  diff: {} → {} nodes ({} → {} outputs)",
        before.len(),
        after.len(),
        before.outputs.len(),
        after.outputs.len(),
    )
    .unwrap();

    let count_kinds = |g: &Graph| {
        let mut h: BTreeMap<String, i32> = BTreeMap::new();
        for n in g.nodes() {
            *h.entry(format!("{:?}", n.op.kind())).or_insert(0) += 1;
        }
        h
    };
    let b = count_kinds(before);
    let a = count_kinds(after);
    let mut keys: Vec<String> = b.keys().chain(a.keys()).cloned().collect();
    keys.sort();
    keys.dedup();
    let mut changes = Vec::new();
    for k in keys {
        let d = a.get(&k).copied().unwrap_or(0) - b.get(&k).copied().unwrap_or(0);
        if d != 0 {
            changes.push(format!("{k}{d:+}"));
        }
    }
    if !changes.is_empty() {
        writeln!(out, "  op delta: {}", changes.join(", ")).unwrap();
    }
    out
}

/// Annotated LIR dump: optimized MIR + buffer plan + schedule.
pub fn inspect_lir(lir: &LirModule) -> String {
    let mut out = String::new();
    writeln!(out, "lir @{} {{", lir.name()).unwrap();
    writeln!(out, "  fingerprint: {:016x}", lir.fingerprint().0).unwrap();
    writeln!(out).unwrap();
    out.push_str(&inspect_buffer_plan(&lir.buffers));
    if !lir.buffers.phases.is_empty() {
        writeln!(out).unwrap();
        out.push_str(&inspect_phases(&lir.buffers));
    }
    if !lir.buffers.io.inputs.is_empty() || !lir.buffers.io.params.is_empty() {
        writeln!(out).unwrap();
        out.push_str(&inspect_io_manifest(&lir.buffers));
    }
    writeln!(out).unwrap();
    writeln!(out, "--- mir ---").unwrap();
    out.push_str(&pretty_print(lir.as_graph()));
    if !out.ends_with('\n') {
        out.push('\n');
    }
    write!(out, "}}").unwrap();
    out
}

/// Annotated graph dump (MIR body). Alias for [`pretty_print`].
pub fn inspect_graph(g: &Graph) -> String {
    pretty_print(g)
}

/// One-line HIR summary (header + op histogram).
pub fn inspect_hir_stats(hir: &HirModule) -> String {
    format!(
        "hir @{} ({} nodes, {} outputs, fusion={:?})\n{}",
        hir.name,
        hir.len(),
        hir.outputs.len(),
        hir.fusion_policy,
        hir_op_kinds_line(hir),
    )
}

/// One-line MIR summary.
pub fn inspect_mir_stats(mir: &MirModule) -> String {
    let g = mir.as_graph();
    format!(
        "mir @{} — {}\n{}",
        mir.name(),
        header_line(g),
        op_kinds_line(g),
    )
}

/// Buffer plan section for LIR inspection.
pub fn inspect_buffer_plan(plan: &LirBufferPlan) -> String {
    let mut out = String::new();
    let saved = plan.bytes_saved();
    let naive = plan.total_unshared_bytes();
    writeln!(
        out,
        "  arena: {} bytes (saved {} vs {} naive, align={})",
        plan.arena_size, saved, naive, plan.alignment,
    )
    .unwrap();
    writeln!(
        out,
        "  schedule: {} nodes, {} views",
        plan.schedule.len(),
        plan.view_aliases.len(),
    )
    .unwrap();
    if !plan.dynamic_symbols.is_empty() {
        let syms: Vec<String> = plan
            .dynamic_symbols
            .iter()
            .map(|s| format!("?{s}"))
            .collect();
        writeln!(out, "  dynamic: {}", syms.join(", ")).unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "  # offset\tsize\tnode").unwrap();

    let mut rows: Vec<(usize, usize, NodeId)> = plan
        .assignments
        .iter()
        .map(|(id, slot)| (slot.offset, slot.size, *id))
        .collect();
    rows.sort_by_key(|(off, _, _)| *off);
    for (off, sz, id) in rows {
        let sched = plan
            .schedule
            .iter()
            .position(|&n| n == id)
            .map(|i| format!(" sched={i}"))
            .unwrap_or_default();
        let view = plan
            .view_aliases
            .get(&id)
            .map(|LirViewAlias { root, byte_offset }| {
                format!(" view→{root}+{byte_offset}")
            })
            .unwrap_or_default();
        let phase = plan
            .phases
            .get(id)
            .map(|p| format!(" {p:?}"))
            .unwrap_or_default();
        writeln!(out, "  {off}\t{sz}\t{id}{sched}{view}{phase}").unwrap();
    }
    out
}

fn inspect_phases(plan: &LirBufferPlan) -> String {
    let mut out = String::from("  phases:\n");
    for phase in [Phase::Prologue, Phase::SteadyState, Phase::Epilogue] {
        let nodes = plan.nodes_in_phase(phase);
        if !nodes.is_empty() {
            writeln!(out, "    {phase:?}: {nodes:?}").unwrap();
        }
    }
    out
}

fn inspect_io_manifest(plan: &LirBufferPlan) -> String {
    let mut out = String::from("  io:\n");
    for (name, id) in &plan.io.inputs {
        writeln!(out, "    input \"{name}\" → {id}").unwrap();
    }
    for (name, id) in &plan.io.params {
        writeln!(out, "    param \"{name}\" → {id}").unwrap();
    }
    if !plan.io.outputs.is_empty() {
        write!(out, "    outputs: {:?}", plan.io.outputs).unwrap();
        out.push('\n');
    }
    out
}

fn hir_op_kinds_line(hir: &HirModule) -> String {
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for node in hir.nodes() {
        *hist.entry(hir_op_kind(&node.op)).or_insert(0) += 1;
    }
    let parts: Vec<String> = hist
        .into_iter()
        .map(|(k, c)| format!("{k}={c}"))
        .collect();
    format!("  block ops: {}", parts.join(", "))
}

fn hir_op_kind(op: &HirOp) -> String {
    match op {
        HirOp::Input { .. } => "Input".into(),
        HirOp::Param { .. } => "Param".into(),
        HirOp::Constant { .. } => "Constant".into(),
        HirOp::Linear { .. } => "Linear".into(),
        HirOp::LinearFused { .. } => "LinearFused".into(),
        HirOp::SharedLinearPair { .. } => "SharedLinearPair".into(),
        HirOp::SwiGLU => "SwiGLU".into(),
        HirOp::ResidualRmsNorm { .. } => "ResidualRmsNorm".into(),
        HirOp::Attention { .. } => "Attention".into(),
        HirOp::DepthwiseConv1dCausal { .. } => "DepthwiseConv1dCausal".into(),
        HirOp::DequantMatMul { .. } => "DequantMatMul".into(),
        HirOp::GatedDeltaNet { .. } => "GatedDeltaNet".into(),
        HirOp::RoPE { .. } => "RoPE".into(),
        HirOp::RmsNorm { .. } => "RmsNorm".into(),
        HirOp::Mir(_) => "Mir".into(),
        HirOp::LlamaDecoderBlock { .. } => "LlamaDecoderBlock".into(),
        HirOp::Qwen35MtpHead { .. } => "Qwen35MtpHead".into(),
    }
}

fn hir_node_tag(node: &HirNode) -> String {
    let label: Option<String> = match &node.op {
        HirOp::Input { name } => Some(format!("input \"{name}\"")),
        HirOp::Param { name } => Some(format!("param \"{name}\"")),
        _ => node.name.as_deref().map(|s| format!("\"{s}\"")),
    };
    match label {
        Some(s) => format!("{} [{s}]", node.id),
        None => format!("{}", node.id),
    }
}

fn format_hir_op(op: &HirOp) -> String {
    match op {
        HirOp::Input { name } => format!("input(\"{name}\")"),
        HirOp::Param { name } => format!("param(\"{name}\")"),
        HirOp::Constant { data } => format!("constant({} bytes)", data.len()),
        HirOp::Linear {
            activation,
            has_bias,
        } => {
            let mut s = String::from("linear");
            if *has_bias {
                s.push_str("+bias");
            }
            if let Some(act) = activation {
                write!(s, "+{act:?}").unwrap();
            }
            s
        }
        HirOp::LinearFused { activation } => match activation {
            Some(act) => format!("linear_fused({act:?})"),
            None => "linear_fused".into(),
        },
        HirOp::SharedLinearPair { slot } => format!("shared_linear_pair(out={slot})"),
        HirOp::SwiGLU => "swiglu_ffn".into(),
        HirOp::ResidualRmsNorm { eps } => format!("residual_rms_norm(eps={eps})"),
        HirOp::Attention {
            num_heads,
            head_dim,
            mask,
        } => format!("attention(heads={num_heads}, dim={head_dim}, mask={mask:?})"),
        HirOp::DepthwiseConv1dCausal { kernel_size } => {
            format!("depthwise_conv1d_causal(k={kernel_size})")
        }
        HirOp::DequantMatMul { scheme } => format!("dequant_matmul({scheme})"),
        HirOp::GatedDeltaNet {
            state_size,
            carry_state,
        } => {
            if *carry_state {
                format!("gated_delta_net(n={state_size},carry)")
            } else {
                format!("gated_delta_net(n={state_size})")
            }
        }
        HirOp::RoPE { head_dim, n_rot } => format!("rope(d={head_dim}, n_rot={n_rot})"),
        HirOp::RmsNorm { eps } => format!("rms_norm(eps={eps})"),
        HirOp::LlamaDecoderBlock {
            num_heads,
            head_dim,
            num_kv_heads,
            eps,
            mask,
        } => format!(
            "llama_decoder_block(heads={num_heads}, dim={head_dim}, kv={num_kv_heads}, eps={eps}, mask={mask:?})"
        ),
        HirOp::Qwen35MtpHead {
            num_heads,
            head_dim,
            mtp_vocab,
            ..
        } => format!(
            "qwen35_mtp_head(heads={num_heads}, dim={head_dim}, vocab={mtp_vocab})"
        ),
        HirOp::Mir(inner) => format!("mir({inner})"),
    }
}

// ── convenience methods on pipeline types ───────────────────────────────

impl HirModule {
    /// Text dump for inspection. Alias for [`inspect_hir`].
    pub fn inspect(&self) -> String {
        inspect_hir(self)
    }
}

impl MirModule {
    /// Text dump for inspection. Alias for [`inspect_mir`].
    pub fn inspect(&self) -> String {
        inspect_mir(self)
    }
}

impl LirModule {
    /// Text dump for inspection. Alias for [`inspect_lir`].
    pub fn inspect(&self) -> String {
        inspect_lir(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use crate::Shape;

    fn f32_shape(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn inspect_hir_includes_blocks_and_outputs() {
        let mut hir = HirModule::new("layer");
        let x = hir.input("x", f32_shape(&[2, 128]));
        let w = hir.param("w", f32_shape(&[128, 128]));
        let h = hir.linear(x, w, None, None, f32_shape(&[2, 128]));
        hir.outputs = vec![h];

        let text = inspect_hir(&hir);
        assert!(text.contains("hir @layer"));
        assert!(text.contains("linear"));
        assert!(text.contains("← output"));
        assert!(text.contains("fusion=Direct"));
    }

    #[test]
    fn inspect_mir_wraps_pretty_print() {
        let mut hir = HirModule::new("m");
        let x = hir.input("x", f32_shape(&[4]));
        hir.outputs = vec![x];
        let mir = hir.lower_to_mir().expect("lower");

        let text = inspect_mir(&mir);
        assert!(text.contains("mir @m"));
        assert!(text.contains("graph @m"));
        assert!(text.contains("input(\"x\")"));
    }

    #[test]
    fn named_block_appears_in_hir_dump() {
        let mut hir = HirModule::new("layer");
        let x = hir.input("x", f32_shape(&[2, 8]));
        let w = hir.param("w", f32_shape(&[8, 8]));
        let out = hir.named("layer0.ffn", |h| {
            h.linear(x, w, None, None, f32_shape(&[2, 8]))
        });
        hir.outputs = vec![out];

        let text = inspect_hir(&hir);
        assert!(text.contains("layer0.ffn"));
    }

    #[test]
    fn provenance_survives_lower() {
        let mut hir = HirModule::new("m");
        let x = hir.input("x", f32_shape(&[2, 8]));
        let w = hir.param("w", f32_shape(&[8, 8]));
        let out = hir.named("block", |h| h.linear(x, w, None, None, f32_shape(&[2, 8])));
        hir.outputs = vec![out];

        let mir = hir.lower_to_mir().expect("lower");
        let text = inspect_mir(&mir);
        assert!(text.contains("hir=h"));
        assert!(text.contains("block"));
    }

    #[test]
    fn inspect_lir_includes_buffer_plan() {
        use crate::lir::{LirBufferPlan, LirBufferSlot, LirIoManifest};

        let mut hir = HirModule::new("l");
        let x = hir.input("x", f32_shape(&[4]));
        hir.outputs = vec![x];
        let mir = hir.lower_to_mir().expect("lower");
        let plan = LirBufferPlan {
            arena_size: 16,
            assignments: [(NodeId(0), LirBufferSlot { offset: 0, size: 16 })]
                .into_iter()
                .collect(),
            schedule: vec![NodeId(0)],
            io: LirIoManifest {
                inputs: vec![("x".into(), NodeId(0))],
                ..Default::default()
            },
            ..Default::default()
        };
        let lir = LirModule::new(mir, plan);

        let text = inspect_lir(&lir);
        assert!(text.contains("lir @l"));
        assert!(text.contains("arena: 16 bytes"));
        assert!(text.contains("fingerprint:"));
        assert!(text.contains("--- mir ---"));
    }
}
