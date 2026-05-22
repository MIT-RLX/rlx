// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Backend lowering transparency — which ops run native, via common IR, or are missing.
//!
//! Use before or during compile to see what will be fast vs decomposed vs blocking.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use rlx_ir::logical_kernel::{
    self, KernelDispatchConfig, KernelDispatchPolicy, registered_logical_kernels,
    should_lower_to_common,
};
use rlx_ir::{Graph, NodeId, OpKind};

use crate::legalize::legalize_for_backend;
use crate::rewrite::rewrite_for_backend_with_config;

/// How a logical / fused op reaches the backend executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchPath {
    /// Claimed in `supported_ops` (or backend makes no op claim).
    Native,
    /// Registered logical kernel lowered to primitive MIR (portable, often slower).
    CommonIr,
    /// Removed by structural rewrite (unfuse, LowerDotGeneral, …) into other kinds.
    Rewritten,
    /// Still not in `supported_ops` after rewrite — compile will fail.
    Unsupported,
}

impl DispatchPath {
    pub fn label(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::CommonIr => "common-ir",
            Self::Rewritten => "rewritten",
            Self::Unsupported => "unsupported",
        }
    }
}

/// Per-`OpKind` summary for one graph + backend claim set.
#[derive(Debug, Clone)]
pub struct KindDispatchSummary {
    pub kind: OpKind,
    pub node_count: usize,
    pub path: DispatchPath,
    /// Set when [`DispatchPath::CommonIr`] (see [`registered_logical_kernels`]).
    pub logical_name: Option<&'static str>,
}

/// Full report after rewrite + legalization probe (same path as [`crate::rewrite::legalize_or_rewrite_for_backend_with_config`]).
#[derive(Debug, Clone)]
pub struct KernelDispatchReport {
    pub backend_name: String,
    pub policy: KernelDispatchPolicy,
    /// Length of `supported_ops` slice (0 = accept all kinds at legalize).
    pub supported_claim_count: usize,
    pub summaries: Vec<KindDispatchSummary>,
    /// Kinds that will use common IR lowering on this compile.
    pub common_lowered_kinds: Vec<OpKind>,
    /// Offenders after all rewrites (empty when compile-ready).
    pub still_unsupported: Vec<(NodeId, OpKind)>,
    pub compile_ready: bool,
}

fn logical_name(kind: OpKind) -> Option<&'static str> {
    registered_logical_kernels()
        .iter()
        .find(|e| e.kind == kind)
        .map(|e| e.name)
}

fn count_kinds(graph: &Graph) -> HashMap<OpKind, usize> {
    let mut m = HashMap::new();
    for node in graph.nodes() {
        *m.entry(node.op.kind()).or_default() += 1;
    }
    m
}

fn classify_kind(
    kind: OpKind,
    supported: &[OpKind],
    config: KernelDispatchConfig,
    common_set: &HashSet<OpKind>,
    before: &HashMap<OpKind, usize>,
    after: &HashMap<OpKind, usize>,
    unsupported_kinds: &HashSet<OpKind>,
) -> DispatchPath {
    if should_lower_to_common(kind, supported, config) || common_set.contains(&kind) {
        return DispatchPath::CommonIr;
    }
    if unsupported_kinds.contains(&kind) {
        return DispatchPath::Unsupported;
    }
    if before.contains_key(&kind) && !after.contains_key(&kind) {
        return DispatchPath::Rewritten;
    }
    if supported.is_empty() || supported.contains(&kind) {
        return DispatchPath::Native;
    }
    if after.contains_key(&kind) {
        return DispatchPath::Native;
    }
    DispatchPath::Unsupported
}

/// Analyze the graph **before** rewrite (static — does not run unfuse passes).
pub fn analyze_dispatch(
    graph: &Graph,
    backend_name: &str,
    supported: &[OpKind],
    config: KernelDispatchConfig,
) -> KernelDispatchReport {
    let before = count_kinds(graph);
    let common_lowered = logical_kernel::logical_kinds_in_graph(graph, supported, config);
    let common_set: HashSet<OpKind> = common_lowered.iter().copied().collect();
    let unsupported_set = HashSet::new();

    let mut summaries: Vec<KindDispatchSummary> = before
        .iter()
        .map(|(&kind, &node_count)| {
            let path = classify_kind(
                kind,
                supported,
                config,
                &common_set,
                &before,
                &before,
                &unsupported_set,
            );
            KindDispatchSummary {
                kind,
                node_count,
                path,
                logical_name: logical_name(kind),
            }
        })
        .collect();
    summaries.sort_by_key(|s| format!("{:?}", s.kind));

    KernelDispatchReport {
        backend_name: backend_name.to_string(),
        policy: config.policy,
        supported_claim_count: supported.len(),
        summaries,
        common_lowered_kinds: common_lowered,
        still_unsupported: Vec::new(),
        // Static probe only — common-ir is compile-ready; use prepare_* for hard failures.
        compile_ready: true,
    }
}

/// Rewrite toward `supported`, then report native / common / rewritten / missing.
pub fn prepare_graph_for_backend_with_report(
    graph: Graph,
    backend_name: &str,
    supported: &[OpKind],
    config: KernelDispatchConfig,
) -> (Graph, KernelDispatchReport) {
    let before = count_kinds(&graph);
    let common_lowered = logical_kernel::logical_kinds_in_graph(&graph, supported, config);
    let common_set: HashSet<OpKind> = common_lowered.iter().copied().collect();

    let rewritten = rewrite_for_backend_with_config(graph, supported, config);
    let after = count_kinds(&rewritten);
    let still_unsupported = legalize_for_backend(&rewritten, supported)
        .err()
        .unwrap_or_default();
    let unsupported_set: HashSet<OpKind> =
        still_unsupported.iter().map(|(_, k)| *k).collect();

    let mut summaries: Vec<KindDispatchSummary> = before
        .iter()
        .map(|(&kind, &node_count)| {
            let path = classify_kind(
                kind,
                supported,
                config,
                &common_set,
                &before,
                &after,
                &unsupported_set,
            );
            KindDispatchSummary {
                kind,
                node_count,
                path,
                logical_name: logical_name(kind),
            }
        })
        .collect();
    summaries.sort_by_key(|s| format!("{:?}", s.kind));

    let compile_ready = still_unsupported.is_empty();
    let report = KernelDispatchReport {
        backend_name: backend_name.to_string(),
        policy: config.policy,
        supported_claim_count: supported.len(),
        summaries,
        common_lowered_kinds: common_lowered,
        still_unsupported,
        compile_ready,
    };
    (rewritten, report)
}

/// Human-readable report for logs / CI / REPL.
pub fn format_dispatch_report(report: &KernelDispatchReport) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "rlx dispatch report — backend {:?}, policy {:?}, supported_ops claim={}",
        report.backend_name,
        report.policy,
        report.supported_claim_count
    );
    if report.supported_claim_count == 0 {
        let _ = writeln!(
            s,
            "  (empty claim = legalize accepts all kinds; native/common split is advisory only)"
        );
    }

    if !report.common_lowered_kinds.is_empty() {
        let _ = writeln!(
            s,
            "  common-ir lowering (portable, add to supported_ops for native fast path):"
        );
        for kind in &report.common_lowered_kinds {
            let name = logical_name(*kind).unwrap_or("?");
            let _ = writeln!(s, "    - {kind:?} ({name})");
        }
    }

    let mut by_path: [Vec<&KindDispatchSummary>; 4] = [vec![], vec![], vec![], vec![]];
    for sum in &report.summaries {
        let idx = match sum.path {
            DispatchPath::Native => 0,
            DispatchPath::CommonIr => 1,
            DispatchPath::Rewritten => 2,
            DispatchPath::Unsupported => 3,
        };
        by_path[idx].push(sum);
    }

    for (label, entries) in [
        ("native", &by_path[0]),
        ("common-ir", &by_path[1]),
        ("rewritten", &by_path[2]),
        ("unsupported", &by_path[3]),
    ] {
        if entries.is_empty() {
            continue;
        }
        let _ = writeln!(s, "  {label}:");
        for e in entries {
            let extra = e
                .logical_name
                .map(|n| format!(" logical={n}"))
                .unwrap_or_default();
            let _ = writeln!(
                s,
                "    - {:?} ×{} nodes{extra}",
                e.kind, e.node_count
            );
        }
    }

    if !report.still_unsupported.is_empty() {
        let _ = writeln!(
            s,
            "  still unsupported after rewrite ({} node(s)) — compile will fail:",
            report.still_unsupported.len()
        );
        for (id, kind) in &report.still_unsupported {
            let _ = writeln!(s, "    - node {id:?}: {kind:?}");
        }
        let _ = writeln!(
            s,
            "  Fix: implement native thunk + add to Backend::supported_ops, or add a \
             rewrite/common body in rlx-fusion."
        );
    } else {
        let _ = writeln!(s, "  compile-ready: yes");
    }

    s
}

/// Print when `RLX_VERBOSE=1` or `RLX_DISPATCH_REPORT=1`.
pub fn maybe_log_dispatch_report(report: &KernelDispatchReport) {
    if rlx_ir::env::flag("RLX_DISPATCH_REPORT") || rlx_ir::env::flag("RLX_VERBOSE") {
        eprintln!("{}", format_dispatch_report(report));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::infer::GraphExt;
    use rlx_ir::*;

    #[test]
    fn common_lowered_when_not_in_supported() {
        use rlx_ir::ops::splat::{GaussianSplatInputs, GaussianSplatRenderParams};
        let mut g = Graph::new("splat");
        let n = 2usize;
        let f = DType::F32;
        let positions = g.input("pos", Shape::new(&[n, 3], f));
        let scales = g.input("scale", Shape::new(&[n, 3], f));
        let rotations = g.input("rot", Shape::new(&[n, 4], f));
        let opacities = g.input("opa", Shape::new(&[n], f));
        let colors = g.input("col", Shape::new(&[n, 3], f));
        let sh_coeffs = g.input("sh", Shape::new(&[n, 3], f));
        let meta = g.input("meta", Shape::new(&[23], f));
        let out = g.gaussian_splat_render(
            GaussianSplatInputs {
                positions,
                scales,
                rotations,
                opacities,
                colors,
                sh_coeffs,
                meta,
            },
            GaussianSplatRenderParams {
                width: 8,
                height: 8,
                ..Default::default()
            },
        );
        g.set_outputs(vec![out]);

        let supported = &[OpKind::Input, OpKind::Param, OpKind::MatMul];
        let report = analyze_dispatch(&g, "test", supported, KernelDispatchConfig::default());
        assert!(report
            .common_lowered_kinds
            .contains(&OpKind::GaussianSplatRender));
        assert!(report
            .summaries
            .iter()
            .any(|s| s.kind == OpKind::GaussianSplatRender && s.path == DispatchPath::CommonIr));
    }

    #[test]
    fn prepare_marks_rewritten_fused_op() {
        let f = DType::F32;
        let mut g = Graph::new("fused");
        let x = g.input("x", Shape::new(&[2, 8], f));
        let w = g.param("w", Shape::new(&[8, 4], f));
        let b = g.param("b", Shape::new(&[4], f));
        let out = g.fused_matmul_bias_act(x, w, b, None, Shape::new(&[2, 4], f));
        g.set_outputs(vec![out]);

        let supported = &[
            OpKind::Input,
            OpKind::Param,
            OpKind::MatMul,
            OpKind::Binary,
            OpKind::Expand,
            OpKind::Activation,
        ];
        let (rewritten, report) = prepare_graph_for_backend_with_report(
            g,
            "cpu",
            supported,
            KernelDispatchConfig::default(),
        );
        assert!(report.compile_ready);
        assert!(!rewritten
            .nodes()
            .iter()
            .any(|n| n.op.kind() == OpKind::FusedMatMulBiasAct));
        assert!(report.summaries.iter().any(|s| {
            s.kind == OpKind::FusedMatMulBiasAct && s.path == DispatchPath::Rewritten
        }));
    }
}
