// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Static graph checker — the analysis behind `cargo rlx check` and the
//! `#[rlx_model(check)]` self-check hook.
//!
//! rlx already *computes* everything a language server would surface, but only
//! exposes it through env-gated `eprintln!` (`RLX_DISPATCH_REPORT`,
//! `RLX_FUSION_REPORT`, `RLX_LINT_NUMERICS`) or compile-time panics. This module
//! folds those same pure functions into one structured [`CheckReport`].
//!
//! Six axes, all but execution-legality fully device-free (no GPU, no driver):
//! * **shape / dtype** — [`rlx_ir::verify::verify_all`] (errors).
//! * **backend dispatch** — per backend, ops that run native vs. portable
//!   common-IR (a perf note) vs. can't be lowered (an error), resolved from the
//!   backend's real op claim via the registry.
//! * **fusion** — patterns that should have collapsed but didn't (warnings).
//! * **numerics** — constant subgraphs that provably fold to NaN/Inf (warnings).
//! * **representation** — [`rlx_ir::repr_check`]: a consumer whose producer
//!   declares a different stride / encoding / arity (errors).
//! * **schedule** — GPU tile utilization from [`rlx_gpu_dispatch::cost`]
//!   (notes).
//!
//! The severities are the design, not decoration. They are the three
//! dispositions a pre-compile analysis can have, and collapsing them would make
//! at least one unusable:
//!
//! * **gate** (errors) — shape and representation. These do not degrade output,
//!   they compute a *different function*, so compilation must not proceed.
//! * **report** (warnings) — fusion and numerics. The program is well-formed
//!   but something is probably not what the author meant.
//! * **hint** (notes) — schedule. Nothing is wrong. The answer is correct and
//!   the machine is being wasted producing it. A finding that could fail a
//!   build here would mean rejecting correct programs for being slow.
//!
//! Execution legality is only reported for backends compiled into the build.
//! CPU is always available and portable; other backends are opt-in behind the
//! consuming crate's Cargo features (the claim is read driver-free — backend
//! factories build unit structs and `supported_ops()` is a const).

use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};

use rlx_ir::verify::VerifyError;
use rlx_ir::{Graph, node_label};
use rlx_opt::rlx_compile::KernelDispatchConfig;
use rlx_opt::rlx_compile::dispatch_report::{DispatchPath, prepare_graph_for_backend_with_report};
use rlx_opt::rlx_compile::fusion_pipeline::{Fuse, FusionTarget};
use serde::Serialize;

use crate::Device;
use crate::registry::backend_for;

/// Diagnostic severity, ordered most-severe first for rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
    Note,
}

impl Severity {
    fn rank(self) -> u8 {
        match self {
            Severity::Error => 0,
            Severity::Warning => 1,
            Severity::Note => 2,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
    }
}

/// A single finding about the graph.
#[derive(Debug, Clone, Serialize)]
pub struct Diagnostic {
    pub severity: Severity,
    /// Stable kebab-case slug, e.g. `shape`, `unsupported-op`, `missed-fusion`, `numeric`.
    pub code: String,
    pub message: String,
    /// Offending node id (the `%N` in the graph), when one applies.
    pub node: Option<u32>,
    /// Node label / name for context (`node_label`).
    pub context: Option<String>,
    /// Actionable remedy, reused from the underlying analysis where available.
    pub hint: Option<String>,
    /// Backend this pertains to (dispatch findings); `None` = backend-agnostic.
    pub backend: Option<String>,
}

/// Execution-legality rollup against a backend's real op claim.
///
/// Only present when the backend is compiled into the build. Resolved
/// statically from the registry — no live driver is needed, since backend
/// factories build unit structs and `supported_ops()` is a const.
#[derive(Debug, Clone, Serialize)]
pub struct Legality {
    /// True when no op is left unsupported after rewrite.
    pub compile_ready: bool,
    pub native_kinds: usize,
    pub common_ir_kinds: usize,
    pub rewritten_kinds: usize,
    pub unsupported_kinds: usize,
}

/// Per-backend rollup: fusion coverage (always) + execution legality (when the
/// backend is compiled in).
#[derive(Debug, Clone, Serialize)]
pub struct BackendSummary {
    pub backend: String,
    /// `None` when this backend isn't compiled into the build — fusion coverage
    /// is still reported; execution legality is not.
    pub legality: Option<Legality>,
    /// Count of fused ops produced by the fusion pipeline for this target.
    pub fused_ops: usize,
    pub missed_fusions: usize,
}

/// The full result of [`check_graph`].
#[derive(Debug, Clone, Serialize)]
pub struct CheckReport {
    pub graph: String,
    pub nodes: usize,
    pub diagnostics: Vec<Diagnostic>,
    pub backends: Vec<BackendSummary>,
}

/// What to check and against which backends.
#[derive(Debug, Clone)]
pub struct CheckOptions {
    /// Backends whose op claim + fusion pipeline to analyze against.
    pub backends: Vec<FusionTarget>,
    pub dispatch: bool,
    pub fusion: bool,
    pub numeric: bool,
    /// Producer–consumer representation compatibility
    /// ([`rlx_ir::repr_check`]). On by default: it is a static, device-free
    /// check whose whole purpose is to catch representation mismatches BEFORE a
    /// backend computes a wrong number from them.
    pub repr: bool,
    /// Physical-schedule quality for GPU dispatch
    /// ([`rlx_gpu_dispatch::cost`]). Notes only — nothing here is wrong, it is
    /// work the hardware will do and throw away.
    pub schedule: bool,
    /// Program-safety and schedule-semantics gates over the memory plan
    /// ([`rlx_compile::plan_check`]). Errors: an overlapping live buffer does
    /// not degrade output, it returns another tensor's bytes.
    pub plan: bool,
}

impl Default for CheckOptions {
    fn default() -> Self {
        Self {
            backends: default_backends(),
            dispatch: true,
            fusion: true,
            numeric: true,
            repr: true,
            schedule: true,
            plan: true,
        }
    }
}

/// A representative cross-vendor default set (CPU + the three big GPU families).
pub fn default_backends() -> Vec<FusionTarget> {
    vec![
        FusionTarget::Cpu,
        FusionTarget::Metal,
        FusionTarget::Cuda,
        FusionTarget::Wgpu,
    ]
}

/// Every fusion target rlx models a static op claim for.
pub fn all_backends() -> Vec<FusionTarget> {
    vec![
        FusionTarget::Cpu,
        FusionTarget::Metal,
        FusionTarget::Mlx,
        FusionTarget::Wgpu,
        FusionTarget::Cuda,
        FusionTarget::Rocm,
        FusionTarget::Tpu,
    ]
}

/// Stable lowercase name for a backend target.
pub fn backend_name(t: FusionTarget) -> &'static str {
    match t {
        FusionTarget::Cpu => "cpu",
        FusionTarget::Metal => "metal",
        FusionTarget::Mlx => "mlx",
        FusionTarget::Wgpu => "wgpu",
        FusionTarget::Cuda => "cuda",
        FusionTarget::Rocm => "rocm",
        FusionTarget::Tpu => "tpu",
    }
}

/// Parse a backend name (with common aliases) into a [`FusionTarget`].
pub fn parse_backend(s: &str) -> Option<FusionTarget> {
    match s.trim().to_ascii_lowercase().as_str() {
        "cpu" => Some(FusionTarget::Cpu),
        "metal" | "mps" | "mtl" => Some(FusionTarget::Metal),
        "mlx" => Some(FusionTarget::Mlx),
        "wgpu" | "gpu" | "webgpu" => Some(FusionTarget::Wgpu),
        "cuda" | "nvidia" => Some(FusionTarget::Cuda),
        "rocm" | "hip" | "amd" => Some(FusionTarget::Rocm),
        "tpu" => Some(FusionTarget::Tpu),
        _ => None,
    }
}

/// The execution [`Device`] whose op claim answers legality for a fusion target.
pub fn backend_device(t: FusionTarget) -> Device {
    match t {
        FusionTarget::Cpu => Device::Cpu,
        FusionTarget::Metal => Device::Metal,
        FusionTarget::Mlx => Device::Mlx,
        FusionTarget::Wgpu => Device::Gpu,
        FusionTarget::Cuda => Device::Cuda,
        FusionTarget::Rocm => Device::Rocm,
        FusionTarget::Tpu => Device::Tpu,
    }
}

/// The backend's real execution op-claim, or `None` if it isn't compiled in.
///
/// Driver-free: the factory builds a unit struct and `supported_ops()` is a
/// const, so this works for any compiled-in backend even without its hardware.
fn execution_claim(device: Device) -> Option<&'static [rlx_ir::OpKind]> {
    catch_unwind(AssertUnwindSafe(|| {
        backend_for(device).map(|be| be.supported_ops())
    }))
    .ok()
    .flatten()
}

impl CheckReport {
    pub fn count(&self, sev: Severity) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == sev)
            .count()
    }

    pub fn errors(&self) -> usize {
        self.count(Severity::Error)
    }

    pub fn warnings(&self) -> usize {
        self.count(Severity::Warning)
    }

    pub fn has_errors(&self) -> bool {
        self.errors() > 0
    }

    /// Compact JSON for machine consumers / editors.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    /// rustc-style human report.
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "rlx check — graph \"{}\" ({} node{})",
            self.graph,
            self.nodes,
            if self.nodes == 1 { "" } else { "s" }
        );

        let mut diags: Vec<&Diagnostic> = self.diagnostics.iter().collect();
        diags.sort_by_key(|d| d.severity.rank());

        if diags.is_empty() {
            let _ = writeln!(s, "\n  no findings.");
        } else {
            let _ = writeln!(s);
            for d in diags {
                let on = d
                    .backend
                    .as_deref()
                    .map(|b| format!(" on {b}"))
                    .unwrap_or_default();
                let _ = writeln!(s, "{}[{}]{on}: {}", d.severity.label(), d.code, d.message);
                if let Some(n) = d.node {
                    // Skip the label when it's just the node id again (unnamed node).
                    match &d.context {
                        Some(c) if !c.is_empty() && *c != format!("%{n}") => {
                            let _ = writeln!(s, "  --> %{n} ({c})");
                        }
                        _ => {
                            let _ = writeln!(s, "  --> %{n}");
                        }
                    }
                }
                if let Some(h) = &d.hint {
                    let _ = writeln!(s, "  = help: {h}");
                }
            }
        }

        if !self.backends.is_empty() {
            let _ = writeln!(s, "\nbackends:");
            for b in &self.backends {
                match &b.legality {
                    Some(l) => {
                        let status = if l.compile_ready {
                            "ready  "
                        } else {
                            "BLOCKED"
                        };
                        let _ = writeln!(
                            s,
                            "  {:<6} {status}  native={} common-ir={} rewritten={} unsupported={}  fused={} missed={}",
                            b.backend,
                            l.native_kinds,
                            l.common_ir_kinds,
                            l.rewritten_kinds,
                            l.unsupported_kinds,
                            b.fused_ops,
                            b.missed_fusions,
                        );
                    }
                    None => {
                        let _ = writeln!(
                            s,
                            "  {:<6} legality n/a (build --features {})           fused={} missed={}",
                            b.backend, b.backend, b.fused_ops, b.missed_fusions,
                        );
                    }
                }
            }
        }

        let _ = writeln!(
            s,
            "\nsummary: {} error(s), {} warning(s), {} note(s) across {} backend(s)",
            self.errors(),
            self.warnings(),
            self.count(Severity::Note),
            self.backends.len(),
        );
        s
    }
}

/// `Some(n)` for a statically-known extent, `None` for a dynamic one.
fn static_dim(d: rlx_ir::Dim) -> Option<usize> {
    match d {
        rlx_ir::Dim::Static(n) => Some(n),
        rlx_ir::Dim::Dynamic(_) => None,
    }
}

/// Whether this target dispatches through the shared GPU tile table at all.
/// CPU and TPU do not, so a tile note would be meaningless for them.
fn is_gpu_target(t: FusionTarget) -> bool {
    matches!(
        t,
        FusionTarget::Metal
            | FusionTarget::Mlx
            | FusionTarget::Wgpu
            | FusionTarget::Cuda
            | FusionTarget::Rocm
    )
}

/// Modelled speedup a non-default tile must show before the graph is worth
/// mentioning at all.
///
/// Higher than the tuner's own `MIN_SPEEDUP` (1.02) on purpose. The tuner is
/// deciding what to install from real timings and can afford a thin margin; a
/// static hint is spending the reader's attention on an *uncalibrated* ranking,
/// so it should only speak when the prize is clearly worth a tuning run.
const MIN_MODELLED_GAIN: f64 = 1.25;

/// A single note on GPU tile fit for the whole graph.
///
/// Device-free: it compares the compile-time default tile against the rest of
/// the candidate set, which is what runs on every arch absent a measured
/// override.
///
/// **One note, not one per node — and not one per shape.** Two earlier shapes
/// of this analysis were both noise. Per-node reporting emitted 196 identical
/// lines for a 28-layer decode graph. Per-shape reporting cut that to 3, but
/// the modelled gain turns out to be near-constant within a regime (~1.56x
/// across every decode shape, ~1.27x across every prefill shape), so those 3
/// lines were one fact about rlx's default tile restated per shape. The
/// information content is "this graph has untuned GEMM shapes, run the tuner",
/// which is worth saying once.
///
/// It also reports *available gain*, not *wasted arithmetic*, because the two
/// are not related the way the wording implied: a shape using 1.6% of the
/// tile's MACs has 1.56x available, while one using 93.8% has 1.16x. "98.4%
/// discarded" reads like a 60x opportunity and is not one.
fn schedule_notes(graph: &Graph) -> Vec<Diagnostic> {
    use rlx_gpu_dispatch::cost::TileCostModel;
    use rlx_gpu_dispatch::dispatch::Workload;
    use rlx_gpu_dispatch::tiles::{MATMUL_TILE_CANDIDATES, TileParams};

    let default_tile = TileParams::DEFAULT_MATMUL;
    // `structural()`, not `sm86()`: this walks a graph with no device in sight,
    // so claiming a calibrated model would claim a measurement never taken.
    let model = TileCostModel::structural();

    // (shape, gain, winning tile, node count, first node) per distinct shape.
    let mut shapes: Vec<(
        (usize, usize, usize),
        f64,
        TileParams,
        usize,
        rlx_ir::NodeId,
    )> = Vec::new();

    for node in graph.nodes() {
        if !matches!(node.op, rlx_ir::Op::MatMul) || node.inputs.len() != 2 {
            continue;
        }
        // A @ B: the output's trailing dims are [m, n]; A's trailing dim is k.
        let out_dims = node.shape.dims();
        let lhs_dims = graph.shape(node.inputs[0]).dims();
        if out_dims.len() < 2 || lhs_dims.is_empty() {
            continue;
        }
        // A dynamic extent has no schedule to assess — the tile that runs
        // depends on a value this analysis cannot see. Skipping is the honest
        // answer; guessing a size would produce a confident claim about a shape
        // that may never occur.
        let (Some(m), Some(n), Some(k)) = (
            static_dim(out_dims[out_dims.len() - 2]),
            static_dim(out_dims[out_dims.len() - 1]),
            static_dim(lhs_dims[lhs_dims.len() - 1]),
        ) else {
            continue;
        };

        if let Some(slot) = shapes.iter_mut().find(|(sh, ..)| *sh == (m, k, n)) {
            slot.3 += 1;
            continue;
        }

        let w = Workload::Matmul { m, k, n };
        let Some(base) = model.estimate(&default_tile, &w) else {
            continue;
        };
        let Some((best_tile, best)) = MATMUL_TILE_CANDIDATES
            .iter()
            .filter_map(|t| model.estimate(t, &w).map(|c| (*t, c)))
            .min_by(|a, b| {
                a.1.score
                    .partial_cmp(&b.1.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        else {
            continue;
        };
        if best.score <= 0.0 {
            continue;
        }
        shapes.push(((m, k, n), base.score / best.score, best_tile, 1, node.id));
    }

    let worth_tuning: Vec<_> = shapes
        .iter()
        .filter(|(_, gain, ..)| *gain >= MIN_MODELLED_GAIN)
        .collect();
    let Some(worst) = worth_tuning
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Vec::new();
    };

    let nodes: usize = worth_tuning.iter().map(|(_, _, _, c, _)| *c).sum();
    let ((wm, wk, wn), gain, best_tile, _, anchor) = **worst;

    vec![Diagnostic {
        severity: Severity::Note,
        code: "schedule-tile".to_string(),
        message: format!(
            "{} matmul shape(s) across {nodes} node(s) model faster on a non-default tile; \
             largest modelled gain {gain:.2}x at {wm}x{wk}x{wn} ({} vs the default {})",
            worth_tuning.len(),
            best_tile.label(),
            default_tile.label()
        ),
        node: Some(anchor.0),
        context: Some(node_label(graph, anchor)),
        hint: Some(
            "these are UNCALIBRATED model rankings, not measurements — run \
             `tune_dispatch` on the target device to confirm and install; the dispatch \
             table only accepts a measured winner that also clears a held-out check"
                .to_string(),
        ),
        backend: None,
    }]
}

fn verify_diag(graph: &Graph, e: &VerifyError, code: &str) -> Diagnostic {
    let hint = if e.message.contains("shape mismatch") {
        Some(
            "declared out-shape disagrees with the inferred shape — fix the builder's \
             out_shape argument (or a dtype mismatch between the operands)"
                .to_string(),
        )
    } else if e.message.contains("not a DAG") {
        Some("an input references a later node — build nodes in topological order".to_string())
    } else if e.message.contains("expects") && e.message.contains("inputs") {
        Some("wrong operand count for this op".to_string())
    } else {
        None
    };
    Diagnostic {
        severity: Severity::Error,
        code: code.to_string(),
        message: e.message.clone(),
        node: e.node.map(|n| n.0),
        context: e.node.map(|n| node_label(graph, n)),
        hint,
        backend: None,
    }
}

/// Run every enabled check against `graph` and collect the findings.
///
/// Pure: no backend is instantiated beyond reading its (driver-free) op claim.
/// Backend dispatch + fusion analysis are skipped when the graph fails
/// structural or shape verification — fix those first, since downstream passes
/// assume a well-formed, shape-consistent graph.
pub fn check_graph(graph: &Graph, opts: &CheckOptions) -> CheckReport {
    let mut diagnostics = Vec::new();

    // 1. Structural integrity (DAG, arity, output refs) — device-independent.
    let structural = rlx_ir::verify::verify(graph);
    for e in &structural {
        diagnostics.push(verify_diag(graph, e, "graph-structure"));
    }

    // 2. Shape / dtype consistency — device-independent.
    let shape_errors = if structural.is_empty() {
        rlx_ir::verify::verify_shapes(graph)
    } else {
        Vec::new()
    };
    for e in &shape_errors {
        diagnostics.push(verify_diag(graph, e, "shape"));
    }

    let graph_ok = structural.is_empty() && shape_errors.is_empty();

    // 3. Provable numeric blow-ups (constant folds to NaN/Inf) — device-independent.
    if opts.numeric && graph_ok {
        for l in rlx_opt::rlx_compile::lint_numerics(graph) {
            diagnostics.push(Diagnostic {
                severity: Severity::Warning,
                code: "numeric".to_string(),
                message: format!("{} produced here — {}", l.kind.as_str(), l.reason),
                node: Some(l.node.0),
                context: Some(l.label),
                hint: l.fix.map(str::to_string),
                backend: None,
            });
        }
    }

    // 3b. Producer–consumer representation compatibility — device-free.
    //
    // Reported as ERRORS, not warnings: a stride/encoding/arity mismatch does
    // not degrade output, it produces a different function. Every finding here
    // corresponds to a class of defect this tree has actually shipped.
    if opts.repr && graph_ok {
        let report = rlx_ir::repr_check::check_graph(graph);
        for f in &report.findings {
            diagnostics.push(Diagnostic {
                severity: Severity::Error,
                code: format!("repr-{}", f.kind.as_str()),
                message: format!(
                    "{:?} expects {} but its producer declares {}",
                    f.op, f.expected, f.actual
                ),
                node: Some(f.node.0),
                context: f.input.map(|i| format!("input[{i}]")),
                hint: Some(f.why.to_string()),
                backend: None,
            });
        }
    }

    // 3c. Physical-schedule quality for GPU dispatch — device-free.
    //
    // Reported as NOTES, and the distinction from 3b is the point. A repr
    // mismatch computes a different function; a bad tile computes the right
    // one and wastes the machine doing it. CAKE separates these dispositions
    // for the same reason: something that must block compilation and something
    // that should tell you what to change are not the same signal, and folding
    // them together makes one of them unusable.
    if opts.schedule && graph_ok && opts.backends.iter().any(|t| is_gpu_target(*t)) {
        diagnostics.extend(schedule_notes(graph));
    }

    // 3d. Program safety + schedule semantics over the memory plan.
    //
    // Errors, like 3b: a buffer that overlaps a live neighbour does not
    // produce a worse answer, it produces another tensor's bytes. Both
    // categories are reported under distinct codes because they name
    // different repair targets — one is where a buffer sits, the other is
    // what order the schedule runs in.
    if opts.plan && graph_ok {
        let plan = rlx_opt::rlx_compile::memory::plan_memory(graph);
        let report = rlx_opt::rlx_compile::plan_check::check_plan(graph, &plan);
        for f in &report.findings {
            diagnostics.push(Diagnostic {
                severity: Severity::Error,
                code: format!("plan-{}", f.kind.as_str()),
                message: f.detail.clone(),
                node: Some(f.node.0),
                context: Some(node_label(graph, f.node)),
                hint: Some(
                    if f.kind.is_program_safety() {
                        "a memory-use hazard: the arena assignment lets one tensor's bytes \
                         be read or written as another's"
                    } else {
                        "a schedule-structure violation: the declared execution order does \
                         not satisfy the graph's dependencies"
                    }
                    .to_string(),
                ),
                backend: None,
            });
        }
    }

    // 4. Per-backend: fusion coverage (device-free) + execution legality
    //    (only for backends compiled into this build).
    let mut backends = Vec::new();
    // Missed fusions are largely authoring-level (structural); dedup across
    // targets so the same miss isn't reported once per backend.
    let mut seen_miss: HashSet<(String, u32, String)> = HashSet::new();
    let mut legality_gaps: Vec<&'static str> = Vec::new();

    if graph_ok {
        for &t in &opts.backends {
            let name = backend_name(t);
            let mut summary = BackendSummary {
                backend: name.to_string(),
                legality: None,
                fused_ops: 0,
                missed_fusions: 0,
            };

            // -- execution legality against the backend's REAL op claim --
            match execution_claim(backend_device(t)) {
                Some(claim) => {
                    let (_g, report) = prepare_graph_for_backend_with_report(
                        graph.clone(),
                        name,
                        claim,
                        KernelDispatchConfig::default(),
                    );
                    let mut leg = Legality {
                        compile_ready: report.compile_ready,
                        native_kinds: 0,
                        common_ir_kinds: 0,
                        rewritten_kinds: 0,
                        unsupported_kinds: 0,
                    };
                    for kind_summary in &report.summaries {
                        match kind_summary.path {
                            DispatchPath::Native => leg.native_kinds += 1,
                            DispatchPath::CommonIr => leg.common_ir_kinds += 1,
                            DispatchPath::Rewritten => leg.rewritten_kinds += 1,
                            DispatchPath::Unsupported => leg.unsupported_kinds += 1,
                        }
                    }
                    if opts.dispatch {
                        for (id, kind) in &report.still_unsupported {
                            diagnostics.push(Diagnostic {
                                severity: Severity::Error,
                                code: "unsupported-op".to_string(),
                                message: format!(
                                    "{kind:?} cannot be lowered on {name} (still unsupported after rewrite)"
                                ),
                                node: Some(id.0),
                                context: Some(node_label(graph, *id)),
                                hint: Some(
                                    "add a native thunk + list the OpKind in Backend::supported_ops, \
                                     or add a rewrite/common-IR body in rlx-fusion"
                                        .to_string(),
                                ),
                                backend: Some(name.to_string()),
                            });
                        }
                        for kind in &report.common_lowered_kinds {
                            diagnostics.push(Diagnostic {
                                severity: Severity::Note,
                                code: "common-ir".to_string(),
                                message: format!(
                                    "{kind:?} runs via portable common-IR on {name} (correct, but off the native fast path)"
                                ),
                                node: None,
                                context: None,
                                hint: Some(
                                    "list this OpKind in the backend's supported_ops to dispatch a native kernel"
                                        .to_string(),
                                ),
                                backend: Some(name.to_string()),
                            });
                        }
                    }
                    summary.legality = Some(leg);
                }
                None => legality_gaps.push(name),
            }

            // -- fusion: what fused, what was left on the table (device-free) --
            let fusion = catch_unwind(AssertUnwindSafe(|| {
                Fuse::new(t).run_with_report(graph.clone())
            }));
            if let Ok((_fused, freport)) = fusion {
                summary.fused_ops = freport.fused_matmul_bias_act
                    + freport.fused_swiglu
                    + freport.fused_residual_ln
                    + freport.fused_residual_rms_norm
                    + freport.fused_attention_block
                    + freport.fused_transformer_layer;
                summary.missed_fusions = freport.missed.len();
                if opts.fusion {
                    for m in &freport.missed {
                        let key = (m.pattern.to_string(), m.node.0, format!("{:?}", m.reason));
                        if seen_miss.insert(key) {
                            diagnostics.push(Diagnostic {
                                severity: Severity::Warning,
                                code: "missed-fusion".to_string(),
                                message: format!(
                                    "{} fusion not applied ({:?})",
                                    m.pattern, m.reason
                                ),
                                node: Some(m.node.0),
                                context: m.context.clone(),
                                hint: m.hint.clone(),
                                backend: None,
                            });
                        }
                    }
                }
            }

            backends.push(summary);
        }

        // One note (not one-per-backend) for backends whose real op claim isn't
        // compiled in — those rows show fusion only.
        if !legality_gaps.is_empty() {
            diagnostics.push(Diagnostic {
                severity: Severity::Note,
                code: "legality-unavailable".to_string(),
                message: format!(
                    "execution legality not checked for [{}] — those backends aren't compiled \
                     into this build (fusion coverage is still shown)",
                    legality_gaps.join(", ")
                ),
                node: None,
                context: None,
                hint: Some(
                    "rebuild with the matching backend feature to check native/unsupported dispatch"
                        .to_string(),
                ),
                backend: None,
            });
        }
    }

    CheckReport {
        graph: graph.name.clone(),
        nodes: graph.len(),
        diagnostics,
        backends,
    }
}

/// Self-check hook injected by `#[rlx_model(check)]` right after the model's
/// graph is traced. Runs [`check_graph`] and reports findings to stderr.
///
/// Controlled at runtime by `RLX_CHECK`:
/// * unset / `warn` / `1` — print the report when there are findings (default).
/// * `off` / `0` / `false` — do nothing (production escape hatch).
/// * `all` — check every backend target and always print.
/// * `strict` — additionally **panic** on any error-level finding.
///
/// Focused on the CPU reference backend by default (that's what `#[rlx_model]`
/// compiles for); `RLX_CHECK=all` widens to every target.
pub fn model_self_check(name: &str, graph: &Graph) {
    let mode = rlx_ir::env::var("RLX_CHECK").unwrap_or_default();
    let mode = mode.trim().to_ascii_lowercase();
    if matches!(mode.as_str(), "0" | "off" | "false") {
        return;
    }
    let strict = mode == "strict";
    let all = mode == "all";
    let verbose = all || mode == "verbose";

    let opts = CheckOptions {
        backends: if all {
            all_backends()
        } else {
            vec![FusionTarget::Cpu]
        },
        ..CheckOptions::default()
    };
    let report = check_graph(graph, &opts);

    if verbose || report.errors() > 0 || report.warnings() > 0 {
        eprint!("rlx-check [{name}]\n{}", report.render());
    }
    if strict && report.has_errors() {
        panic!(
            "rlx-check: model `{name}` has {} error-level finding(s) — see report above \
             (RLX_CHECK=strict)",
            report.errors()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::infer::GraphExt;
    use rlx_ir::{DType, Shape};

    fn f32s(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn clean_mlp_is_cpu_ready() {
        let mut g = Graph::new("mlp");
        let x = g.input("x", f32s(&[4, 16]));
        let w = g.param("w", f32s(&[16, 8]));
        let h = g.mm(x, w);
        let y = g.gelu(h);
        g.set_outputs(vec![y]);

        let r = check_graph(&g, &CheckOptions::default());
        assert_eq!(r.errors(), 0, "unexpected errors: {:#?}", r.diagnostics);
        let cpu = r.backends.iter().find(|b| b.backend == "cpu").unwrap();
        let leg = cpu.legality.as_ref().expect("cpu legality");
        assert!(leg.compile_ready);
        assert_eq!(leg.unsupported_kinds, 0);
    }

    #[test]
    fn self_check_hook_runs_without_panic() {
        // Default RLX_CHECK behavior must never panic on a clean graph.
        let mut g = Graph::new("hooked");
        let x = g.input("x", f32s(&[2, 4]));
        let w = g.param("w", f32s(&[4, 4]));
        let y = g.mm(x, w);
        g.set_outputs(vec![y]);
        model_self_check("hooked", &g);
    }
}
