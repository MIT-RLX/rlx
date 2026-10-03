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

/// What one run of this graph costs on one architecture.
///
/// Device-free, like the rest of the checker: every number comes from the same
/// memory planner and dispatch claim the backend will use, not from a probe.
/// Nothing here is *wrong* — that is the point of reporting it as notes. It is
/// the difference between a graph that fits a phone and one that does not, and
/// between a kernel that runs on the accelerator and one that quietly round
/// trips through the host.
#[derive(Debug, Clone, Serialize)]
pub struct Budget {
    /// Arena bytes the planner asks for under this backend's width policy.
    pub arena_bytes: usize,
    /// Bytes liveness-aware slot reuse saved versus one buffer per node.
    pub reuse_saved_bytes: usize,
    /// Extra arena bytes this backend's width policy costs over storing every
    /// tensor at its native dtype width, with every other planner option held
    /// fixed. Non-zero only on the f32-uniform backends, and only for graphs
    /// that actually carry F16/BF16 activations. Deliberately *not*
    /// `arena_bytes` minus the native plan's: those two plans also differ in
    /// pinning, and that delta would report a saving where there is none.
    pub width_overhead_bytes: usize,
    /// Name of the width policy in force (`native` / `f32-uniform` / `hybrid`).
    pub width_policy: &'static str,
    /// Bytes that must cross the host boundary every run: graph inputs read in
    /// plus graph outputs read back. Parameters are excluded — they are
    /// uploaded once and stay resident.
    pub host_io_bytes: usize,
    /// Ops that will not run on this backend's native fast path (portable
    /// common-IR or still unsupported), and the output bytes they touch. On a
    /// GPU backend these are the candidates for a host round trip.
    pub off_fast_path_ops: usize,
    pub off_fast_path_bytes: usize,
    /// `Cast` nodes in the graph — every one is a full pass over a tensor whose
    /// only product is a different dtype.
    pub cast_nodes: usize,
    /// Parameter bytes the graph expects to find already on the device.
    ///
    /// Separate from `host_io_bytes` on purpose, and reported rather than
    /// folded in, because "weights are uploaded once and stay resident" is an
    /// *assumption* — and it is the assumption that fails. When residency
    /// silently does not hold, this whole number becomes per-run traffic, and a
    /// boundary figure that already included it could not show the difference.
    pub param_bytes: usize,
    /// Ops left after this backend's fusion pipeline, excluding graph leaves.
    ///
    /// An estimate of dispatch count, which on a GPU backend is the quantity
    /// that multiplies the per-dispatch floor. It is an estimate because one op
    /// is not always one dispatch — a multi-pass reduction is several — so
    /// treat it as a floor on dispatches, not an exact count.
    pub dispatches: usize,
    /// Bytes the dispatched ops read and write, summed over the fused graph.
    ///
    /// An upper bound: it assumes nothing survives in cache or registers
    /// between ops. Useful as a ratio against `arena_bytes` — a graph moving
    /// many times its own arena is re-reading the same tensors.
    pub dram_bytes: usize,
}

/// Per-backend rollup: fusion coverage (always) + execution legality (when the
/// backend is compiled in) + resource budget.
#[derive(Debug, Clone, Serialize)]
pub struct BackendSummary {
    pub backend: String,
    /// `None` when this backend isn't compiled into the build — fusion coverage
    /// is still reported; execution legality is not.
    pub legality: Option<Legality>,
    /// Count of fused ops produced by the fusion pipeline for this target.
    pub fused_ops: usize,
    pub missed_fusions: usize,
    /// `None` when the graph did not verify — a budget computed from a graph
    /// that is not well-formed is a number nobody should act on.
    pub budget: Option<Budget>,
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
    /// Per-architecture resource budget: arena bytes, host-boundary traffic and
    /// precision churn ([`Budget`]). Notes only — this axis reports what a
    /// correct program will cost, never that it is wrong.
    pub budget: bool,
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
            budget: true,
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
                // Budget on its own line: it answers a different question from
                // the legality row above it — not "will this run" but "what
                // will it spend to run".
                if let Some(bu) = &b.budget {
                    let _ = write!(
                        s,
                        "         arena={} ({}, reuse saved {})  params={}  host-io/run={}  dispatch={} moving {}",
                        human_bytes(bu.arena_bytes),
                        bu.width_policy,
                        human_bytes(bu.reuse_saved_bytes),
                        human_bytes(bu.param_bytes),
                        human_bytes(bu.host_io_bytes),
                        bu.dispatches,
                        human_bytes(bu.dram_bytes),
                    );
                    if bu.width_overhead_bytes > 0 {
                        let _ = write!(s, "  width cost={}", human_bytes(bu.width_overhead_bytes));
                    }
                    if bu.off_fast_path_ops > 0 {
                        let _ = write!(
                            s,
                            "  off-fast-path={} ops/{}",
                            bu.off_fast_path_ops,
                            human_bytes(bu.off_fast_path_bytes)
                        );
                    }
                    if bu.cast_nodes > 0 {
                        let _ = write!(s, "  casts={}", bu.cast_nodes);
                    }
                    let _ = writeln!(s);
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

/// Bytes at a glance. Budget rows sit next to each other and the reader is
/// comparing magnitudes, not auditing exact counts — the JSON carries those.
fn human_bytes(n: usize) -> String {
    const UNITS: [(&str, usize); 3] = [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)];
    for (suffix, scale) in UNITS {
        if n >= scale {
            return format!("{:.1} {suffix}", n as f64 / scale as f64);
        }
    }
    format!("{n} B")
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
/// Bytes a node's tensor occupies at its declared dtype.
fn node_bytes(graph: &Graph, id: rlx_ir::NodeId) -> usize {
    graph.node(id).shape.size_bytes().unwrap_or(0)
}

/// Per-architecture resource budget for one graph. See [`Budget`].
///
/// `off_path_kinds` are the `OpKind` debug names the dispatch claim resolved to
/// common-IR or left unsupported, passed in rather than re-derived so the bytes
/// below are the bytes of those exact nodes, not a share-of-kinds estimate.
fn compute_budget(
    graph: &Graph,
    t: FusionTarget,
    off_path_kinds: &HashSet<String>,
    fused: Option<&Graph>,
) -> Budget {
    use rlx_opt::rlx_compile::memory::{
        ArenaWidthPolicy, arena_width_policy, plan_memory_aligned, plan_memory_f32_uniform,
        plan_memory_hybrid, plan_memory_with_policy,
    };

    let policy = arena_width_policy(t);
    // 64 B is what `plan_memory` uses; holding it fixed keeps the three plans
    // below comparable, which is the whole point of `width_overhead_bytes`.
    const ALIGN: usize = 64;
    let plan = match policy {
        ArenaWidthPolicy::Native => plan_memory_aligned(graph, ALIGN),
        ArenaWidthPolicy::F32Uniform => plan_memory_f32_uniform(graph, ALIGN),
        ArenaWidthPolicy::Hybrid => plan_memory_hybrid(graph, ALIGN),
        ArenaWidthPolicy::NativeHalfWidened => {
            plan_memory_with_policy(graph, ALIGN, ArenaWidthPolicy::NativeHalfWidened)
        }
        ArenaWidthPolicy::NativeBf16Widened => {
            plan_memory_with_policy(graph, ALIGN, ArenaWidthPolicy::NativeBf16Widened)
        }
    };
    // What this architecture's storage convention costs, isolated from every
    // other planner decision: `plan_memory_with_policy` holds the options fixed
    // and varies only the width. Differencing the real plans instead would fold
    // in `pin_output_ancestors` and report a saving on f32 graphs, where the
    // width policy is free by construction. A native-policy backend is its own
    // baseline, so skip both plans.
    let width_overhead_bytes = if policy == ArenaWidthPolicy::Native {
        0
    } else {
        plan_memory_with_policy(graph, ALIGN, policy)
            .arena_size
            .saturating_sub(
                plan_memory_with_policy(graph, ALIGN, ArenaWidthPolicy::Native).arena_size,
            )
    };

    // Host boundary: inputs in, declared outputs back. `Param` is excluded —
    // a weight is uploaded once and stays resident, so counting it here would
    // swamp the per-run number with a one-off.
    let mut host_io_bytes = 0usize;
    for node in graph.nodes() {
        if matches!(node.op, rlx_ir::Op::Input { .. }) {
            host_io_bytes += node.shape.size_bytes().unwrap_or(0);
        }
    }
    for id in &graph.outputs {
        host_io_bytes += node_bytes(graph, *id);
    }

    // Off the native fast path: ops the backend lowers through portable
    // common-IR or cannot lower at all. Counted in bytes as well as ops
    // because one big tensor off the fast path costs more than ten small ones.
    let mut off_fast_path_ops = 0usize;
    let mut off_fast_path_bytes = 0usize;
    if !off_path_kinds.is_empty() {
        for node in graph.nodes() {
            if off_path_kinds.contains(&format!("{:?}", node.op.kind())) {
                off_fast_path_ops += 1;
                off_fast_path_bytes += node.shape.size_bytes().unwrap_or(0);
            }
        }
    }

    let cast_nodes = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, rlx_ir::Op::Cast { .. }))
        .count();

    let param_bytes: usize = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, rlx_ir::Op::Param { .. }))
        .map(|n| n.shape.size_bytes().unwrap_or(0))
        .sum();

    // Leaves are not dispatched: they are where data already is.
    let is_leaf = |op: &rlx_ir::Op| {
        matches!(
            op,
            rlx_ir::Op::Input { .. } | rlx_ir::Op::Param { .. } | rlx_ir::Op::Constant { .. }
        )
    };
    // Fall back to the unfused graph when fusion did not run (it is wrapped in
    // `catch_unwind`), so the figure is conservative rather than absent.
    let sched = fused.unwrap_or(graph);
    let dispatches = sched.nodes().iter().filter(|n| !is_leaf(&n.op)).count();
    let dram_bytes: usize = sched
        .nodes()
        .iter()
        .filter(|n| !is_leaf(&n.op))
        .map(|n| {
            let out = n.shape.size_bytes().unwrap_or(0);
            let ins: usize = n
                .inputs
                .iter()
                .map(|id| sched.node(*id).shape.size_bytes().unwrap_or(0))
                .sum();
            out + ins
        })
        .sum();

    Budget {
        arena_bytes: plan.arena_size,
        reuse_saved_bytes: plan.bytes_saved(),
        width_overhead_bytes,
        width_policy: match policy {
            ArenaWidthPolicy::Native => "native",
            ArenaWidthPolicy::F32Uniform => "f32-uniform",
            ArenaWidthPolicy::Hybrid => "hybrid",
            ArenaWidthPolicy::NativeHalfWidened => "native-half-widened",
            ArenaWidthPolicy::NativeBf16Widened => "native-bf16-widened",
        },
        host_io_bytes,
        off_fast_path_ops,
        off_fast_path_bytes,
        cast_nodes,
        param_bytes,
        dispatches,
        dram_bytes,
    }
}

/// Notes — never errors. A budget finding says the answer is correct and
/// something is being spent to get it.
fn budget_notes(backend: &str, b: &Budget) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    let note = |code: &str, message: String, hint: &str| Diagnostic {
        severity: Severity::Note,
        code: code.to_string(),
        message,
        node: None,
        context: None,
        hint: Some(hint.to_string()),
        backend: Some(backend.to_string()),
    };

    // Only worth saying when it is a material share of the arena; a few bytes
    // of alignment padding is not a finding.
    if b.width_overhead_bytes * 10 > b.arena_bytes && b.width_overhead_bytes > 0 {
        out.push(note(
            "budget-width",
            format!(
                "{}'s {} arena costs {} extra bytes ({:.0}% of {}) over native dtype widths",
                backend,
                b.width_policy,
                b.width_overhead_bytes,
                100.0 * b.width_overhead_bytes as f64 / b.arena_bytes.max(1) as f64,
                b.arena_bytes,
            ),
            "this architecture stores activations f32-uniform; a hybrid arena would pack              F16/BF16 activations natively, at the cost of a layout change for anything              that indexes the arena by f32 element",
        ));
    }
    if b.off_fast_path_ops > 0 {
        out.push(note(
            "budget-io",
            format!(
                "{} node(s) run off {backend}'s native fast path ({} bytes of tensor traffic)",
                b.off_fast_path_ops, b.off_fast_path_bytes,
            ),
            "a kind lowered through common-IR or left unsupported is where a host round              trip appears; `RLX_DISPATCH_REPORT=1` names them per node",
        ));
    }
    // Residency is a claim, not a guarantee. When the weights dwarf the per-run
    // boundary, whether they actually stay put is the only IO question that
    // matters — re-uploading them per step costs orders of magnitude more than
    // anything at the input/output edge.
    if b.param_bytes > 4 * b.host_io_bytes.max(1) {
        out.push(note(
            "budget-residency",
            format!(
                "{} of parameters vs {} of per-run boundary traffic on {backend}",
                human_bytes(b.param_bytes),
                human_bytes(b.host_io_bytes),
            ),
            "the per-run figure assumes these stay resident; if residency does not hold they \
             become per-step uploads, which is the difference between a fast decode and a \
             host-bound one",
        ));
    }
    // A graph moving many times its own arena is re-reading the same tensors.
    if b.arena_bytes > 0 && b.dram_bytes > 4 * b.arena_bytes {
        out.push(note(
            "budget-traffic",
            format!(
                "{} dispatch(es) move {} against a {} arena ({:.1}x) on {backend}",
                b.dispatches,
                human_bytes(b.dram_bytes),
                human_bytes(b.arena_bytes),
                b.dram_bytes as f64 / b.arena_bytes as f64,
            ),
            "fusion is what removes a round trip through memory between two ops; the \
             missed-fusion warnings above name the ones that did not collapse",
        ));
    }
    if b.cast_nodes > 0 {
        out.push(note(
            "budget-precision",
            format!(
                "{} Cast node(s): full passes over a tensor whose only product is a dtype",
                b.cast_nodes
            ),
            "casts at a precision boundary are expected; a cast that immediately feeds its              own inverse is not, and folds away",
        ));
    }
    out
}

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
                budget: None,
            };
            // Filled by the legality pass below when the backend is compiled in;
            // empty otherwise, which makes the budget's fast-path figures zero
            // rather than wrong.
            let mut off_path_kinds: HashSet<String> = HashSet::new();

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
                        // Keyed on the debug name: `OpKind` is not `Ord`, and
                        // the name is what the budget has to match nodes on.
                        if matches!(
                            kind_summary.path,
                            DispatchPath::CommonIr | DispatchPath::Unsupported
                        ) {
                            off_path_kinds.insert(format!("{:?}", kind_summary.kind));
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
            let mut fused_graph: Option<Graph> = None;
            if let Ok((fused, freport)) = fusion {
                fused_graph = Some(fused);
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

            // -- resource budget: what this graph costs on this architecture --
            //
            // Device-free by construction: the same planner the backend will
            // run, keyed on the same width policy it plans with, and the op
            // claim already resolved above.
            if graph_ok {
                let b = compute_budget(graph, t, &off_path_kinds, fused_graph.as_ref());
                if opts.budget {
                    diagnostics.extend(budget_notes(name, &b));
                }
                summary.budget = Some(b);
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
