// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pass infrastructure — trait + pipeline runner.
//!
//! A [`Pass`] is a graph-to-graph transformation. Beyond the transformation
//! itself, the runner needs to know **whether the pass changed anything**:
//! without that signal every downstream consumer must assume the worst —
//! re-verify the graph, re-stamp provenance, and throw away every cached
//! analysis — after a pass that may well have been a no-op. In a ~20-pass
//! pipeline where most passes fire on a handful of graphs, that is nearly all
//! wasted work.
//!
//! [`Pass::run_with_status`] supplies the signal, and its default is the
//! conservative one — "assume changed" — so an unmodified `impl Pass` behaves
//! exactly as before. A pass reports precisely by overriding it, which is
//! nearly free for the many passes that already begin with a trigger scan.
//!
//! Deriving the answer automatically is available as
//! [`Pass::run_detecting_change`], but it is not the default on purpose: a
//! structural fingerprint costs more than the whole-graph verification it
//! would let the runner skip (~64µs vs ~25µs on a 97-node graph), so making it
//! automatic would be a net loss in debug builds and pure overhead in release.
//! Measure before reaching for it.

use std::sync::{Arc, OnceLock, RwLock};

use rlx_ir::{Graph, OpKind};

use crate::analysis::{AnalysisManager, OpKindIndex, PreservedAnalyses};

/// Whether a pass changed the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IRStatus {
    /// The pass left the graph structurally identical.
    #[default]
    Unchanged,
    /// The pass rewrote the graph.
    Changed,
}

impl IRStatus {
    /// True when this is [`IRStatus::Changed`].
    pub fn changed(self) -> bool {
        self == IRStatus::Changed
    }
}

impl From<bool> for IRStatus {
    fn from(changed: bool) -> Self {
        if changed {
            IRStatus::Changed
        } else {
            IRStatus::Unchanged
        }
    }
}

impl From<IRStatus> for bool {
    fn from(status: IRStatus) -> Self {
        status.changed()
    }
}

impl std::ops::BitOr for IRStatus {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        IRStatus::from(self.changed() || rhs.changed())
    }
}

impl std::ops::BitOrAssign for IRStatus {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

/// What a pass produced, and what it did.
#[derive(Debug)]
pub struct PassResult {
    /// The transformed graph.
    pub graph: Graph,
    /// Whether [`graph`](Self::graph) differs from the input.
    pub ir_changed: IRStatus,
    /// Analyses the pass guarantees are still valid. Ignored when
    /// `ir_changed` is [`IRStatus::Unchanged`] — nothing can have gone stale.
    pub preserved: PreservedAnalyses,
}

impl PassResult {
    /// A result that changed nothing. Preserves every analysis.
    pub fn unchanged(graph: Graph) -> Self {
        Self {
            graph,
            ir_changed: IRStatus::Unchanged,
            preserved: PreservedAnalyses::all(),
        }
    }

    /// A result that rewrote the graph and invalidates every analysis.
    pub fn changed(graph: Graph) -> Self {
        Self {
            graph,
            ir_changed: IRStatus::Changed,
            preserved: PreservedAnalyses::none(),
        }
    }

    /// Build from an explicit status, choosing the matching default
    /// preservation set.
    pub fn from_status(graph: Graph, status: IRStatus) -> Self {
        match status {
            IRStatus::Unchanged => Self::unchanged(graph),
            IRStatus::Changed => Self::changed(graph),
        }
    }

    /// Narrow the preserved set (builder form).
    pub fn preserving(mut self, preserved: PreservedAnalyses) -> Self {
        self.preserved = preserved;
        self
    }
}

/// A graph-to-graph transformation pass.
pub trait Pass {
    /// Human-readable name for logging.
    fn name(&self) -> &str;

    /// Transform the graph. Returns a new graph (or the same if no changes).
    fn run(&self, graph: Graph) -> Graph;

    /// Transform the graph and report what happened.
    ///
    /// The default is **conservative**: it reports [`IRStatus::Changed`], which
    /// costs nothing and preserves the pre-existing behaviour exactly (re-verify,
    /// invalidate everything). Override it to report precisely.
    ///
    /// Deriving the answer automatically — fingerprint before, fingerprint
    /// after — is available as [`run_detecting_change`](Self::run_detecting_change),
    /// but it is deliberately *not* the default, because measurement says it
    /// does not pay for itself: on a 97-node graph a structural fingerprint
    /// costs ~64µs against ~25µs for the whole-graph verification it would let
    /// you skip, so two fingerprints per pass lose ~100µs to save 25µs — and in
    /// release builds `debug_assert_valid!` compiles to nothing, making the
    /// hashing pure overhead. Use it only where the answer is worth more than
    /// two graph walks, as the backend decomposition loop does to avoid
    /// spinning to its round cap.
    ///
    /// The cheap way to report precisely is to answer from the trigger check
    /// that most `Lower*` passes already perform:
    ///
    /// ```
    /// # use rlx_fusion::pass::{Pass, PassResult};
    /// # use rlx_ir::{Graph, OpKind};
    /// # struct LowerThing;
    /// # impl LowerThing { fn rewrite(&self, g: Graph) -> Graph { g } }
    /// impl Pass for LowerThing {
    ///     fn name(&self) -> &str { "lower_thing" }
    ///
    ///     fn run(&self, graph: Graph) -> Graph {
    ///         self.run_with_status(graph).graph
    ///     }
    ///
    ///     fn run_with_status(&self, graph: Graph) -> PassResult {
    ///         if !graph.nodes().iter().any(|n| n.op.kind() == OpKind::Pad) {
    ///             return PassResult::unchanged(graph);
    ///         }
    ///         PassResult::changed(self.rewrite(graph))
    ///     }
    /// }
    /// ```
    fn run_with_status(&self, graph: Graph) -> PassResult {
        if !self.can_fire(&graph) {
            return PassResult::unchanged(graph);
        }
        PassResult::changed(self.run(graph))
    }

    /// Op kinds whose presence this pass requires in order to do anything.
    ///
    /// Declaring them is the cheap way to report status precisely: most
    /// `Lower*` passes already open `run` with exactly this scan and return the
    /// graph untouched, so lifting it here is behaviour-preserving and lets the
    /// runner skip re-verification and analysis invalidation for the ~18 of 19
    /// passes that do not fire on a typical graph.
    ///
    /// An empty slice (the default) means "no cheap pre-filter" — the pass is
    /// always run and conservatively reported as changing the IR.
    ///
    /// This is a **correctness claim**: naming a kind here asserts the pass is
    /// a no-op without it. Listing a kind the pass does not truly require would
    /// silently skip a lowering and surface later as a legalization failure, so
    /// only lift a scan the pass already performs.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[]
    }

    /// Could this pass fire on `graph`? `true` when it declares no triggers.
    fn can_fire(&self, graph: &Graph) -> bool {
        let triggers = self.trigger_kinds();
        triggers.is_empty()
            || graph
                .nodes()
                .iter()
                .any(|n| triggers.contains(&n.op.kind()))
    }

    /// [`run_with_status`](Self::run_with_status) with access to cached
    /// analyses.
    ///
    /// The default answers the trigger check from a cached [`OpKindIndex`]
    /// rather than walking the graph. That turns ~20 independent `O(n)` scans
    /// per pipeline into one index build plus `O(1)` lookups — and because a
    /// pass that does not fire preserves every analysis, the index survives to
    /// answer the next pass too.
    fn run_with_analyses(&self, graph: Graph, analyses: &mut AnalysisManager) -> PassResult {
        let triggers = self.trigger_kinds();
        if !triggers.is_empty() && !analyses.get::<OpKindIndex>(&graph).contains_any(triggers) {
            return PassResult::unchanged(graph);
        }
        self.run_with_status(graph)
    }

    /// Run the pass and derive [`IRStatus`] by comparing structural
    /// fingerprints, ignoring whatever [`run_with_status`](Self::run_with_status)
    /// would report.
    ///
    /// Correct for any pass with no cooperation from it, at the price of two
    /// `O(graph)` hashes. Worth it only when a wrong "changed" costs more than
    /// that — the decomposition fixpoint in `rlx_compile::rewrite` uses it
    /// because a spurious "changed" there means up to 16 wasted rebuild rounds.
    /// For ordinary pipeline use, prefer overriding `run_with_status`.
    fn run_detecting_change(&self, graph: Graph) -> PassResult {
        let before = graph.fingerprint();
        let graph = self.run(graph);
        let status = IRStatus::from(graph.fingerprint() != before);
        PassResult::from_status(graph, status)
    }
}

/// Registry of downstream-registered IR passes, run **after** the built-in
/// fusion pipeline (so core fusion invariants hold) but **before** backend
/// legalization (so a pass's output — e.g. a custom fused op — is still lowered
/// / legalized). Empty by default: zero cost until a downstream crate registers
/// one. This is the extension seam for custom fusion / rewrite rules without
/// editing the core pass list.
///
/// A registered pass should fast-path return the graph unchanged when its
/// trigger pattern is absent — it runs on *every* compiled graph in the process.
static IR_PASS_REGISTRY: OnceLock<RwLock<Vec<Arc<dyn Pass + Send + Sync>>>> = OnceLock::new();

fn ir_pass_registry() -> &'static RwLock<Vec<Arc<dyn Pass + Send + Sync>>> {
    IR_PASS_REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

/// Register a downstream IR pass to run after the built-in fusion passes.
pub fn register_ir_pass(pass: Arc<dyn Pass + Send + Sync>) {
    ir_pass_registry().write().unwrap().push(pass);
}

/// Snapshot of registered downstream passes, in registration order.
pub fn registered_ir_passes() -> Vec<Arc<dyn Pass + Send + Sync>> {
    ir_pass_registry().read().unwrap().clone()
}

/// Run every registered downstream pass over `graph`, in registration order.
/// A no-op (returns the graph untouched) when none are registered.
pub fn run_registered_ir_passes(mut graph: Graph) -> Graph {
    for pass in registered_ir_passes() {
        let result = pass.run_with_status(graph);
        graph = result.graph;
        rlx_ir::stamp_pass_origins(&mut graph, pass.name());
        // A pass that changed nothing cannot have broken an invariant that
        // held on entry, so the (debug-only, whole-graph) verifier is
        // redundant — and a no-op registered pass is the common case.
        if result.ir_changed.changed() {
            rlx_ir::debug_assert_valid!(&graph, format!("after registered pass `{}`", pass.name()));
        }
    }
    graph
}

/// Run a sequence of passes, printing the graph after each if `verbose`.
///
/// When `RLX_FUSION_REPORT=1`, prints a [`fusion_report::FusionReport`]
/// comparing the input graph to the fused result.
///
/// In debug builds, the verifier (lifted from MAX) runs
/// after every pass via [`rlx_ir::debug_assert_valid!`] — so any optimizer
/// bug that introduces a malformed graph is caught at the boundary where it
/// was introduced. In release builds the check is not compiled in.
pub fn run_passes(graph: Graph, passes: &[&dyn Pass], verbose: bool) -> Graph {
    let mut analyses = AnalysisManager::default();
    run_passes_tracked(graph, passes, verbose, &mut analyses).graph
}

/// [`run_passes`] threading an [`AnalysisManager`], and reporting whether any
/// pass in the sequence changed the IR.
///
/// The manager is invalidated per pass according to what that pass preserved,
/// so analyses computed by one pass survive into the next whenever it is sound
/// for them to. Callers that drive their own fixpoint loop (see
/// `rlx_compile::rewrite`) should use this form: the returned [`IRStatus`] is
/// the loop's termination signal, replacing "did the unsupported-op set
/// shrink?" proxies.
pub fn run_passes_tracked(
    mut graph: Graph,
    passes: &[&dyn Pass],
    verbose: bool,
    analyses: &mut AnalysisManager,
) -> PassResult {
    let report = rlx_ir::env::flag("RLX_FUSION_REPORT");
    let before = report.then(|| graph.clone());
    let mut overall = IRStatus::Unchanged;
    let mut fired: Vec<&str> = Vec::new();

    for pass in passes {
        if verbose {
            eprintln!("--- before {} ---\n{graph}", pass.name());
        }

        let result = pass.run_with_analyses(graph, analyses);
        graph = result.graph;
        overall |= result.ir_changed;

        // Unconditional: `stamp_pass_origins` only fills nodes whose origin is
        // still `None`, so it is idempotent and near-free once stamped — and
        // skipping it on the first pass would shift attribution.
        rlx_ir::stamp_pass_origins(&mut graph, pass.name());

        if result.ir_changed.changed() {
            fired.push(pass.name());
            // Only re-verify what actually changed: a no-op pass cannot have
            // broken an invariant that held when it started.
            rlx_ir::debug_assert_valid!(&graph, format!("after pass `{}`", pass.name()));
            analyses.retain(&result.preserved, || graph.fingerprint());
        }
    }

    if verbose {
        eprintln!("--- final ---\n{graph}");
    }
    if let Some(before) = before {
        eprintln!(
            "{}",
            crate::fusion_report::FusionReport::analyze(&before, &graph)
        );
        let (hits, misses) = analyses.stats();
        eprintln!(
            "passes: {} of {} changed the IR ({}); analyses: {hits} hits / {misses} computed",
            fired.len(),
            passes.len(),
            if fired.is_empty() {
                "none".to_string()
            } else {
                fired.join(", ")
            },
        );
    }

    PassResult {
        graph,
        ir_changed: overall,
        preserved: PreservedAnalyses::none(),
    }
}

#[cfg(test)]
mod status_tests {
    use super::*;
    use crate::analysis::{OpKindIndex, UseCounts};
    use rlx_ir::{DType, Op, Shape};

    fn one_node(name: &str) -> Graph {
        let mut g = Graph::new(name);
        let x = g.input("x", Shape::new(&[4], DType::F32));
        g.set_outputs(vec![x]);
        g
    }

    /// Does nothing, and says so — the shape of a pass whose trigger is absent.
    struct NoOp;
    impl Pass for NoOp {
        fn name(&self) -> &str {
            "noop"
        }
        fn run(&self, graph: Graph) -> Graph {
            graph
        }
        fn run_with_status(&self, graph: Graph) -> PassResult {
            PassResult::unchanged(graph)
        }
    }

    /// Reports nothing, so the conservative default applies.
    struct SilentNoOp;
    impl Pass for SilentNoOp {
        fn name(&self) -> &str {
            "silent_noop"
        }
        fn run(&self, graph: Graph) -> Graph {
            graph
        }
    }

    /// Rebuilds the graph node-for-node. Structurally identical output, so it
    /// must still report `Unchanged` — "returned a fresh `Graph` value" is not
    /// the same question as "changed the IR", and conflating them would
    /// invalidate the analysis cache on every rebuild-style pass.
    struct Rebuild;
    impl Pass for Rebuild {
        fn name(&self) -> &str {
            "rebuild"
        }
        fn run(&self, graph: Graph) -> Graph {
            let mut out = Graph::new(&graph.name);
            for node in graph.nodes() {
                out.add_node(node.op.clone(), node.inputs.clone(), node.shape.clone());
            }
            out.set_outputs(graph.outputs.clone());
            out
        }
        fn run_with_status(&self, graph: Graph) -> PassResult {
            PassResult::unchanged(self.run(graph))
        }
    }

    struct AppendNeg;
    impl Pass for AppendNeg {
        fn name(&self) -> &str {
            "append_neg"
        }
        fn run(&self, mut graph: Graph) -> Graph {
            let out = graph.outputs[0];
            let shape = graph.node(out).shape.clone();
            let neg = graph.add_node(
                Op::Activation(rlx_ir::op::Activation::Neg),
                vec![out],
                shape,
            );
            graph.set_outputs(vec![neg]);
            graph
        }
    }

    /// Reports precisely instead of paying for two fingerprints, and keeps the
    /// analyses its edits cannot have invalidated.
    struct SelfReporting;
    impl Pass for SelfReporting {
        fn name(&self) -> &str {
            "self_reporting"
        }
        fn run(&self, graph: Graph) -> Graph {
            self.run_with_status(graph).graph
        }
        fn run_with_status(&self, graph: Graph) -> PassResult {
            PassResult::unchanged(graph).preserving(PreservedAnalyses::preserving::<UseCounts>())
        }
    }

    #[test]
    fn detection_helper_sees_a_noop() {
        let result = NoOp.run_detecting_change(one_node("g"));
        assert_eq!(result.ir_changed, IRStatus::Unchanged);
        assert!(result.preserved.is_all());
    }

    #[test]
    fn detection_helper_sees_through_a_structure_preserving_rebuild() {
        let result = Rebuild.run_detecting_change(one_node("g"));
        assert_eq!(
            result.ir_changed,
            IRStatus::Unchanged,
            "a rebuilt-but-identical graph is not a change"
        );
    }

    #[test]
    fn detection_helper_sees_a_real_edit() {
        let result = AppendNeg.run_detecting_change(one_node("g"));
        assert_eq!(result.ir_changed, IRStatus::Changed);
        assert!(!result.preserved.is_all());
    }

    #[test]
    fn override_is_honoured() {
        let result = SelfReporting.run_with_status(one_node("g"));
        assert_eq!(result.ir_changed, IRStatus::Unchanged);
        assert!(result.preserved.preserves::<UseCounts>());
        assert!(!result.preserved.preserves::<OpKindIndex>());
    }

    #[test]
    fn runner_reports_whether_any_pass_fired() {
        let mut analyses = AnalysisManager::default();

        let quiet = run_passes_tracked(one_node("g"), &[&NoOp, &Rebuild], false, &mut analyses);
        assert_eq!(quiet.ir_changed, IRStatus::Unchanged);

        let loud = run_passes_tracked(one_node("g"), &[&NoOp, &AppendNeg], false, &mut analyses);
        assert_eq!(loud.ir_changed, IRStatus::Changed);
    }

    #[test]
    fn analyses_survive_a_pipeline_of_noops() {
        let graph = one_node("g");
        let mut analyses = AnalysisManager::default();
        let _ = analyses.get::<UseCounts>(&graph);
        let _ = analyses.get::<OpKindIndex>(&graph);

        let result = run_passes_tracked(graph, &[&NoOp, &Rebuild, &NoOp], false, &mut analyses);
        assert_eq!(result.ir_changed, IRStatus::Unchanged);
        assert_eq!(
            analyses.len(),
            2,
            "no pass changed the IR, so nothing may be invalidated"
        );
    }

    #[test]
    fn a_changing_pass_invalidates_the_cache() {
        let graph = one_node("g");
        let mut analyses = AnalysisManager::default();
        let _ = analyses.get::<UseCounts>(&graph);
        let _ = analyses.get::<OpKindIndex>(&graph);

        let result = run_passes_tracked(graph, &[&AppendNeg], false, &mut analyses);
        assert_eq!(result.ir_changed, IRStatus::Changed);
        assert!(analyses.is_empty(), "AppendNeg preserves nothing");

        // And the cache is now keyed to the rewritten graph, so a fresh
        // lookup is sound (this would trip the debug fingerprint check if
        // `retain` had not re-baselined it).
        let _ = analyses.get::<UseCounts>(&result.graph);
    }

    #[test]
    fn the_default_is_conservative_not_clever() {
        // An unmodified `impl Pass` must keep the old behaviour: assume the
        // worst, so nothing downstream can act on a stale analysis.
        let result = SilentNoOp.run_with_status(one_node("g"));
        assert_eq!(result.ir_changed, IRStatus::Changed);
        assert!(!result.preserved.is_all());

        // ...and the opt-in helper still gets the true answer for it.
        assert_eq!(
            SilentNoOp.run_detecting_change(one_node("g")).ir_changed,
            IRStatus::Unchanged
        );
    }

    #[test]
    fn ir_status_or_accumulates() {
        let mut status = IRStatus::Unchanged;
        status |= IRStatus::Unchanged;
        assert_eq!(status, IRStatus::Unchanged);
        status |= IRStatus::Changed;
        assert_eq!(status, IRStatus::Changed);
        status |= IRStatus::Unchanged;
        assert_eq!(status, IRStatus::Changed, "changed is sticky");
    }
}

#[cfg(test)]
mod registry_tests {
    use super::*;
    use rlx_ir::{DType, Op, Shape};

    /// Sentinel-gated pass: only transforms a graph containing an input named
    /// `__ir_pass_sentinel`; a strict pass-through otherwise, so registering it
    /// globally cannot contaminate other tests' graphs. When triggered it
    /// negates the single output.
    struct SentinelNegate;
    impl Pass for SentinelNegate {
        fn name(&self) -> &str {
            "sentinel_negate"
        }
        fn run(&self, mut graph: Graph) -> Graph {
            let has_sentinel = graph
                .nodes()
                .iter()
                .any(|n| matches!(&n.op, Op::Input { name } if name == "__ir_pass_sentinel"));
            if !has_sentinel {
                return graph; // fast-path: untouched, safe for unrelated graphs
            }
            let out = graph.outputs[0];
            let shape = graph.node(out).shape.clone();
            let neg = graph.add_node(
                Op::Activation(rlx_ir::op::Activation::Neg),
                vec![out],
                shape,
            );
            graph.set_outputs(vec![neg]);
            graph
        }
    }

    #[test]
    fn registered_pass_runs_only_on_its_trigger() {
        register_ir_pass(Arc::new(SentinelNegate));

        // Graph WITHOUT the sentinel: the registered pass must leave it alone.
        let mut plain = Graph::new("plain");
        let a = plain.input("a", Shape::new(&[2], DType::F32));
        plain.set_outputs(vec![a]);
        let plain_len = plain.len();
        let plain_out = run_registered_ir_passes(plain);
        assert_eq!(
            plain_out.len(),
            plain_len,
            "unrelated graph must be untouched"
        );

        // Graph WITH the sentinel: the pass appends a Neg and repoints the output.
        let mut g = Graph::new("triggered");
        let s = g.input("__ir_pass_sentinel", Shape::new(&[2], DType::F32));
        g.set_outputs(vec![s]);
        let out = run_registered_ir_passes(g);
        assert!(
            matches!(
                out.node(out.outputs[0]).op,
                Op::Activation(rlx_ir::op::Activation::Neg)
            ),
            "registered pass should have negated the output"
        );
    }
}

/// Every built-in pass, keyed by its [`Pass::name`].
///
/// Exists so a test fixture can name the passes it wants by string rather than
/// by Rust type — the missing piece between [`rlx_ir::text`] (which can parse
/// and print IR) and pass tests that live as files instead of as code.
///
/// Only passes that are constructible with no configuration appear here;
/// parameterised ones (`UnfuseElementwiseRegions` variants beyond the CPU
/// default, the dispatch-configured lowerings) still need a Rust harness.
pub fn builtin_passes() -> Vec<&'static dyn Pass> {
    use crate::fusion::*;
    use crate::{
        control_flow::{LowerControlFlow, LowerScan},
        lower_axial_rope2d::LowerAxialRope2d,
        lower_backward_ops::LowerBackwardOps,
        lower_cumulative::LowerCumulative,
        lower_dot_general::LowerDotGeneral,
        lower_fake_quantize::LowerFakeQuantize,
        lower_fma::LowerFma,
        lower_histogram::LowerHistogram,
        lower_loss_ops::LowerSoftmaxCrossEntropy,
        lower_pad::LowerPad,
        lower_reduce_axes::LowerNonLastAxisReduce,
        lower_scaled_grouped_matmul::LowerScaledGroupedMatMul,
        lower_slice::LowerSlice,
        lower_spectral::LowerSpectral,
        lower_spline_activation::LowerSplineActivation,
        lower_spline_backward::LowerSplineActivationBackward,
        lower_structural::LowerStructural,
        lower_synth_matmul::LowerSynthMatMul,
        lower_synth_matmul_backward::LowerSynthMatMulBackward,
        lower_synth_reconstruct::LowerSynthReconstruct,
        lower_vae_ops::{LowerBatchNormInference, LowerGroupNorm, LowerResizeNearest2x},
    };

    vec![
        &LowerAxialRope2d,
        &LowerBackwardOps,
        &LowerControlFlow,
        &LowerCumulative,
        &LowerDotGeneral,
        &LowerFakeQuantize,
        &LowerFma,
        &LowerHistogram,
        &LowerNonLastAxisReduce,
        &LowerPad,
        &LowerScaledGroupedMatMul,
        &LowerScan,
        &LowerSlice,
        &LowerSoftmaxCrossEntropy,
        &LowerSpectral,
        &LowerSplineActivation,
        &LowerSplineActivationBackward,
        &LowerStructural,
        &LowerSynthMatMul,
        &LowerSynthMatMulBackward,
        &LowerSynthReconstruct,
        &LowerBatchNormInference,
        &LowerGroupNorm,
        &LowerResizeNearest2x,
        &FuseAdaLayerNorm,
        &FuseAttentionBlock,
        &FuseConvAffineAct,
        &FuseConvBiasAct,
        &FuseGatedResidual,
        &FuseMatMulBiasAct,
        &FuseMatMulResidual,
        &FuseResidualLN,
        &FuseResidualRmsNorm,
        &FuseRmsNormReshape,
        &FuseSharedInputMatMul,
        &FuseSwiGLU,
        &FuseSwiGLUDualMatmul,
        &FuseTransformerLayer,
        &MarkElementwiseRegions,
        &UnfuseElementwiseRegions::FOR_CPU,
    ]
}

/// Look a built-in pass up by the name it reports.
pub fn pass_by_name(name: &str) -> Option<&'static dyn Pass> {
    builtin_passes().into_iter().find(|p| p.name() == name)
}
