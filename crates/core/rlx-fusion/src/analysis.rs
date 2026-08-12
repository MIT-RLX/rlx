// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cached graph analyses with pass-driven invalidation.
//!
//! A [`CompilePipeline`](../../rlx_compile/compiler/struct.CompilePipeline.html)
//! run is ~20 passes deep, and most of them open by asking the same two
//! questions: *who uses this node?* and *does this graph even contain the op I
//! rewrite?* Answered directly those are `O(n²)` and `O(n)` respectively, per
//! pass, recomputed from scratch every time —
//! [`Graph::users`](rlx_ir::Graph::users) walks every node's input list to
//! answer for one node, so a pass that asks for every node walks the graph `n`
//! times.
//!
//! [`AnalysisManager`] computes each such fact once and hands out cached
//! references until a pass reports that it changed the IR. Invalidation is
//! driven by the pass runner (see [`crate::pass::run_passes_tracked`]), not by
//! content addressing: a pass returning
//! [`IRStatus::Unchanged`](crate::pass::IRStatus::Unchanged) preserves
//! everything, and a pass returning `Changed` drops every analysis it did not
//! explicitly name in its [`PreservedAnalyses`].
//!
//! # Adding an analysis
//!
//! Implement [`Analysis`] and ask for it by type:
//!
//! ```
//! use rlx_fusion::analysis::{Analysis, AnalysisManager, UseCounts};
//! # use rlx_ir::{DType, Graph, Shape};
//! # let mut graph = Graph::new("g");
//! # let x = graph.input("x", Shape::new(&[4], DType::F32));
//! # graph.set_outputs(vec![x]);
//! let mut analyses = AnalysisManager::default();
//! let uses = analyses.get::<UseCounts>(&graph);
//! assert_eq!(uses.use_count(x), 0);
//! ```

use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};

use rlx_ir::{Graph, NodeId, OpKind};

/// A fact derived from a [`Graph`], cached by [`AnalysisManager`] until a pass
/// invalidates it.
///
/// Implementations must be pure functions of the graph — the cache assumes
/// that recomputing on an unchanged graph would produce an identical result.
pub trait Analysis: Any + Send + Sync {
    /// Derive the analysis from scratch.
    fn compute(graph: &Graph) -> Self
    where
        Self: Sized;

    /// Human-readable name, for cache statistics and logs.
    fn name() -> &'static str
    where
        Self: Sized;
}

/// The set of analyses a pass leaves intact when it reports
/// [`IRStatus::Changed`](crate::pass::IRStatus::Changed).
///
/// Default is [`none`](Self::none) — the conservative choice. Naming an
/// analysis here is a correctness claim: it says the pass's edits cannot have
/// changed that fact.
#[derive(Debug, Clone, Default)]
pub struct PreservedAnalyses {
    all: bool,
    ids: HashSet<TypeId>,
}

impl PreservedAnalyses {
    /// Nothing survives — every cached analysis is recomputed on next use.
    pub fn none() -> Self {
        Self {
            all: false,
            ids: HashSet::new(),
        }
    }

    /// Everything survives. Only correct when the pass genuinely did not touch
    /// the IR; the pass runner applies this automatically for
    /// [`IRStatus::Unchanged`](crate::pass::IRStatus::Unchanged).
    pub fn all() -> Self {
        Self {
            all: true,
            ids: HashSet::new(),
        }
    }

    /// Preserve exactly one analysis.
    pub fn preserving<A: Analysis>() -> Self {
        Self::none().and::<A>()
    }

    /// Add another preserved analysis (builder form).
    pub fn and<A: Analysis>(mut self) -> Self {
        self.ids.insert(TypeId::of::<A>());
        self
    }

    /// Does this set preserve `A`?
    pub fn preserves<A: Analysis>(&self) -> bool {
        self.all || self.ids.contains(&TypeId::of::<A>())
    }

    fn preserves_id(&self, id: TypeId) -> bool {
        self.all || self.ids.contains(&id)
    }

    /// True when nothing at all is invalidated.
    pub fn is_all(&self) -> bool {
        self.all
    }
}

/// Cache of [`Analysis`] results for one graph.
///
/// Not thread-safe by design — a pass pipeline is sequential, and an
/// `RwLock` per lookup would cost more than the analyses being cached.
#[derive(Default)]
pub struct AnalysisManager {
    cache: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    /// Structural fingerprint the cached entries were computed against.
    ///
    /// Debug-only, and gated as such on purpose: it exists solely to serve the
    /// misuse assertion in [`get`](Self::get), while maintaining it costs a
    /// `graph.fingerprint()` — the most expensive hash in the crate. Left
    /// ungated it is computed and never read in release builds, which measured
    /// as ~20% of `fuse()` at 6k nodes.
    #[cfg(debug_assertions)]
    valid_for: Option<u64>,
    /// Debug-only: has the current baseline been checked against a real graph
    /// yet? Reset whenever the baseline moves. See [`AnalysisManager::get`].
    #[cfg(debug_assertions)]
    verified_against_baseline: bool,
    hits: usize,
    misses: usize,
}

impl AnalysisManager {
    /// Get `A` for `graph`, computing and caching it on first use.
    ///
    /// In debug builds this asserts that the graph has not silently changed
    /// underneath the cache — passing a rewritten graph without going through
    /// [`retain`](Self::retain) or [`invalidate_all`](Self::invalidate_all) is
    /// a bug that would otherwise surface as stale analysis data much later.
    ///
    /// The check runs **once per baseline**, not once per lookup: a structural
    /// fingerprint costs more than the analyses being cached, so verifying on
    /// every `get` would cost more than the cache saves even in debug builds.
    /// Each time [`retain`](Self::retain) re-baselines, the next `get`
    /// re-verifies.
    pub fn get<A: Analysis>(&mut self, graph: &Graph) -> &A {
        #[cfg(debug_assertions)]
        if !self.verified_against_baseline
            && let Some(expected) = self.valid_for
        {
            self.verified_against_baseline = true;
            debug_assert_eq!(
                expected,
                graph.fingerprint(),
                "AnalysisManager holds results for a different graph: the IR was \
                 rewritten without invalidating the cache. Run passes through \
                 `run_passes_tracked`, or call `invalidate_all` after an \
                 out-of-band edit."
            );
        }

        let id = TypeId::of::<A>();
        if !self.cache.contains_key(&id) {
            self.misses += 1;
            let computed = A::compute(graph);
            self.cache.insert(id, Box::new(computed));
            #[cfg(debug_assertions)]
            if self.valid_for.is_none() {
                self.valid_for = Some(graph.fingerprint());
            }
        } else {
            self.hits += 1;
        }

        self.cache
            .get(&id)
            .and_then(|boxed| boxed.downcast_ref::<A>())
            .expect("analysis cached under its own TypeId")
    }

    /// Drop every cached analysis.
    pub fn invalidate_all(&mut self) {
        self.cache.clear();
        #[cfg(debug_assertions)]
        {
            self.valid_for = None;
            self.verified_against_baseline = false;
        }
    }

    /// Drop one cached analysis.
    pub fn invalidate<A: Analysis>(&mut self) {
        self.cache.remove(&TypeId::of::<A>());
    }

    /// Apply a pass's [`PreservedAnalyses`]: drop everything it does not name.
    ///
    /// `fingerprint` yields the fingerprint of the graph *after* the pass, so
    /// surviving entries stay paired with the IR they describe. It is a
    /// closure, and it is only called when something actually survives —
    /// computing a structural hash costs more than most passes save, and an
    /// empty cache has no baseline to maintain. A pipeline that asks for no
    /// analyses therefore pays nothing at all for this call.
    pub fn retain(&mut self, preserved: &PreservedAnalyses, fingerprint: impl FnOnce() -> u64) {
        if !preserved.is_all() {
            self.cache.retain(|id, _| preserved.preserves_id(*id));
        }
        #[cfg(debug_assertions)]
        {
            self.verified_against_baseline = false;
            self.valid_for = if self.cache.is_empty() {
                None
            } else {
                Some(fingerprint())
            };
        }
        // `fingerprint` is deliberately unused in release: nothing reads the
        // baseline there, so paying for the hash would be pure overhead.
        #[cfg(not(debug_assertions))]
        let _ = fingerprint;
    }

    /// `(hits, misses)` since construction — for pipeline reporting.
    pub fn stats(&self) -> (usize, usize) {
        (self.hits, self.misses)
    }

    /// Is `A` currently cached? Lets a test observe whether a pass ever asked
    /// for an analysis, which is the only external signal that a trigger check
    /// short-circuited rather than the matcher simply finding nothing.
    pub fn is_cached<A: Analysis>(&self) -> bool {
        self.cache.contains_key(&TypeId::of::<A>())
    }

    /// Number of analyses currently cached.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// True when nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

// ── Shipped analyses ────────────────────────────────────────────

/// Consumers of every node, computed in one pass.
///
/// [`Graph::users`](rlx_ir::Graph::users) scans all `n` nodes to answer for a
/// single node, so the common "for each node, who uses it?" loop is `O(n²)`.
/// This builds the whole relation in `O(edges)` once.
///
/// A node used twice by the same consumer (e.g. `Mul(x, x)`) counts twice in
/// [`use_count`](Self::use_count) and appears once in [`users`](Self::users) —
/// matching [`Graph::use_count`](rlx_ir::Graph::use_count), which counts
/// consumer nodes rather than edges.
pub struct UseCounts {
    users: Vec<Vec<NodeId>>,
    counts: Vec<usize>,
    /// Nodes reachable from the graph outputs (i.e. not dead).
    live: Vec<bool>,
}

impl Analysis for UseCounts {
    fn name() -> &'static str {
        "use_counts"
    }

    fn compute(graph: &Graph) -> Self {
        let n = graph.len();
        let mut users = vec![Vec::new(); n];
        let mut counts = vec![0usize; n];

        for node in graph.nodes() {
            // Dedup within one consumer so `Mul(x, x)` lists that consumer
            // once, matching `Graph::users`.
            let mut seen: Vec<NodeId> = Vec::with_capacity(node.inputs.len());
            for &input in &node.inputs {
                let idx = input.0 as usize;
                if idx >= n {
                    continue; // malformed graph; `verify` reports it properly
                }
                if !seen.contains(&input) {
                    seen.push(input);
                    users[idx].push(node.id);
                    counts[idx] += 1;
                }
            }
        }

        // Liveness: reverse reachability from the outputs. Nodes are in topo
        // order, so one reverse sweep suffices.
        let mut live = vec![false; n];
        for out in &graph.outputs {
            if (out.0 as usize) < n {
                live[out.0 as usize] = true;
            }
        }
        for node in graph.nodes().iter().rev() {
            if live[node.id.0 as usize] {
                for &input in &node.inputs {
                    if (input.0 as usize) < n {
                        live[input.0 as usize] = true;
                    }
                }
            }
        }

        Self {
            users,
            counts,
            live,
        }
    }
}

impl UseCounts {
    /// Nodes consuming `id`'s output, in topological order.
    pub fn users(&self, id: NodeId) -> &[NodeId] {
        self.users
            .get(id.0 as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// How many nodes consume `id`'s output.
    pub fn use_count(&self, id: NodeId) -> usize {
        self.counts.get(id.0 as usize).copied().unwrap_or(0)
    }

    /// True when `id` has exactly one consumer — the precondition almost every
    /// fusion pattern checks before absorbing a producer into its consumer.
    pub fn has_single_use(&self, id: NodeId) -> bool {
        self.use_count(id) == 1
    }

    /// Is `id` reachable from the graph outputs?
    pub fn is_live(&self, id: NodeId) -> bool {
        self.live.get(id.0 as usize).copied().unwrap_or(false)
    }

    /// Every node not reachable from the outputs, in topological order.
    pub fn dead_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.live
            .iter()
            .enumerate()
            .filter(|&(_, &live)| !live)
            .map(|(i, _)| NodeId(i as u32))
    }
}

/// [`UseCounts`] built on first use.
///
/// A fusion pass asks "how many consumers does this node have?" only once it
/// has a pattern candidate in hand — and most passes find no candidate at all
/// on any given graph. Building the relation eagerly at the top of every pass
/// costs an `O(edges)` sweep per pass to answer zero questions, which measured
/// as a **30% regression** on the CPU pipeline (6.4ms → 8.4ms at 6k nodes)
/// when it was done that way.
///
/// This defers the build to the first query, so a pass that never matches
/// pays nothing, and one that matches repeatedly still builds it once.
///
/// ```
/// # use rlx_fusion::analysis::LazyUseCounts;
/// # use rlx_ir::{DType, Graph, Shape};
/// # let mut g = Graph::new("g");
/// # let x = g.input("x", Shape::new(&[4], DType::F32));
/// # g.set_outputs(vec![x]);
/// let uses = LazyUseCounts::new(&g);
/// assert!(!uses.was_built());
/// assert_eq!(uses.use_count(x), 0);
/// assert!(uses.was_built());
/// ```
pub struct LazyUseCounts<'g> {
    source: Source<'g>,
    cell: std::cell::OnceCell<UseCounts>,
}

enum Source<'g> {
    /// Compute from this graph on first query.
    Deferred(&'g Graph),
    /// Already computed — typically by [`AnalysisManager`], shared across the
    /// whole pipeline rather than rebuilt per pass.
    Shared(&'g UseCounts),
}

impl<'g> LazyUseCounts<'g> {
    pub fn new(graph: &'g Graph) -> Self {
        Self {
            source: Source::Deferred(graph),
            cell: std::cell::OnceCell::new(),
        }
    }

    /// Wrap counts someone else already computed.
    pub fn shared(uses: &'g UseCounts) -> Self {
        Self {
            source: Source::Shared(uses),
            cell: std::cell::OnceCell::new(),
        }
    }

    /// Use `shared` when a cached relation is available, otherwise defer to a
    /// per-pass build against `graph`.
    ///
    /// This is the signature a pass wants: taking `Option<&UseCounts>` rather
    /// than borrowing the [`AnalysisManager`] is what lets the caller do
    /// `let u = analyses.get::<UseCounts>(&graph); pass.fuse_with(graph, Some(u))`
    /// — the `get` borrows the *manager*, so `graph` stays free to move into
    /// the pass.
    pub fn from_shared(shared: Option<&'g UseCounts>, graph: &'g Graph) -> Self {
        match shared {
            Some(uses) => Self::shared(uses),
            None => Self::new(graph),
        }
    }

    /// The underlying counts, computing them if this is the first query.
    pub fn get(&self) -> &UseCounts {
        match self.source {
            Source::Shared(uses) => uses,
            Source::Deferred(graph) => self.cell.get_or_init(|| UseCounts::compute(graph)),
        }
    }

    /// Did a query actually force a build? Always `false` for a shared source,
    /// which had nothing to build. For tests and reporting.
    pub fn was_built(&self) -> bool {
        self.cell.get().is_some()
    }

    pub fn users(&self, id: NodeId) -> &[NodeId] {
        self.get().users(id)
    }

    pub fn use_count(&self, id: NodeId) -> usize {
        self.get().use_count(id)
    }

    pub fn has_single_use(&self, id: NodeId) -> bool {
        self.get().has_single_use(id)
    }

    pub fn is_live(&self, id: NodeId) -> bool {
        self.get().is_live(id)
    }
}

/// Which [`OpKind`]s the graph contains, and where.
///
/// Every `Lower*` pass opens by scanning for its trigger kind so it can
/// fast-path out. With ~30 such passes in a pipeline that is ~30 full graph
/// walks before any work happens; this replaces them with one.
pub struct OpKindIndex {
    by_kind: HashMap<OpKind, Vec<NodeId>>,
}

impl Analysis for OpKindIndex {
    fn name() -> &'static str {
        "op_kind_index"
    }

    fn compute(graph: &Graph) -> Self {
        let mut by_kind: HashMap<OpKind, Vec<NodeId>> = HashMap::new();
        for node in graph.nodes() {
            by_kind.entry(node.op.kind()).or_default().push(node.id);
        }
        Self { by_kind }
    }
}

impl OpKindIndex {
    /// Does the graph contain any node of this kind?
    pub fn contains(&self, kind: OpKind) -> bool {
        self.by_kind.contains_key(&kind)
    }

    /// Does the graph contain any node of any of these kinds?
    pub fn contains_any(&self, kinds: &[OpKind]) -> bool {
        kinds.iter().any(|&k| self.contains(k))
    }

    /// Every node of this kind, in topological order.
    pub fn nodes_of(&self, kind: OpKind) -> &[NodeId] {
        self.by_kind.get(&kind).map(Vec::as_slice).unwrap_or(&[])
    }

    /// How many distinct op kinds the graph uses.
    pub fn distinct_kinds(&self) -> usize {
        self.by_kind.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::{Activation, BinaryOp};
    use rlx_ir::{DType, Op, Shape};

    fn diamond() -> Graph {
        // x ─┬─> gelu ─┐
        //    └─> relu ─┴─> add   (plus a dead `neg` off x)
        let mut g = Graph::new("diamond");
        let shape = Shape::new(&[4], DType::F32);
        let x = g.input("x", shape.clone());
        let gelu = g.add_node(Op::Activation(Activation::Gelu), vec![x], shape.clone());
        let relu = g.add_node(Op::Activation(Activation::Relu), vec![x], shape.clone());
        let add = g.add_node(Op::Binary(BinaryOp::Add), vec![gelu, relu], shape.clone());
        let _dead = g.add_node(Op::Activation(Activation::Neg), vec![x], shape);
        g.set_outputs(vec![add]);
        g
    }

    #[test]
    fn use_counts_match_the_graph_helpers() {
        let g = diamond();
        let uses = UseCounts::compute(&g);
        for node in g.nodes() {
            assert_eq!(
                uses.use_count(node.id),
                g.use_count(node.id),
                "use_count mismatch at {}",
                node.id
            );
            assert_eq!(uses.users(node.id), g.users(node.id).as_slice());
        }
    }

    #[test]
    fn repeated_operand_counts_once_like_graph_users() {
        let mut g = Graph::new("square");
        let shape = Shape::new(&[4], DType::F32);
        let x = g.input("x", shape.clone());
        let sq = g.add_node(Op::Binary(BinaryOp::Mul), vec![x, x], shape);
        g.set_outputs(vec![sq]);

        let uses = UseCounts::compute(&g);
        assert_eq!(uses.use_count(x), g.use_count(x));
        assert_eq!(uses.users(x), g.users(x).as_slice());
    }

    #[test]
    fn liveness_finds_the_dead_node() {
        let g = diamond();
        let uses = UseCounts::compute(&g);
        let dead: Vec<_> = uses.dead_nodes().collect();
        assert_eq!(dead, vec![NodeId(4)], "only the unused Neg is dead");
        assert!(uses.is_live(NodeId(0)));
        assert!(uses.has_single_use(NodeId(1)));
    }

    #[test]
    fn op_kind_index_locates_nodes() {
        let g = diamond();
        let idx = OpKindIndex::compute(&g);
        assert!(idx.contains(OpKind::Activation));
        assert!(idx.contains(OpKind::Binary));
        assert!(!idx.contains(OpKind::MatMul));
        assert_eq!(idx.nodes_of(OpKind::Binary), &[NodeId(3)]);
        assert_eq!(idx.nodes_of(OpKind::Activation).len(), 3);
        assert!(idx.contains_any(&[OpKind::MatMul, OpKind::Binary]));
    }

    #[test]
    fn manager_caches_and_reports_hits() {
        let g = diamond();
        let mut am = AnalysisManager::default();
        let _ = am.get::<UseCounts>(&g);
        let _ = am.get::<UseCounts>(&g);
        let _ = am.get::<OpKindIndex>(&g);
        assert_eq!(am.stats(), (1, 2), "one hit, two distinct computations");
        assert_eq!(am.len(), 2);
    }

    #[test]
    fn retain_drops_only_unpreserved() {
        let g = diamond();
        let mut am = AnalysisManager::default();
        let _ = am.get::<UseCounts>(&g);
        let _ = am.get::<OpKindIndex>(&g);

        am.retain(&PreservedAnalyses::preserving::<UseCounts>(), || {
            g.fingerprint()
        });
        assert_eq!(am.len(), 1);

        // The surviving one is a cache hit; the dropped one recomputes.
        let (h0, m0) = am.stats();
        let _ = am.get::<UseCounts>(&g);
        let _ = am.get::<OpKindIndex>(&g);
        assert_eq!(am.stats(), (h0 + 1, m0 + 1));
    }

    #[test]
    fn preserve_all_keeps_everything() {
        let g = diamond();
        let mut am = AnalysisManager::default();
        let _ = am.get::<UseCounts>(&g);
        let _ = am.get::<OpKindIndex>(&g);
        am.retain(&PreservedAnalyses::all(), || g.fingerprint());
        assert_eq!(am.len(), 2);
    }
}
