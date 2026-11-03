// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Taps**: observe and rewrite every graph compiled inside a scope, and get
//! extra outputs back from it, without the code that compiles and runs it
//! changing a line.
//!
//! The motivating case: exposing a model's pre-head representation (see
//! [`rlx_ir::head`]) from crates whose forward builds a graph inline, compiles
//! it and reads its outputs its own way (`pop()`, an index, a named list).
//! Before this module each such crate carried two hook lines around its
//! compile and run. Now a caller wraps the *unchanged* forward:
//!
//! ```ignore
//! let out = rlx_runtime::taps::with_tap(Rc::new(my_tap), || model.forward(input, batch));
//! ```
//!
//! and every [`Session::compile`](crate::Session::compile) /
//! [`Session::compile_with`](crate::Session::compile_with) inside the closure
//! passes its graph through [`GraphTap::on_compile`], which may append outputs.
//! The [`CompiledGraph`](crate::CompiledGraph) remembers how many; its `run` /
//! `run_typed` strip exactly those trailing outputs before returning — the
//! caller sees the outputs it always did — and hand them to
//! [`GraphTap::on_outputs`].
//!
//! Outside a [`with_tap`] scope every hook is a no-op, so normal compiles and
//! runs are untouched. Scopes are per thread and nest (the inner tap wins).

use std::cell::RefCell;
use std::rc::Rc;

use rlx_ir::{DType, Graph};

/// A scoped observer of compiled graphs.
pub trait GraphTap {
    /// Called with each graph compiled in scope. Return the graph to compile
    /// and the number of outputs the tap APPENDED to it (stripped from every
    /// run of the compiled graph, and passed to [`Self::on_outputs`]).
    fn on_compile(&self, graph: Graph) -> (Graph, usize);

    /// Called after each run of a tapped graph with the outputs the tap
    /// appended, as `(little-endian bytes, dtype)` — f32 runs are reported as
    /// `DType::F32` bytes, so one callback serves `run` and `run_typed`.
    fn on_outputs(&self, extra: Vec<(Vec<u8>, DType)>);
}

thread_local! {
    static STACK: RefCell<Vec<Rc<dyn GraphTap>>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` with `tap` observing every graph it compiles and runs.
pub fn with_tap<T>(tap: Rc<dyn GraphTap>, f: impl FnOnce() -> T) -> T {
    struct Pop;
    impl Drop for Pop {
        fn drop(&mut self) {
            STACK.with(|s| {
                s.borrow_mut().pop();
            });
        }
    }
    STACK.with(|s| s.borrow_mut().push(tap));
    let _pop = Pop; // popped even if `f` unwinds
    f()
}

/// [`with_tap`] as a guard: `tap` observes every graph compiled on this
/// thread until the returned scope drops. For a caller that compiles across
/// an `await` (a browser's WebGPU run), where a closure cannot hold the scope;
/// scopes must drop in reverse order of entry, as [`with_tap`]'s nest.
pub fn enter(tap: Rc<dyn GraphTap>) -> TapScope {
    STACK.with(|s| s.borrow_mut().push(tap));
    TapScope(())
}

/// The scope [`enter`] opens; dropping it removes the tap.
pub struct TapScope(());

impl Drop for TapScope {
    fn drop(&mut self) {
        STACK.with(|s| {
            s.borrow_mut().pop();
        });
    }
}

fn current() -> Option<Rc<dyn GraphTap>> {
    STACK.with(|s| s.borrow().last().cloned())
}

/// Hook for the compile paths: the graph to compile and how many outputs the
/// active tap appended (0 outside a scope).
pub(crate) fn on_compile(graph: Graph) -> (Graph, usize) {
    match current() {
        Some(t) => t.on_compile(graph),
        None => (graph, 0),
    }
}

thread_local! {
    static DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Compile through `f`, tapping only the OUTERMOST compile: `compile` calls
/// `compile_with`, which recurses into prepare/main sub-graphs — those must
/// pass through untouched, or a tap would see (and extend) fragments.
pub(crate) fn tapped(
    graph: Graph,
    f: impl FnOnce(Graph) -> crate::CompiledGraph,
) -> crate::CompiledGraph {
    struct Down;
    impl Drop for Down {
        fn drop(&mut self) {
            DEPTH.with(|d| d.set(d.get() - 1));
        }
    }
    let outer = DEPTH.with(|d| {
        let v = d.get();
        d.set(v + 1);
        v == 0
    });
    let _down = Down;
    if outer {
        let (g, extra) = on_compile(graph);
        let mut c = f(g);
        c.set_tap_extra(extra);
        c
    } else {
        f(graph)
    }
}

/// Hook for `run`: strip `extra` trailing outputs and report them.
pub(crate) fn on_run(mut outs: Vec<Vec<f32>>, extra: usize) -> Vec<Vec<f32>> {
    if extra == 0 || outs.len() < extra {
        return outs;
    }
    let tail = outs.split_off(outs.len() - extra);
    if let Some(t) = current() {
        let bytes = tail
            .into_iter()
            .map(|v| (v.iter().flat_map(|x| x.to_le_bytes()).collect(), DType::F32))
            .collect();
        t.on_outputs(bytes);
    }
    outs
}

/// Hook for `run_typed`: strip `extra` trailing outputs and report them.
pub(crate) fn on_run_typed(mut outs: Vec<(Vec<u8>, DType)>, extra: usize) -> Vec<(Vec<u8>, DType)> {
    if extra == 0 || outs.len() < extra {
        return outs;
    }
    let tail = outs.split_off(outs.len() - extra);
    if let Some(t) = current() {
        t.on_outputs(tail);
    }
    outs
}

/// Decode a tap output to f32, whatever its dtype (F64 narrowed, F32 as is).
pub fn to_f32(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
        _ => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
    }
}

/// One tapped node's value in one run: what [`NodeDigests`] records.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeDigest {
    /// Which tapped compile (0, 1, … in compile order) the run belongs to.
    pub compile: usize,
    pub node: rlx_ir::NodeId,
    /// `Debug` of the node's op, cut to 48 characters.
    pub op: String,
    /// Bytes of the value.
    pub len: usize,
    /// FNV-1a of the value's little-endian bytes.
    pub hash: u64,
}

impl std::fmt::Display for NodeDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {} {} {:016x} {}",
            self.compile, self.node.0, self.len, self.hash, self.op
        )
    }
}

impl std::str::FromStr for NodeDigest {
    type Err = String;
    /// The [`Display`](std::fmt::Display) form: `compile node len hash op…`.
    fn from_str(s: &str) -> Result<Self, String> {
        let mut it = s.trim().splitn(5, ' ');
        let mut field = |name: &str| it.next().ok_or_else(|| format!("no {name} in `{s}`"));
        let compile = field("compile")?.parse().map_err(|e| format!("{e}"))?;
        let node = rlx_ir::NodeId(field("node")?.parse().map_err(|e| format!("{e}"))?);
        let len = field("len")?.parse().map_err(|e| format!("{e}"))?;
        let hash = u64::from_str_radix(field("hash")?, 16).map_err(|e| format!("{e}"))?;
        let op = it.next().unwrap_or_default().to_string();
        Ok(Self {
            compile,
            node,
            op,
            len,
            hash,
        })
    }
}

/// A tap that digests (FNV-1a of the bytes) every interior node, or a chosen
/// subset, of every graph compiled in scope, on each run — for bisecting a
/// divergence between two backends, or a native and a browser run, to the
/// first node that differs. Each run is matched to the compile it ran:
///
/// ```ignore
/// let d = Rc::new(taps::NodeDigests::default());
/// taps::with_tap(d.clone(), || model.forward(&x));
/// // the other side: the same, on another device / in the browser
/// let first = taps::first_divergence(&d.digests(), &other);
/// ```
///
/// [`NodeDigests::digests`] lines print with `Display` and parse back with
/// `FromStr`, so a browser run can log them and a native one compare.
///
/// **Tapping changes fusion.** A tapped node becomes a graph output, and a
/// fused kernel cannot expose its interior, so a fully tapped graph runs the
/// unfused kernels — a bug only in a fused kernel can vanish under it. Once a
/// region is suspect, tap only its boundary ([`NodeDigests::subset`]) to keep
/// the fusion inside it.
#[derive(Default)]
pub struct NodeDigests {
    only: Option<std::collections::HashSet<rlx_ir::NodeId>>,
    compiles: std::cell::RefCell<Vec<Vec<(rlx_ir::NodeId, String)>>>,
    digests: std::cell::RefCell<Vec<NodeDigest>>,
}

impl NodeDigests {
    /// Digest only `nodes` (those that exist in a compiled graph).
    pub fn subset(nodes: impl IntoIterator<Item = rlx_ir::NodeId>) -> Self {
        Self {
            only: Some(nodes.into_iter().collect()),
            ..Self::default()
        }
    }

    /// Everything recorded so far, in run order and, within a run, node
    /// order (topological).
    pub fn digests(&self) -> Vec<NodeDigest> {
        self.digests.borrow().clone()
    }

    /// [`Self::digests`] as text, one [`NodeDigest`] per line.
    pub fn to_text(&self) -> String {
        self.digests
            .borrow()
            .iter()
            .map(|d| format!("{d}\n"))
            .collect()
    }

    /// Parse [`Self::to_text`] output (blank lines skipped).
    pub fn parse(text: &str) -> Result<Vec<NodeDigest>, String> {
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(str::parse)
            .collect()
    }
}

impl GraphTap for NodeDigests {
    fn on_compile(&self, mut graph: Graph) -> (Graph, usize) {
        let taken: Vec<rlx_ir::NodeId> = graph
            .nodes()
            .iter()
            .filter(|n| !n.inputs.is_empty() && !graph.outputs.contains(&n.id))
            .filter(|n| self.only.as_ref().is_none_or(|s| s.contains(&n.id)))
            .map(|n| n.id)
            .collect();
        let mut compiles = self.compiles.borrow_mut();
        // The compile's index rides along as one more output, so each run
        // reports to the graph it ran — compile-then-run-many and
        // compile-many-then-run-each alike — with no guess from its width.
        let tag = graph.add_node(
            rlx_ir::Op::Constant {
                data: (compiles.len() as f32).to_le_bytes().to_vec(),
            },
            vec![],
            rlx_ir::Shape::new(&[1], DType::F32),
        );
        compiles.push(
            taken
                .iter()
                .map(|&id| {
                    let op: String = format!("{:?}", graph.node(id).op)
                        .chars()
                        .take(48)
                        .collect();
                    (id, op)
                })
                .collect(),
        );
        let mut outs = graph.outputs.clone();
        outs.extend(&taken);
        outs.push(tag);
        graph.set_outputs(outs);
        (graph, taken.len() + 1)
    }

    fn on_outputs(&self, mut extra: Vec<(Vec<u8>, DType)>) {
        let Some((tag, dtype)) = extra.pop() else {
            return;
        };
        let k = to_f32(&tag, dtype).first().copied().unwrap_or(-1.0);
        let compiles = self.compiles.borrow();
        let Some(nodes) = (k >= 0.0).then(|| compiles.get(k as usize)).flatten() else {
            return;
        };
        let fnv = |b: &[u8]| {
            b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &v| {
                (h ^ v as u64).wrapping_mul(0x100_0000_01b3)
            })
        };
        let mut out = self.digests.borrow_mut();
        for ((node, op), (bytes, _)) in nodes.iter().zip(&extra) {
            out.push(NodeDigest {
                compile: k as usize,
                node: *node,
                op: op.clone(),
                len: bytes.len(),
                hash: fnv(bytes),
            });
        }
    }
}

/// The first position where two digest runs disagree — a different hash, or
/// a different node there (the graphs diverged structurally) — with both
/// sides' digests. `None` when they agree for as long as both run.
pub fn first_divergence<'a>(
    a: &'a [NodeDigest],
    b: &'a [NodeDigest],
) -> Option<(usize, &'a NodeDigest, &'a NodeDigest)> {
    a.iter()
        .zip(b)
        .enumerate()
        .find(|(_, (x, y))| {
            (x.compile, x.node, x.len, x.hash) != (y.compile, y.node, y.len, y.hash)
        })
        .map(|(i, (x, y))| (i, x, y))
}
