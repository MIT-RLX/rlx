// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thread-local compile cache.
//!
//! Compiling a graph (fuse → memory-plan → backend lower) is far more
//! expensive than running it, so every materialization (`Tensor::to_vec`,
//! `Tensor::grad` + eval, `Func::run`) routes through here. A graph that was
//! already compiled — same structure, same constant bytes, same device — gets
//! its [`CompiledGraph`] back instead of recompiling. For [`crate::Func`] this
//! turns repeated `run` calls into a real `jit`: compile once, execute many
//! inputs.
//!
//! The cache is per-thread. Available with the `eval` feature.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use rlx_ir::{Graph, NodeId, Op};
use rlx_runtime::{CompiledGraph, Device, PrecisionPolicy, Session};

/// Clear the cache once it grows past this many distinct graphs, a simple
/// backstop against unbounded growth in long-lived threads.
const MAX_ENTRIES: usize = 512;

thread_local! {
    static CACHE: RefCell<HashMap<u64, Rc<RefCell<CompiledGraph>>>> =
        RefCell::new(HashMap::new());
    static HITS: Cell<u64> = const { Cell::new(0) };
    static MISSES: Cell<u64> = const { Cell::new(0) };
}

/// Structural fingerprint covering everything that affects the compiled
/// result: op fields (via `Debug`), constant bytes, operands, shapes, output
/// set, and target device.
fn fingerprint(graph: &Graph, outputs: &[NodeId], device: Device) -> u64 {
    let mut h = DefaultHasher::new();
    device.hash(&mut h);
    graph.name.hash(&mut h);
    for node in graph.nodes() {
        node.id.0.hash(&mut h);
        match &node.op {
            // Hash bytes directly — avoids a huge Debug string for weights.
            Op::Constant { data } => {
                0u8.hash(&mut h);
                data.hash(&mut h);
            }
            other => format!("{other:?}").hash(&mut h),
        }
        for inp in &node.inputs {
            inp.0.hash(&mut h);
        }
        format!("{:?}", node.shape).hash(&mut h);
    }
    for out in outputs {
        out.0.hash(&mut h);
    }
    h.finish()
}

fn lookup(key: u64) -> Option<Rc<RefCell<CompiledGraph>>> {
    let hit = CACHE.with(|c| c.borrow().get(&key).cloned());
    if hit.is_some() {
        HITS.with(|n| n.set(n.get() + 1));
    } else {
        MISSES.with(|n| n.set(n.get() + 1));
    }
    hit
}

/// Opt-in mixed-precision policy from `RLX_PRECISION` (default: F32, unchanged).
/// `auto` = the Metal-safe policy (elementwise/reduction in F16, **matmul stays
/// F32** — F16 matmul diverges on the residual/RoPE/gather paths); `f16`/`bf16`
/// push matmul into low precision too (faster, but numerically riskier — bf16
/// keeps F32's exponent range so it's the safer of the two).
fn precision_policy_from_env() -> Option<PrecisionPolicy> {
    match rlx_ir::env::var("RLX_PRECISION").as_deref() {
        Some("auto") => Some(PrecisionPolicy::AutoMixed),
        Some("f16") => Some(PrecisionPolicy::AutoMixedConservative),
        Some("bf16") => Some(PrecisionPolicy::AutoMixedBf16),
        Some("f16-all") => Some(PrecisionPolicy::AlwaysF16),
        _ => None,
    }
}

fn store(key: u64, device: Device, graph: Graph) -> Rc<RefCell<CompiledGraph>> {
    let mut session = Session::new(device);
    if let Some(policy) = precision_policy_from_env() {
        session = session.with_policy(policy);
    }
    let compiled = Rc::new(RefCell::new(session.compile(graph)));
    CACHE.with(|c| {
        let mut map = c.borrow_mut();
        if map.len() >= MAX_ENTRIES {
            map.clear();
        }
        map.insert(key, compiled.clone());
    });
    compiled
}

/// Compiled graph for `(graph, device)` where `graph` already carries its
/// outputs (the `Func` path). Cloned only on a cache miss.
pub(crate) fn compiled(graph: &Graph, device: Device) -> Rc<RefCell<CompiledGraph>> {
    let key = fingerprint(graph, &graph.outputs, device);
    lookup(key).unwrap_or_else(|| store(key, device, graph.clone()))
}

/// Compiled graph that emits a single `output` node from a shared graph —
/// the `Tensor::to_vec` path. **The graph is borrowed (not cloned) for the
/// fingerprint**, so cache hits copy zero constant bytes; the clone +
/// `set_outputs` happens only on a miss.
pub(crate) fn compiled_output(
    graph: &Graph,
    output: NodeId,
    device: Device,
) -> Rc<RefCell<CompiledGraph>> {
    let key = fingerprint(graph, &[output], device);
    lookup(key).unwrap_or_else(|| {
        let mut g = graph.clone();
        g.set_outputs(vec![output]);
        store(key, device, g)
    })
}

/// `(hits, misses)` for the calling thread's cache. Useful for tests and
/// confirming `jit`-style reuse.
pub fn cache_stats() -> (u64, u64) {
    (HITS.with(Cell::get), MISSES.with(Cell::get))
}

/// Drop all cached compiled graphs and reset stats on the calling thread.
pub fn clear_cache() {
    CACHE.with(|c| c.borrow_mut().clear());
    HITS.with(|n| n.set(0));
    MISSES.with(|n| n.set(0));
}
