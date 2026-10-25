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
