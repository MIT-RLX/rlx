// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Combining and splitting graphs: `rlx_splice`, `rlx_parallel`,
//! `rlx_sequential`, `rlx_extract` and the `RlxPartition` handle.
//!
//! Every graph argument is borrowed. A graph may be passed twice (spliced
//! into itself): the source is copied before the target is touched.

use std::ffi::c_char;

use rlx_bind::runtime::compose::{self, Partition};

use crate::ffi::{RlxU32s, arg, borrow, guard, handle, out, release};
use crate::graph::{RlxGraph, graph_mut};

fn new_graph(g: rlx_ir::Graph) -> *mut RlxGraph {
    handle(RlxGraph { inner: Some(g) })
}

/// Copy `source` into `target` and write `source`'s outputs, as `target`
/// node ids, to `out`. `options_json`: `{"bind": {leaf: node}, "prefix",
/// "adapt_shapes"}` or `NULL`. On error `target` is unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_splice(
    target: *mut RlxGraph,
    source: *mut RlxGraph,
    options_json: *const c_char,
    out: *mut RlxU32s,
) -> i32 {
    guard(|| unsafe {
        let opts = arg::json(options_json, "rlx_splice options")?;
        let source = graph_mut(source)?.clone();
        let ids = compose::splice(graph_mut(target)?, &source, &opts)?;
        out::put(out, ids.into(), "rlx_splice out")
    })
}

/// `a` and `b` side by side (`a`'s outputs, then `b`'s); `name` may be `NULL`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_parallel(
    a: *mut RlxGraph,
    b: *mut RlxGraph,
    name: *const c_char,
    out: *mut *mut RlxGraph,
) -> i32 {
    guard(|| unsafe {
        let name = arg::opt_str(name, "rlx_parallel name")?;
        let b = graph_mut(b)?.clone();
        let g = compose::parallel(graph_mut(a)?, &b, name.as_deref())?;
        out::put(out, new_graph(g), "rlx_parallel out")
    })
}

/// `b` after `a`: `b`'s inputs fed by `a`'s outputs; `name` may be `NULL`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_sequential(
    a: *mut RlxGraph,
    b: *mut RlxGraph,
    name: *const c_char,
    out: *mut *mut RlxGraph,
) -> i32 {
    guard(|| unsafe {
        let name = arg::opt_str(name, "rlx_sequential name")?;
        let b = graph_mut(b)?.clone();
        let g = compose::sequential(graph_mut(a)?, &b, name.as_deref())?;
        out::put(out, new_graph(g), "rlx_sequential out")
    })
}

/// The subgraph computing the `n` nodes `outputs`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_extract(
    g: *mut RlxGraph,
    outputs: *const u32,
    n: usize,
    out: *mut *mut RlxGraph,
) -> i32 {
    guard(|| unsafe {
        let ids = arg::vec(outputs, n, "rlx_extract outputs")?;
        let r = compose::extract(graph_mut(g)?, ids)?;
        out::put(out, new_graph(r), "rlx_extract out")
    })
}

/// A graph split into parts that run in order.
pub struct RlxPartition(Partition);

/// Split `g`, node `i` going to part `assign[i]` (`n` = the node count).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_partition_new(
    g: *mut RlxGraph,
    assign: *const u32,
    n: usize,
    out: *mut *mut RlxPartition,
) -> i32 {
    guard(|| unsafe {
        let assign = arg::slice(assign, n, "rlx_partition_new assign")?;
        let p = Partition::new(graph_mut(g)?, assign)?;
        out::put(out, handle(RlxPartition(p)), "rlx_partition_new out")
    })
}

/// Two parts: the `n` nodes `cut` and everything they read, then the rest.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_partition_split_at(
    g: *mut RlxGraph,
    cut: *const u32,
    n: usize,
    out: *mut *mut RlxPartition,
) -> i32 {
    guard(|| unsafe {
        let cut = arg::vec(cut, n, "rlx_partition_split_at cut")?;
        let p = Partition::split_at(graph_mut(g)?, cut)?;
        out::put(out, handle(RlxPartition(p)), "rlx_partition_split_at out")
    })
}

/// One part per independent subgraph.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_partition_components(
    g: *mut RlxGraph,
    out: *mut *mut RlxPartition,
) -> i32 {
    guard(|| unsafe {
        let p = Partition::components(graph_mut(g)?)?;
        out::put(out, handle(RlxPartition(p)), "rlx_partition_components out")
    })
}

/// Free a partition (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_partition_free(p: *mut RlxPartition) {
    unsafe { release(p) }
}

/// The number of parts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_partition_num_parts(p: *mut RlxPartition, out: *mut usize) -> i32 {
    guard(|| unsafe {
        let n = borrow(p, "partition")?.0.len();
        out::put(out, n, "rlx_partition_num_parts out")
    })
}

/// A copy of part `part`'s graph.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_partition_graph(
    p: *mut RlxPartition,
    part: usize,
    out: *mut *mut RlxGraph,
) -> i32 {
    guard(|| unsafe {
        let g = borrow(p, "partition")?.0.graph(part)?;
        out::put(out, new_graph(g), "rlx_partition_graph out")
    })
}

/// The boundary names part `part` reads, as a JSON array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_partition_imports(
    p: *mut RlxPartition,
    part: usize,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let names = borrow(p, "partition")?.0.imports(part)?;
        out::strings(out, &names, "rlx_partition_imports out")
    })
}

/// The name of each of part `part`'s outputs, as a JSON array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_partition_exports(
    p: *mut RlxPartition,
    part: usize,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let names = borrow(p, "partition")?.0.exports(part)?;
        out::strings(out, &names, "rlx_partition_exports out")
    })
}

/// Where each source output went: `[[part, output index], …]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_partition_outputs(p: *mut RlxPartition, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let v = borrow(p, "partition")?.0.outputs();
        out::json(out, &v, "rlx_partition_outputs out")
    })
}

/// The parts as one graph again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_partition_merge(p: *mut RlxPartition, out: *mut *mut RlxGraph) -> i32 {
    guard(|| unsafe {
        let g = borrow(p, "partition")?.0.merge()?;
        out::put(out, new_graph(g), "rlx_partition_merge out")
    })
}
