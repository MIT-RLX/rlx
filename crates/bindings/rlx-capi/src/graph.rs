// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The graph handle. Builder functions are in `graph_ops` (generated).

use std::ffi::c_char;

use rlx_bind::ApiResult;
use rlx_ir::Graph;

use crate::ffi::{arg, borrow, guard, handle, out, release};

/// A graph, or the hole left after something consumed it (`rlx_session_compile`,
/// a body-graph argument).
pub struct RlxGraph {
    pub(crate) inner: Option<Graph>,
}

/// `&mut Graph` behind a handle, or an error naming why not.
pub(crate) unsafe fn graph_mut<'a>(g: *mut RlxGraph) -> ApiResult<&'a mut Graph> {
    let h = unsafe { borrow(g, "graph") }?;
    h.inner.as_mut().ok_or_else(consumed)
}

/// Move the graph out of its handle (the handle stays valid, and empty).
pub(crate) unsafe fn take_graph(g: *mut RlxGraph, what: &str) -> ApiResult<Graph> {
    let h = unsafe { borrow(g, what) }?;
    h.inner.take().ok_or_else(consumed)
}

fn consumed() -> rlx_bind::ApiError {
    rlx_bind::ApiError::runtime(
        "this graph was consumed (by a compile, or as a body graph) — build a new one",
    )
}

/// Create an empty graph named `name`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_graph_new(name: *const c_char, out: *mut *mut RlxGraph) -> i32 {
    guard(|| unsafe {
        let name = arg::str_or(name, "graph", "rlx_graph_new name")?;
        out::put(
            out,
            handle(RlxGraph {
                inner: Some(Graph::new(name)),
            }),
            "rlx_graph_new out",
        )
    })
}

/// Free a graph handle (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_graph_free(g: *mut RlxGraph) {
    unsafe { release(g) }
}

/// A deep copy — compile one, keep building on the other.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_graph_clone(g: *mut RlxGraph, out: *mut *mut RlxGraph) -> i32 {
    guard(|| unsafe {
        let copy = graph_mut(g)?.clone();
        out::put(
            out,
            handle(RlxGraph { inner: Some(copy) }),
            "rlx_graph_clone out",
        )
    })
}

/// Whether the graph was consumed (by a compile, or as a body graph).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_graph_is_consumed(g: *mut RlxGraph) -> bool {
    unsafe { g.as_ref() }.is_none_or(|h| h.inner.is_none())
}
