// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx.Graph` — the symbolic IR builder.
//!
//! Every method returns an integer node id, so JavaScript composes graphs the
//! same way the Rust builder does:
//!
//! ```js
//! const g = new rlx.Graph("mlp");
//! const x = g.input("x", [1, 4], "f32");
//! const w = g.param("w", [4, 2], "f32");
//! g.setOutputs([g.relu(g.matmul(x, w))]);
//! ```
//!
//! The methods are not written here: they are generated from the op table in
//! `rlx-bind` (one row per op, shared with Python, C and Swift) by
//! [`crate::js_graph_ops!`]. A row's snake_case name is camelCased
//! (`attention_kind_vdim` → `attentionKindVdim`), positional arguments read in
//! order, and a row's options group is one trailing object
//! (`conv2d(x, w, {kernelSize: 3, stride: 2})`). `undefined` / `null` in any
//! slot means "the default".
//!
//! Only what is JS-specific lives in this file: the handle plumbing, and
//! `toString` / `toJSON`.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_ir::{DType, Graph};

use crate::convert::*;
use crate::handle::{self, CLASS_GRAPH};

/// A graph, or the hole left after `Session.compile` moved it out.
pub struct GraphSlot {
    pub inner: Option<Graph>,
}

impl GraphSlot {
    pub fn new(graph: Graph) -> Self {
        Self { inner: Some(graph) }
    }
}

/// `&mut Graph` behind `this`, or a `TypeError`.
pub(crate) fn graph_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut Graph> {
    let slot: &mut GraphSlot = handle::borrow_mut(ctx, this, CLASS_GRAPH, "rlx.Graph")?;
    match slot.inner.as_mut() {
        Some(g) => Ok(g),
        None => ctx.throw_type(
            "this Graph was consumed by Session.compile() — build a new one, or call \
             Session.compile on a fresh graph",
        ),
    }
}

/// Read-only view, for `grad` / `deviceReport` which borrow rather than consume.
pub fn graph_ref<'a>(ctx: &mut Context, v: &Value) -> JsResult<&'a Graph> {
    let slot: &mut GraphSlot = handle::borrow_mut(ctx, v, CLASS_GRAPH, "rlx.Graph")?;
    match slot.inner.as_ref() {
        Some(g) => Ok(g),
        None => ctx.throw_type("Graph has already been consumed by Session.compile()"),
    }
}

/// Move the graph out — `Session.compile` takes ownership, as rlx-runtime does.
pub fn take_graph(ctx: &mut Context, v: &Value) -> JsResult<Graph> {
    let slot: &mut GraphSlot = handle::borrow_mut(ctx, v, CLASS_GRAPH, "rlx.Graph")?;
    match slot.inner.take() {
        Some(g) => Ok(g),
        None => ctx.throw_type("Graph has already been consumed by Session.compile()"),
    }
}

pub fn wrap_graph(ctx: &mut Context, graph: Graph) -> Value {
    handle::wrap(ctx, CLASS_GRAPH, GraphSlot::new(graph))
}

// ── dtype names (shared with Session / Compiled) ────────────

pub fn parse_dtype(ctx: &mut Context, s: &str) -> JsResult<DType> {
    let r = rlx_bind::enums::dtype(s);
    crate::api::bind::lift(ctx, r)
}

pub fn dtype_label(d: DType) -> &'static str {
    rlx_bind::enums::dtype_label(d)
}

// ── JS-only methods ─────────────────────────────────────────

fn m_to_string(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let text = {
        let g = graph_of(ctx, this)?;
        format!("[rlx.Graph {} nodes={}]", g.name, g.nodes().len())
    };
    Ok(new_string(ctx, &text))
}

/// Serialize to the graph-JSON that `rlx-check` and the Rust tools read.
///
/// Behind the `graph-json` feature: it needs `rlx-ir/serialize`, and turning
/// that on here would switch it on for every crate in a `--workspace` build
/// through feature unification.
#[cfg(feature = "graph-json")]
fn m_to_json(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let json = match serde_json::to_string(&*graph_of(ctx, this)?) {
        Ok(text) => text,
        Err(e) => return ctx.throw_internal(&format!("toJSON: {e}")),
    };
    Ok(new_string(ctx, &json))
}

/// `clone()` — a deep copy: compile one, keep building on the other.
fn m_clone(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let copy = graph_of(ctx, this)?.clone();
    Ok(wrap_graph(ctx, copy))
}

/// `isConsumed()` — whether a compile (or a body-graph argument) took it.
fn m_is_consumed(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let slot: &mut GraphSlot = handle::borrow_mut(ctx, this, CLASS_GRAPH, "rlx.Graph")?;
    Ok(Value::Bool(slot.inner.is_none()))
}

// ── installation ────────────────────────────────────────────

/// Build the `Graph` prototype and the `rlx.Graph` constructor.
pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    let proto = js_class! {
        ctx, namespace;
        name: "Graph",
        class: CLASS_GRAPH,
        ctor: construct,
        methods: {
            "toString" => m_to_string, 0;
            "clone" => m_clone, 0;
            "isConsumed" => m_is_consumed, 0;
        }
    };
    rlx_bind::for_each_graph_op!(crate::js_graph_ops, ctx, &proto);
    #[cfg(feature = "graph-json")]
    ctx.define_method(&proto, "toJSON", m_to_json, 0, 0);
}

/// `new rlx.Graph("name")` — also callable without `new`.
fn construct(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = to_string_or(ctx, arg(args, 0), "graph")?;
    Ok(wrap_graph(ctx, Graph::new(name)))
}
