// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Combining and splitting graphs: `rlx.splice`, `parallel`, `sequential`,
//! `extract` and `rlx.Partition`.
//!
//! ```js
//! const g = new rlx.Graph("stack");
//! let [h] = rlx.splice(g, layer, { prefix: "l0." });
//! [h] = rlx.splice(g, layer, { prefix: "l1.", bind: { x: h } });
//! g.setOutputs([h]);
//!
//! const p = rlx.Partition.splitAt(g, [cut]);
//! for (let i = 0; i < p.numParts(); i++) { const part = p.graph(i); … }
//! ```

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_bind::runtime::compose::{self, Partition};
use rlx_ir::Graph;

use crate::api::bind::{from_json, lift, to_json};
use crate::api::graph::{GraphSlot, graph_ref, wrap_graph};
use crate::convert::*;
use crate::handle::{self, CLASS_GRAPH, CLASS_PARTITION};

/// The graph behind `v`, mutably (`splice` grows its target).
fn graph_mut<'a>(ctx: &mut Context, v: &Value) -> JsResult<&'a mut Graph> {
    let slot: &mut GraphSlot = handle::borrow_mut(ctx, v, CLASS_GRAPH, "rlx.Graph")?;
    match slot.inner.as_mut() {
        Some(g) => Ok(g),
        None => ctx.throw_type("Graph has already been consumed by Session.compile()"),
    }
}

fn opt_name(ctx: &mut Context, v: &Value) -> JsResult<Option<String>> {
    if is_nullish(v) {
        Ok(None)
    } else {
        Ok(Some(ctx.to_rust_string(v)?))
    }
}

/// `splice(target, source, {bind, prefix, adaptShapes})` → `source`'s
/// outputs as `target` node ids. `target` grows; `source` is untouched.
fn f_splice(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let opts = to_json(ctx, arg(args, 2))?;
    // A copy first: `source` may be `target` itself.
    let source = graph_ref(ctx, arg(args, 1))?.clone();
    let target = graph_mut(ctx, arg(args, 0))?;
    let r = compose::splice(target, &source, &opts);
    let ids = lift(ctx, r)?;
    let ids: Vec<usize> = ids.into_iter().map(|i| i as usize).collect();
    Ok(new_usize_array(ctx, &ids))
}

/// `parallel(a, b, name?)` — a's outputs, then b's.
fn f_parallel(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = opt_name(ctx, arg(args, 2))?;
    let b = graph_ref(ctx, arg(args, 1))?.clone();
    let a = graph_ref(ctx, arg(args, 0))?;
    let g = lift(ctx, compose::parallel(a, &b, name.as_deref()))?;
    Ok(wrap_graph(ctx, g))
}

/// `sequential(a, b, name?)` — b's inputs fed by a's outputs.
fn f_sequential(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = opt_name(ctx, arg(args, 2))?;
    let b = graph_ref(ctx, arg(args, 1))?.clone();
    let a = graph_ref(ctx, arg(args, 0))?;
    let g = lift(ctx, compose::sequential(a, &b, name.as_deref()))?;
    Ok(wrap_graph(ctx, g))
}

/// `extract(graph, outputs)` — the subgraph computing `outputs`.
fn f_extract(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let ids = to_u32_vec(ctx, arg(args, 1), "extract outputs")?;
    let g = graph_ref(ctx, arg(args, 0))?;
    let r = lift(ctx, compose::extract(g, ids))?;
    Ok(wrap_graph(ctx, r))
}

fn partition_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut Partition> {
    handle::borrow_mut(ctx, this, CLASS_PARTITION, "rlx.Partition")
}

/// `new rlx.Partition(graph, assign)` — node `i` goes to part `assign[i]`.
fn partition_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let assign = to_u32_vec(ctx, arg(args, 1), "Partition assign")?;
    let g = graph_ref(ctx, arg(args, 0))?;
    let p = lift(ctx, Partition::new(g, &assign))?;
    Ok(handle::wrap(ctx, CLASS_PARTITION, p))
}

/// `Partition.splitAt(graph, cut)`.
fn partition_split_at(
    ctx: &mut Context,
    _this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let cut = to_u32_vec(ctx, arg(args, 1), "Partition.splitAt cut")?;
    let g = graph_ref(ctx, arg(args, 0))?;
    let p = lift(ctx, Partition::split_at(g, cut))?;
    Ok(handle::wrap(ctx, CLASS_PARTITION, p))
}

/// `Partition.components(graph)`.
fn partition_components(
    ctx: &mut Context,
    _this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let g = graph_ref(ctx, arg(args, 0))?;
    let p = lift(ctx, Partition::components(g))?;
    Ok(handle::wrap(ctx, CLASS_PARTITION, p))
}

fn m_num_parts(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let n = partition_of(ctx, this)?.len();
    Ok(Value::number(n as f64))
}

fn m_graph(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let i = to_usize(ctx, arg(args, 0), "Partition.graph part")?;
    let r = partition_of(ctx, this)?.graph(i);
    let g = lift(ctx, r)?;
    Ok(wrap_graph(ctx, g))
}

fn m_imports(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let i = to_usize(ctx, arg(args, 0), "Partition.imports part")?;
    let r = partition_of(ctx, this)?.imports(i);
    let names = lift(ctx, r)?;
    Ok(new_string_array(ctx, &names))
}

fn m_exports(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let i = to_usize(ctx, arg(args, 0), "Partition.exports part")?;
    let r = partition_of(ctx, this)?.exports(i);
    let names = lift(ctx, r)?;
    Ok(new_string_array(ctx, &names))
}

fn m_outputs(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let v = partition_of(ctx, this)?.outputs();
    Ok(from_json(ctx, &v))
}

fn m_merge(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let r = partition_of(ctx, this)?.merge();
    let g = lift(ctx, r)?;
    Ok(wrap_graph(ctx, g))
}

fn m_to_string(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let text = {
        let p = partition_of(ctx, this)?;
        format!("[rlx.Partition {} parts={}]", p.0.name, p.len())
    };
    Ok(new_string(ctx, &text))
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_class! {
        ctx, namespace;
        name: "Partition",
        class: CLASS_PARTITION,
        ctor: partition_new,
        methods: {
            "numParts" => m_num_parts, 0;
            "graph" => m_graph, 1;
            "imports" => m_imports, 1;
            "exports" => m_exports, 1;
            "outputs" => m_outputs, 0;
            "merge" => m_merge, 0;
            "toString" => m_to_string, 0;
        }
    };
    js_statics! {
        ctx, namespace, "Partition";
        "splitAt" => partition_split_at, 2;
        "components" => partition_components, 1;
    };
    js_functions! {
        ctx, namespace;
        "splice" => f_splice, 3;
        "parallel" => f_parallel, 3;
        "sequential" => f_sequential, 3;
        "extract" => f_extract, 2;
    };
}
