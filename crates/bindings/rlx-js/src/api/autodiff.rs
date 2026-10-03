// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Graph-to-graph transforms: `grad`, `jvp`, `hvp`, `vmap`.
//!
//! Each borrows the forward graph and returns a new one, so the same forward
//! can be differentiated against several `wrt` sets:
//!
//! ```js
//! const bwd = rlx.grad(fwd, [w, b]);     // outputs: [loss, dW, dB]
//! const c = new rlx.Session("cpu").compile(bwd);
//! const [loss, dW, dB] = c.run({ x, d_output: new Float32Array([1]) });
//! ```

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_ir::NodeId;

use crate::api::graph::{graph_ref, wrap_graph};
use crate::convert::*;

/// `grad(graph, wrt)` → a graph whose outputs are `[loss, ...dWrt]`.
///
/// The returned graph gains an input named `d_output` for the upstream
/// gradient — seed it with `[1]` to differentiate the loss directly.
fn f_grad(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let wrt = to_u32_vec(ctx, arg(args, 1), "grad wrt")?;
    let forward = graph_ref(ctx, arg(args, 0))?;
    let ids: Vec<NodeId> = wrt.into_iter().map(NodeId).collect();
    // A `wrt` the loss does not depend on, or an op with no VJP rule, is a
    // `panic!` inside rlx — surfaced here instead of aborting the host.
    let backward = crate::panics::catch(ctx, "grad", || {
        rlx_opt::autodiff::grad_with_loss(forward, &ids)
    })?;
    Ok(wrap_graph(ctx, backward))
}

/// `jvp(graph, tangentFor)` → `[...primals, ...tangents]`, with one fresh
/// `tangent_<name>` input per seeded node.
fn f_jvp(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let wrt = to_u32_vec(ctx, arg(args, 1), "jvp tangentFor")?;
    let forward = graph_ref(ctx, arg(args, 0))?;
    let ids: Vec<NodeId> = wrt.into_iter().map(NodeId).collect();
    let out = crate::panics::catch(ctx, "jvp", || rlx_opt::autodiff_fwd::jvp(forward, &ids))?;
    Ok(wrap_graph(ctx, out))
}

/// `hvp(graph, wrt)` — Hessian-vector product, forward-over-reverse.
fn f_hvp(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let wrt = to_u32_vec(ctx, arg(args, 1), "hvp wrt")?;
    let forward = graph_ref(ctx, arg(args, 0))?;
    let ids: Vec<NodeId> = wrt.into_iter().map(NodeId).collect();
    let out = crate::panics::catch(ctx, "hvp", || rlx_opt::autodiff_fwd::hvp(forward, &ids))?;
    Ok(wrap_graph(ctx, out))
}

/// `vmap(graph, ["x"], batchSize)` — batch the named inputs along a new axis 0.
fn f_vmap(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let names = to_string_vec(ctx, arg(args, 1))?;
    let batch = to_usize(ctx, arg(args, 2), "vmap batchSize")?;
    let forward = graph_ref(ctx, arg(args, 0))?;
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let out = crate::panics::catch(ctx, "vmap", || rlx_opt::vmap::vmap(forward, &refs, batch))?;
    Ok(wrap_graph(ctx, out))
}

/// `nthOrderGrad(graph, "x", order)` — scalar higher-order derivative.
fn f_nth_order_grad(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 1))?;
    let order = to_usize(ctx, arg(args, 2), "nthOrderGrad order")?;
    let forward = graph_ref(ctx, arg(args, 0))?;
    let out = crate::panics::catch(ctx, "nthOrderGrad", || {
        rlx_opt::rlx_autodiff::higher_order::nth_order_grad(forward, &name, order)
    })?;
    Ok(wrap_graph(ctx, out))
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_functions! {
        ctx, namespace;
        "grad" => f_grad, 2;
        "jvp" => f_jvp, 2;
        "hvp" => f_hvp, 2;
        "vmap" => f_vmap, 3;
        "nthOrderGrad" => f_nth_order_grad, 3;
    };
}
