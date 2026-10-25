// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-backend execution: one graph, a per-device compile cache, and a
//! policy deciding which backend a given run lands on.
//!
//! ```js
//! const r = new rlx.Runner(g, { deny: ["cpu"], benchmarkRuns: 5 });
//! r.setParams(weights);
//! console.log(r.benchmark(inputs, 5));   // [{device, compileMs, execMs}, …]
//! const { device, outputs } = r.run(inputs);
//! ```
//!
//! `rlx.Runner` wraps `GraphDevices` — explicit device choice, warm-all,
//! benchmark-pick. `rlx.Router` wraps `DeviceRouter`, which adds a fallback
//! chain: when a backend fails to compile or run, the next one is tried and
//! the error names every attempt rather than just the last.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_bind::runtime::devices::{self, Router, Runner};
use rlx_runtime::DevicePolicy;

use crate::api::bind::{lift, report, to_json};
use crate::api::graph::take_graph;
use crate::convert::*;
use crate::handle::{self, CLASS_ROUTER, CLASS_RUNNER};

fn runner_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut Runner> {
    handle::borrow_mut(ctx, this, CLASS_RUNNER, "rlx.Runner")
}

fn router_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut Router> {
    handle::borrow_mut(ctx, this, CLASS_ROUTER, "rlx.Router")
}

/// `"env"`, `"all"`, or `{only, deny, prefer, benchmarkRuns}`.
pub fn parse_policy(ctx: &mut Context, v: &Value) -> JsResult<DevicePolicy> {
    let json = to_json(ctx, v)?;
    lift(ctx, devices::policy(&json))
}

fn optional_device(ctx: &mut Context, v: &Value) -> JsResult<Option<String>> {
    if is_nullish(v) {
        Ok(None)
    } else {
        Ok(Some(ctx.to_rust_string(v)?))
    }
}

fn collect(ctx: &mut Context, bag: &Value) -> JsResult<Vec<(String, Vec<f32>)>> {
    let names = own_keys(ctx, bag)?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let value = field(ctx, bag, &name)?;
        let data = to_f32_vec(ctx, &value, &format!("input '{name}'"))?;
        out.push((name, data));
    }
    Ok(out)
}

fn pairs(owned: &[(String, Vec<f32>)]) -> Vec<(&str, &[f32])> {
    owned
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect()
}

/// `{device, outputs: [Float32Array, …]}`.
fn ran(ctx: &mut Context, device: &str, outs: Vec<Vec<f32>>) -> Value {
    let label = new_string(ctx, device);
    let values: Vec<Value> = outs.iter().map(|o| new_f32_array(ctx, o)).collect();
    let outputs = new_array(ctx, values);
    new_object(ctx, vec![("device", label), ("outputs", outputs)])
}

// ── Runner (GraphDevices) ───────────────────────────────────

fn runner_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let policy = parse_policy(ctx, arg(args, 1))?;
    let graph = take_graph(ctx, arg(args, 0))?;
    Ok(handle::wrap(ctx, CLASS_RUNNER, Runner::new(graph, policy)))
}

/// `Runner.fromEnv(graph)` — the policy from `RLX_DEVICE*`.
fn runner_from_env(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let graph = take_graph(ctx, arg(args, 0))?;
    Ok(handle::wrap(ctx, CLASS_RUNNER, Runner::from_env(graph)))
}

fn runner_devices(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let names = runner_of(ctx, this)?.devices();
    Ok(new_string_array(ctx, &names))
}

fn runner_fastest(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let label = runner_of(ctx, this)?.fastest();
    Ok(new_string(ctx, label))
}

fn runner_resolve(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let hint = optional_device(ctx, arg(args, 0))?;
    let r = runner_of(ctx, this)?.resolve(hint.as_deref());
    let label = lift(ctx, r)?;
    Ok(new_string(ctx, label))
}

fn runner_set_param(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let data = to_f32_vec(ctx, arg(args, 1), &format!("setParam('{name}')"))?;
    runner_of(ctx, this)?.set_param(&name, &data);
    Ok(Value::Undefined)
}

fn runner_set_params(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let params = collect(ctx, &bag)?;
    let runner = runner_of(ctx, this)?;
    for (name, data) in &params {
        runner.set_param(name, data);
    }
    Ok(Value::number(params.len() as f64))
}

fn runner_set_param_typed(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let bytes = to_bytes(ctx, arg(args, 1), &format!("setParamTyped('{name}')"))?;
    let label = ctx.to_rust_string(arg(args, 2))?;
    let r = runner_of(ctx, this)?.set_param_typed(&name, &bytes, &label);
    lift(ctx, r)?;
    Ok(Value::Undefined)
}

fn runner_warm_all(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let r = runner_of(ctx, this)?.warm_all();
    let names = lift(ctx, r)?;
    Ok(new_string_array(ctx, &names))
}

/// `benchmark(inputs, runs = 3)` → `[{device, compileMs, execMs}]`.
fn runner_benchmark(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let owned = collect(ctx, &bag)?;
    let runs = if is_nullish(arg(args, 1)) {
        3
    } else {
        to_usize(ctx, arg(args, 1), "benchmark runs")?
    };
    let r = runner_of(ctx, this)?.benchmark(&pairs(&owned), runs);
    let json = lift(ctx, r)?;
    Ok(report(ctx, &json))
}

/// `run(inputs, device?)` → `{device, outputs}`.
fn runner_run(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let owned = collect(ctx, &bag)?;
    let hint = optional_device(ctx, arg(args, 1))?;
    let r = runner_of(ctx, this)?.run(&pairs(&owned), hint.as_deref());
    let (device, outs) = lift(ctx, r)?;
    Ok(ran(ctx, device, outs))
}

/// `runOn(device, inputs)` → `[Float32Array, …]` — exactly that device.
fn runner_run_on(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let device = ctx.to_rust_string(arg(args, 0))?;
    let bag = arg(args, 1).clone();
    let owned = collect(ctx, &bag)?;
    let r = runner_of(ctx, this)?.run_on(&device, &pairs(&owned));
    let outs = lift(ctx, r)?;
    let values: Vec<Value> = outs.iter().map(|o| new_f32_array(ctx, o)).collect();
    Ok(new_array(ctx, values))
}

/// `runTry(["cuda", "cpu"], inputs)` → `{device, outputs}` from the first that runs.
fn runner_run_try(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let chain = to_string_vec(ctx, arg(args, 0))?;
    let bag = arg(args, 1).clone();
    let owned = collect(ctx, &bag)?;
    let r = runner_of(ctx, this)?.run_try(&chain, &pairs(&owned));
    let (device, outs) = lift(ctx, r)?;
    Ok(ran(ctx, device, outs))
}

/// `runChain(inputs, device?)` → `{device, outputs}`, falling back down the policy.
fn runner_run_chain(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let owned = collect(ctx, &bag)?;
    let hint = optional_device(ctx, arg(args, 1))?;
    let r = runner_of(ctx, this)?.run_chain(&pairs(&owned), hint.as_deref());
    let (device, outs) = lift(ctx, r)?;
    Ok(ran(ctx, device, outs))
}

/// `runWithInputs(inputs, device?)` → `[Float32Array, …]`, resolving with the
/// inputs in hand (input-shape-aware).
fn runner_run_with_inputs(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let owned = collect(ctx, &bag)?;
    let hint = optional_device(ctx, arg(args, 1))?;
    let r = runner_of(ctx, this)?.run_with_inputs(&pairs(&owned), hint.as_deref());
    let outs = lift(ctx, r)?;
    let values: Vec<Value> = outs.iter().map(|o| new_f32_array(ctx, o)).collect();
    Ok(new_array(ctx, values))
}

fn runner_report(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let json = runner_of(ctx, this)?.report();
    Ok(report(ctx, &json))
}

// ── Router (DeviceRouter) ───────────────────────────────────

fn router_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let policy = parse_policy(ctx, arg(args, 1))?;
    let graph = take_graph(ctx, arg(args, 0))?;
    let router = lift(ctx, Router::new(graph, policy))?;
    Ok(handle::wrap(ctx, CLASS_ROUTER, router))
}

/// `Router.fromEnv(graph)` — the policy from `RLX_DEVICE*`.
fn router_from_env(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let graph = take_graph(ctx, arg(args, 0))?;
    let router = lift(ctx, Router::from_env(graph))?;
    Ok(handle::wrap(ctx, CLASS_ROUTER, router))
}

fn router_devices(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let names = router_of(ctx, this)?.devices();
    Ok(new_string_array(ctx, &names))
}

fn router_set_param(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let data = to_f32_vec(ctx, arg(args, 1), &format!("setParam('{name}')"))?;
    router_of(ctx, this)?.set_param(&name, &data);
    Ok(Value::Undefined)
}

fn router_set_params(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let params = collect(ctx, &bag)?;
    let router = router_of(ctx, this)?;
    for (name, data) in &params {
        router.set_param(name, data);
    }
    Ok(Value::number(params.len() as f64))
}

fn router_set_param_typed(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let name = ctx.to_rust_string(arg(args, 0))?;
    let bytes = to_bytes(ctx, arg(args, 1), &format!("setParamTyped('{name}')"))?;
    let label = ctx.to_rust_string(arg(args, 2))?;
    let r = router_of(ctx, this)?.set_param_typed(&name, &bytes, &label);
    lift(ctx, r)?;
    Ok(Value::Undefined)
}

/// `runOn(device, inputs)` → `[Float32Array, …]` — exactly that device.
fn router_run_on(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let device = ctx.to_rust_string(arg(args, 0))?;
    let bag = arg(args, 1).clone();
    let owned = collect(ctx, &bag)?;
    let r = router_of(ctx, this)?.run_on(&device, &pairs(&owned));
    let outs = lift(ctx, r)?;
    let values: Vec<Value> = outs.iter().map(|o| new_f32_array(ctx, o)).collect();
    Ok(new_array(ctx, values))
}

fn router_run(ctx: &mut Context, this: &Value, args: &[Value], magic: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let owned = collect(ctx, &bag)?;
    let hint = optional_device(ctx, arg(args, 1))?;
    let router = router_of(ctx, this)?;
    let r = if magic == 0 {
        router.run(&pairs(&owned), hint.as_deref())
    } else {
        router.run_chain(&pairs(&owned), hint.as_deref())
    };
    let (device, outs) = lift(ctx, r)?;
    Ok(ran(ctx, device, outs))
}

fn router_run_policy(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    router_run(ctx, this, args, 0)
}

fn router_run_chain(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    router_run(ctx, this, args, 1)
}

fn router_rebench(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let enabled = to_bool(arg(args, 0));
    router_of(ctx, this)?.set_rebench_on_throttle(enabled);
    Ok(Value::Undefined)
}

// ── installation ────────────────────────────────────────────

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    js_class! {
        ctx, namespace;
        name: "Runner",
        class: CLASS_RUNNER,
        ctor: runner_new,
        methods: {
            "devices" => runner_devices, 0;
            "fastest" => runner_fastest, 0;
            "resolve" => runner_resolve, 1;
            "report" => runner_report, 0;
            "setParam" => runner_set_param, 2;
            "setParams" => runner_set_params, 1;
            "setParamTyped" => runner_set_param_typed, 3;
            "warmAll" => runner_warm_all, 0;
            "benchmark" => runner_benchmark, 2;
            "run" => runner_run, 2;
            "runOn" => runner_run_on, 2;
            "runTry" => runner_run_try, 2;
            "runChain" => runner_run_chain, 2;
            "runWithInputs" => runner_run_with_inputs, 2;
        }
    };
    js_statics! { ctx, namespace, "Runner"; "fromEnv" => runner_from_env, 1; };
    js_class! {
        ctx, namespace;
        name: "Router",
        class: CLASS_ROUTER,
        ctor: router_new,
        methods: {
            "devices" => router_devices, 0;
            "setParam" => router_set_param, 2;
            "setParams" => router_set_params, 1;
            "setRebenchOnThrottle" => router_rebench, 1;
            "run" => router_run_policy, 2;
            "runChain" => router_run_chain, 2;
            "runOn" => router_run_on, 2;
            "setParamTyped" => router_set_param_typed, 3;
        }
    };
    js_statics! { ctx, namespace, "Router"; "fromEnv" => router_from_env, 1; };
}
