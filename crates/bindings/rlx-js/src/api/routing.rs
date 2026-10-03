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
//! [`Runner`] wraps `GraphDevices` — explicit device choice, warm-all,
//! benchmark-pick. [`Router`] wraps `DeviceRouter`, which adds a fallback
//! chain: when a backend fails to compile or run, the next one is tried and
//! the error names every attempt rather than just the last.

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_runtime::{Device, DevicePolicy, DeviceRouter, GraphDevices};

use crate::api::graph::{parse_dtype, take_graph};
use crate::convert::*;
use crate::handle::{self, CLASS_ROUTER, CLASS_RUNNER};

fn runner_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut GraphDevices> {
    handle::borrow_mut(ctx, this, CLASS_RUNNER, "rlx.Runner")
}

fn router_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut DeviceRouter> {
    handle::borrow_mut(ctx, this, CLASS_ROUTER, "rlx.Router")
}

fn device_list(ctx: &mut Context, v: &Value, what: &str) -> JsResult<Vec<Device>> {
    let names = to_string_vec(ctx, v)?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        match rlx_runtime::parse_device(&name) {
            Ok(d) => out.push(d),
            Err(e) => return ctx.throw_type(&format!("{what}: {e}")),
        }
    }
    Ok(out)
}

/// `{only, deny, prefer, benchmarkRuns}` — or `"env"` to read `RLX_DEVICES`
/// and friends. Omitted entirely means every device this build has.
pub fn parse_policy(ctx: &mut Context, v: &Value) -> JsResult<DevicePolicy> {
    if is_nullish(v) {
        return Ok(DevicePolicy::all());
    }
    if let Some(s) = v.as_string() {
        let label = s.to_string_lossy();
        return match label.trim().to_ascii_lowercase().as_str() {
            "env" => Ok(DevicePolicy::from_env()),
            "all" => Ok(DevicePolicy::all()),
            other => ctx.throw_type(&format!(
                "device policy must be 'env', 'all' or an object, got '{other}'"
            )),
        };
    }

    let only_v = field(ctx, v, "only")?;
    let deny_v = field(ctx, v, "deny")?;
    let prefer_v = field(ctx, v, "prefer")?;
    let runs_v = field(ctx, v, "benchmarkRuns")?;

    let mut policy = if is_nullish(&only_v) {
        DevicePolicy::all()
    } else {
        DevicePolicy::only(device_list(ctx, &only_v, "policy.only")?)
    };
    if !is_nullish(&deny_v) {
        policy = policy.with_deny(device_list(ctx, &deny_v, "policy.deny")?);
    }
    if !is_nullish(&prefer_v) {
        policy = policy.with_prefer(device_list(ctx, &prefer_v, "policy.prefer")?);
    }
    if !is_nullish(&runs_v) {
        policy = policy.with_benchmark_pick(to_usize(ctx, &runs_v, "policy.benchmarkRuns")?);
    }
    Ok(policy)
}

fn optional_device(ctx: &mut Context, v: &Value, what: &str) -> JsResult<Option<Device>> {
    if is_nullish(v) {
        return Ok(None);
    }
    let name = ctx.to_rust_string(v)?;
    match rlx_runtime::parse_device(&name) {
        Ok(d) => Ok(Some(d)),
        Err(e) => ctx.throw_type(&format!("{what}: {e}")),
    }
}

/// `{name: Float32Array}` → owned buffers, converted before any handle borrow.
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

fn outputs_to_js(ctx: &mut Context, outs: Vec<Vec<f32>>) -> Value {
    let values: Vec<Value> = outs.iter().map(|o| new_f32_array(ctx, o)).collect();
    new_array(ctx, values)
}

// ── Runner (GraphDevices) ───────────────────────────────────

fn runner_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let policy = parse_policy(ctx, arg(args, 1))?;
    let graph = take_graph(ctx, arg(args, 0))?;
    Ok(handle::wrap(
        ctx,
        CLASS_RUNNER,
        GraphDevices::with_policy(graph, policy),
    ))
}

fn runner_devices(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let names: Vec<&'static str> = runner_of(ctx, this)?
        .devices()
        .iter()
        .map(|d| rlx_runtime::device_label(*d))
        .collect();
    Ok(new_string_array(ctx, &names))
}

fn runner_fastest(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let label = rlx_runtime::device_label(runner_of(ctx, this)?.fastest());
    Ok(new_string(ctx, label))
}

/// `resolve(hint?)` — what the policy would actually pick.
fn runner_resolve(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let hint = optional_device(ctx, arg(args, 0), "resolve hint")?;
    match runner_of(ctx, this)?.resolve(hint) {
        Ok(d) => {
            let label = rlx_runtime::device_label(d);
            Ok(new_string(ctx, label))
        }
        Err(e) => ctx.throw_internal(&e),
    }
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
    let dtype = parse_dtype(ctx, &label)?;
    runner_of(ctx, this)?.set_param_typed(&name, &bytes, dtype);
    Ok(Value::Undefined)
}

/// Compile every viable backend up front, so a later `run` is not paying for
/// a first compile inside whatever you are timing.
fn runner_warm_all(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    match runner_of(ctx, this)?.warm_all() {
        Ok(devices) => {
            let names: Vec<&'static str> = devices
                .iter()
                .map(|d| rlx_runtime::device_label(*d))
                .collect();
            Ok(new_string_array(ctx, &names))
        }
        Err(e) => ctx.throw_internal(&e),
    }
}

/// `benchmark(inputs, runs)` → `[{device, compileMs, execMs}]`, median exec.
fn runner_benchmark(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let owned = collect(ctx, &bag)?;
    let runs = if is_nullish(arg(args, 1)) {
        3
    } else {
        to_usize(ctx, arg(args, 1), "benchmark runs")?
    };
    let results = {
        let pairs: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        match runner_of(ctx, this)?.benchmark(&pairs, runs) {
            Ok(r) => r,
            Err(e) => return ctx.throw_internal(&e),
        }
    };
    let rows: Vec<Value> = results
        .into_iter()
        .map(|r| {
            let device = new_string(ctx, r.label);
            new_object(
                ctx,
                vec![
                    ("device", device),
                    ("compileMs", Value::number(r.compile_ns as f64 / 1e6)),
                    ("execMs", Value::number(r.median_exec_ns as f64 / 1e6)),
                ],
            )
        })
        .collect();
    Ok(new_array(ctx, rows))
}

/// `run(inputs, device?)` → `{device, outputs}`. Without a device the policy
/// decides; the answer is reported back so a caller never has to guess which
/// backend produced the numbers.
fn runner_run(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let owned = collect(ctx, &bag)?;
    let hint = optional_device(ctx, arg(args, 1), "run device")?;
    let (device, outs) = {
        let pairs: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        let runner = runner_of(ctx, this)?;
        let device = match runner.resolve(hint) {
            Ok(d) => d,
            Err(e) => return ctx.throw_internal(&e),
        };
        match runner.run(device, &pairs) {
            Ok(outs) => (device, outs),
            Err(e) => return ctx.throw_internal(&e),
        }
    };
    let label = new_string(ctx, rlx_runtime::device_label(device));
    let outputs = outputs_to_js(ctx, outs);
    Ok(new_object(
        ctx,
        vec![("device", label), ("outputs", outputs)],
    ))
}

fn runner_report(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let report = runner_of(ctx, this)?.report();
    let rows: Vec<Value> = report
        .into_iter()
        .map(|c| {
            let device = new_string(ctx, c.label);
            let blocker = match &c.blocker {
                Some(text) => new_string(ctx, text),
                None => Value::Null,
            };
            new_object(
                ctx,
                vec![
                    ("device", device),
                    ("available", Value::Bool(c.available)),
                    ("supportsGraph", Value::Bool(c.supports_graph)),
                    ("recommended", Value::Bool(c.recommended)),
                    ("blocker", blocker),
                ],
            )
        })
        .collect();
    Ok(new_array(ctx, rows))
}

// ── Router (DeviceRouter) ───────────────────────────────────

fn router_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let policy = parse_policy(ctx, arg(args, 1))?;
    let graph = take_graph(ctx, arg(args, 0))?;
    match DeviceRouter::new(graph, policy) {
        Ok(router) => Ok(handle::wrap(ctx, CLASS_ROUTER, router)),
        Err(e) => ctx.throw_internal(&e),
    }
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

/// `run(inputs, hint?)` (magic 0) — policy pick.
/// `runChain(inputs, hint?)` (magic 1) — fallback chain; on total failure the
/// error lists every device tried and why each one declined.
fn router_run(ctx: &mut Context, this: &Value, args: &[Value], magic: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let owned = collect(ctx, &bag)?;
    let hint = optional_device(ctx, arg(args, 1), "run device")?;
    let (device, outs) = {
        let pairs: Vec<(&str, &[f32])> = owned
            .iter()
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();
        let router = router_of(ctx, this)?;
        if magic == 0 {
            match router.run(&pairs, hint) {
                Ok(v) => v,
                Err(e) => return ctx.throw_internal(&e),
            }
        } else {
            match router.run_chain(&pairs, hint) {
                Ok(v) => v,
                Err(e) => return ctx.throw_internal(&format!("{e}")),
            }
        }
    };
    let label = new_string(ctx, rlx_runtime::device_label(device));
    let outputs = outputs_to_js(ctx, outs);
    Ok(new_object(
        ctx,
        vec![("device", label), ("outputs", outputs)],
    ))
}

/// Policy pick.
fn router_run_policy(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    router_run(ctx, this, args, 0)
}

/// Fallback chain: on total failure the error lists every device tried.
fn router_run_chain(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    router_run(ctx, this, args, 1)
}

/// Re-benchmark when the picked device starts thermal-throttling.
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
        }
    };
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
        }
    };
}
