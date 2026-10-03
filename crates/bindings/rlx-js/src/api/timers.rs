// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `setTimeout`, `queueMicrotask`, `rlx.sleep` — and the loop that drains them.
//!
//! QuickJS-ng resolves promises through a microtask queue, but nothing *drives*
//! it and there are no timers at all, so `await` worked only for promises that
//! were already settled. `(async () => 1)()` came back as `"[object Promise]"`.
//!
//! This is a real, if small, event loop: one timer queue, drained in due order,
//! with the microtask queue run to exhaustion between each. Single-threaded, so
//! it buys cooperative scheduling — a long training run can yield to a progress
//! reporter — not parallelism. Nothing here pretends otherwise.
//!
//! Timers live in a JS array on the hidden registry rather than in Rust, so the
//! callbacks are traced by the cycle collector like any other reachable value.

use std::time::{Duration, Instant};

use quickrs_core::context::Context;
use quickrs_core::object::PropFlags;
use quickrs_core::value::{JsResult, PropKey, Value};

const TIMERS: &str = "__rlx_timers";
const NEXT_ID: &str = "__rlx_timer_id";

fn started() -> Instant {
    // One base for the whole process: `due` is milliseconds from here.
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *START.get_or_init(Instant::now)
}

fn now_ms() -> f64 {
    started().elapsed().as_secs_f64() * 1000.0
}

fn timers(ctx: &mut Context) -> Option<Value> {
    let key = PropKey::from_string(&ctx.intern(TIMERS));
    let global = Value::Object(ctx.global());
    ctx.get_property(&global, &key)
        .ok()
        .filter(|v| v.is_object())
}

/// `setTimeout(fn, ms, ...args)` → a numeric id.
fn f_set_timeout(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let callback = crate::convert::arg(args, 0).clone();
    if !callback.is_object() {
        return ctx.throw_type("setTimeout: the first argument must be a function");
    }
    let delay = if crate::convert::is_nullish(crate::convert::arg(args, 1)) {
        0.0
    } else {
        ctx.to_number(crate::convert::arg(args, 1))?.max(0.0)
    };
    let extra: Vec<Value> = args.iter().skip(2).cloned().collect();

    let global = Value::Object(ctx.global());
    let id_key = PropKey::from_string(&ctx.intern(NEXT_ID));
    let id = ctx
        .get_property(&global, &id_key)
        .ok()
        .and_then(|v| match v {
            Value::Int(n) => Some(n as f64),
            Value::Float(x) => Some(x),
            _ => None,
        })
        .unwrap_or(1.0);
    if let Value::Object(g) = &global {
        ctx.define_value(g, NEXT_ID, Value::number(id + 1.0), PropFlags::C_W);
    }

    let entry = ctx.new_object();
    ctx.define_value(&entry, "id", Value::number(id), PropFlags::C_W_E);
    ctx.define_value(
        &entry,
        "due",
        Value::number(now_ms() + delay),
        PropFlags::C_W_E,
    );
    ctx.define_value(&entry, "fn", callback, PropFlags::C_W_E);
    let args_array = crate::convert::new_array(ctx, extra);
    ctx.define_value(&entry, "args", args_array, PropFlags::C_W_E);

    if let Some(list) = timers(ctx) {
        let len = ctx.length_of(&list)? as u32;
        ctx.set_property(&list, &PropKey::Index(len), Value::Object(entry), false)?;
    }
    Ok(Value::number(id))
}

/// `clearTimeout(id)` — drops the entry if it has not fired.
fn f_clear_timeout(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let id = ctx.to_number(crate::convert::arg(args, 0))?;
    let Some(list) = timers(ctx) else {
        return Ok(Value::Undefined);
    };
    let len = ctx.length_of(&list)? as usize;
    for i in 0..len {
        let entry = ctx.get_property(&list, &PropKey::Index(i as u32))?;
        let entry_id = crate::convert::field(ctx, &entry, "id")?;
        if ctx.to_number(&entry_id)? == id {
            if let Some(obj) = entry.as_object().cloned() {
                ctx.define_value(&obj, "cleared", Value::Bool(true), PropFlags::C_W_E);
            }
            break;
        }
    }
    Ok(Value::Undefined)
}

/// `queueMicrotask(fn)` — straight onto the promise job queue.
fn f_queue_microtask(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let callback = crate::convert::arg(args, 0).clone();
    if !callback.is_object() {
        return ctx.throw_type("queueMicrotask: the argument must be a function");
    }
    ctx.enqueue_job(callback, Vec::new());
    Ok(Value::Undefined)
}

/// `rlx.sleep(ms)` → a promise that settles after `ms`.
///
/// The idiomatic way to yield: `await rlx.sleep(0)` lets pending microtasks and
/// due timers run before the next chunk of work.
fn f_sleep(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let delay = if crate::convert::is_nullish(crate::convert::arg(args, 0)) {
        0.0
    } else {
        ctx.to_number(crate::convert::arg(args, 0))?.max(0.0)
    };
    let promise = ctx.new_promise();

    // The resolver is a native function carrying the promise in a slot, so the
    // timer queue holds an ordinary callable and needs no special case.
    let resolver = ctx.new_native_function(settle_slot, "resolveSleep", 0, 0);
    ctx.set_slot(&resolver, "__promise", Value::Object(promise.clone()));

    let set_timeout = ctx.global_value("setTimeout");
    ctx.call(
        set_timeout,
        Value::Undefined,
        &[Value::Object(resolver), Value::number(delay)],
    )?;
    Ok(Value::Object(promise))
}

fn settle_slot(ctx: &mut Context, _this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let promise = ctx.callee_slot("__promise")?;
    if let Some(obj) = promise.as_object().cloned() {
        ctx.settle_promise(&obj, Value::Undefined, false);
    }
    Ok(Value::Undefined)
}

/// Run microtasks, then due timers, until both are empty or `budget` elapses.
///
/// Returns false if it gave up with work outstanding, so a caller can say
/// "timers still pending" rather than hanging.
pub fn drain(ctx: &mut Context, budget: Duration) -> JsResult<bool> {
    let deadline = Instant::now() + budget;
    loop {
        ctx.run_jobs()?;

        // Earliest un-fired timer.
        let Some(list) = timers(ctx) else {
            return Ok(true);
        };
        let len = ctx.length_of(&list)? as usize;
        let mut best: Option<(usize, f64)> = None;
        for i in 0..len {
            let entry = ctx.get_property(&list, &PropKey::Index(i as u32))?;
            if !entry.is_object() {
                continue;
            }
            let cleared = crate::convert::field(ctx, &entry, "cleared")?;
            if cleared.to_boolean() {
                continue;
            }
            let due = crate::convert::field(ctx, &entry, "due")?;
            let due = ctx.to_number(&due)?;
            if best.is_none_or(|(_, b)| due < b) {
                best = Some((i, due));
            }
        }
        let Some((index, due)) = best else {
            return Ok(true);
        };

        let wait = due - now_ms();
        if wait > 0.0 {
            if Instant::now() + Duration::from_secs_f64(wait / 1000.0) > deadline {
                return Ok(false);
            }
            std::thread::sleep(Duration::from_secs_f64(wait / 1000.0));
        } else if Instant::now() > deadline {
            return Ok(false);
        }

        let entry = ctx.get_property(&list, &PropKey::Index(index as u32))?;
        // Mark fired before calling: a callback that throws must not re-run.
        if let Some(obj) = entry.as_object().cloned() {
            ctx.define_value(&obj, "cleared", Value::Bool(true), PropFlags::C_W_E);
        }
        let callback = crate::convert::field(ctx, &entry, "fn")?;
        let extra = crate::convert::field(ctx, &entry, "args")?;
        let n = ctx.length_of(&extra).unwrap_or(0) as usize;
        let mut call_args = Vec::with_capacity(n);
        for i in 0..n {
            call_args.push(ctx.get_property(&extra, &PropKey::Index(i as u32))?);
        }
        ctx.call(callback, Value::Undefined, &call_args)?;
    }
}

pub fn install(
    ctx: &mut Context,
    namespace: &quickrs_core::gc::Gc<quickrs_core::object::JsObject>,
) {
    let _ = started();
    let list = ctx.new_array();
    let global = ctx.global();
    ctx.define_value(&global, TIMERS, Value::Object(list), PropFlags::NONE);
    ctx.define_value(&global, NEXT_ID, Value::number(1.0), PropFlags::C_W);

    // Global, because that is where a script expects them.
    ctx.define_method(&global, "setTimeout", f_set_timeout, 2, 0);
    ctx.define_method(&global, "clearTimeout", f_clear_timeout, 1, 0);
    ctx.define_method(&global, "setInterval", f_set_timeout, 2, 0);
    ctx.define_method(&global, "queueMicrotask", f_queue_microtask, 1, 0);
    ctx.define_method(namespace, "sleep", f_sleep, 1, 0);
}
