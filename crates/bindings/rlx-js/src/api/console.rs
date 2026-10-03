// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `console.log` / `console.error` / `print`.
//!
//! QuickJS-ng has no host bindings of its own — a bare context cannot write a
//! byte. These are the minimum a script needs to be debuggable, and they are
//! separate from the `rlx` namespace so an embedder can install its own
//! (routing to a log crate, a UI pane, a test buffer) instead.

use quickrs_core::context::Context;
use quickrs_core::object::PropFlags;
use quickrs_core::value::{JsResult, Value};

/// Space-joined string form of every argument, as `console.log` specifies.
fn joined(ctx: &mut Context, args: &[Value]) -> JsResult<String> {
    let mut out = String::new();
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&ctx.to_rust_string(a)?);
    }
    Ok(out)
}

fn log(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    println!("{}", joined(ctx, args)?);
    Ok(Value::Undefined)
}

fn error(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    eprintln!("{}", joined(ctx, args)?);
    Ok(Value::Undefined)
}

pub fn install(ctx: &mut Context) {
    let console = ctx.new_object();
    js_functions! {
        ctx, &console;
        "log" => log, 1;
        "info" => log, 1;
        "debug" => log, 1;
        "warn" => error, 1;
        "error" => error, 1;
    };
    let global = ctx.global();
    ctx.define_value(&global, "console", Value::Object(console), PropFlags::C_W);
    ctx.define_method(&global, "print", log, 1, 0);
}
