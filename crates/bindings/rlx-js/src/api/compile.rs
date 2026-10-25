// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-compile options: fusion toggles, kernel-dispatch policy, precision
//! policy.
//!
//! A plain options object rather than a class — `{fusion: {skipFusion: true}}`
//! is how a JS caller expects to configure a call, and it keeps A/B sweeps to
//! one object literal:
//!
//! ```js
//! for (const nativeFk of [false, true]) {
//!   const c = session.compileWith(build(), { fusion: { nativeFkRegions: nativeFk } });
//!   …
//! }
//! ```
//!
//! The object is handed to `rlx-bind` as JSON and parsed there, so its shape is
//! the same one pyrlx and the C ABI take. A policy is a preset name or an
//! object of op kinds:
//!
//! ```js
//! new rlx.Session({ device: "metal", policy: "mixed" })
//! new rlx.Session({ device: "metal", policy: { compute: "bf16", reduction: "f32" } })
//! ```

use quickrs_core::context::Context;
use quickrs_core::value::{JsResult, Value};
use rlx_bind::runtime::{CompileRequest, policy};
use rlx_runtime::PrecisionPolicy;

use crate::api::bind::{lift, to_json};

/// A precision policy from a preset name or an object of op kinds.
pub fn parse_precision_policy(ctx: &mut Context, v: &Value) -> JsResult<Option<PrecisionPolicy>> {
    let json = to_json(ctx, v)?;
    lift(ctx, policy::parse(&json))
}

/// Refuse the policies a backend gets wrong (bf16 on Metal), naming what goes wrong.
pub fn check_policy_supported(
    ctx: &mut Context,
    device: rlx_runtime::Device,
    policy: &PrecisionPolicy,
) -> JsResult<()> {
    lift(ctx, policy::check_supported(device, policy))
}

pub fn policy_label(policy: &PrecisionPolicy) -> &'static str {
    policy::label(policy)
}

/// `{fusion, kernelDispatch, policy}` → what a compile call overrides.
pub fn compile_request(ctx: &mut Context, options: &Value) -> JsResult<CompileRequest> {
    let json = to_json(ctx, options)?;
    lift(ctx, CompileRequest::from_json(&json))
}
