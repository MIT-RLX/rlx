// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-compile knobs: fusion toggles and kernel-dispatch policy.
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

use quickrs_core::context::Context;
use quickrs_core::value::{JsResult, Value};
use std::collections::HashMap;

use rlx_ir::logical_kernel::KernelDispatchPolicy;
use rlx_opt::FusionOptions;
// Two different `Precision` types are in play: `rlx_runtime::Precision` is the
// session-wide one, `rlx_opt::Precision` is what a per-op-kind policy holds.
// Aliasing the second makes which is which visible at every use.
use rlx_opt::Precision as OpPrecision;
use rlx_runtime::{CompileOptions, OpKind, Precision, PrecisionPolicy};

use crate::convert::*;

fn parse_kernel_dispatch(ctx: &mut Context, s: &str) -> JsResult<KernelDispatchPolicy> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "native" | "force_native" => KernelDispatchPolicy::ForceNative,
        "common" | "force_common" => KernelDispatchPolicy::ForceCommon,
        "prefer_native" | "prefer" | "default" => KernelDispatchPolicy::PreferNative,
        other => {
            return ctx.throw_type(&format!(
                "unknown kernelDispatch '{other}' (native, common, prefer_native)"
            ));
        }
    })
}

/// Read one boolean field, falling back to the `FusionOptions` default rather
/// than to `false` — `fkFusion` and the two `fuse*` flags default **on**, and
/// silently clearing them would make `{}` a different compile from no options
/// at all.
fn flag(ctx: &mut Context, options: &Value, name: &str, default: bool) -> JsResult<bool> {
    let v = field(ctx, options, name)?;
    Ok(if is_nullish(&v) { default } else { to_bool(&v) })
}

fn parse_fusion(ctx: &mut Context, options: &Value) -> JsResult<FusionOptions> {
    let base = FusionOptions::default();
    Ok(FusionOptions {
        skip_fusion: flag(ctx, options, "skipFusion", base.skip_fusion)?,
        unfuse_elementwise_regions: flag(
            ctx,
            options,
            "unfuseElementwiseRegions",
            base.unfuse_elementwise_regions,
        )?,
        keep_elementwise_regions: flag(
            ctx,
            options,
            "keepElementwiseRegions",
            base.keep_elementwise_regions,
        )?,
        decompose_fusion_regions: flag(
            ctx,
            options,
            "decomposeFusionRegions",
            base.decompose_fusion_regions,
        )?,
        fk_fusion: flag(ctx, options, "fkFusion", base.fk_fusion)?,
        fuse_region_prologue: flag(
            ctx,
            options,
            "fuseRegionPrologue",
            base.fuse_region_prologue,
        )?,
        fuse_batch_preprocess: flag(
            ctx,
            options,
            "fuseBatchPreprocess",
            base.fuse_batch_preprocess,
        )?,
        native_fk_regions: flag(ctx, options, "nativeFkRegions", base.native_fk_regions)?,
        disable_conv_bias_act_fusion: flag(
            ctx,
            options,
            "disableConvBiasActFusion",
            base.disable_conv_bias_act_fusion,
        )?,
        ..base
    })
}

/// Parse a per-op-kind precision policy.
///
/// A string names one of the presets; an object gives an explicit policy per op
/// kind. `Boundary` is what a script sees, so it stays f32 unless the caller
/// says otherwise — a graph whose inputs and outputs silently became f16 would
/// hand back values that do not round-trip.
///
/// ```js
/// new rlx.Session({ device: "metal", policy: "mixed" })
/// new rlx.Session({ device: "metal", policy: { compute: "bf16", reduction: "f32" } })
/// ```
pub fn parse_precision_policy(ctx: &mut Context, v: &Value) -> JsResult<Option<PrecisionPolicy>> {
    if is_nullish(v) {
        return Ok(None);
    }
    if v.as_string().is_some() {
        let label = ctx.to_rust_string(v)?;
        return Ok(Some(
            match label.trim().to_ascii_lowercase().replace('_', "-").as_str() {
                "f32" | "always-f32" | "off" => PrecisionPolicy::AlwaysF32,
                "f16" | "always-f16" => PrecisionPolicy::AlwaysF16,
                // The tuned default: compute and data movement stay f32 because
                // f16 there diverged on real decode paths; elementwise and
                // reduction go f16, and the reduction kernels still accumulate
                // in f32 internally.
                "mixed" | "auto" | "amp" | "auto-mixed" => PrecisionPolicy::AutoMixed,
                "mixed-conservative" | "conservative" => PrecisionPolicy::AutoMixedConservative,
                "mixed-bf16" | "bf16" => PrecisionPolicy::AutoMixedBf16,
                other => {
                    return ctx.throw_type(&format!(
                        "unknown precision policy '{other}' (f32, f16, mixed,                          mixed-conservative, mixed-bf16, or an object of op kinds)"
                    ));
                }
            },
        ));
    }

    // Explicit per-op-kind.
    let mut map: HashMap<OpKind, OpPrecision> = HashMap::new();
    let kinds = [
        ("compute", OpKind::Compute),
        ("reduction", OpKind::Reduction),
        ("elementwise", OpKind::Elementwise),
        ("dataMovement", OpKind::DataMovement),
        ("boundary", OpKind::Boundary),
    ];
    let mut named = 0usize;
    for (key, kind) in kinds {
        let entry = field(ctx, v, key)?;
        if is_nullish(&entry) {
            // Unnamed kinds default to f32 rather than to the lowest precision
            // present: a partial policy should not silently downcast what it
            // did not mention.
            map.insert(kind, OpPrecision::F32);
            continue;
        }
        let label = ctx.to_rust_string(&entry)?;
        let precision = match label.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" => OpPrecision::F32,
            "f16" | "float16" | "half" => OpPrecision::F16,
            "bf16" | "bfloat16" => OpPrecision::BF16,
            other => {
                return ctx.throw_type(&format!(
                    "precision policy {key}: unknown precision '{other}' (f32, f16, bf16)"
                ));
            }
        };
        map.insert(kind, precision);
        named += 1;
    }
    if named == 0 {
        return ctx.throw_type(
            "precision policy object named no op kinds (compute, reduction, elementwise,              dataMovement, boundary)",
        );
    }
    Ok(Some(PrecisionPolicy::Custom(map)))
}

/// Build `CompileOptions` from `{fusion, kernelDispatch, policy}`, keeping the
/// session's precision.
pub fn build_compile_options(
    ctx: &mut Context,
    precision: Precision,
    options: &Value,
) -> JsResult<CompileOptions> {
    let mut opts = CompileOptions::new().precision(precision);
    let fusion = field(ctx, options, "fusion")?;
    if !is_nullish(&fusion) {
        opts.fusion_opts = parse_fusion(ctx, &fusion)?;
    }
    let dispatch = field(ctx, options, "kernelDispatch")?;
    if !is_nullish(&dispatch) {
        let label = ctx.to_rust_string(&dispatch)?;
        opts.kernel_dispatch.policy = parse_kernel_dispatch(ctx, &label)?;
    }
    let policy = field(ctx, options, "policy")?;
    if let Some(policy) = parse_precision_policy(ctx, &policy)? {
        opts.policy = Some(policy);
    }
    Ok(opts)
}

/// Op kinds a policy asks to run in BF16.
///
/// Metal has no bf16 compute kernels at all: `HalfFlag` is `{F32, F16}` and
/// maps BF16 to F32, so a bf16-tagged node runs an f32 kernel over bf16 bytes
/// and two bf16 values are read as one f32.
pub fn bf16_kinds(policy: &PrecisionPolicy) -> Vec<&'static str> {
    use rlx_runtime::OpKind::*;
    [
        (Compute, "compute"),
        (Reduction, "reduction"),
        (Elementwise, "elementwise"),
        (DataMovement, "dataMovement"),
        (Boundary, "boundary"),
    ]
    .into_iter()
    .filter(|(kind, _)| policy.precision_for(*kind) == OpPrecision::BF16)
    .map(|(_, name)| name)
    .collect()
}

/// Refuse the policies a backend gets wrong, naming what goes wrong.
///
/// This used to refuse **f16** on Metal as well, because a single
/// `Op::Attention` under `AlwaysF16` returned all zeros and a transformer block
/// NaN'd in `silu`. Both were one bug: `Op::Attention` and `Op::LayerNorm2d`
/// were missing from the "Metal kernels are still f32-only" list in
/// `rlx-compile/src/precision.rs`, so the pass retagged them F16 and an f32
/// kernel then read f16 bytes. With those listed, f16 measures 1.3e-3 relative
/// on a transformer block and 5e-4 to 1.3e-3 across matmul / softmax / rms_norm
/// / layer_norm / gelu / silu (`rlx-metal/tests/precision_policy_f16_attention.rs`).
/// So f16 is allowed now.
///
/// BF16 is still refused: it is not a missing entry in a list, it is a missing
/// dtype in `HalfFlag`. `rlx-metal` asserts the same thing at compile time; this
/// check exists so the JS caller gets a TypeError with somewhere to go rather
/// than a panic crossing the FFI boundary.
pub fn check_policy_supported(
    ctx: &mut Context,
    device: rlx_runtime::Device,
    policy: &PrecisionPolicy,
) -> JsResult<()> {
    if !matches!(device, rlx_runtime::Device::Metal) {
        return Ok(());
    }
    let offenders = bf16_kinds(policy);
    if offenders.is_empty() {
        return Ok(());
    }
    ctx.throw_type(&format!(
        "precision policy '{}' asks for bf16 on {}, and Metal has no bf16 compute \
         kernels — an f32 kernel would read two bf16 values as one f32 and return \
         non-finite garbage rather than an error. Use 'f16' (measured 1.3e-3 \
         relative on a transformer block) or 'mixed' (compute stays f32).",
        policy_label(policy),
        offenders.join(", ")
    ))
}

pub fn policy_label(policy: &PrecisionPolicy) -> &'static str {
    match policy {
        PrecisionPolicy::AlwaysF32 => "f32",
        PrecisionPolicy::AlwaysF16 => "f16",
        PrecisionPolicy::AutoMixed => "mixed",
        PrecisionPolicy::AutoMixedConservative => "mixed-conservative",
        PrecisionPolicy::AutoMixedBf16 => "mixed-bf16",
        PrecisionPolicy::AutoMixedBf16Safe => "mixed-bf16-safe",
        PrecisionPolicy::Custom(_) => "custom",
    }
}
