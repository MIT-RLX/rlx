// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Precisions and per-op-kind precision policies.
//!
//! A policy is a preset name or an object giving a precision per op kind:
//!
//! ```text
//! "mixed"
//! {"compute": "f16", "reduction": "f32"}
//! ```
//!
//! Op kinds an object does not name stay f32 rather than following the lowest
//! precision present: a partial policy must not silently downcast what it did
//! not mention — `boundary` above all, since inputs and outputs that became
//! f16 would not round-trip.

use std::collections::HashMap;

use rlx_opt::Precision as OpPrecision;
use rlx_runtime::{Device, OpKind, Precision, PrecisionPolicy};
use serde_json::Value;

use crate::enums::key;
use crate::{ApiError, ApiResult};

/// The session-wide compute precision: `"f32"`, `"f16"`, `"bf16"`.
pub fn precision(s: &str) -> ApiResult<Precision> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" | "float" => Precision::F32,
        "f16" | "float16" | "half" => Precision::F16,
        "bf16" | "bfloat16" => Precision::BF16,
        other => {
            return Err(ApiError::type_(format!(
                "unknown precision '{other}' (f32, f16, bf16)"
            )));
        }
    })
}

pub fn precision_label(p: Precision) -> &'static str {
    match p {
        Precision::F32 => "f32",
        Precision::F16 => "f16",
        Precision::BF16 => "bf16",
    }
}

const KINDS: [(&str, OpKind); 5] = [
    ("compute", OpKind::Compute),
    ("reduction", OpKind::Reduction),
    ("elementwise", OpKind::Elementwise),
    ("data_movement", OpKind::DataMovement),
    ("boundary", OpKind::Boundary),
];

/// A policy from a preset name, an object of op kinds, or null (no policy).
pub fn parse(v: &Value) -> ApiResult<Option<PrecisionPolicy>> {
    match v {
        Value::Null => Ok(None),
        Value::String(label) => preset(label).map(Some),
        Value::Object(map) => {
            for k in map.keys() {
                if !KINDS.iter().any(|(name, _)| key(name) == key(k)) {
                    return Err(ApiError::type_(format!(
                        "precision policy: unknown op kind '{k}' (compute, reduction, \
                         elementwise, data_movement, boundary)"
                    )));
                }
            }
            let mut out: HashMap<OpKind, OpPrecision> = HashMap::new();
            let mut named = 0usize;
            for (name, kind) in KINDS {
                let entry = map
                    .iter()
                    .find(|(k, _)| key(k) == key(name))
                    .map(|(_, v)| v)
                    .filter(|v| !v.is_null());
                let Some(entry) = entry else {
                    out.insert(kind, OpPrecision::F32);
                    continue;
                };
                let Some(label) = entry.as_str() else {
                    return Err(ApiError::type_(format!(
                        "precision policy {name}: expected a precision name, got {}",
                        super::opts::type_name(entry)
                    )));
                };
                let p = match label.trim().to_ascii_lowercase().as_str() {
                    "f32" | "float32" => OpPrecision::F32,
                    "f16" | "float16" | "half" => OpPrecision::F16,
                    "bf16" | "bfloat16" => OpPrecision::BF16,
                    other => {
                        return Err(ApiError::type_(format!(
                            "precision policy {name}: unknown precision '{other}' (f32, f16, bf16)"
                        )));
                    }
                };
                out.insert(kind, p);
                named += 1;
            }
            if named == 0 {
                return Err(ApiError::type_(
                    "precision policy object named no op kinds (compute, reduction, \
                     elementwise, data_movement, boundary)",
                ));
            }
            Ok(Some(PrecisionPolicy::Custom(out)))
        }
        other => Err(ApiError::type_(format!(
            "precision policy must be a preset name or an object of op kinds, got {}",
            super::opts::type_name(other)
        ))),
    }
}

fn preset(label: &str) -> ApiResult<PrecisionPolicy> {
    Ok(
        match label.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "f32" | "always-f32" | "off" => PrecisionPolicy::AlwaysF32,
            "f16" | "always-f16" => PrecisionPolicy::AlwaysF16,
            // Compute and data movement stay f32 (f16 there diverged on real
            // decode paths); elementwise and reduction go f16, and the
            // reduction kernels still accumulate in f32 internally.
            "mixed" | "auto" | "amp" | "auto-mixed" => PrecisionPolicy::AutoMixed,
            "mixed-conservative" | "conservative" => PrecisionPolicy::AutoMixedConservative,
            "mixed-bf16" | "bf16" => PrecisionPolicy::AutoMixedBf16,
            "mixed-bf16-safe" | "bf16-safe" => PrecisionPolicy::AutoMixedBf16Safe,
            other => {
                return Err(ApiError::type_(format!(
                    "unknown precision policy '{other}' (f32, f16, mixed, mixed-conservative, \
                     mixed-bf16, mixed-bf16-safe, or an object of op kinds)"
                )));
            }
        },
    )
}

pub fn label(policy: &PrecisionPolicy) -> &'static str {
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

/// Op kinds a policy asks to run in BF16.
pub fn bf16_kinds(policy: &PrecisionPolicy) -> Vec<&'static str> {
    KINDS
        .into_iter()
        .filter(|(_, kind)| policy.precision_for(*kind) == OpPrecision::BF16)
        .map(|(name, _)| name)
        .collect()
}

/// Refuse the policies a backend gets wrong, naming what goes wrong.
///
/// Metal has no bf16 compute kernels: `HalfFlag` is `{F32, F16}` and maps BF16
/// to F32, so a bf16-tagged node runs an f32 kernel over bf16 bytes and two
/// bf16 values are read as one f32. f16 is fine there (1.3e-3 relative on a
/// transformer block, `rlx-metal/tests/precision_policy_f16_attention.rs`).
pub fn check_supported(device: Device, policy: &PrecisionPolicy) -> ApiResult<()> {
    if !matches!(device, Device::Metal) {
        return Ok(());
    }
    let offenders = bf16_kinds(policy);
    if offenders.is_empty() {
        return Ok(());
    }
    Err(ApiError::type_(format!(
        "precision policy '{}' asks for bf16 on {}, and Metal has no bf16 compute \
         kernels — an f32 kernel would read two bf16 values as one f32 and return \
         non-finite garbage rather than an error. Use 'f16' (measured 1.3e-3 \
         relative on a transformer block) or 'mixed' (compute stays f32).",
        label(policy),
        offenders.join(", ")
    )))
}
