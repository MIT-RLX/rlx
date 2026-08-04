// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structured observability / audit spine (`tracing` feature, default-off).
//!
//! RLX's diagnostics are ~2600 ad-hoc `eprintln!`/`println!` sites: unstructured
//! and impossible to feed into an audit trail. This module is the typed, opt-in
//! alternative — a thin facade over the [`tracing`] crate that emits STRUCTURED
//! events with a stable name plus a metadata detail. It is the foundation for
//! the audit-logging a HIPAA/FDA validated pipeline needs (per-inference
//! records: model + weight hash, engine version, device, shapes, timing).
//!
//! ## PHI-safety invariant (load-bearing)
//! Events carry METADATA ONLY — ids, content hashes, shapes, dtypes, op kinds,
//! counts, timings. NEVER pass tensor contents, input values, decoded text, or
//! raw file paths that could embed PHI. Reference data by a content hash, never
//! the data itself. A leaked value in a log is a reportable breach, so the facade
//! deliberately takes a formatted *detail* the caller controls rather than
//! slurping arbitrary state.
//!
//! ## Zero-overhead default
//! Without the `tracing` feature every entry point compiles to a no-op (the
//! optimizer removes the call), exactly like [`crate::kernel_trace`]. Turn on
//! `--features tracing` to get structured logs; a downstream product installs
//! its own [`tracing`] subscriber to route events to an append-only audit sink.
//!
//! ```ignore
//! rlx_runtime::obs::event("inference", format_args!(
//!     "model={model_id} weight_sha={weight_sha} engine={engine} device={dev}"
//! ));
//! ```
//!
//! This is a *foundation*: one representative call site (`maybe_log_fusion`) is
//! wired through it. Migrating the remaining diagnostics off `eprintln!` is
//! tracked follow-up — each converted site becomes audit-visible for free.

/// Install a default `tracing` subscriber (idempotent). No-op without the
/// `tracing` feature. A downstream product typically installs its OWN
/// subscriber (to route to an audit sink) and never calls this; it exists so
/// `--features tracing` is usable standalone (tests, local debugging).
#[cfg(feature = "tracing")]
pub fn init() {
    use std::sync::OnceLock;
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        // Best-effort: if the host app already installed a subscriber, keep it.
        let _ = tracing_subscriber::fmt().with_target(true).try_init();
    });
}

/// No-op without the `tracing` feature.
#[cfg(not(feature = "tracing"))]
pub fn init() {}

/// Emit a structured observability/audit event. `kind` is a stable event name
/// (e.g. `"compile.fusion"`, `"inference"`); `detail` is a caller-formatted,
/// METADATA-ONLY payload (see the PHI-safety invariant). No-op without the
/// `tracing` feature — the optimizer removes the call.
#[cfg(feature = "tracing")]
#[inline]
pub fn event(kind: &str, detail: std::fmt::Arguments<'_>) {
    tracing::info!(target: "rlx", event = %kind, detail = %detail);
}

/// No-op without the `tracing` feature.
#[cfg(not(feature = "tracing"))]
#[inline]
pub fn event(_kind: &str, _detail: std::fmt::Arguments<'_>) {}

/// Structured event with a caller-controlled metadata detail. Expands to a
/// no-op without the `tracing` feature (the optimizer removes it), mirroring
/// [`crate::ktrace`]. Keep the detail metadata-only — never tensor values.
#[macro_export]
macro_rules! obs_event {
    ($kind:expr, $($arg:tt)+) => {{
        $crate::obs::event($kind, format_args!($($arg)+));
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The facade must be safe to call in BOTH build configs: a no-op without
    /// the feature, a real `tracing` event with it. (With the feature on and no
    /// subscriber installed, `tracing` events are simply dropped — still safe.)
    #[test]
    fn event_is_callable_in_any_config() {
        init();
        event("test.event", format_args!("k={} shape={}", 1, "[2,3]"));
        obs_event!(
            "test.event2",
            "engine={} weight_sha={}",
            "0.2.14",
            "deadbeef"
        );
    }
}
