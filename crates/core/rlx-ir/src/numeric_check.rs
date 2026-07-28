// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runtime NaN/Inf localization — the compiler-level debug epilogue.
//!
//! When a downstream model or developer produces a NaN/Inf, a raw bad
//! float is useless on its own — it says *what* but not *where* or *why*.
//! [`check_node`] turns it into a localized, actionable diagnostic:
//!
//!   * **which** op first produced it (callers scan in topological order,
//!     so the first trip is the origin — not the hundreds of downstream
//!     ops that merely inherited the NaN),
//!   * **culprit vs propagator** — were the op's inputs already non-finite
//!     (it propagated one) or were they clean and the output went bad (this
//!     op *created* it)? This single bit is the most useful localization
//!     signal: it tells the developer whether to look here or upstream,
//!   * **provenance** back to the user's source via [`node_label`], and
//!   * a one-line **fix hint** keyed off the op kind.
//!
//! It is deliberately backend-agnostic: it operates on already-materialized
//! `f32` host slices plus the [`Graph`] for provenance, so every backend can
//! call it from its run loop by handing over the input+output buffers it
//! already has. Detection is env-gated by callers (`RLX_DEBUG_NANS`) so it
//! costs nothing in production — see the CPU executor for the reference wiring.

use crate::op::{Activation, BinaryOp};
use crate::provenance::node_label;
use crate::{Graph, NodeId, Op};

/// What was found in a buffer that fails the finiteness check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BadValue {
    Nan,
    PosInf,
    NegInf,
}

impl BadValue {
    /// Classify a single value, or `None` if it is finite.
    #[inline]
    fn classify(v: f32) -> Option<BadValue> {
        if v.is_nan() {
            Some(BadValue::Nan)
        } else if v.is_infinite() {
            Some(if v > 0.0 {
                BadValue::PosInf
            } else {
                BadValue::NegInf
            })
        } else {
            None
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            BadValue::Nan => "NaN",
            BadValue::PosInf => "+inf",
            BadValue::NegInf => "-inf",
        }
    }
}

/// The first non-finite value in a buffer: what and where.
#[derive(Debug, Clone, Copy)]
pub struct BadHit {
    pub kind: BadValue,
    pub index: usize,
}

/// Scan a slice for the first non-finite value. `None` ⇒ all finite.
///
/// Returns on the first hit — O(n) worst case but usually much less, and
/// only ever called when a caller opts into debug checking.
#[inline]
pub fn first_bad(data: &[f32]) -> Option<BadHit> {
    for (i, &v) in data.iter().enumerate() {
        if let Some(kind) = BadValue::classify(v) {
            return Some(BadHit { kind, index: i });
        }
    }
    None
}

/// A localized NaN/Inf diagnostic tied to a specific graph node.
#[derive(Debug, Clone)]
pub struct NanReport {
    /// The node whose output tripped the check.
    pub node: NodeId,
    /// Best-effort provenance label (origin label, node name, or id).
    pub label: String,
    /// Short op description (`Rsqrt`, `Div`, …).
    pub op: String,
    pub kind: BadValue,
    /// Flat index of the first bad element in the output buffer.
    pub index: usize,
    /// The input node that was *already* non-finite, if this op merely
    /// propagated a bad value. `None` ⇒ inputs were clean ⇒ this op is the
    /// culprit that produced the NaN/Inf.
    pub source_input: Option<NodeId>,
    /// A remedy hint, present only for culprit ops (a propagator's fix lives
    /// upstream, at whichever op is eventually flagged as the culprit).
    pub fix: Option<&'static str>,
}

impl std::fmt::Display for NanReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} at index {} of {} {} \"{}\"",
            self.kind.as_str(),
            self.index,
            self.node,
            self.op,
            self.label
        )?;
        match self.source_input {
            Some(src) => write!(
                f,
                "\n  → propagated: input {src} was already non-finite (look upstream)"
            ),
            None => {
                write!(f, "\n  → inputs finite, this op produced it")?;
                if let Some(fix) = self.fix {
                    write!(f, "\n  fix: {fix}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for NanReport {}

/// Short, human-readable op tag for diagnostics — keeps the specific
/// activation/binary (`Rsqrt`, not just `Activation`) that matters for the
/// fix hint, without dumping a `Constant`'s full byte payload.
fn op_short(op: &Op) -> String {
    match op {
        Op::Activation(a) => format!("{a:?}"),
        Op::Binary(b) => format!("{b:?}"),
        other => format!("{:?}", other.kind()),
    }
}

/// Map an op that *produced* a NaN/Inf to a concrete remedy. Deliberately
/// covers the common downstream mistakes; returns `None` for ops with no
/// single obvious fix (the location + culprit flag still localize it).
pub fn fix_hint(op: &Op) -> Option<&'static str> {
    match op {
        Op::Activation(Activation::Rsqrt | Activation::Sqrt) => Some(
            "rsqrt/sqrt of a negative or zero — norm variance underflow; raise eps or clamp input ≥ 0",
        ),
        Op::Activation(Activation::Log) => {
            Some("log of ≤ 0 — clamp the input to a small positive floor (e.g. 1e-12) or add eps")
        }
        Op::Activation(Activation::Exp) => {
            Some("exp overflow → +inf — subtract the row max before exp (unstable softmax?)")
        }
        Op::Binary(BinaryOp::Div) => Some(
            "division by zero — guard the denominator with eps, or mask with where(denom != 0, …)",
        ),
        Op::Binary(BinaryOp::Pow) => {
            Some("pow of a negative base or huge exponent — check base sign / exponent magnitude")
        }
        Op::Softmax { .. } => {
            Some("all-masked row or -inf logits — use a finite mask fill (-1e9), not f32 -inf")
        }
        Op::Constant { .. } => Some(
            "a Constant already holds NaN/inf — likely baked by constant-folding; run with RLX_LINT_NUMERICS to name the source op",
        ),
        _ => None,
    }
}

/// How a backend run loop should react to a localized NaN/Inf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugMode {
    /// Scanning off — the whole epilogue is skipped.
    Off,
    /// Print the first bad value's diagnostic to stderr and keep running.
    Warn,
    /// Print and then panic on the first bad value (fail-fast, like JAX
    /// `jax_debug_nans`), so a backtrace points at the call site.
    Abort,
}

/// Shared NaN/Inf debug epilogue for every backend run loop.
///
/// Centralizes the `RLX_DEBUG_NANS` policy so each backend adds only a few
/// lines: build one with [`DebugScanner::from_env`] before the loop, gate on
/// [`enabled`](Self::enabled), and hand each computed node's output + operand
/// slices to [`check`](Self::check). The env var is read once (constructing
/// the scanner), never per-node.
///
/// `RLX_DEBUG_NANS` values: unset / `0` / empty ⇒ [`Off`](DebugMode::Off);
/// `abort` ⇒ [`Abort`](DebugMode::Abort); anything else truthy ⇒
/// [`Warn`](DebugMode::Warn).
#[derive(Debug, Clone, Copy)]
pub struct DebugScanner {
    pub mode: DebugMode,
    /// Short backend tag for the message prefix (e.g. "cpu", "metal").
    backend: &'static str,
}

impl DebugScanner {
    /// Construct with an explicit mode (bypassing the env), tagging messages
    /// with `backend`. Useful in tests and when a caller drives the mode.
    pub fn with_mode(mode: DebugMode, backend: &'static str) -> Self {
        Self { mode, backend }
    }

    /// Build from `RLX_DEBUG_NANS`, tagging messages with `backend`.
    pub fn from_env(backend: &'static str) -> Self {
        let mode = match crate::env::var("RLX_DEBUG_NANS").as_deref() {
            None => DebugMode::Off,
            Some(v) if v.is_empty() || v == "0" => DebugMode::Off,
            Some("abort") => DebugMode::Abort,
            Some(_) => DebugMode::Warn,
        };
        Self { mode, backend }
    }

    /// True when scanning is active — gate the (possibly expensive) readback
    /// and gather on this so production pays nothing.
    #[inline]
    pub fn enabled(&self) -> bool {
        self.mode != DebugMode::Off
    }

    /// Scan a whole-graph run's outputs — the universal fallback for backends
    /// that execute opaquely (MPSGraph, MLX, CoreML MIL, PJRT/HLO) and can't
    /// hook per-op. Reports which graph output first went non-finite, with its
    /// provenance. `outputs` is zipped against `graph.outputs` positionally, as
    /// every backend returns them in that order. No-op when scanning is off.
    ///
    /// For *internal* localization on those backends, run the identical graph
    /// on the CPU backend with `RLX_DEBUG_NANS` — provenance is backend-neutral.
    pub fn check_outputs(&self, graph: &Graph, outputs: &[Vec<f32>]) {
        if self.mode == DebugMode::Off {
            return;
        }
        for (buf, &id) in outputs.iter().zip(graph.outputs.iter()) {
            self.check(graph, id, buf, &[]);
        }
    }

    /// Scan one node's output; on the first bad value, print a localized
    /// diagnostic and (in [`Abort`](DebugMode::Abort) mode) panic. Returns the
    /// report so callers may also collect/stop. No-op when the output is clean
    /// or scanning is off.
    pub fn check(
        &self,
        graph: &Graph,
        node: NodeId,
        output: &[f32],
        inputs: &[(NodeId, &[f32])],
    ) -> Option<NanReport> {
        if self.mode == DebugMode::Off {
            return None;
        }
        match check_node(graph, node, output, inputs) {
            Ok(()) => None,
            Err(report) => {
                eprintln!("rlx nan-check [{}]: {report}", self.backend);
                if self.mode == DebugMode::Abort {
                    panic!(
                        "rlx nan-check [{}]: NaN/Inf localized — aborting\n{report}",
                        self.backend
                    );
                }
                Some(report)
            }
        }
    }
}

/// Localize a NaN/Inf to a graph node.
///
/// `output` is the node's freshly-computed output buffer; `inputs` are the
/// `(id, buffer)` pairs the caller already has for the operands (any subset
/// is fine — e.g. non-`f32` operands may be omitted). Returns `Ok(())` when
/// the output is clean, or a [`NanReport`] pinpointing the first bad element
/// and classifying the node as culprit or propagator.
pub fn check_node(
    graph: &Graph,
    node: NodeId,
    output: &[f32],
    inputs: &[(NodeId, &[f32])],
) -> Result<(), NanReport> {
    let Some(hit) = first_bad(output) else {
        return Ok(());
    };
    // Culprit vs propagator: was any operand already non-finite? If so this
    // op just carried a NaN forward; if not, it manufactured one here.
    let source_input = inputs
        .iter()
        .find(|(_, buf)| first_bad(buf).is_some())
        .map(|(id, _)| *id);
    let op = &graph.node(node).op;
    Err(NanReport {
        node,
        label: node_label(graph, node),
        op: op_short(op),
        kind: hit.kind,
        index: hit.index,
        source_input,
        fix: if source_input.is_none() {
            fix_hint(op)
        } else {
            None
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::Activation;
    use crate::{DType, Shape};

    fn rsqrt_graph() -> (Graph, NodeId, NodeId) {
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[2], DType::F32));
        let r = g.activation(Activation::Rsqrt, x, Shape::new(&[2], DType::F32));
        g.set_outputs(vec![r]);
        (g, x, r)
    }

    #[test]
    fn clean_output_passes() {
        let (g, x, r) = rsqrt_graph();
        let out = [1.0f32, 0.5];
        let inp = [1.0f32, 4.0];
        assert!(check_node(&g, r, &out, &[(x, &inp)]).is_ok());
    }

    #[test]
    fn culprit_when_inputs_clean() {
        let (g, x, r) = rsqrt_graph();
        // Finite (but negative) input → NaN output ⇒ this op is the culprit.
        let out = [f32::NAN, 0.5];
        let inp = [-1.0f32, 4.0];
        let err = check_node(&g, r, &out, &[(x, &inp)]).unwrap_err();
        assert_eq!(err.kind, BadValue::Nan);
        assert_eq!(err.index, 0);
        assert!(err.source_input.is_none(), "should be flagged as culprit");
        assert!(err.fix.is_some(), "culprit should carry a fix hint");
        assert!(err.op.contains("Rsqrt"));
    }

    #[test]
    fn propagator_when_input_already_bad() {
        let (g, x, r) = rsqrt_graph();
        // Input already NaN ⇒ this op merely propagated; no local fix.
        let out = [f32::NAN, 0.5];
        let inp = [f32::NAN, 4.0];
        let err = check_node(&g, r, &out, &[(x, &inp)]).unwrap_err();
        assert_eq!(err.source_input, Some(x));
        assert!(err.fix.is_none(), "propagator's fix lives upstream");
    }

    #[test]
    fn scanner_modes_and_policy() {
        let (g, x, r) = rsqrt_graph();
        // Off: never reports, even on a bad output.
        let off = DebugScanner::with_mode(DebugMode::Off, "test");
        assert!(!off.enabled());
        assert!(off.check(&g, r, &[f32::NAN], &[(x, &[1.0])]).is_none());
        // Warn: reports (and prints) but does not panic.
        let warn = DebugScanner::with_mode(DebugMode::Warn, "test");
        assert!(warn.enabled());
        let rep = warn.check(&g, r, &[f32::NAN], &[(x, &[1.0])]);
        assert!(rep.is_some());
        // Clean output → no report in any mode.
        assert!(warn.check(&g, r, &[0.5], &[(x, &[4.0])]).is_none());
    }

    #[test]
    #[should_panic(expected = "aborting")]
    fn scanner_abort_panics_on_bad() {
        let (g, x, r) = rsqrt_graph();
        let abort = DebugScanner::with_mode(DebugMode::Abort, "test");
        abort.check(&g, r, &[f32::NAN], &[(x, &[1.0])]);
    }

    #[test]
    fn detects_pos_inf() {
        assert_eq!(
            first_bad(&[f32::INFINITY, 0.0]).unwrap().kind,
            BadValue::PosInf
        );
        assert_eq!(
            first_bad(&[0.0, f32::NEG_INFINITY]).unwrap().kind,
            BadValue::NegInf
        );
        assert!(first_bad(&[1.0, 2.0, -3.0]).is_none());
    }
}
