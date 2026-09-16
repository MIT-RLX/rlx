// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **How long the GPU was actually busy for the last run.**
//!
//! ```no_run
//! # let mut exe: rlx_metal::backend::MetalExecutable = todo!();
//! exe.run(&[("x", &[0.0f32][..])]);
//! if let Some(ms) = rlx_metal::gpu_span::last_ms() {
//!     println!("device busy {ms:.4} ms");
//! }
//! ```
//!
//! # Why a wall clock is not good enough
//!
//! Every Metal benchmark in this tree times `Instant::now()` around
//! `run()`, which measures **host encode + objc bridging + queue wait + GPU
//! execution**. On a busy machine the first three dominate, and the number stops
//! being about the kernel at all.
//!
//! That is not hypothetical. `reference_perf` on this box at load 50 reported
//! 24–47% spread and withheld its own aggregate; `cost.rs` puts objc bridging
//! alone at 5–20 µs per dispatch, which is a large fraction of a 0.25 ms decode
//! pass. A kernel change and a host-scheduling change look identical through a
//! wall clock.
//!
//! `MTLCommandBuffer` reports `GPUStartTime`/`GPUEndTime` — when the device
//! actually began and finished — so this isolates the part a kernel change can
//! move. It is CAKE's *"CUPTI timing"* in the form Apple provides.
//!
//! # What it does not fix
//!
//! A contended **GPU**. If something else is submitting work, the device span
//! for a given command buffer still stretches. Isolating host cost is not the
//! same as isolating device cost, and a run on a busy GPU is still not a
//! measurement — check `ioreg -c IOAccelerator | grep 'Device Utilization'`.
//!
//! # Scope
//!
//! Records the span of the **last completed `run()`**: earliest `GPUStartTime`
//! to latest `GPUEndTime` across that run's command buffers, so a multi-buffer
//! schedule is covered end to end. Gaps *between* buffers are included, which is
//! deliberate — a schedule that leaves the GPU idle mid-run is slower, and
//! summing per-buffer spans would hide exactly that.

use std::sync::atomic::{AtomicU64, Ordering};

/// `f64` bits, or `NOT_RECORDED` when the last run did not report one.
static LAST: AtomicU64 = AtomicU64::new(NOT_RECORDED);

/// Sentinel: no span recorded. A real span is never this bit pattern.
const NOT_RECORDED: u64 = u64::MAX;

/// Record a device span in milliseconds. Called by the backend after a run.
pub fn record(ms: f64) {
    // Metal returns 0 for both timestamps when the buffer has not completed, or
    // on a driver that does not report them. Storing that would silently look
    // like a 0 ms kernel — the fastest possible result — so it is refused.
    if ms.is_finite() && ms > 0.0 {
        LAST.store(ms.to_bits(), Ordering::Relaxed);
    } else {
        LAST.store(NOT_RECORDED, Ordering::Relaxed);
    }
}

/// Device span of the last completed run, in milliseconds.
///
/// `None` when the backend did not record one — a coverage limitation the
/// caller should report rather than paper over with a wall clock.
pub fn last_ms() -> Option<f64> {
    match LAST.load(Ordering::Relaxed) {
        NOT_RECORDED => None,
        bits => Some(f64::from_bits(bits)),
    }
}

/// Forget the last span. Use before a run whose span must not be confused with
/// an earlier one.
pub fn reset() {
    LAST.store(NOT_RECORDED, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recorded_span_round_trips() {
        reset();
        assert_eq!(last_ms(), None);
        record(1.25);
        assert_eq!(last_ms(), Some(1.25));
    }

    /// Metal reports 0 for both timestamps on an incomplete buffer. Storing
    /// that would read as an infinitely fast kernel — the most flattering
    /// possible wrong answer, and the kind a benchmark would happily print.
    #[test]
    fn a_zero_or_nonfinite_span_is_refused_rather_than_recorded() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            record(1.0);
            record(bad);
            assert_eq!(last_ms(), None, "{bad} should not be recorded as a span");
        }
    }
}
