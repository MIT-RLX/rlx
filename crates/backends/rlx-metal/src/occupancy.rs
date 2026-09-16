// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Will more threadgroup memory pay for itself on this Apple GPU?**
//!
//! One question, one function, one answer:
//!
//! ```no_run
//! use rlx_metal::occupancy::{predict, AppleGpuFamily, Verdict};
//!
//! match predict(AppleGpuFamily::M4, 2048, 6144) {
//!     Verdict::Slower(x)      => println!("don't bother: ~{x:.2}x"),
//!     Verdict::Faster(x)      => println!("worth measuring: ~{x:.2}x"),
//!     Verdict::TooCloseToCall => println!("measure it"),
//!     Verdict::Unknown(why)   => println!("no model: {why}"),
//! }
//! ```
//!
//! # Why this exists
//!
//! CAKE §3.1: *"A calibrated cost model estimates candidate performance and
//! returns high-level bottleneck attribution... used to rank and filter
//! candidates; on-device measurement and profiling remain the final ground
//! truth."*
//!
//! CAKE's own model is NVIDIA-shaped, because NVIDIA GPUs hide memory latency
//! with software pipelining — deeper staging is usually a *win* there, and it
//! was: `+8.4%` at three stages on an RTX 3080 Ti.
//!
//! Apple GPUs hide latency with **occupancy**, and threadgroup memory is the
//! resource occupancy is made of. Spending it to build a pipeline buys the
//! latency hiding you already had and pays for it with parallelism. That is not
//! a theory; it is what two independent toolchains measured on an M4 Pro (see
//! [`EVIDENCE`](crate::occupancy::EVIDENCE)).
//!
//! So this model answers the Apple question — *does the extra threadgroup
//! memory pay?* — rather than transliterating the NVIDIA one.
//!
//! # What it is not
//!
//! A throughput predictor. It does **not** know FLOPs, bandwidth, tile shape or
//! instruction mix, and it will not tell you how fast a kernel is. It answers
//! exactly one thing: how a schedule's throughput moves when its threadgroup
//! footprint changes, everything else held equal. That is the question a
//! `stages`-deep rotation poses, and it is the one worth filtering on before
//! spending device time.
//!
//! An unrecognised chip returns [`Verdict::Unknown`](crate::occupancy::Verdict::Unknown)
//! rather than a number.
//! Inventing one would be `cost-model-uncalibrated-ranking`, a defect this tree
//! has already shipped once.
//!
//! # Where this model was measured WRONG
//!
//! It was scored against a tile-edge sweep it had no business ranking, and the
//! result is worth keeping because it maps the model's edge precisely.
//!
//! | candidate | predicted | measured (geomean) | measured (small m) | measured (large m) |
//! |---|---|---|---|---|
//! | `tile=8` (512 B) | 1.190x | 1.118x | **1.55 - 1.82x** | 0.96 - 0.99x |
//! | `tile=32` (8192 B) | 0.810x | 0.910x | 0.52 - 0.57x | **1.08 - 1.10x** |
//!
//! Directionally right on both, and *usefully* right where occupancy dominates
//! — at `m < 32` the small tile wins by up to 1.8x, which is the regime the
//! model describes. But at large `m` the big tile **wins** 1.08 - 1.10x where
//! the model predicted 0.81x, because a larger tile reuses each staged element
//! more and that gain outweighs the occupancy it costs once there is enough
//! work to fill the machine.
//!
//! That is not a bug; it is the documented scope holding. This model prices
//! threadgroup memory and nothing else — it explicitly does not know tile shape
//! or data reuse, and a tile-edge change moves both. Use it for a *rotation*,
//! where the footprint changes and the reuse does not; use a measurement for
//! anything that changes the tile. [`crate::apple_params::AppleKernelParams::for_shape`]
//! carries the measured answer for tile edge, and does not consult this model.

/// Apple GPU family — different bandwidth, tensor units and threadgroup-memory
/// characteristics per generation.
///
/// Lives here rather than in `cost` because this module is **not** gated on
/// `rlx_metal_host`: naming a chip is pure string matching with no Metal
/// dependency, and a prediction is most useful *before* you are on the device.
/// `cost::AppleGpuFamily` re-exports it, so existing call sites are unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleGpuFamily {
    Unknown,
    M1,    // M1 (8-core GPU baseline)
    M1Pro, // M1 Pro/Max (16-32 core)
    M2,    // M2 family
    M3,    // M3 family — added dynamic caching
    M4,    // M4 family — improved tensor units
}

impl AppleGpuFamily {
    /// Classify from a Metal device name (`MTLDevice.name`).
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("m4") {
            Self::M4
        } else if lower.contains("m3") {
            Self::M3
        } else if lower.contains("m2") {
            Self::M2
        } else if lower.contains("m1 pro") || lower.contains("m1 max") || lower.contains("m1 ultra")
        {
            Self::M1Pro
        } else if lower.contains("m1") {
            Self::M1
        } else {
            Self::Unknown
        }
    }
}

/// A measured (threadgroup bytes -> relative throughput) point.
///
/// Recorded in the source so the fit is auditable: anyone can check the
/// constant against the data instead of trusting it.
#[derive(Debug, Clone, Copy)]
pub struct Point {
    /// Where it was measured.
    pub chip: &'static str,
    /// How it was measured.
    pub via: &'static str,
    pub threadgroup_bytes: usize,
    /// Throughput relative to that row's own 1-stage baseline.
    pub relative: f64,
}

/// The measurements behind [`LOSS_PER_DOUBLING`].
///
/// Two toolchains, same chip, same transformation. They were produced by
/// `rlx-metal/examples/schedule_sgemm_ab.rs` and
/// `rlx-wgpu/examples/wgpu_schedule_matmul_ab.rs`, each gated on bit-exactness
/// against its shipping kernel and each carrying a `serial` control arm that
/// read 0.98-1.00x — so the deltas below are the treatment, not the harness.
pub const EVIDENCE: &[Point] = &[
    // Native MSL, 16x16 threadgroup, geomean over 12 shapes.
    Point {
        chip: "M4 Pro",
        via: "MSL",
        threadgroup_bytes: 2048,
        relative: 1.000,
    },
    Point {
        chip: "M4 Pro",
        via: "MSL",
        threadgroup_bytes: 4096,
        relative: 0.906,
    },
    Point {
        chip: "M4 Pro",
        via: "MSL",
        threadgroup_bytes: 6144,
        relative: 0.842,
    },
    // WGSL through naga -> Metal, 8x8 workgroup, geomean over 12 shapes.
    // Held out of the fit and used to validate it — see the tests.
    Point {
        chip: "M4 Pro",
        via: "WGSL",
        threadgroup_bytes: 4096,
        relative: 1.000,
    },
    Point {
        chip: "M4 Pro",
        via: "WGSL",
        threadgroup_bytes: 8192,
        relative: 0.952,
    },
    Point {
        chip: "M4 Pro",
        via: "WGSL",
        threadgroup_bytes: 12288,
        relative: 0.811,
    },
];

/// Fractional throughput lost per **doubling** of threadgroup memory.
///
/// Fitted on the MSL rows of [`EVIDENCE`] only; the WGSL rows are held out and
/// used as validation (`the_model_predicts_the_held_out_wgsl_run`). Fitting on
/// everything and then reporting agreement would be circular.
///
/// The two toolchains bracket this value rather than agreeing on it — MSL is
/// steeper at 2x, WGSL steeper at 3x — so treat it as a ranking signal with
/// roughly +/-5 points of slack, which is why [`predict`] has a dead band.
pub const LOSS_PER_DOUBLING: f64 = 0.095;

/// Predictions inside this band are not claims. Below it the two toolchains
/// disagree by as much as the effect, so the honest answer is "measure it".
pub const DEAD_BAND: f64 = 0.05;

/// What the model thinks of a change in threadgroup footprint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    /// Predicted faster by this ratio (> 1.0). Worth device time.
    Faster(f64),
    /// Predicted slower by this ratio (< 1.0). Filter it out first.
    Slower(f64),
    /// Inside the dead band — the model cannot separate them.
    TooCloseToCall,
    /// No calibration for this chip. Carries the reason.
    Unknown(&'static str),
}

impl Verdict {
    /// The predicted ratio, when there is one.
    pub fn ratio(self) -> Option<f64> {
        match self {
            Self::Faster(x) | Self::Slower(x) => Some(x),
            _ => None,
        }
    }

    /// Whether this candidate is worth spending GPU time on.
    ///
    /// `TooCloseToCall` and `Unknown` are both `true`: a model that cannot
    /// distinguish two candidates must not be the thing that discards one.
    pub fn worth_measuring(self) -> bool {
        !matches!(self, Self::Slower(_))
    }
}

/// Predicted throughput of `candidate_bytes` relative to `baseline_bytes`.
///
/// Everything except the threadgroup footprint is assumed equal — same tile,
/// same arithmetic, same access pattern. That is exactly true of a
/// `stages`-deep rotation, which is what this is for.
pub fn predict(chip: AppleGpuFamily, baseline_bytes: usize, candidate_bytes: usize) -> Verdict {
    // Only the M-series generations with measurements behind them. The others
    // are not "probably similar" — they are unmeasured, and saying so is the
    // whole point of `CostCalibration::Uncalibrated` existing in this tree.
    match chip {
        AppleGpuFamily::M4 => {}
        AppleGpuFamily::Unknown => {
            return Verdict::Unknown("unrecognised Apple GPU — no occupancy calibration");
        }
        _ => {
            return Verdict::Unknown(
                "occupancy is calibrated on M4 only; earlier families have different \
                 threadgroup-memory-per-core and were not measured",
            );
        }
    }
    if baseline_bytes == 0 || candidate_bytes == 0 {
        return Verdict::Unknown("a zero threadgroup footprint has no ratio");
    }
    let doublings = (candidate_bytes as f64 / baseline_bytes as f64).log2();
    // Linear in log2(footprint): the shape both toolchains showed.
    let ratio = (1.0 - LOSS_PER_DOUBLING * doublings).max(0.0);
    if (ratio - 1.0).abs() <= DEAD_BAND {
        Verdict::TooCloseToCall
    } else if ratio > 1.0 {
        Verdict::Faster(ratio)
    } else {
        Verdict::Slower(ratio)
    }
}

/// One-line summary for a report.
pub fn explain(chip: AppleGpuFamily, baseline_bytes: usize, candidate_bytes: usize) -> String {
    match predict(chip, baseline_bytes, candidate_bytes) {
        Verdict::Faster(x) => format!(
            "predicted {x:.3}x ({baseline_bytes} -> {candidate_bytes} B threadgroup) — worth measuring"
        ),
        Verdict::Slower(x) => format!(
            "predicted {x:.3}x ({baseline_bytes} -> {candidate_bytes} B threadgroup) — \
             occupancy cost exceeds the gain on this chip"
        ),
        Verdict::TooCloseToCall => {
            format!("within +/-{:.0}% — measure it", DEAD_BAND * 100.0)
        }
        Verdict::Unknown(why) => format!("no prediction: {why}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msl(bytes: usize) -> f64 {
        EVIDENCE
            .iter()
            .find(|p| p.via == "MSL" && p.threadgroup_bytes == bytes)
            .expect("evidence point")
            .relative
    }

    /// The fit must reproduce the data it was fitted on. Necessary, not
    /// sufficient — see the held-out test below.
    #[test]
    fn the_model_reproduces_the_msl_run_it_was_fitted_on() {
        for bytes in [4096usize, 6144] {
            let got = predict(AppleGpuFamily::M4, 2048, bytes)
                .ratio()
                .expect("M4 is calibrated");
            let want = msl(bytes);
            assert!(
                (got - want).abs() < 0.02,
                "{bytes} B: model {got:.3} vs measured {want:.3}"
            );
        }
    }

    /// **The real test.** The WGSL run is a different toolchain, a different
    /// workgroup shape and a different absolute footprint, and it was held out
    /// of the fit. If the model only worked on its own training data it would
    /// be a lookup table wearing a formula's clothes.
    #[test]
    fn the_model_predicts_the_held_out_wgsl_run() {
        for (bytes, measured) in [(8192usize, 0.952), (12288usize, 0.811)] {
            let got = predict(AppleGpuFamily::M4, 4096, bytes)
                .ratio()
                .expect("M4 is calibrated");
            assert!(
                (got - measured).abs() < 0.06,
                "held-out {bytes} B: model {got:.3} vs measured {measured:.3} — the fit does \
                 not generalise across toolchains"
            );
        }
    }

    /// The result that motivated the model: it must call the rotation a loss
    /// *before* anyone runs it. This is the filtering CAKE §3.1 asks for.
    #[test]
    fn the_pipelined_rotation_is_filtered_out_on_apple() {
        for stages in 2..=4 {
            let v = predict(AppleGpuFamily::M4, 2048, 2048 * stages);
            assert!(
                matches!(v, Verdict::Slower(_)),
                "{stages} stages should be predicted slower on Apple, got {v:?}"
            );
            assert!(!v.worth_measuring());
        }
    }

    /// Deeper is worse, monotonically — the shape both runs showed.
    #[test]
    fn deeper_rotations_are_predicted_worse() {
        let mut last = f64::INFINITY;
        for stages in 1..=6 {
            let r = predict(AppleGpuFamily::M4, 2048, 2048 * stages)
                .ratio()
                .unwrap_or(1.0);
            assert!(r < last, "{stages} stages: {r:.3} not below {last:.3}");
            last = r;
        }
    }

    /// An unrecognised or unmeasured chip gets no number.
    ///
    /// `cost-model-uncalibrated-ranking` is in this tree's evolution ledger
    /// because a model once invented throughput for hardware it had never seen.
    #[test]
    fn an_uncalibrated_chip_refuses_to_predict() {
        for chip in [
            AppleGpuFamily::Unknown,
            AppleGpuFamily::M1,
            AppleGpuFamily::M3,
        ] {
            let v = predict(chip, 2048, 6144);
            assert!(matches!(v, Verdict::Unknown(_)), "{chip:?} predicted {v:?}");
            assert!(v.ratio().is_none());
            // And it must NOT be treated as a reason to skip the candidate.
            assert!(
                v.worth_measuring(),
                "{chip:?} must not filter on no evidence"
            );
        }
    }

    /// No change in footprint is no change.
    #[test]
    fn an_unchanged_footprint_is_too_close_to_call() {
        assert_eq!(
            predict(AppleGpuFamily::M4, 4096, 4096),
            Verdict::TooCloseToCall
        );
    }

    /// Every recorded point must be a real measurement with a source, or the
    /// audit trail is decoration.
    #[test]
    fn every_evidence_point_names_its_chip_and_toolchain() {
        assert!(EVIDENCE.len() >= 6);
        for p in EVIDENCE {
            assert!(!p.chip.is_empty() && !p.via.is_empty());
            assert!(p.threadgroup_bytes > 0);
            assert!(p.relative > 0.0 && p.relative <= 1.0);
        }
    }
}
