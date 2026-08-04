// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! MEASURED execution time — the empirical complement to [`crate::shapes`]'s
//! analytical roofline. `shapes.rs` *predicts* "memory-bound" from arithmetic
//! intensity vs a fixed ridge point; this module *measures* achieved bandwidth so
//! the claim carries a number, and attributes real wall-clock to op regions.
//!
//! Everything here is pure `std` (a timer, a DRAM-bandwidth probe, and roofline
//! arithmetic) — the caller supplies the closure that compiles+runs a real graph,
//! so this stays backend-agnostic and unit-testable without a runtime dep.

use std::hint::black_box;
use std::time::Instant;

/// Median wall-clock of `runs` timed calls to `f`, in **milliseconds**, after
/// discarding `warmup` untimed calls (page-faults, cache fill, JIT/first-touch).
/// Median (not mean) rejects the odd scheduler hiccup.
pub fn median_ms(warmup: usize, runs: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let runs = runs.max(1);
    let mut ts = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        f();
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[ts.len() / 2]
}

/// Sustained **copy bandwidth** (GB/s): stream a read+write over buffers far
/// larger than any cache, so it hits DRAM. Not a peak-STREAM number (a copy reads
/// one stream and writes another), but a stable *same-machine reference* to judge
/// a forward's achieved bandwidth against — if a kernel reaches a large fraction
/// of this, it is genuinely bandwidth-limited, not compute- or launch-limited.
///
/// Reported **best-of-N** (max GB/s = min time): a bandwidth CEILING is estimated
/// from the least-contended pass, so a busy machine understates it less. A single
/// averaged run drops with background load, poisoning every downstream "% of peak".
pub fn measure_bandwidth_gbps() -> f64 {
    let n = (96usize << 20) / 4; // 96 MB of f32 per buffer ≫ LLC
    let mut a = vec![1.0f32; n];
    let mut b = vec![0.0f32; n];
    b.copy_from_slice(&a); // fault both buffers in
    black_box(b.as_ptr());
    best_copy_gbps(n as f64 * 4.0 * 2.0, |it| {
        a[it] = it as f32; // perturb → the copy is never provably redundant
        let src = black_box(&a); // opaque src ⇒ LLVM can't elide the copy
        b.copy_from_slice(src);
        black_box(b.as_ptr());
    })
}

/// Empirical roofline point for one measured region.
#[derive(Clone, Debug)]
pub struct Roofline {
    pub ms: f64,
    /// Achieved compute throughput (GFLOP/s) = analytical FLOPs / measured time.
    pub achieved_gflops: f64,
    /// Achieved memory throughput (GB/s) = analytical bytes / measured time.
    pub achieved_gbps: f64,
    /// Arithmetic intensity (FLOP/byte) — the analytical roofline x-axis.
    pub intensity: f64,
    /// Achieved GB/s as a fraction of the machine's measured copy bandwidth.
    pub bw_frac: f64,
    /// Empirical verdict from `bw_frac` (measured, not predicted).
    pub bound: &'static str,
}

/// Turn a measured `ms` plus the region's analytical `flops`/`bytes` and the
/// machine `peak_gbps` (from [`measure_bandwidth_gbps`]) into an empirical
/// roofline point. A region whose achieved GB/s is ≥ half the machine copy
/// bandwidth is bandwidth-limited *in measurement*, which either confirms or
/// refutes `shapes.rs`'s analytical classification.
pub fn empirical_roofline(ms: f64, flops: u64, bytes: u64, peak_gbps: f64) -> Roofline {
    let s = (ms / 1e3).max(1e-12);
    let achieved_gflops = flops as f64 / 1e9 / s;
    let achieved_gbps = bytes as f64 / 1e9 / s;
    let bw_frac = if peak_gbps > 0.0 {
        achieved_gbps / peak_gbps
    } else {
        0.0
    };
    let intensity = if bytes > 0 {
        flops as f64 / bytes as f64
    } else {
        0.0
    };
    let bound = if bw_frac >= 0.5 {
        "bandwidth-bound (measured)"
    } else if intensity >= crate::shapes::DEFAULT_RIDGE {
        "compute-bound (measured)"
    } else {
        "latency/overhead-bound (measured)"
    };
    Roofline {
        ms,
        achieved_gflops,
        achieved_gbps,
        intensity,
        bw_frac,
        bound,
    }
}

/// One-directional host bandwidth (GB/s): `memcpy` a big buffer into a fresh
/// destination — the closest pure-host proxy for a host→device **staging** copy
/// (what a discrete-GPU H2D upload pays on the host side, and roughly the whole
/// cost on Apple unified memory). Best-of-N ceiling, like [`measure_bandwidth_gbps`].
pub fn measure_memcpy_gbps() -> f64 {
    let n = 96usize << 20; // 96 MB of bytes ≫ LLC
    let mut src = vec![7u8; n];
    let mut dst = vec![0u8; n];
    dst.copy_from_slice(&src); // fault both buffers in
    black_box(dst.as_ptr());
    best_copy_gbps(n as f64 * 2.0, |it| {
        src[it] = it as u8; // perturb → non-redundant
        let s = black_box(&src); // opaque src ⇒ copy can't be elided
        dst.copy_from_slice(s);
        black_box(dst.as_ptr());
    })
}

/// Best-of-8 bandwidth (GB/s) for a `copy` closure moving `bytes` per call.
/// Rejects implausibly-fast iterations (> 2 TB/s ⇒ the compiler elided the work
/// or the timer glitched) so a single bad sample can't blow up the max.
fn best_copy_gbps(bytes: f64, mut copy: impl FnMut(usize)) -> f64 {
    let mut best = 0.0f64;
    for it in 0..8 {
        let t = Instant::now();
        copy(it);
        let secs = t.elapsed().as_secs_f64();
        let gbps = bytes / 1e9 / secs.max(1e-9);
        if gbps < 2000.0 {
            best = best.max(gbps);
        }
    }
    best.max(1.0)
}

/// Offload-profitability roofline: is running a region on an accelerator worth the
/// host↔device IO? Weights upload **once** and stay resident; each forward then
/// pays only per-step boundary IO (inputs up + outputs back) plus device compute.
#[derive(Clone, Debug)]
pub struct OffloadRoofline {
    /// Per-step accelerator time incl. per-step transfer (ms).
    pub device_step_ms: f64,
    /// Per-step host time (ms).
    pub host_step_ms: f64,
    /// Per-step boundary transfer time (ms) — inputs + outputs.
    pub per_step_transfer_ms: f64,
    /// One-time weight upload time (ms).
    pub weight_upload_ms: f64,
    /// Steady-state speedup (host / device incl. per-step IO), ignoring warm-up.
    pub steady_speedup: f64,
    /// Forward passes needed to amortize the one-time weight upload (`f64::INFINITY`
    /// if the device is not faster per step, so upload never pays back).
    pub break_even_steps: f64,
}

/// Model offload economics from measured pieces: `host_step_ms` (CPU forward),
/// `device_compute_ms` (accelerator compute, measured or BW-modeled), the one-time
/// `weight_bytes` upload, the `per_step_bytes` crossing each forward (input +
/// logits readback), and the measured `transfer_gbps`.
pub fn offload_roofline(
    host_step_ms: f64,
    device_compute_ms: f64,
    weight_bytes: u64,
    per_step_bytes: u64,
    transfer_gbps: f64,
) -> OffloadRoofline {
    let bps = (transfer_gbps * 1e9).max(1.0);
    let per_step_transfer_ms = per_step_bytes as f64 / bps * 1e3;
    let weight_upload_ms = weight_bytes as f64 / bps * 1e3;
    let device_step_ms = device_compute_ms + per_step_transfer_ms;
    let steady_speedup = if device_step_ms > 0.0 {
        host_step_ms / device_step_ms
    } else {
        0.0
    };
    let per_step_gain = host_step_ms - device_step_ms;
    let break_even_steps = if per_step_gain > 0.0 {
        weight_upload_ms / per_step_gain
    } else {
        f64::INFINITY
    };
    OffloadRoofline {
        device_step_ms,
        host_step_ms,
        per_step_transfer_ms,
        weight_upload_ms,
        steady_speedup,
        break_even_steps,
    }
}

/// A measured region's share of a whole: `(this_region_ms, total_ms) → percent`.
/// Clamped to `[0, 100]` — a differential attribution (`T_full − T_without`) can
/// go slightly negative on timing noise; report 0 rather than a spurious sign.
pub fn region_pct(region_ms: f64, total_ms: f64) -> f64 {
    if total_ms <= 0.0 {
        0.0
    } else {
        (region_ms / total_ms * 100.0).clamp(0.0, 100.0)
    }
}

/// One measured op-region: a named slice of the graph (e.g. "attention",
/// "mlp"), its differential wall-clock, and its analytical weight-byte share —
/// so the caller can print "time follows bytes" (the memory-bound signature).
#[derive(Clone, Debug)]
pub struct RegionTime {
    pub name: String,
    /// Differential wall-clock attributed to this region (ms).
    pub ms: f64,
    /// The region's share of measured time (%).
    pub time_pct: f64,
    /// The region's share of analytical weight bytes (%) — compare to `time_pct`.
    pub byte_pct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_ms_is_positive_and_ignores_warmup() {
        // Time a fixed-work closure; median must be finite and > 0.
        let ms = median_ms(2, 5, || {
            let mut acc = 0u64;
            for i in 0..50_000u64 {
                acc = acc.wrapping_add(i);
            }
            black_box(acc);
        });
        assert!(ms >= 0.0 && ms.is_finite());
    }

    #[test]
    fn bandwidth_probe_is_plausible() {
        let gbps = measure_bandwidth_gbps();
        // Any real machine clears 1 GB/s and stays under 100 TB/s — a sanity
        // band, not a benchmark assertion.
        assert!(
            gbps > 1.0 && gbps < 100_000.0,
            "implausible bandwidth {gbps}"
        );
    }

    #[test]
    fn empirical_roofline_arithmetic() {
        // 2 GFLOP + 8 GB in 100 ms → 20 GFLOP/s, 80 GB/s.
        let r = empirical_roofline(100.0, 2_000_000_000, 8_000_000_000, 100.0);
        assert!((r.achieved_gflops - 20.0).abs() < 1e-6);
        assert!((r.achieved_gbps - 80.0).abs() < 1e-6);
        assert!((r.bw_frac - 0.8).abs() < 1e-6);
        assert!((r.intensity - 0.25).abs() < 1e-9);
        assert_eq!(r.bound, "bandwidth-bound (measured)");
    }

    #[test]
    fn empirical_roofline_compute_vs_latency() {
        // High intensity, tiny achieved BW → compute-bound.
        let hi = empirical_roofline(10.0, 1_000_000_000_000, 1_000_000, 100.0);
        assert_eq!(hi.bound, "compute-bound (measured)");
        // Low intensity AND tiny achieved BW → latency/overhead-bound.
        let lo = empirical_roofline(1000.0, 10, 1_000_000, 100.0);
        assert_eq!(lo.bound, "latency/overhead-bound (measured)");
    }

    #[test]
    fn region_pct_clamps() {
        assert_eq!(region_pct(5.0, 10.0), 50.0);
        assert_eq!(region_pct(-1.0, 10.0), 0.0); // noise → clamp, not negative
        assert_eq!(region_pct(20.0, 10.0), 100.0);
        assert_eq!(region_pct(1.0, 0.0), 0.0);
    }

    #[test]
    fn memcpy_probe_is_plausible() {
        let gbps = measure_memcpy_gbps();
        assert!(
            gbps > 1.0 && gbps < 100_000.0,
            "implausible memcpy BW {gbps}"
        );
    }

    #[test]
    fn offload_pays_when_device_is_faster() {
        // Host 100 ms/step; device compute 10 ms; 1 GB weights + 1 MB/step at 10 GB/s.
        // upload = 1e9/1e10 = 100 ms; per-step transfer = 1e6/1e10 = 0.1 ms.
        let r = offload_roofline(100.0, 10.0, 1_000_000_000, 1_000_000, 10.0);
        assert!((r.weight_upload_ms - 100.0).abs() < 1e-6);
        assert!((r.per_step_transfer_ms - 0.1).abs() < 1e-6);
        assert!((r.device_step_ms - 10.1).abs() < 1e-6);
        assert!(r.steady_speedup > 9.0); // ~9.9×
        // gain/step ≈ 89.9 ms → break-even ≈ 100/89.9 ≈ 1.11 steps
        assert!(r.break_even_steps > 1.0 && r.break_even_steps < 2.0);
    }

    #[test]
    fn offload_never_pays_when_device_slower() {
        // Device compute already slower than host → break-even infinite.
        let r = offload_roofline(10.0, 100.0, 1_000_000_000, 1_000_000, 10.0);
        assert!(r.steady_speedup < 1.0);
        assert!(r.break_even_steps.is_infinite());
    }
}
