// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional one-shot calibration tool.
//!
//! Usage:
//!   cargo run --release --example metal_calibrate -p rlx-metal
//!
//! Measures actual GPU throughput on this hardware, writes to
//! ~/.cache/rlx/metal-calib-<hwid>.json. The cost model picks it up
//! automatically on subsequent program runs.
//!
//! Without running this, RLX uses per-family compile-time defaults
//! (M1/M2/M3/M4 each have their own constants). Those are usually fine —
//! run this only if you want measured-on-this-machine accuracy.
//!
//! # It refuses to SAVE a contended measurement
//!
//! The cache it writes is permanent and silently trusted by every later
//! process, which makes a bad calibration far worse than no calibration: the
//! model stops falling back to a sane default and starts confidently using a
//! wrong number nobody re-checks.
//!
//! That is not hypothetical. Calibrating this machine twice minutes apart while
//! another workload held the GPU at ~60% gave 706 and 1510 GFLOP/s for the same
//! attention shape — a 2.1x spread, either of which would have been persisted.
//! `tune_dispatch` already refuses to write its cache under contention for
//! exactly this reason (`RLX_ALLOW_THROTTLE` bypasses both).
//!
//! Measuring under load is still allowed — the numbers print. Only the *write*
//! is gated, because looking is harmless and persisting is not.

#[cfg(target_os = "macos")]
fn main() {
    use rlx_metal::calibrate::Calibration;
    use std::time::Instant;

    /// Host + GPU contention, or `Ok` with a description of a quiet machine.
    ///
    /// Both axes matter and they fail independently: a busy CPU inflates encode
    /// time, and a busy GPU inflates the device span. Today the CPU sat at load
    /// ~45 while another process held the GPU at 60%, and only the second one
    /// shows up in a device-span measurement.
    fn contention_check() -> Result<String, String> {
        if rlx_ir::env::flag("RLX_ALLOW_THROTTLE") {
            return Ok("RLX_ALLOW_THROTTLE=1 — gate bypassed".into());
        }
        let mut problems = Vec::new();

        // 1-minute load average via `sysctl`, the same signal
        // `scripts/check-throttle.sh` uses.
        let load = std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| {
                s.split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse::<f64>().ok())
            });
        if let Some(l) = load
            && l > 4.0
        {
            problems.push(format!("1-min load average {l:.1} (limit 4.0)"));
        }

        // GPU utilization from IOKit. No sudo needed, unlike powermetrics.
        let gpu = std::process::Command::new("ioreg")
            .args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| {
                s.split("\"Device Utilization %\"=")
                    .nth(1)?
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse::<u32>()
                    .ok()
            });
        if let Some(u) = gpu
            && u > 15
        {
            problems.push(format!("GPU {u}% busy (limit 15%)"));
        }

        if problems.is_empty() {
            Ok(format!(
                "quiet: load {}, GPU {}% busy",
                load.map(|l| format!("{l:.1}")).unwrap_or("?".into()),
                gpu.map(|u| u.to_string()).unwrap_or("?".into())
            ))
        } else {
            Err(problems.join("; "))
        }
    }

    let host_state = contention_check();
    match &host_state {
        Ok(s) => println!("host: {s}"),
        Err(why) => println!("host: CONTENDED — {why}"),
    }

    println!("Measuring Metal sgemm throughput on this hardware...");
    let t0 = Instant::now();
    let cal = Calibration::measure();
    let elapsed = t0.elapsed();

    println!();
    println!("GPU: {}", cal.gpu_name);
    println!("Measured throughput:");
    println!(
        "  sgemm_simd_4x4 :  {:>6.0} GFLOP/s",
        cal.sgemm_simd_4x4_flops / 1e9
    );
    println!(
        "  sgemm_simd     :  {:>6.0} GFLOP/s",
        cal.sgemm_simd_flops / 1e9
    );
    println!(
        "  sgemm_padded   :  {:>6.0} GFLOP/s",
        cal.sgemm_padded_flops / 1e9
    );
    println!(
        "  sgemm_tiled    :  {:>6.0} GFLOP/s",
        cal.sgemm_tiled_flops / 1e9
    );
    println!(
        "  roundtrip      :  {:>6.1} µs",
        cal.roundtrip_overhead_ns / 1000.0
    );
    println!();
    println!("Calibration took {:?}", elapsed);

    if let Err(why) = &host_state {
        eprintln!();
        eprintln!("NOT SAVING: the machine is contended ({why}).");
        eprintln!(
            "The numbers above are indicative only. A calibration cache is permanent and\n\
             silently trusted by every later process, so persisting a contended one is\n\
             worse than having none — the cost model would stop falling back to a sane\n\
             default and start confidently using a wrong number."
        );
        eprintln!("Re-run on an idle machine, or set RLX_ALLOW_THROTTLE=1 to save anyway.");
        return;
    }

    match cal.save() {
        Ok(()) => {
            let home = std::env::var("HOME").unwrap_or_default();
            println!(
                "Saved to {}/.cache/rlx/metal-calib-{:x}.json",
                home, cal.registry_id
            );
            println!("The cost model will use these values on subsequent runs.");
        }
        Err(e) => eprintln!("Warning: failed to save calibration: {e}"),
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Metal calibration is macOS-only");
}
