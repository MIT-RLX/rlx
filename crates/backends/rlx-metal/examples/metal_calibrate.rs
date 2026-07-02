// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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

#[cfg(target_os = "macos")]
fn main() {
    use rlx_metal::calibrate::Calibration;
    use std::time::Instant;

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
