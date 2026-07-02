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

//! Quick check that Metal device detection works on this machine.
//! cargo run --example detect -p rlx-metal --release

#[cfg(target_os = "macos")]
fn main() {
    if let Some(dev) = rlx_metal::device::metal_device() {
        println!("Metal device detected:");
        println!("  Name: {}", dev.name);
        println!("  Registry ID: {}", dev.registry_id);
        println!("  Unified memory: {}", dev.has_unified_memory);
        println!(
            "  Max working set: {} MB",
            dev.max_working_set / (1024 * 1024)
        );

        // Compile MSL kernels
        let _ = rlx_metal::kernels::kernels();
        println!("  MSL kernels compiled OK");

        // Cost model uses Tier 1 family-tuned constants by default;
        // values are upgraded to measured ones if a calibration cache exists.
        let hw = rlx_metal::cost::hw_model();
        let calibrated = std::path::PathBuf::from(format!(
            "{}/.cache/rlx/metal-calib-{:x}.json",
            std::env::var("HOME").unwrap_or_default(),
            dev.registry_id
        ))
        .exists();
        println!();
        println!(
            "Cost model throughput estimates (source: {}):",
            if calibrated {
                "measured"
            } else {
                "M-family defaults"
            }
        );
        println!(
            "  sgemm_simd_4x4 : {:>6.0} GFLOP/s",
            hw.sgemm_simd_4x4_flops / 1e9
        );
        println!(
            "  sgemm_simd     : {:>6.0} GFLOP/s",
            hw.sgemm_simd_flops / 1e9
        );
        println!(
            "  sgemm_padded   : {:>6.0} GFLOP/s",
            hw.sgemm_padded_flops / 1e9
        );
        println!(
            "  sgemm_tiled    : {:>6.0} GFLOP/s",
            hw.sgemm_tiled_flops / 1e9
        );
        if !calibrated {
            println!();
            println!("Run `cargo run --release --example metal_calibrate -p rlx-metal`");
            println!("to replace defaults with measured values for this exact GPU.");
        }
    } else {
        println!("No Metal device available");
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("Metal only available on macOS");
}
