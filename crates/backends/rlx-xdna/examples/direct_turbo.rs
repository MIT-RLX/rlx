// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// TURBO power mode via the direct amdxdna ioctl (`SET_STATE`/`SET_POWER_MODE`).
// This clocks the NPU array to its maximum DPM and is INDEPENDENT of the (blocked)
// direct exec path — it raises the device clock for the working XRT compute path.
// Device-global; hold the fd for the session so the mode isn't reset.
//
//   cargo run -p rlx-xdna --features direct --example direct_turbo

fn main() {
    #[cfg(all(feature = "direct", target_os = "linux"))]
    {
        use rlx_xdna::direct::Npu;
        let npu = Npu::open("").expect("open /dev/accel/accel0");
        match npu.set_turbo() {
            Ok(()) => println!(
                "[turbo] SET_STATE(POWER_MODE_TURBO) OK — NPU clocked to max DPM (hold this fd to keep it)"
            ),
            Err(e) => println!("[turbo] SET_STATE(POWER_MODE_TURBO) FAILED: {e}"),
        }
        // Also exercise DEFAULT so we know the ioctl round-trips both ways.
        match npu.set_power_mode(0) {
            Ok(()) => {
                println!("[turbo] SET_STATE(POWER_MODE_DEFAULT) OK — reset to calculated DPM")
            }
            Err(e) => println!("[turbo] SET_STATE(POWER_MODE_DEFAULT) FAILED: {e}"),
        }
    }
    #[cfg(not(all(feature = "direct", target_os = "linux")))]
    println!("direct_turbo requires --features direct on Linux");
}
