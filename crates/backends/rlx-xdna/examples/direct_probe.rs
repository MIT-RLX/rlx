// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Level-1a hardware probe for the DIRECT amdxdna ioctl path (no XRT).
// Exercises device open + query ioctls + BO mmap roundtrip + syncobj wait, and
// reports what the live kernel driver accepted — the foundation the EXEC_CMD /
// UMQ-doorbell submit path builds on.
//
//   cargo run -p rlx-xdna --features direct --example direct_probe
//
// Needs read/write on /dev/accel/accel0 (the `render` group).

fn main() {
    match rlx_xdna::direct::probe() {
        Ok(report) => {
            print!("{report}");
        }
        Err(e) => {
            eprintln!("direct probe FAILED: {e}");
            std::process::exit(1);
        }
    }
}
