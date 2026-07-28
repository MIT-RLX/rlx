// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

fn main() {
    println!("cargo::rustc-check-cfg=cfg(rlx_mlx_host)");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // MLX has a real backend on macOS / Linux / Windows and on iOS (device +
    // simulator — MLX's CMake builds a Metal backend there). tvOS / watchOS /
    // visionOS are not MLX-supported, so they keep the non-host stub.
    if matches!(os.as_str(), "macos" | "linux" | "windows" | "ios") {
        println!("cargo:rustc-cfg=rlx_mlx_host");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
