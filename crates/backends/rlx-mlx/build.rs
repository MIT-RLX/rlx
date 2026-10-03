// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

fn main() {
    println!("cargo::rustc-check-cfg=cfg(rlx_mlx_host)");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // MLX has a real backend on macOS / Linux / Windows and on the Apple
    // platforms that ship Metal — iOS, tvOS and visionOS (device + simulator).
    // **watchOS keeps the stub**: no public Metal API, so MLX has no backend to
    // build there.
    //
    // Three build scripts gate on this same fact and must move together —
    // rlx-mlx/build.rs, rlx-runtime/build.rs (the module + registry entry) and
    // rlx-mlx-sys/build.rs (whether libmlx is cross-compiled at all). Out of
    // sync gives "no backend registered for MLX" at runtime, or a link against
    // symbols nothing built.
    if matches!(
        os.as_str(),
        "macos" | "linux" | "windows" | "ios" | "tvos" | "visionos"
    ) {
        println!("cargo:rustc-cfg=rlx_mlx_host");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
