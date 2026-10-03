// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single source of truth for "is the native Metal backend available on this
//! target?". Metal is an Apple-only API, so the real backend (device/blas/
//! kernels/thunk/…) only builds where it can run. Everywhere else the crate
//! compiles to the `is_available() == false` stub.
//!
//! Mirrors `rlx-mlx`'s `rlx_mlx_host` cfg so platform gating is consistent and
//! centralized — modules use `#[cfg(rlx_metal_host)]` instead of each repeating
//! a `target_os` check (which is how `ms_deform_attn` was once left ungated).

fn main() {
    // Register the custom cfgs so they're known names (no `unexpected_cfgs` noise).
    println!("cargo::rustc-check-cfg=cfg(rlx_metal_host)");
    println!("cargo::rustc-check-cfg=cfg(rlx_mps_graph_host)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    // "sim" on every Apple simulator target, empty on device targets.
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    // The native Metal backend targets every Apple platform that ships Metal +
    // MetalPerformanceShaders: macOS, iOS, tvOS and visionOS (device +
    // simulator). metal-rs, MPS and MPSGraph are all available there, so the
    // real backend builds. **watchOS is excluded** — it has no public Metal
    // API — and every non-Apple target keeps compiling to the
    // `is_available() == false` stub. This is the single place that decides
    // where the real backend is built.
    let metal_host = target_vendor == "apple" && target_os != "watchos";
    if metal_host {
        println!("cargo:rustc-cfg=rlx_metal_host");
    }

    // MPSGraph is a *separate* framework from MPS, and the tvOS **simulator**
    // SDK is the one place Apple ships one without the other:
    //
    //   appletvos         Metal  MetalPerformanceShaders  MetalPerformanceShadersGraph
    //   appletvsimulator  Metal  MetalPerformanceShaders  —
    //
    // `cargo check` cannot see this — it is a link-time failure
    // (`ld: framework 'MetalPerformanceShadersGraph' not found`), which is why
    // it only showed up once `just test-apple-sim` started *running* on the
    // tvOS simulator. Nothing in `mps_graph` binds a framework symbol at link
    // time (every entry point goes through the Objective-C runtime), so the
    // module still compiles where this is off; `mps_graph_supported()` then
    // answers false because the class is not registered, which is the truth on
    // that target.
    if metal_host && !(target_os == "tvos" && target_env == "sim") {
        println!("cargo:rustc-cfg=rlx_mps_graph_host");
    }

    println!("cargo:rerun-if-changed=build.rs");
}
