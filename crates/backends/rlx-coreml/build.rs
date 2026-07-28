// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Build script for `rlx-coreml`.
//!
//! 1. Compiles the focused CoreML protobuf schema (`proto/coreml.proto`)
//!    into Rust types via prost-build. `protoc` is supplied by the
//!    `protoc-bin-vendored` crate — no system install required.
//! 2. On Apple platforms, compiles the Objective-C CoreML shim
//!    (`csrc/coreml_shim.m`) and links CoreML + Foundation. Everywhere
//!    else this is skipped; the Rust side stubs the FFI out under the
//!    same `cfg` so the crate still type-checks for cross-builds.

fn main() {
    // --- protobuf -> Rust -------------------------------------------------
    let protoc = protoc_bin_vendored::protoc_bin_path()
        .expect("protoc-bin-vendored: no binary for this target");
    // SAFETY: single-threaded build script; the path is absolute.
    unsafe {
        std::env::set_var("PROTOC", protoc);
    }
    prost_build::Config::new()
        .compile_protos(&["proto/coreml.proto"], &["proto"])
        .expect("prost-build: compile_protos failed");
    println!("cargo:rerun-if-changed=proto/coreml.proto");

    // --- Objective-C CoreML shim (Apple only) -----------------------------
    // CoreML.framework + Foundation ship on macOS, iOS, tvOS and visionOS, so
    // build + link the shim there. **watchOS is excluded**: it marks the
    // runtime model-compilation API (`compileModelAtURL:`) unavailable, and
    // the shim relies on it — watchOS falls back to the CPU/Accelerate backend.
    let target_vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_coreml_capable = target_vendor == "apple" && target_os != "watchos";
    if is_coreml_capable {
        cc::Build::new()
            .file("csrc/coreml_shim.m")
            .flag("-fobjc-arc")
            .flag("-fmodules")
            .compile("rlx_coreml_shim");
        println!("cargo:rerun-if-changed=csrc/coreml_shim.m");
        println!("cargo:rerun-if-changed=csrc/coreml_shim.h");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
