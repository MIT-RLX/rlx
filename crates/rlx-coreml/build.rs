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
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let is_apple = matches!(target_os.as_str(), "macos" | "ios");
    if is_apple {
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
