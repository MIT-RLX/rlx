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
    // Ordered maps: prost encodes a HashMap field in its (per-process random)
    // iteration order, so the same model serialised to different bytes in
    // every process and the compiled-model cache, keyed by those bytes,
    // never hit.
    prost_build::Config::new()
        .btree_map(["."])
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
        // The shim's `@available` checks compile to
        // `___isPlatformVersionAtLeast`, which lives in clang's runtime
        // (`libclang_rt.<os>.a`). Rust links with `-nodefaultlibs`, so name it
        // here. It used to arrive only when rlx-mlx-sys was also linked; a
        // CoreML-without-MLX build failed with an undefined symbol.
        if target_os == "macos" {
            link_clang_rt("osx");
        }
    }
}

/// Add `libclang_rt.<variant>.a` from the active clang to the link line.
fn link_clang_rt(variant: &str) {
    let libname = format!("libclang_rt.{variant}.a");
    let out = std::process::Command::new("clang")
        .arg(format!("--print-file-name={libname}"))
        .output();
    let path = match out {
        Ok(o) if o.status.success() => {
            std::path::PathBuf::from(String::from_utf8_lossy(&o.stdout).trim())
        }
        _ => {
            println!("cargo:warning=rlx-coreml: could not run `clang --print-file-name={libname}`");
            return;
        }
    };
    match path.parent() {
        Some(dir) if path.is_file() => {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=static=clang_rt.{variant}");
        }
        _ => println!(
            "cargo:warning=rlx-coreml: {libname} not found ({})",
            path.display()
        ),
    }
}
