// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Disk cache for compiled MSL → metallib (avoids ~1s MSL compile on cold start).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

use metal::{CompileOptions, DeviceRef, Library};

pub fn cache_enabled() -> bool {
    !matches!(
        rlx_ir::env::var("RLX_METAL_PIPELINE_CACHE"),
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no")
    )
}

fn cache_root() -> PathBuf {
    match rlx_ir::env::var("RLX_METAL_PIPELINE_CACHE") {
        Some(p) => PathBuf::from(p),
        None => std::env::temp_dir().join("rlx-metal-pipelines"),
    }
}

fn hash_msl(msl: &str) -> u64 {
    let mut h = DefaultHasher::new();
    msl.hash(&mut h);
    h.finish()
}

fn cache_paths(msl: &str) -> (PathBuf, PathBuf) {
    let root = cache_root();
    let tag = format!("{:016x}", hash_msl(msl));
    (
        root.join(format!("{tag}.metal")),
        root.join(format!("{tag}.metallib")),
    )
}

fn compile_msl_file_to_metallib(src: &Path, out: &Path) -> bool {
    Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            src.to_str().unwrap_or_default(),
            "-o",
            out.to_str().unwrap_or_default(),
        ])
        .status()
        .ok()
        .is_some_and(|s| s.success())
}

pub fn load_or_compile_library(device: &DeviceRef, msl: &str) -> Library {
    let opts = CompileOptions::new();
    if cache_enabled() {
        let (src_path, lib_path) = cache_paths(msl);
        if lib_path.is_file() {
            if let Ok(lib) = device.new_library_with_file(&lib_path) {
                return lib;
            }
        }
        if let Some(parent) = lib_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if std::fs::write(&src_path, msl).is_ok()
            && compile_msl_file_to_metallib(&src_path, &lib_path)
            && lib_path.is_file()
        {
            if let Ok(lib) = device.new_library_with_file(&lib_path) {
                return lib;
            }
        }
    }
    device
        .new_library_with_source(msl, &opts)
        .expect("MSL compilation failed")
}

/// Force metallib + pipeline init (call once at process load).
pub fn prewarm_library() {
    let _ = crate::kernels::prewarm();
}
