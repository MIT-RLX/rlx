// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every test that touches the Apple-only API must carry the crate's target
//! gate, or it fails to *compile* off Apple — and one test target that does not
//! compile aborts the whole `cargo test --workspace` build, `--no-fail-fast`
//! included. `coreml_transpose_reduce.rs` and `coreml_bn_chain.rs` shipped
//! without it, and the Linux rig ran zero tests across the workspace.
//!
//! Device-free and ungated on purpose: it runs on the Mac where those files
//! compile fine, so the omission is caught before it reaches a Linux host.

use std::path::PathBuf;

/// Items that exist only under `cfg(all(target_vendor = "apple", not(target_os = "watchos")))`.
const APPLE_ONLY: &[&str] = &["CoremlExecutable", "rlx_coreml::backend"];
const GATE: &str = r#"#![cfg(all(target_vendor = "apple", not(target_os = "watchos")))]"#;

#[test]
fn tests_using_the_apple_only_api_carry_the_target_gate() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut missing = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("tests/ dir").flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs")
            || path.file_name() == Some(std::ffi::OsStr::new(file!().rsplit('/').next().unwrap()))
        {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap_or_default();
        if APPLE_ONLY.iter().any(|item| src.contains(item)) && !src.contains(GATE) {
            missing.push(path.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    missing.sort();
    assert!(
        missing.is_empty(),
        "these tests use the Apple-only CoreML API without `{GATE}` and break the \
         workspace build on Linux: {missing:?}"
    );
}
