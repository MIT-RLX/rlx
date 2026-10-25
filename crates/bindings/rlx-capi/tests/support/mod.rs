// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared by the rlx-capi integration tests: the C library and the reference
//! model plugin, built for this test binary's profile.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

/// `target/<profile>`, from this test binary's location (`…/<profile>/deps/<test>`).
fn profile_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("test binary path");
    exe.parent()
        .and_then(|deps| deps.parent())
        .expect("target dir")
        .to_path_buf()
}

/// `target/<profile>`, once `cargo build -p rlx-capi -p rlx-model-plugin` for
/// the same profile has put `librlx_capi` (shared and static) and the plugin
/// there. `cargo test` alone leaves the shared library in `deps/` and never
/// copies it up, so the bindings would otherwise load whatever an earlier
/// `cargo build` left behind — or nothing, after a clean.
pub fn target_dir() -> PathBuf {
    static BUILT: OnceLock<PathBuf> = OnceLock::new();
    BUILT
        .get_or_init(|| {
            let dir = profile_dir();
            let profile = match dir.file_name().and_then(|n| n.to_str()) {
                Some("debug") | None => "dev".to_string(),
                Some(p) => p.to_string(),
            };
            let status = Command::new(env!("CARGO"))
                .args(["build", "-q", "-p", "rlx-capi", "-p", "rlx-model-plugin"])
                .args(["--profile", &profile, "--target-dir"])
                .arg(dir.parent().expect("target root"))
                .status()
                .expect("running cargo build");
            assert!(
                status.success(),
                "building rlx-capi and rlx-model-plugin ({profile}) failed"
            );
            dir
        })
        .clone()
}

/// The reference model plugin (`rlx-model-plugin`): every language's `models`
/// example opens its `demo:count` model.
#[allow(dead_code)] // not every test binary runs the models example
pub fn plugin_lib() -> PathBuf {
    target_dir().join(format!(
        "{}rlx_model_plugin{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ))
}
