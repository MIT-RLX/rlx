// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! Emits `backends_manifest.json` describing which backend Cargo features
//! were enabled for this build (plan: deploy-time manifest).

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR");
    let out = Path::new(&out_dir);
    let manifest_path = out.join("backends_manifest.json");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    println!("cargo::rustc-check-cfg=cfg(rlx_mlx_host)");
    // Must mirror rlx-mlx's own build.rs: the MLX backend is real on macOS,
    // Linux, Windows and iOS. Keep in sync, or rlx-runtime gates the MLX
    // backend module / registry differently from where rlx-mlx actually builds.
    if matches!(target_os.as_str(), "macos" | "linux" | "windows" | "ios") {
        println!("cargo:rustc-cfg=rlx_mlx_host");
    }

    let backends: Vec<&str> = [
        ("cpu", "cpu"),
        ("metal", "metal"),
        ("mlx", "mlx"),
        ("gpu", "gpu"),
        ("cuda", "cuda"),
        ("rocm", "rocm"),
        ("tpu", "tpu"),
        ("vulkan", "vulkan"),
        ("oneapi", "oneapi"),
        ("opengl", "opengl"),
        ("directx", "directx"),
        ("webgpu", "webgpu"),
        ("ane", "ane"),
        ("blas-accelerate", "blas-accelerate"),
        ("blas-mkl", "blas-mkl"),
        ("blas-openblas", "blas-openblas"),
    ]
    .into_iter()
    .filter_map(|(feature, label)| {
        let key = format!("CARGO_FEATURE_{}", feature.replace('-', "_").to_uppercase());
        env::var(key).ok().map(|_| label)
    })
    .collect();

    let json = serde_json::json!({
        "crate": "rlx-runtime",
        "version": env!("CARGO_PKG_VERSION"),
        "backends": backends,
    });
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&json).expect("json"),
    )
    .expect("write backends_manifest.json");
    println!(
        "cargo:rustc-env=RLX_BACKENDS_MANIFEST_PATH={}",
        manifest_path.display()
    );
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
}
