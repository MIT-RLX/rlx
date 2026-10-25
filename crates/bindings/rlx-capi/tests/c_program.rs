// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compiles `examples/train_linear.c` against `include/rlx.h` and the shared
//! library cargo just built, runs it, and checks what it printed — the C ABI
//! as a C program sees it, generated builders and runtime both.

use std::path::Path;
use std::process::Command;

mod support;
use support::{plugin_lib, target_dir};

#[test]
fn a_c_program_builds_runs_differentiates_and_trains_a_graph() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("skip: no C compiler ({cc})");
        return;
    }
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lib_dir = target_dir();
    let exe = lib_dir.join("rlx_capi_train_linear");
    let status = Command::new(&cc)
        .arg(here.join("examples/train_linear.c"))
        .arg("-I")
        .arg(here.join("include"))
        .arg("-L")
        .arg(&lib_dir)
        .arg("-lrlx_capi")
        .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
        .arg("-lm")
        .arg("-o")
        .arg(&exe)
        .status()
        .expect("running the C compiler");
    assert!(
        status.success(),
        "the C program did not compile against rlx.h"
    );

    let out = Command::new(&exe).output().expect("running the C program");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "C program failed:\n{stdout}\n{stderr}"
    );
    for want in [
        r#"shape_of w: {"dims":[2,2],"dtype":"f32"}"#,
        "forward loss: 2.5000",
        "grad w: 1.0 2.0 2.0 4.0",
        "bad node: RLX_ERR_RANGE named",
        "check: no errors",
        "trained: converged (summary has steps=200)",
        "steps: 200",
        "consumed: yes",
        concat!("version: ", env!("CARGO_PKG_VERSION")),
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in:\n{stdout}\n{stderr}"
        );
    }
}

/// examples/compose.c: splice, split, run the parts in order, merge.
#[test]
fn a_c_program_combines_and_splits_graphs() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("skip: no C compiler ({cc})");
        return;
    }
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lib_dir = target_dir();
    let exe = lib_dir.join("rlx_capi_compose");
    let status = Command::new(&cc)
        .arg(here.join("examples/compose.c"))
        .arg("-I")
        .arg(here.join("include"))
        .arg("-L")
        .arg(&lib_dir)
        .arg("-lrlx_capi")
        .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
        .arg("-o")
        .arg(&exe)
        .status()
        .expect("running the C compiler");
    assert!(status.success(), "compose.c did not compile against rlx.h");
    let out = Command::new(&exe).output().expect("running the C program");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "compose.c failed:\n{stdout}\n{stderr}"
    );
    for want in [
        "stacked: 2 8 18",
        "boundary: matches",
        "parts: 2, split: 2 8 18",
        "merged: 2 8 18",
        r#"sequential params: ["w"]"#,
        "bad bind: RLX_ERR_TYPE named unchanged",
    ] {
        assert!(
            stdout.contains(want),
            "missing `{want}` in:\n{stdout}\n{stderr}"
        );
    }
}

#[test]
fn a_c_program_saves_opens_and_runs_models() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("skip: no C compiler ({cc})");
        return;
    }
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let lib_dir = target_dir();
    let exe = lib_dir.join("rlx_capi_models");
    let status = Command::new(&cc)
        .arg(here.join("examples/models.c"))
        .arg("-I")
        .arg(here.join("include"))
        .arg("-L")
        .arg(&lib_dir)
        .arg("-lrlx_capi")
        .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
        .arg("-o")
        .arg(&exe)
        .status()
        .expect("running the C compiler");
    assert!(status.success(), "models.c did not compile against rlx.h");
    let out = Command::new(&exe)
        .env("RLX_MODEL_PLUGINS", plugin_lib())
        .output()
        .expect("running the C program");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "models.c failed:\n{stdout}\n{stderr}");
    for want in MODELS_PRINTS {
        assert!(
            stdout.contains(want),
            "missing `{want}` in:\n{stdout}\n{stderr}"
        );
    }
}

/// What examples/models.c prints with the reference plugin's models.
const MODELS_PRINTS: [&str; 9] = [
    "model: affine (graph) on cpu",
    "model run: 2.000 3.000",
    "plugin model: count (text-generation)",
    "model text: defg",
    "model ids: 3 4 5",
    "chat: defg",
    "voice: 5 samples at 16000 Hz",
    "heard: hello",
    "embed: 5 2 3 0",
];

/// The libraries a static Rust library leaves to the C link, which rustc
/// prints only when it builds the library: kept beside it for the next run.
fn native_static_libs(manifest: &Path, target: &Path) -> Vec<String> {
    let out = Command::new(env!("CARGO"))
        .args(["rustc", "-q", "--lib", "--manifest-path"])
        .arg(manifest)
        .arg("--target-dir")
        .arg(target)
        .args(["--", "--print", "native-static-libs"])
        .output()
        .expect("running cargo");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "building the linked library:\n{stderr}"
    );
    let saved = target.join("native-static-libs.txt");
    let libs = match stderr
        .lines()
        .find_map(|l| l.split_once("native-static-libs: "))
    {
        Some((_, libs)) => {
            std::fs::write(&saved, libs).expect("saving the link list");
            libs.to_string()
        }
        None => std::fs::read_to_string(&saved).expect("the saved link list"),
    };
    libs.split_whitespace().map(str::to_string).collect()
}

/// For a host that cannot load a plugin (iOS, wasm): the plugin is linked into
/// a static librlx_capi, and a C program sees its models without
/// RLX_MODEL_PLUGINS once it calls rlx_link_model_plugins. Built without LTO,
/// where the C linker drops a static library's start-up registrations.
#[test]
fn a_static_c_program_sees_its_linked_model_plugins() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    if Command::new(&cc).arg("--version").output().is_err() {
        eprintln!("skip: no C compiler ({cc})");
        return;
    }
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = target_dir()
        .parent()
        .expect("target root")
        .join("linked-plugins");
    let libs = native_static_libs(&here.join("tests/linked/Cargo.toml"), &target);
    let exe = target.join("rlx_capi_linked_models");
    let status = Command::new(&cc)
        .arg("-DRLX_LINKED_PLUGINS")
        .arg(here.join("examples/models.c"))
        .arg("-I")
        .arg(here.join("include"))
        .arg(target.join("debug/librlx_capi_linked.a"))
        .args(&libs)
        .arg("-o")
        .arg(&exe)
        .status()
        .expect("running the C compiler");
    assert!(
        status.success(),
        "models.c did not link against the static library"
    );
    let out = Command::new(&exe)
        .env_remove("RLX_MODEL_PLUGINS")
        .env("RLX_HOME", target.join("no-home"))
        .output()
        .expect("running the C program");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "models.c failed:\n{stdout}\n{stderr}");
    for want in MODELS_PRINTS {
        assert!(
            stdout.contains(want),
            "missing `{want}` in:\n{stdout}\n{stderr}"
        );
    }
}
