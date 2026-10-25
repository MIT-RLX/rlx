// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Swift binding (`crates/bindings/rlx-swift`) against the library cargo
//! just built: its XCTest suite, its examples, and a check that every function
//! in `rlx.h` has a Swift wrapper. The first two are skipped where there is no
//! Swift toolchain; the coverage check needs none.

use std::path::Path;
use std::process::Command;

mod support;
use support::{plugin_lib, target_dir};

/// Each example (`rlx-swift/Examples/<Name>`) and lines its output must hold.
const EXAMPLES: &[(&str, &[&str])] = &[
    (
        "rlx-example-inference",
        &[
            "compiled on: cpu",
            "probs sum: 1.0000",
            "typed run: f32 matches",
        ],
    ),
    (
        "rlx-example-training",
        &[
            "trainer w: 1.999 -0.999 0.498",
            "checkpoint: 3 tensors",
            "frozen: [\"w\"] unchanged: true",
            "manual loss: 0.000000 w: 2.000 -1.000 0.500",
        ],
    ),
    (
        "rlx-example-loading",
        &[
            "gguf architecture: demo",
            "gguf f32 exact: true",
            "safetensors: [\"proj.weight\"] exact: true",
            "argmax: 1",
        ],
    ),
    (
        "rlx-example-distributed",
        &[
            "all_reduce sum: [6.0]",
            "all_gather: [1.0, 2.0, 3.0]",
            "workers: rank 1",
            "rank 0 w: 1.500 -2.000 0.500",
            "rank 1 w agrees: true",
            "rank 2 w agrees: true",
        ],
    ),
    (
        "rlx-example-backends",
        &["session cpu: max|Δ| vs first = 0.00e+00", "router ran on:"],
    ),
    (
        "rlx-example-models",
        &[
            "model: affine (graph) on cpu",
            "model run: 2.000 3.000",
            "plugin model: count (text-generation)",
            "model text: defg",
            "model ids: 3 4 5",
            "chat: defg",
            "voice: 5 samples at 16000 Hz",
            "heard: hello",
            "embed: 5 2 3 0",
        ],
    ),
];

#[test]
fn the_swift_examples_run_against_the_c_abi() {
    if Command::new("swift").arg("--version").output().is_err() {
        eprintln!("skip: no swift toolchain");
        return;
    }
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("../rlx-swift");
    let plugin = plugin_lib();
    for (example, wants) in EXAMPLES {
        let out = Command::new("swift")
            .args(["run", "--package-path"])
            .arg(&package)
            .arg(example)
            .env("RLX_CAPI_LIB_DIR", target_dir())
            .env("RLX_MODEL_PLUGINS", &plugin)
            .output()
            .expect("running swift");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "swift run {example} failed:\n{stdout}\n{stderr}"
        );
        for want in *wants {
            assert!(
                stdout.contains(want),
                "{example}: missing `{want}` in:\n{stdout}\n{stderr}"
            );
        }
    }
}

#[test]
fn the_swift_test_suite_passes() {
    if Command::new("swift").arg("--version").output().is_err() {
        eprintln!("skip: no swift toolchain");
        return;
    }
    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("../rlx-swift");
    let out = Command::new("swift")
        .args(["test", "--package-path"])
        .arg(&package)
        .env("RLX_CAPI_LIB_DIR", target_dir())
        .output()
        .expect("running swift test");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.contains("with 0 failures"),
        "swift test failed:\n{stdout}\n{stderr}"
    );
}

/// Every function the C ABI exports is reachable from Swift — a function added
/// to rlx-capi without a Swift wrapper fails here. (The graph builders are
/// generated into `GraphOps.swift`, so they are covered by construction.)
#[test]
fn every_c_function_has_a_swift_wrapper() {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let header = std::fs::read_to_string(here.join("include/rlx.h")).unwrap();
    let mut swift = String::new();
    for entry in std::fs::read_dir(here.join("../rlx-swift/Sources/RLX")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "swift") {
            swift.push_str(&std::fs::read_to_string(path).unwrap());
        }
    }
    let missing: Vec<&str> = header
        .lines()
        .filter_map(|line| {
            let open = line.find('(')?;
            let name = line[..open].rsplit([' ', '*']).next()?;
            (name.starts_with("rlx_") && !line.starts_with("typedef")).then_some(name)
        })
        // The `_release` forms are for FFIs without struct-by-value; Swift
        // has it and calls the `_free` ones.
        .filter(|name| !name.ends_with("_release"))
        .filter(|name| !swift.contains(&format!("{name}(")))
        .collect();
    assert!(
        missing.is_empty(),
        "C functions with no Swift wrapper in rlx-swift/Sources/RLX: {missing:?}"
    );
}
