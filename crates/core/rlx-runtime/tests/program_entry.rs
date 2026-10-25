// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An entry of kind `program`: a WebAssembly module (rlx-program's test
//! guest, built here for wasm32) answering the model's tasks around the
//! package's own function. It loops on the data calling the function, keeps
//! a tensor in rlx's slots across runs, reads a file of the package, and its
//! errors come back as errors.
#![cfg(all(feature = "program", feature = "cpu"))]

use std::path::{Path, PathBuf};
use std::process::Command;

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::model::{Tensor, open_source, rlxp};
use serde_json::json;

/// The test guest, built for wasm32.
fn guest() -> Vec<u8> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let manifest = root.join("crates/io/rlx-program/tests/guest/Cargo.toml");
    let target = root.join("target/rlx-program-guest");
    let status = Command::new(env!("CARGO"))
        .args([
            "build",
            "--quiet",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
        ])
        .arg("--manifest-path")
        .arg(&manifest)
        .env("CARGO_TARGET_DIR", &target)
        .status()
        .expect("cargo");
    assert!(status.success(), "building the test guest");
    std::fs::read(target.join("wasm32-unknown-unknown/release/rlx_program_test_guest.wasm"))
        .unwrap()
}

/// `double(x) = 2x`, symbolic in `x`'s length.
fn double() -> Graph {
    let at = |n: usize| {
        let mut g = Graph::new("double");
        let x = g.input("x", Shape::new(&[n], DType::F32));
        let two = g.constant(2.0, DType::F32);
        let y = g.mul(x, two);
        g.set_outputs(vec![y]);
        g
    };
    rlx_ir::dynamic::generalize(&at(37), &at(53), &[(0, 37, 53)]).unwrap()
}

#[test]
fn a_program_answers_around_the_package_functions() {
    let dir: PathBuf = std::env::temp_dir().join(format!("rlx-program-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut m = rlx_ir::rlxmodel::Module::new(vec![double()]);
    m.dims = vec![("n".into(), 0)];
    m.model = Some(json!({
        "name": "toy", "family": "toy",
        "entries": {"toy": {
            "kind": "program",
            "module": "sidecar:program.wasm",
            "tasks": ["run", "synthesize", "generate", "embed", "score"],
            "task": "graph",
            "config": {"limit": 20.0, "notes": "sidecar:notes.txt"},
        }},
    }));
    let path = dir.join("toy.rlxp");
    rlxp::save_package_module(
        &path,
        &m,
        &[],
        &[
            ("program.wasm", "application/wasm", guest()),
            ("notes.txt", "text/plain", b"from the package".to_vec()),
        ],
    )
    .unwrap();
    let mut model = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    assert_eq!(model.info().metadata["tasks"][0], "run");

    // 3 → 6 → 12 → 24: three rounds through the package's `double`, each a
    // length it has not been compiled at once.
    let out = model
        .run(&[Tensor::f32("x", vec![2], &[1.0, 2.0])], &json!({}))
        .unwrap();
    assert_eq!(out[0].to_f32().unwrap(), [8.0, 16.0]);
    assert_eq!(out[1].to_f32().unwrap(), [3.0]);
    let out = model
        .run(&[Tensor::f32("x", vec![3], &[5.0, 5.0, 5.0])], &json!({}))
        .unwrap();
    assert_eq!(out[1].to_f32().unwrap(), [1.0]);

    let audio = model.synthesize("AB", &json!({})).unwrap();
    assert_eq!(audio.samples, [130.0, 132.0]);
    assert_eq!(audio.sample_rate, 8000);
    assert_eq!(
        model.generate_text("hi", &json!({})).unwrap(),
        "from the package|hi"
    );

    // Kept in an rlx slot across three runs: 2, 5 → 16, 40.
    let e = model.embed(&["ab", "hello"], &json!({})).unwrap();
    assert_eq!(e, [vec![16.0], vec![40.0]]);

    // A task it does not list, and one it lists but refuses.
    let e = model.chat(&[], &json!({})).unwrap_err().to_string();
    assert!(e.contains("not 'chat'"), "{e}");
    let e = model
        .score("q", &["p"], &json!({}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("the toy does not score"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}
