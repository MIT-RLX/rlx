// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runtime examples (`examples/{inference,training,loading,distributed,
//! backends,models}.js`) run under the `rlx-js` CLI and print what they claim. The
//! Swift and Python examples of the same names check the same lines, so the
//! bindings agree on the answers too. `distributed.js` runs as a real mesh:
//! three processes, one rank each, on loopback.

use std::path::Path;
use std::process::{Command, Stdio};

fn example(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

fn run(name: &str, args: &[&str]) -> String {
    run_env(name, args, &[])
}

fn run_env(name: &str, args: &[&str], env: &[(&str, &str)]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_rlx-js"))
        .arg(example(name))
        .args(args)
        .envs(env.iter().copied())
        .output()
        .expect("running rlx-js");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{name} failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

fn expect(name: &str, stdout: &str, wants: &[&str]) {
    for want in wants {
        assert!(
            stdout.contains(want),
            "{name}: missing `{want}` in:\n{stdout}"
        );
    }
}

#[test]
fn inference_and_backends_examples_run() {
    expect(
        "inference.js",
        &run("inference.js", &[]),
        &["compiled on: cpu", "probs sum: 1.0000"],
    );
    expect(
        "backends.js",
        &run("backends.js", &[]),
        &["session cpu: max|Δ| vs first = 0.00e+0", "router ran on:"],
    );
}

#[cfg(all(feature = "training", feature = "gguf"))]
#[test]
fn training_example_runs() {
    expect(
        "training.js",
        &run("training.js", &[]),
        &[
            "trainer w: 1.999 -0.999 0.498",
            "checkpoint: 3 tensors",
            "unchanged: true",
            "manual loss: 0.000000 w: 2.000 -1.000 0.500",
        ],
    );
}

#[cfg(all(feature = "gguf", feature = "weights", feature = "text"))]
#[test]
fn loading_example_runs() {
    expect(
        "loading.js",
        &run("loading.js", &[]),
        &[
            "gguf architecture: demo",
            "gguf f32 exact: true",
            "exact: true",
            "argmax: 1",
        ],
    );
}

/// The reference model plugin (`rlx-model-plugin`), built into this
/// workspace's target dir.
#[cfg(feature = "models")]
fn model_plugin() -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let st = Command::new(env!("CARGO"))
        .args(["build", "-q", "-p", "rlx-model-plugin"])
        .current_dir(&root)
        .status()
        .expect("cargo");
    assert!(st.success(), "building rlx-model-plugin");
    let target = std::env::var_os("CARGO_TARGET_DIR").map_or(root.join("target"), Into::into);
    target.join("debug").join(format!(
        "{}rlx_model_plugin{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ))
}

#[cfg(feature = "models")]
#[test]
fn models_example_runs() {
    let plugin = model_plugin();
    expect(
        "models.js",
        &run_env(
            "models.js",
            &[],
            &[("RLX_MODEL_PLUGINS", plugin.to_str().unwrap())],
        ),
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
    );
}

#[test]
fn distributed_example_runs_as_a_three_process_mesh() {
    let peers = rlx_bind::runtime::dist::loopback_peers(3)
        .unwrap()
        .join(",");
    let spawn = |rank: &str| {
        Command::new(env!("CARGO_BIN_EXE_rlx-js"))
            .arg(example("distributed.js"))
            .args([rank, "3", &peers])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawning a rank")
    };
    let workers = [spawn("1"), spawn("2")];
    let coordinator = spawn("0").wait_with_output().expect("rank 0");
    let rank0 = String::from_utf8_lossy(&coordinator.stdout).into_owned();
    assert!(
        coordinator.status.success(),
        "rank 0 failed:\n{rank0}\n{}",
        String::from_utf8_lossy(&coordinator.stderr)
    );
    expect(
        "distributed.js rank 0",
        &rank0,
        &[
            "all_reduce sum: 6",
            "all_gather: 1,2,3",
            "workers: rank 1",
            "rank 0 w: 1.500 -2.000 0.500",
        ],
    );
    for (i, w) in workers.into_iter().enumerate() {
        let out = w.wait_with_output().expect("worker");
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(out.status.success(), "rank {} failed:\n{text}", i + 1);
        expect(
            "distributed.js worker",
            &text,
            &[&format!("rank {} w: 1.500 -2.000 0.500", i + 1)],
        );
    }
}
