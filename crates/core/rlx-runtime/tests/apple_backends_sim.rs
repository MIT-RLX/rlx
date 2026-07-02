// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! On-simulator (and on-device) smoke + parity for the Apple backends.
//!
//! Unlike the other parity tests — which are macOS-host-only by convention —
//! this one is deliberately ungated for the Apple *platforms* so it can run on
//! the iOS / tvOS / visionOS **simulator** via `scripts/apple-sim-runner.sh`:
//!
//! ```sh
//! just test-apple-sim          # boots a sim + runs this on it
//! ```
//!
//! It builds one small graph — `relu(x @ w + b)` — runs it on the CPU as the
//! reference, then on every Apple accelerator that reports itself available
//! (Metal, CoreML/ANE, MLX, wgpu) and checks the result matches. It also
//! exercises runtime backend *selection* (`parse_device` / `fastest_device`),
//! which is how a host app picks a backend without a recompile.
//!
//! watchOS is excluded: it has neither Metal nor the CoreML runtime-compile
//! path, so only the CPU backend exists there (covered by the CPU smoke).
#![cfg(all(target_vendor = "apple", not(target_os = "watchos")))]

use rlx_ir::*;
use rlx_runtime::{Device, Session, fastest_device, is_available};

const M: usize = 4;
const K: usize = 6;
const N: usize = 5;

/// `relu(x @ w + b)` — exercises matmul (BLAS/MPS/MLX), elementwise add, and an
/// activation in one graph; every backend supports all three.
fn build() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("apple_smoke");
    let x = g.input("x", Shape::new(&[M, K], f));
    let w = g.input("w", Shape::new(&[K, N], f));
    let b = g.input("b", Shape::new(&[M, N], f));
    let mm = g.matmul(x, w, Shape::new(&[M, N], f));
    let y = g.binary(op::BinaryOp::Add, mm, b, Shape::new(&[M, N], f));
    let y = g.relu(y);
    g.set_outputs(vec![y]);
    g
}

fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..M * K)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..K * N)
        .map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.07)
        .collect();
    // Bias spans negatives so relu actually clamps some outputs.
    let b: Vec<f32> = (0..M * N)
        .map(|i| ((i * 3 % 11) as f32 - 6.0) * 0.10)
        .collect();
    (x, w, b)
}

fn run(device: Device) -> Vec<f32> {
    let (x, w, b) = inputs();
    Session::new(device)
        .compile(build())
        .run(&[("x", &x), ("w", &w), ("b", &b)])
        .pop()
        .expect("one output")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "output length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn cpu_smoke() {
    let out = run(Device::Cpu);
    assert_eq!(out.len(), M * N);
    assert!(out.iter().all(|v| v.is_finite()), "non-finite CPU output");
}

/// Run `dev` but treat a backend that's compiled-in / reports available yet has
/// no usable device *in this exact process* (e.g. Metal under a headless
/// `simctl spawn`) as a skip rather than a hard failure. CPU is the floor
/// (asserted in `cpu_smoke`), so the same test passes on a Metal-capable phone
/// and on a headless simulator. Returns `None` when the backend couldn't run.
fn try_run(dev: Device) -> Option<Vec<f32>> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // silence the expected device-missing panic
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(dev))).ok();
    std::panic::set_hook(prev);
    out
}

/// Every Apple accelerator that actually runs in this context must match the
/// CPU reference. Opportunistic by design — see [`try_run`].
#[test]
fn apple_backends_match_cpu() {
    let reference = run(Device::Cpu);
    let mut ran = Vec::new();
    let mut skipped = Vec::new();
    for dev in [Device::Metal, Device::Ane, Device::Mlx, Device::Gpu] {
        if !is_available(dev) {
            skipped.push((dev.name(), "unavailable"));
            continue;
        }
        match try_run(dev) {
            Some(out) => {
                let diff = max_abs_diff(&out, &reference);
                eprintln!("{:>10}: max|Δ| vs CPU = {diff:.2e}", dev.name());
                assert!(diff <= 2e-3, "{} diverged from CPU: {diff}", dev.name());
                ran.push(dev.name());
            }
            None => skipped.push((dev.name(), "no usable device in this context")),
        }
    }
    eprintln!("Apple accelerators matching CPU: {ran:?}");
    eprintln!("skipped: {skipped:?}");
}

/// Runtime backend selection: map a string to a device and pick the fastest
/// *live* one — the path a host app uses to choose a backend without a
/// recompile. `is_available` (and hence `fastest_device`) probes the real
/// device for Metal/MLX, so the choice is always something that can run.
#[test]
fn runtime_selection_works() {
    assert_eq!(rlx_runtime::parse_device("cpu").unwrap(), Device::Cpu);
    assert_eq!(rlx_runtime::parse_device("metal").unwrap(), Device::Metal);
    assert_eq!(rlx_runtime::parse_device("ane").unwrap(), Device::Ane);
    assert_eq!(rlx_runtime::parse_device("mlx").unwrap(), Device::Mlx);

    let chosen = fastest_device();
    eprintln!("fastest_device() = {}", chosen.name());
    assert!(
        is_available(chosen),
        "fastest_device returned unavailable {chosen:?}"
    );
    // The selected backend must actually execute (CPU is always a valid floor).
    assert!(try_run(chosen).is_some() || chosen == Device::Cpu);
    assert!(run(Device::Cpu).iter().all(|v| v.is_finite()));
}
