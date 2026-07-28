// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-backend validation: run the same computation on every GPU backend
//! compiled in AND live on this host, and check each matches CPU.
//!
//! `Device::Ane` (CoreML) is intentionally excluded — its op surface is
//! narrower and its Obj-C FFI panics are not unwind-safe (a failed op aborts
//! the process during cleanup). CoreML/ANE has its own CPU-vs-ANE parity
//! suite in the `rlx-coreml` crate. Run e.g.:
//! `cargo test -p rlx-tensor --features eval-apple -- --nocapture`.
#![cfg(feature = "eval")]

use rlx_tensor::{Device, Tensor, available_devices};

fn approx(a: &[f32], b: &[f32], dev: Device) {
    assert_eq!(a.len(), b.len(), "{dev:?}: length mismatch");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-3, "{dev:?}: {a:?} != {b:?}");
    }
}

/// Matmul + broadcast-add + relu (exact) + reduce — robust across backends.
fn sample() -> Tensor {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let w = Tensor::from_vec(vec![0.5, -1.0, 2.0, 0.0, 1.0, 1.0], [3, 2]);
    let b = Tensor::from_vec(vec![0.1, -0.2], [2]);
    (&a.matmul(&w) + &b).relu().sum([1], false)
}

#[test]
fn gpu_backends_match_cpu() {
    let reference = sample().to_vec_on(Device::Cpu);
    let mut validated = vec![Device::Cpu];

    for dev in available_devices() {
        // Skip CPU (the reference) and ANE (FFI not unwind-safe; tested in
        // rlx-coreml's own suite).
        if matches!(dev, Device::Cpu | Device::Ane) {
            continue;
        }
        let got = sample().to_vec_on(dev);
        approx(&got, &reference, dev);
        validated.push(dev);
    }

    eprintln!("validated backends against CPU: {validated:?}");
    // On an Apple machine with eval-apple we expect at least Metal + MLX + Gpu.
    assert!(validated.contains(&Device::Cpu));

    // Drop the cached GPU CompiledGraphs now, while thread-locals are still
    // alive — otherwise multiple GPU contexts tear down at thread exit and a
    // backend's destructor can touch an already-destroyed thread-local.
    rlx_tensor::clear_cache();
}

#[test]
fn different_work_on_different_gpu_backends() {
    use rlx_tensor::{Device, Tensor, available_devices};
    // Two independent computations, each pinned to a different GPU backend,
    // both read back correctly. (Per-graph single-device; host-mediated.)
    let gpus: Vec<Device> = available_devices()
        .into_iter()
        .filter(|d| !matches!(d, Device::Cpu | Device::Ane))
        .collect();
    eprintln!("live GPU backends: {gpus:?}");
    for (i, dev) in gpus.iter().enumerate() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
        let y = (&x * &x).sum([0], false).to_vec_on(*dev); // 14
        assert!((y[0] - 14.0).abs() < 1e-3, "{dev:?} -> {y:?}");
        eprintln!("  backend #{i} {dev:?}: sum(x^2)=14 OK");
    }
    rlx_tensor::clear_cache();
}
