// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Licensed under the GNU General Public License, version 3.

//! Cross-backend parity for `Op::FusedAttentionBlock`.
//!
//! Every backend that claims `OpKind::FusedAttentionBlock` (so the
//! `FuseAttentionBlock` pass is first-class) and lowers it: CPU/MLX natively,
//! everyone else by decomposing to the primitive chain (matmul → narrow →
//! reshape/transpose → \[rope\] → attention → matmul). This test pins all
//! paths to the same numbers. QNN (`Device::Hexagon`) is included — it claims
//! FAB and runs `unfuse_attention_block` before the FFI lower.
//!
//! ## Mask convention
//!
//! The block's mask is input #3 with `MaskKind::Custom`. The native CPU
//! thunk reads it as a `[B, S]` **binary** per-key mask (`v < thr` ⇒ drop);
//! the decompose path and the MLX lowering feed it to `Op::Attention`
//! / SDPA as an **additive** bias. An **all-ones `[B, H, S, S]`** buffer is
//! neutral under *both*: binary `1.0` = keep, and an additive constant is
//! softmax shift-invariant — so a single graph runs identically on the
//! native and decomposed paths and we can compare them directly.

#![cfg(feature = "cpu")]
#![allow(dead_code)]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, GraphDevices, is_available};

/// Build an `Op::FusedAttentionBlock` graph.
///
/// Inputs: `h` (hidden `[B,S,H*D]`), `mask` (`[B,H,S,S]`), and — when
/// `has_rope` — `cos` / `sin` (`[S, D/2]`). Params: `qkv_w` (`[H*D, 3H*D]`),
/// `out_w` (`[H*D, H*D]`), and — when `has_bias` — `qkv_b` / `out_b`.
fn fab_graph(b: usize, s: usize, nh: usize, dh: usize, has_bias: bool, has_rope: bool) -> Graph {
    let inner = nh * dh;
    let mut g = Graph::new("fab_parity");
    let hidden = g.input("h", Shape::new(&[b, s, inner], DType::F32));
    let qkv_w = g.param("qkv_w", Shape::new(&[inner, 3 * inner], DType::F32));
    let out_w = g.param("out_w", Shape::new(&[inner, inner], DType::F32));
    let mask = g.input("mask", Shape::new(&[b, nh, s, s], DType::F32));
    let mut inputs = vec![hidden, qkv_w, out_w, mask];
    if has_bias {
        inputs.push(g.param("qkv_b", Shape::new(&[3 * inner], DType::F32)));
        inputs.push(g.param("out_b", Shape::new(&[inner], DType::F32)));
    }
    if has_rope {
        inputs.push(g.input("cos", Shape::new(&[s, dh / 2], DType::F32)));
        inputs.push(g.input("sin", Shape::new(&[s, dh / 2], DType::F32)));
    }
    let y = g.add_node(
        Op::FusedAttentionBlock {
            num_heads: nh,
            head_dim: dh,
            has_bias,
            has_rope,
        },
        inputs,
        Shape::new(&[b, s, inner], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "{label}[{i}]: {a} vs {b} (tol {tol})\n got={got:?}\nwant={want:?}"
        );
    }
}

/// Identity-weight block (no bias, no rope): Q = identity, K = 0, V = identity,
/// out = identity. K = 0 ⇒ uniform softmax ⇒ each token's output is the mean of
/// the V (= hidden) rows. For `hidden = [[1,2,3,4],[5,6,7,8]]` that mean is
/// `(3,4,5,6)` for both tokens — a closed form independent of the backend.
fn identity_case(device: Device, tol: f32, label: &str) {
    let (b, s, nh, dh) = (1usize, 2usize, 2usize, 4usize);
    let inner = nh * dh;
    let g = fab_graph(b, s, nh, dh, false, false);

    let mut qkv_w = vec![0f32; inner * 3 * inner];
    for i in 0..inner {
        qkv_w[i * 3 * inner + i] = 1.0; // Q = identity
        qkv_w[i * 3 * inner + 2 * inner + i] = 1.0; // V = identity
    }
    let mut out_w = vec![0f32; inner * inner];
    for i in 0..inner {
        out_w[i * inner + i] = 1.0;
    }

    // hidden rows [1..8] and [9..16]; K = 0 ⇒ uniform softmax ⇒ each token's
    // output is the per-channel mean of the two rows = [5..12], for both tokens.
    let hidden: Vec<f32> = (1..=(b * s * inner)).map(|i| i as f32).collect();
    let mask = vec![1.0; b * nh * s * s]; // all-ones ⇒ neutral for both conventions
    let want = vec![
        5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    let mut runner = GraphDevices::new(g);
    runner.set_param("qkv_w", &qkv_w);
    runner.set_param("out_w", &out_w);
    let inputs: &[(&str, &[f32])] = &[("h", &hidden), ("mask", &mask)];

    let cpu = runner.run(Device::Cpu, inputs).expect("cpu run");
    assert_close(
        &cpu[0],
        &want,
        1e-4,
        &format!("{label}/identity cpu-native"),
    );
    let dev = runner.run(device, inputs).expect("device run");
    assert_close(&dev[0], &want, tol, &format!("{label}/identity decompose"));
}

/// Bias variant (`has_bias`, no rope) with deterministic dense weights. No
/// closed form — assert the device matches the native CPU thunk.
fn bias_case(device: Device, tol: f32, label: &str) {
    let (b, s, nh, dh) = (1usize, 3usize, 2usize, 4usize);
    let inner = nh * dh;
    let g = fab_graph(b, s, nh, dh, true, false);

    // Deterministic, well-conditioned weights/inputs.
    let qkv_w: Vec<f32> = (0..inner * 3 * inner)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let out_w: Vec<f32> = (0..inner * inner)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
        .collect();
    let qkv_b: Vec<f32> = (0..3 * inner).map(|i| (i as f32) * 0.01).collect();
    let out_b: Vec<f32> = (0..inner).map(|i| (i as f32) * -0.02).collect();
    let hidden: Vec<f32> = (0..b * s * inner)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.1)
        .collect();
    let mask = vec![1.0; b * nh * s * s];

    let mut runner = GraphDevices::new(g);
    runner.set_param("qkv_w", &qkv_w);
    runner.set_param("out_w", &out_w);
    runner.set_param("qkv_b", &qkv_b);
    runner.set_param("out_b", &out_b);
    let inputs: &[(&str, &[f32])] = &[("h", &hidden), ("mask", &mask)];

    let cpu = runner.run(Device::Cpu, inputs).expect("cpu run");
    let dev = runner.run(device, inputs).expect("device run");
    assert_close(
        &dev[0],
        &cpu[0],
        tol,
        &format!("{label}/bias vs cpu-native"),
    );
}

/// `has_rope` plumbing: identity rotation (`cos = 1`, `sin = 0`) so the result
/// is independent of rope style, but the `cos`/`sin` inputs (#6/#7) and the
/// rope branch of the decomposition are exercised end to end. Asserts device
/// == native CPU.
fn rope_case(device: Device, tol: f32, label: &str) {
    let (b, s, nh, dh) = (1usize, 2usize, 2usize, 4usize);
    let inner = nh * dh;
    let g = fab_graph(b, s, nh, dh, false, true);

    let qkv_w: Vec<f32> = (0..inner * 3 * inner)
        .map(|i| ((i % 6) as f32 - 2.5) * 0.05)
        .collect();
    let out_w: Vec<f32> = (0..inner * inner)
        .map(|i| ((i % 4) as f32 - 1.5) * 0.1)
        .collect();
    let hidden: Vec<f32> = (0..b * s * inner)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let mask = vec![1.0; b * nh * s * s];
    let cos = vec![1.0; s * (dh / 2)]; // identity rotation
    let sin = vec![0.0; s * (dh / 2)];

    let mut runner = GraphDevices::new(g);
    runner.set_param("qkv_w", &qkv_w);
    runner.set_param("out_w", &out_w);
    let inputs: &[(&str, &[f32])] = &[
        ("h", &hidden),
        ("mask", &mask),
        ("cos", &cos),
        ("sin", &sin),
    ];

    let cpu = runner.run(Device::Cpu, inputs).expect("cpu run");
    let dev = runner.run(device, inputs).expect("device run");
    assert_close(
        &dev[0],
        &cpu[0],
        tol,
        &format!("{label}/rope vs cpu-native"),
    );
}

/// Run the full suite (identity + bias + rope) on `device`, comparing against
/// the closed form / native CPU. No-op when the device is unavailable.
fn fab_suite(device: Device, tol: f32, label: &str) {
    if !is_available(device) {
        eprintln!("skip fused_attention_block_parity/{label} ({device:?} unavailable)");
        return;
    }
    identity_case(device, tol, label);
    bias_case(device, tol, label);
    rope_case(device, tol, label);
}

/// CPU runs the block natively (thunk-level fusion) — pin its closed form.
#[test]
fn fab_cpu_native_identity() {
    let (b, s, nh, dh) = (1usize, 2usize, 2usize, 4usize);
    let inner = nh * dh;
    let g = fab_graph(b, s, nh, dh, false, false);
    let mut qkv_w = vec![0f32; inner * 3 * inner];
    for i in 0..inner {
        qkv_w[i * 3 * inner + i] = 1.0;
        qkv_w[i * 3 * inner + 2 * inner + i] = 1.0;
    }
    let mut out_w = vec![0f32; inner * inner];
    for i in 0..inner {
        out_w[i * inner + i] = 1.0;
    }
    let mut runner = GraphDevices::new(g);
    runner.set_param("qkv_w", &qkv_w);
    runner.set_param("out_w", &out_w);
    let hidden: Vec<f32> = (1..=(b * s * inner)).map(|i| i as f32).collect();
    let mask = vec![1.0; b * nh * s * s];
    let out = runner
        .run(Device::Cpu, &[("h", &hidden), ("mask", &mask)])
        .expect("cpu run");
    assert_close(
        &out[0],
        &[
            5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        1e-4,
        "cpu-native/identity",
    );
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn fab_metal_parity() {
    fab_suite(Device::Metal, 1e-3, "metal");
}

#[test]
#[cfg(all(feature = "mlx", rlx_mlx_host))]
fn fab_mlx_parity() {
    fab_suite(Device::Mlx, 1e-3, "mlx");
}

#[test]
#[cfg(feature = "gpu")]
fn fab_wgpu_parity() {
    fab_suite(Device::Gpu, 1e-2, "wgpu");
}

#[test]
#[cfg(feature = "vulkan")]
fn fab_vulkan_parity() {
    fab_suite(Device::Vulkan, 1e-2, "vulkan");
}

#[test]
#[cfg(feature = "oneapi")]
fn fab_oneapi_parity() {
    fab_suite(Device::OneApi, 1e-2, "oneapi");
}

#[test]
#[cfg(any(feature = "coreml", feature = "ane"))]
fn fab_coreml_parity() {
    fab_suite(Device::Ane, 1e-2, "coreml");
}

#[test]
#[cfg(feature = "cuda")]
fn fab_cuda_parity() {
    fab_suite(Device::Cuda, 1e-2, "cuda");
}

#[test]
#[cfg(feature = "rocm")]
fn fab_rocm_parity() {
    fab_suite(Device::Rocm, 1e-2, "rocm");
}

#[test]
#[cfg(feature = "qnn")]
fn fab_qnn_parity() {
    fab_suite(Device::Hexagon, 1e-2, "qnn");
}
