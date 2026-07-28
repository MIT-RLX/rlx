// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::ScanBackward` host-fallback parity: CPU reference vs GPU backends
//! that stage through `HostOpDesc` (Metal / wgpu / MLX).

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Session, is_available};

const F: DType = DType::F32;

/// Geometric growth scan → sum loss → `grad_with_loss` (emits `ScanBackward`).
/// Closed form: ∂loss/∂init[i] = 1.1^length.
fn geometric_scan_bwd(length: u32, n: usize) -> (Graph, NodeId, NodeId) {
    let mut body = Graph::new("scan_bwd_body");
    let x = body.input("carry", Shape::new(&[n], F));
    let scale_bytes: Vec<u8> = (0..n).flat_map(|_| 1.1_f32.to_le_bytes()).collect();
    let scale = body.add_node(
        Op::Constant { data: scale_bytes },
        vec![],
        Shape::new(&[n], F),
    );
    let next = body.binary(BinaryOp::Mul, x, scale, Shape::new(&[n], F));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("scan_bwd_outer");
    let init = g.input("init", Shape::new(&[n], F));
    let final_x = g.scan(init, body, length);
    let loss = g.reduce(final_x, ReduceOp::Sum, vec![0], false, Shape::new(&[1], F));
    g.set_outputs(vec![loss]);

    let bwd = rlx_autodiff::grad_with_loss(&g, &[init]);
    let find = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no node named {want}");
    };
    let init_id = find(&bwd, "init");
    let d_out = find(&bwd, "d_output");
    // Prefer graphs that still contain ScanBackward (host path under test).
    let _ = bwd
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::ScanBackward { .. }));
    (bwd, init_id, d_out)
}

fn run_dinit(dev: Device, bwd: &Graph, init: &[f32], d_seed: &[f32]) -> Vec<f32> {
    // Keep ScanBackward intact (no short-scan unroll of nested AD).
    let opts = CompileOptions::new().scan_unroll_max_length(0);
    let mut sess = Session::new(dev).compile_with(bwd.clone(), &opts);
    let outs = sess.run(&[("init", init), ("d_output", d_seed)]);
    // grad_with_loss outputs: [loss, …, d_init]
    outs.last().expect("d_init output").clone()
}

fn parity_vs_cpu(dev: Device) {
    if !is_available(dev) {
        eprintln!("[scan_backward_parity] {dev:?} unavailable — skip");
        return;
    }
    let length = 5u32;
    let n = 3usize;
    let (bwd, _init_id, _d_out) = geometric_scan_bwd(length, n);
    assert!(
        bwd.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::ScanBackward { .. })),
        "expected ScanBackward in AD graph"
    );

    let init = vec![1.0f32; n];
    let d_seed = [1.0f32];
    let cpu = run_dinit(Device::Cpu, &bwd, &init, &d_seed);
    let got = run_dinit(dev, &bwd, &init, &d_seed);
    assert_eq!(cpu.len(), got.len());
    let want = 1.1f32.powi(length as i32);
    for (i, (a, b)) in cpu.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "dinit[{i}] cpu={a} {dev:?}={b} (closed-form≈{want})"
        );
        assert!((a - want).abs() < 1e-3, "cpu dinit[{i}]={a} want≈{want}");
    }
}

#[test]
fn scan_backward_cpu_matches_closed_form() {
    parity_vs_cpu(Device::Cpu);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn scan_backward_metal_matches_cpu() {
    parity_vs_cpu(Device::Metal);
}

#[cfg(feature = "gpu")]
#[test]
fn scan_backward_wgpu_matches_cpu() {
    parity_vs_cpu(Device::Gpu);
}

#[cfg(feature = "mlx")]
#[test]
fn scan_backward_mlx_matches_cpu() {
    parity_vs_cpu(Device::Mlx);
}

#[cfg(feature = "cuda")]
#[test]
fn scan_backward_cuda_matches_cpu() {
    parity_vs_cpu(Device::Cuda);
}

#[cfg(feature = "vulkan")]
#[test]
fn scan_backward_vulkan_matches_cpu() {
    parity_vs_cpu(Device::Vulkan);
}

#[cfg(feature = "oneapi")]
#[test]
fn scan_backward_oneapi_matches_cpu() {
    parity_vs_cpu(Device::OneApi);
}

#[cfg(feature = "rocm")]
#[test]
fn scan_backward_rocm_matches_cpu() {
    parity_vs_cpu(Device::Rocm);
}
