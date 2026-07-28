// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Short-Scan IR unroll vs host-Scan numeric parity on CPU.
//!
//! GPU backends prefer `maybe_unroll_scans` for `length ≤ scan_unroll_max_length`
//! so the body runs as ordinary device ops. This checks that the unrolled IR
//! matches the native `Op::Scan` executor on CPU for a small cumulative-sum.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

fn cumsum_scan_graph(length: u32) -> Graph {
    let n = 3usize;
    let f = DType::F32;
    let carry_s = Shape::new(&[n], f);
    let mut body = Graph::new("cumsum_body");
    let bc = body.input("carry", carry_s.clone());
    let bx = body.input("x_t", carry_s.clone());
    let by = body.binary(BinaryOp::Add, bc, bx, carry_s.clone());
    body.set_outputs(vec![by]);

    let mut g = Graph::new("cumsum_outer");
    let init = g.input("init", carry_s.clone());
    let xs = g.input("xs", Shape::new(&[length as usize, n], f));
    let y = g.add_node(
        Op::Scan {
            body: Box::new(body),
            length,
            save_trajectory: false,
            num_bcast: 0,
            num_xs: 1,
            num_checkpoints: 0,
        },
        vec![init, xs],
        carry_s,
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn short_scan_unroll_matches_native_scan_on_cpu() {
    let length = 4u32;
    let g = cumsum_scan_graph(length);
    let init = [0.0f32; 3];
    let xs: Vec<f32> = (1..=12).map(|i| i as f32).collect();

    let mut native = Session::new(Device::Cpu).compile(g.clone());
    let out_native = native.run(&[("init", &init[..]), ("xs", &xs[..])]);

    let unrolled = rlx_opt::maybe_unroll_scans(g, length);
    assert!(
        !unrolled
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Scan { .. })),
        "maybe_unroll_scans should erase short Scan"
    );
    let mut unrolled_sess = Session::new(Device::Cpu).compile(unrolled);
    let out_unrolled = unrolled_sess.run(&[("init", &init[..]), ("xs", &xs[..])]);

    assert_eq!(out_native[0].len(), out_unrolled[0].len());
    for (a, b) in out_native[0].iter().zip(out_unrolled[0].iter()) {
        assert!((a - b).abs() < 1e-5, "native={a} unrolled={b}");
    }
}

#[test]
fn scan_unroll_max_length_zero_keeps_scan() {
    let g = cumsum_scan_graph(4);
    let kept = rlx_opt::maybe_unroll_scans(g, 0);
    assert!(
        kept.nodes().iter().any(|n| matches!(n.op, Op::Scan { .. })),
        "max_length=0 must leave Scan intact"
    );
}

#[test]
fn packed_scan_host_matches_session() {
    let length = 4u32;
    let n = 3usize;
    let g = cumsum_scan_graph(length);
    let init = vec![0.0f32; n];
    let xs: Vec<f32> = (1..=(length as usize * n)).map(|i| i as f32).collect();

    let mut sess = Session::new(Device::Cpu).compile(g.clone());
    let out = sess.run(&[("init", &init[..]), ("xs", &xs[..])]);

    let node = g
        .nodes()
        .iter()
        .find(|n| matches!(n.op, Op::Scan { .. }))
        .expect("scan node");
    let (body, save_trajectory, nb, nx) = match &node.op {
        Op::Scan {
            body,
            save_trajectory,
            num_bcast,
            num_xs,
            ..
        } => (
            body,
            *save_trajectory,
            *num_bcast as usize,
            *num_xs as usize,
        ),
        _ => unreachable!(),
    };
    let packed = rlx_cpu::thunk::run_scan_packed_f32(
        body,
        length,
        save_trajectory,
        nb,
        nx,
        &init,
        &[],
        &[xs],
        n,
    );
    assert_eq!(out[0].len(), packed.len());
    for (a, b) in out[0].iter().zip(packed.iter()) {
        assert!((a - b).abs() < 1e-5, "session={a} packed={b}");
    }

    let _opts = CompileOptions::new().scan_unroll_max_length(64);
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    feature = "gpu",
    feature = "mlx",
    feature = "cuda",
    feature = "vulkan",
    feature = "oneapi",
    feature = "rocm",
))]
fn parity_vs_cpu(dev: Device, length: u32, opts: CompileOptions) {
    if !rlx_runtime::is_available(dev) {
        eprintln!("skip {dev:?}: unavailable");
        return;
    }
    let g = cumsum_scan_graph(length);
    let init = [0.0f32; 3];
    let xs: Vec<f32> = (1..=(length as usize * 3)).map(|i| i as f32).collect();
    let inputs: &[(&str, &[f32])] = &[("init", &init[..]), ("xs", &xs[..])];

    let host_opts = CompileOptions::new().scan_unroll_max_length(0);
    let mut cpu = Session::new(Device::Cpu).compile_with(g.clone(), &host_opts);
    let ref_out = cpu.run(inputs);

    let mut sess = Session::new(dev).compile_with(g, &opts);
    let out = sess.run(inputs);
    assert_eq!(ref_out[0].len(), out[0].len());
    for (a, b) in ref_out[0].iter().zip(out[0].iter()) {
        assert!((a - b).abs() < 1e-4, "cpu={a} {dev:?}={b}");
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_short_scan_matches_cpu() {
    parity_vs_cpu(
        Device::Metal,
        4,
        CompileOptions::new().scan_unroll_max_length(64),
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_long_host_scan_matches_cpu() {
    // Force host Scan path (no IR unroll).
    parity_vs_cpu(
        Device::Metal,
        8,
        CompileOptions::new().scan_unroll_max_length(0),
    );
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_short_scan_matches_cpu() {
    parity_vs_cpu(
        Device::Gpu,
        4,
        CompileOptions::new().scan_unroll_max_length(64),
    );
}

#[cfg(feature = "mlx")]
#[test]
fn mlx_short_scan_matches_cpu() {
    parity_vs_cpu(
        Device::Mlx,
        4,
        CompileOptions::new().scan_unroll_max_length(64),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_short_scan_matches_cpu() {
    parity_vs_cpu(
        Device::Cuda,
        4,
        CompileOptions::new().scan_unroll_max_length(64),
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_short_scan_matches_cpu() {
    parity_vs_cpu(
        Device::Vulkan,
        4,
        CompileOptions::new().scan_unroll_max_length(64),
    );
}

#[cfg(feature = "oneapi")]
#[test]
fn oneapi_short_scan_matches_cpu() {
    parity_vs_cpu(
        Device::OneApi,
        4,
        CompileOptions::new().scan_unroll_max_length(64),
    );
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_short_scan_matches_cpu() {
    parity_vs_cpu(
        Device::Rocm,
        4,
        CompileOptions::new().scan_unroll_max_length(64),
    );
}
