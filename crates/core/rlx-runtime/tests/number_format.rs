// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::RoundToFormat` and `CompileOptions::number_format`.
//!
//! The CPU kernel must equal `FloatFormat::round_slice_scaled` bit for bit, and
//! every other compiled-in backend must equal the CPU bit for bit.

mod common;

use rlx_ir::op::Activation;
use rlx_ir::{DType, FloatFormat, FormatScaling, Graph, NumberFormat, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

// 70 000 elements: past the CPU kernel's single-thread cutoff in both the
// elementwise and the per-block paths. COLS is not a multiple of 32, so rows
// end in a short block.
const ROWS: usize = 1000;
const COLS: usize = 70;

fn formats() -> Vec<FloatFormat> {
    [
        "f16", "bf16", "f8e4m3", "bf8", "f6e2m3", "f6e3m2", "f4", "f4e3m0", "f12m7e4", "f5m2e2",
    ]
    .iter()
    .map(|s| FloatFormat::parse(s).unwrap_or_else(|| panic!("{s}")))
    .collect()
}

fn scalings() -> [FormatScaling; 6] {
    [
        FormatScaling::None,
        FormatScaling::PerTensor,
        FormatScaling::Block(32),
        FormatScaling::Block(7),
        FormatScaling::FineBlock(16),
        FormatScaling::FineBlock(7),
    ]
}

/// Edge cases first, then a spread of magnitudes.
fn inputs() -> Vec<f32> {
    let mut v = vec![
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        2.5,
        3.5,
        -2.5,
        0.75,
        1e-8,
        65520.0,
        448.0,
        449.0,
        6.0,
        7.0,
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    while v.len() < ROWS * COLS {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mant = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        let exp = ((state >> 20) % 40) as i32 - 20;
        v.push(mant * 2f32.powi(exp));
    }
    v
}

fn round_graph(format: FloatFormat, scaling: FormatScaling) -> Graph {
    let mut g = Graph::new("round_to_format");
    let shape = Shape::new(&[ROWS, COLS], DType::F32);
    let x = g.input("x", shape.clone());
    let y = rlx_opt::rlx_compile::emulate_format::round_node(&mut g, x, format, scaling);
    g.set_outputs(vec![y]);
    g
}

fn run(device: Device, g: Graph, x: &[f32]) -> Vec<f32> {
    Session::new(device).compile(g).run(&[("x", x)]).remove(0)
}

fn same_bits(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[test]
fn cpu_kernel_is_the_reference() {
    let _gpu = common::serialize_gpu();
    let x = inputs();
    for format in formats() {
        for scaling in scalings() {
            let got = run(Device::Cpu, round_graph(format, scaling), &x);
            let mut want = x.clone();
            format.round_slice_scaled(&mut want, scaling, COLS);
            let bad: Vec<_> = (0..x.len())
                .filter(|&i| !same_bits(got[i], want[i]))
                .take(3)
                .map(|i| format!("x={:e} got={:e} want={:e}", x[i], got[i], want[i]))
                .collect();
            assert!(bad.is_empty(), "{format} {scaling:?}: {bad:?}");
        }
    }
}

/// Backends compiled into this test build, other than the CPU.
#[allow(clippy::vec_init_then_push)]
fn other_devices() -> Vec<Device> {
    #[allow(unused_mut)]
    let mut v = Vec::new();
    #[cfg(feature = "metal")]
    v.push(Device::Metal);
    #[cfg(feature = "mlx")]
    v.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    v.push(Device::Gpu);
    #[cfg(feature = "vulkan")]
    v.push(Device::Vulkan);
    #[cfg(feature = "cuda")]
    v.push(Device::Cuda);
    #[cfg(feature = "rocm")]
    v.push(Device::Rocm);
    #[cfg(feature = "coreml")]
    v.push(Device::Ane);
    v.into_iter()
        .filter(|d| rlx_runtime::is_available(*d))
        .collect()
}

#[test]
fn every_backend_matches_cpu_bit_for_bit() {
    let _gpu = common::serialize_gpu();
    let x = inputs();
    let mut bad = Vec::new();
    for device in other_devices() {
        for format in formats() {
            for scaling in scalings() {
                let cpu = run(Device::Cpu, round_graph(format, scaling), &x);
                let got = run(device, round_graph(format, scaling), &x);
                let mismatches = (0..x.len()).filter(|&i| !same_bits(got[i], cpu[i])).count();
                if mismatches > 0 {
                    let i = (0..x.len()).find(|&i| !same_bits(got[i], cpu[i])).unwrap();
                    bad.push(format!(
                        "{device:?} {format} {scaling:?}: {mismatches} differ, e.g. x={:e} got={:e} cpu={:e}",
                        x[i], got[i], cpu[i]
                    ));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} mismatching runs:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// `number_format` rounds weights and intermediates (not inputs) but returns the
/// last op's value unrounded; `f32` leaves the graph alone.
#[test]
fn compile_option_rewrites_the_graph() {
    let _gpu = common::serialize_gpu();
    let build = || {
        let mut g = Graph::new("mlp");
        let x = g.input("x", Shape::new(&[1, 3], DType::F32));
        let w = g.add_node(
            Op::Param { name: "w".into() },
            vec![],
            Shape::new(&[3, 2], DType::F32),
        );
        let h = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[1, 2], DType::F32));
        let y = g.activation(Activation::Exp, h, Shape::new(&[1, 2], DType::F32));
        g.set_outputs(vec![y]);
        g
    };
    let w = [0.1f32, 0.7, -0.33, 1.9, 0.05, -1.2];
    let x = [1.3f32, -0.6, 2.2];
    let run_with = |opts: CompileOptions| {
        let mut cg = Session::new(Device::Cpu).compile_with(build(), &opts);
        cg.set_param("w", &w);
        cg.run(&[("x", &x)]).remove(0)
    };
    let f32_out = run_with(CompileOptions::baseline());
    assert_eq!(
        run_with(CompileOptions::baseline().number_format(NumberFormat::F32)),
        f32_out
    );

    let f4: NumberFormat = "f4@none".parse().unwrap();
    let NumberFormat::Float { format, .. } = f4 else {
        unreachable!()
    };
    let got = run_with(CompileOptions::baseline().number_format(f4));
    // By hand: round w (inputs stay f32), matmul in f32, round, exp (output unrounded).
    let r = |v: f32| format.round(v);
    let want: Vec<f32> = (0..2)
        .map(|j| {
            let h: f32 = (0..3).map(|k| x[k] * r(w[k * 2 + j])).sum();
            r(h).exp()
        })
        .collect();
    assert_eq!(got, want);
    assert_ne!(got, f32_out);
}

#[test]
fn report_tested_devices() {
    let _gpu = common::serialize_gpu();
    eprintln!(
        "number_format: devices under test besides cpu: {:?}",
        other_devices()
    );
}

/// A rounded weight feeding a realistically sized matmul (CPU sgemm paths
/// differ by size). The rounded weight must reach the matmul.
#[test]
fn rounded_weight_feeds_large_matmul() {
    let _gpu = common::serialize_gpu();
    for (m, k, n) in [(3usize, 4usize, 2usize), (16, 192, 576), (40, 768, 2304)] {
        let build = || {
            let mut g = Graph::new("mm");
            let x = g.input("x", Shape::new(&[m, k], DType::F32));
            let w = g.add_node(
                Op::Param { name: "w".into() },
                vec![],
                Shape::new(&[k, n], DType::F32),
            );
            let y = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[m, n], DType::F32));
            g.set_outputs(vec![y]);
            g
        };
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.01)
            .collect();
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1)
            .collect();
        let fmt: NumberFormat = "f24m15e8".parse().unwrap();
        let run_with = |opts: CompileOptions| {
            let mut cg = Session::new(Device::Cpu).compile_with(build(), &opts);
            cg.set_param("w", &w);
            cg.run(&[("x", &x)]).remove(0)
        };
        let want = run_with(CompileOptions::baseline());
        let mut opts = CompileOptions::baseline().number_format(fmt);
        opts.number_format_options.keep_ops = vec![rlx_ir::OpKind::Input];
        let got = run_with(opts);
        let max = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            err <= 1e-3 * max,
            "m={m} k={k} n={n}: off by {err} (scale {max})"
        );
    }
}

/// Three matmuls sharing one input — the QKV shape `FuseSharedInputMatMul`
/// packs into one wide matmul — with rounded weights, on every backend.
#[test]
fn shared_input_matmuls_with_rounded_weights() {
    let _gpu = common::serialize_gpu();
    let (m, k, n) = (8usize, 64usize, 48usize);
    let build = || {
        let mut g = Graph::new("qkv");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let mut outs = Vec::new();
        for name in ["wq", "wk", "wv"] {
            // `[out, in]` storage + Transpose, as Linear layers are built.
            let w = g.add_node(
                Op::Param { name: name.into() },
                vec![],
                Shape::new(&[n, k], DType::F32),
            );
            let wt = g.add_node(
                Op::Transpose { perm: vec![1, 0] },
                vec![w],
                Shape::new(&[k, n], DType::F32),
            );
            let y = g.add_node(Op::MatMul, vec![x, wt], Shape::new(&[m, n], DType::F32));
            outs.push(g.activation(Activation::Silu, y, Shape::new(&[m, n], DType::F32)));
        }
        let s = g.add_node(
            Op::Concat { axis: 1 },
            outs,
            Shape::new(&[m, 3 * n], DType::F32),
        );
        g.set_outputs(vec![s]);
        g
    };
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1)
        .collect();
    let ws: Vec<Vec<f32>> = (0..3)
        .map(|j| {
            (0..k * n)
                .map(|i| ((i * 7 + j) % 13) as f32 * 0.01 - 0.06)
                .collect()
        })
        .collect();
    let run = |device: Device| {
        let mut opts = CompileOptions::new().number_format("f16".parse().unwrap());
        // Weights rounded inside the graph (QAT's path) rather than on upload.
        opts.number_format_options.params_in_graph =
            std::env::var("NF_TEST_PARAMS_IN_GRAPH").is_ok_and(|v| v == "1");
        let mut cg = Session::new(device).compile_with(build(), &opts);
        for (name, w) in ["wq", "wk", "wv"].iter().zip(&ws) {
            cg.set_param(name, w);
        }
        cg.run(&[("x", &x)]).remove(0)
    };
    let cpu = run(Device::Cpu);
    let scale = cpu.iter().fold(0f32, |a, v| a.max(v.abs()));
    for device in other_devices() {
        let got = run(device);
        let err = got
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            err <= 1e-3 * scale,
            "{device:?}: off by {err} (scale {scale})"
        );
    }
}

/// Rounded weights when Metal re-plans its arena for a separate weight buffer
/// (forced small via `RLX_METAL_WEIGHT_SPLIT_MIN`): the rounding op is not
/// weight-buffer aware, so its weights must stay readable in the arena.
#[test]
fn rounded_weights_survive_metal_weight_split() {
    rlx_runtime::env::set("RLX_METAL_WEIGHT_SPLIT_MIN", "0");
    let (m, k, n) = (4usize, 512usize, 512usize);
    let build = || {
        let mut g = Graph::new("split");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let g1 = g.add_node(
            Op::Param { name: "g".into() },
            vec![],
            Shape::new(&[k], DType::F32),
        );
        let w = g.add_node(
            Op::Param { name: "w".into() },
            vec![],
            Shape::new(&[k, n], DType::F32),
        );
        let xs = g.binary(
            rlx_ir::op::BinaryOp::Mul,
            x,
            g1,
            Shape::new(&[m, k], DType::F32),
        );
        let y = g.add_node(Op::MatMul, vec![xs, w], Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);
        g
    };
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1)
        .collect();
    let gain: Vec<f32> = (0..k).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i * 7) % 13) as f32 * 0.01 - 0.06)
        .collect();
    let run = |device: Device| {
        let opts = CompileOptions::new().number_format("f16".parse().unwrap());
        let mut cg = Session::new(device).compile_with(build(), &opts);
        cg.set_param("g", &gain);
        cg.set_param("w", &w);
        cg.run(&[("x", &x)]).remove(0)
    };
    let cpu = run(Device::Cpu);
    let scale = cpu.iter().fold(0f32, |a, v| a.max(v.abs()));
    let mut bad = Vec::new();
    for device in other_devices() {
        let got = run(device);
        let err = got
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if err > 1e-3 * scale {
            bad.push(format!("{device:?}: off by {err} (scale {scale})"));
        }
    }
    rlx_runtime::env::unset("RLX_METAL_WEIGHT_SPLIT_MIN");
    assert!(bad.is_empty(), "{bad:?}");
}

/// An embedding-sized rounded table read through `Gather` (Qwen3-0.6B shape).
#[test]
#[ignore = "allocates ~1.3 GB; run explicitly"]
fn gather_from_large_rounded_table() {
    let (vocab, dim, n) = (151_936usize, 1024usize, 256usize);
    let build = || {
        let mut g = Graph::new("embed");
        let ids = g.input("ids", Shape::new(&[n], DType::I32));
        let t = g.add_node(
            Op::Param {
                name: "table".into(),
            },
            vec![],
            Shape::new(&[vocab, dim], DType::F32),
        );
        let y = g.add_node(
            Op::Gather { axis: 0 },
            vec![t, ids],
            Shape::new(&[n, dim], DType::F32),
        );
        let z = g.activation(Activation::Silu, y, Shape::new(&[n, dim], DType::F32));
        g.set_outputs(vec![z]);
        g
    };
    let table: Vec<f32> = (0..vocab * dim)
        .map(|i| ((i % 977) as f32 - 488.0) * 1e-3)
        .collect();
    let ids: Vec<f32> = (0..n).map(|i| ((i * 7919) % vocab) as f32).collect();
    let run = |device: Device| {
        let opts = CompileOptions::new().number_format("f16".parse().unwrap());
        let mut cg = Session::new(device).compile_with(build(), &opts);
        cg.set_param("table", &table);
        cg.run(&[("ids", &ids)]).remove(0)
    };
    let cpu = run(Device::Cpu);
    for device in other_devices() {
        let got = run(device);
        let err = got
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let zeros = got.iter().filter(|v| **v == 0.0).count();
        assert!(
            err < 1e-6,
            "{device:?}: off by {err}; {zeros} zeros of {}",
            got.len()
        );
    }
}
