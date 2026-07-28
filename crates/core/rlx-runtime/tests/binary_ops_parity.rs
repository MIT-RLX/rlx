// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `BinaryOp::{Mod, BitAnd, BitOr, BitXor, Shl, Shr}` parity. Mod = C `fmod`
//! (matches Rust `%`); bitwise operate on integer-valued operands.

#![cfg(feature = "cpu")]

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const OPS: [BinaryOp; 6] = [
    BinaryOp::Mod,
    BinaryOp::BitAnd,
    BinaryOp::BitOr,
    BinaryOp::BitXor,
    BinaryOp::Shl,
    BinaryOp::Shr,
];

fn eval(op: BinaryOp, a: f32, b: f32) -> f32 {
    match op {
        BinaryOp::Mod => a % b,
        BinaryOp::BitAnd => ((a as i64) & (b as i64)) as f32,
        BinaryOp::BitOr => ((a as i64) | (b as i64)) as f32,
        BinaryOp::BitXor => ((a as i64) ^ (b as i64)) as f32,
        BinaryOp::Shl => (a as i64).wrapping_shl(b as u32) as f32,
        BinaryOp::Shr => (a as i64).wrapping_shr(b as u32) as f32,
        _ => unreachable!(),
    }
}

fn run(device: Device, op: BinaryOp, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("bin");
    let n = a.len();
    let ai = g.input("a", Shape::new(&[n], DType::F32));
    let bi = g.input("b", Shape::new(&[n], DType::F32));
    let y = g.add_node(Op::Binary(op), vec![ai, bi], Shape::new(&[n], DType::F32));
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("a", a), ("b", b)])
        .pop()
        .unwrap()
}

// Integer-valued operands so bitwise + shift are well-defined; a few also
// exercise fmod's sign-of-dividend behavior (negative `a`).
fn operands() -> (Vec<f32>, Vec<f32>) {
    let a = vec![12.0, 10.0, 7.0, 255.0, -13.0, 33.0, 5.0, 100.0];
    let b = vec![10.0, 3.0, 2.0, 16.0, 4.0, 6.0, 1.0, 7.0];
    (a, b)
}

#[test]
fn binary_ops_cpu_matches_reference() {
    let (a, b) = operands();
    for op in OPS {
        let got = run(Device::Cpu, op, &a, &b);
        for i in 0..a.len() {
            let want = eval(op, a[i], b[i]);
            assert!(
                (got[i] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                "{op:?}[{i}] a={} b={}: got {} want {want}",
                a[i],
                b[i],
                got[i]
            );
        }
    }
}

#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    all(target_os = "macos", feature = "mlx"),
    feature = "gpu",
    feature = "cuda"
))]
fn check_device(device: Device, label: &str) {
    let (a, b) = operands();
    for op in OPS {
        let got = run(device, op, &a, &b);
        let want = run(Device::Cpu, op, &a, &b);
        for i in 0..a.len() {
            assert!(
                (got[i] - want[i]).abs() <= 1e-4 * (1.0 + want[i].abs()),
                "{label} {op:?}[{i}]: got {} want {}",
                got[i],
                want[i]
            );
        }
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn binary_ops_metal_matches_cpu() {
    check_device(Device::Metal, "metal");
}

#[test]
#[cfg(all(target_os = "macos", feature = "mlx"))]
fn binary_ops_mlx_matches_cpu() {
    if !rlx_runtime::is_available(Device::Mlx) {
        return;
    }
    check_device(Device::Mlx, "mlx");
}

#[test]
#[cfg(feature = "gpu")]
fn binary_ops_wgpu_matches_cpu() {
    check_device(Device::Gpu, "wgpu");
}

#[test]
#[cfg(feature = "cuda")]
fn binary_ops_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        return;
    }
    check_device(Device::Cuda, "cuda");
}
