// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The XDNA bf16 matmul (`Precision::BF16`) is exact for its rounded operands:
//! against `Σ bf16(a)·bf16(b)` accumulated in f64 it is off only by f32
//! accumulation. Its distance from the f32 product is bf16 input rounding —
//! which looks large next to the output where the products cancel, but is
//! the format, not the kernel. Shapes cover K below, at and past one 32-wide
//! tile and several K-blocks.

#![cfg(feature = "xdna")]

mod common;

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Precision, Session};

fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

#[test]
fn xdna_bf16_matmul_is_exact_for_rounded_operands() {
    if common::skip_unless(Device::Xdna) {
        return;
    }
    for (m, k, n) in [(64, 2, 100), (512, 17, 32), (64, 64, 64), (32, 256, 32)] {
        let mut g = Graph::new("xdna_bf16_mm");
        let a = g.input("a", Shape::new(&[m, k], DType::F32));
        let b = g.input("b", Shape::new(&[k, n], DType::F32));
        let c = g.add_node(Op::MatMul, vec![a, b], Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![c]);
        let av: Vec<f32> = (0..m * k).map(|j| (j as f32 * 0.37).sin() * 3.0).collect();
        let bv: Vec<f32> = (0..k * n).map(|j| (j as f32 * 0.11).cos()).collect();
        let mut o = CompileOptions::new();
        o.precision = Precision::BF16;
        let got = Session::new(Device::Xdna)
            .compile_with(g, &o)
            .run(&[("a", &av), ("b", &bv)])
            .pop()
            .unwrap();
        let mut worst = 0f64;
        for i in 0..m {
            for j in 0..n {
                let (mut want, mut mag) = (0f64, 0f64);
                for p in 0..k {
                    let prod = bf16(av[i * k + p]) as f64 * bf16(bv[p * n + j]) as f64;
                    want += prod;
                    mag += prod.abs();
                }
                worst = worst.max((got[i * n + j] as f64 - want).abs() / mag.max(1e-30));
            }
        }
        eprintln!("{m}x{k}x{n}: max error vs rounded operands {worst:.2e} of Σ|ab|");
        assert!(worst < 1e-6, "{m}x{k}x{n}: bf16 matmul off by {worst:e}");
    }
}
