// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 3-D convolution networks in f32 on CUDA, against the CPU: the graphs of
//! `rlx-metal/tests/precision_policy_f16_conv3d.rs` (each op of a MONAI-style
//! decoder alone, then together), at volumes up to 96^3 — the size at which
//! medrlx's SegResNet and SwinUNETR disagreed with the CPU on CUDA.
//!

// `!(x < tol)` is deliberate: a NaN error fails the check.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use rlx_runtime::{CompileOptions, Device, PrecisionPolicy, Session};

fn data(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 9) as f32 / 4_194_304.0 - 1.0) * scale
        })
        .collect()
}

struct Case {
    g: Graph,
    feeds: Vec<(String, Vec<f32>)>,
    params: Vec<(String, Vec<f32>)>,
}

impl Case {
    fn new(name: &str) -> Self {
        Self {
            g: Graph::new(name),
            feeds: Vec::new(),
            params: Vec::new(),
        }
    }

    fn input(&mut self, name: &str, dims: &[usize]) -> NodeId {
        let n = dims.iter().product();
        self.feeds
            .push((name.into(), data(n, self.feeds.len() as u32 * 7 + 3, 1.0)));
        self.g.input(name, Shape::new(dims, DType::F32))
    }

    /// A weight scaled like a He initialisation, so activations stay O(1).
    fn param(&mut self, name: &str, dims: &[usize], fan_in: usize) -> NodeId {
        let n = dims.iter().product();
        let scale = (3.0 / fan_in as f32).sqrt();
        self.params.push((
            name.into(),
            data(n, self.params.len() as u32 * 13 + 5, scale),
        ));
        self.g.param(name, Shape::new(dims, DType::F32))
    }

    fn conv(&mut self, x: NodeId, name: &str, c_out: usize, k: usize, groups: usize) -> NodeId {
        let c_in = self.g.shape(x).dim(1).unwrap_static();
        let w = self.param(
            name,
            &[c_out, c_in / groups, k, k, k],
            c_in / groups * k * k * k,
        );
        self.g.conv3d(x, w, [1; 3], [k / 2; 3], [1; 3], groups)
    }

    fn bias(&mut self, x: NodeId, name: &str) -> NodeId {
        let c = self.g.shape(x).dim(1).unwrap_static();
        let b = self.param(name, &[c], 1);
        let b = self.g.reshape_(b, vec![1, c as i64, 1, 1, 1]);
        self.g.add(x, b)
    }

    fn norm(&mut self, x: NodeId, name: &str, groups: usize) -> NodeId {
        let c = self.g.shape(x).dim(1).unwrap_static();
        let gamma = self.param(&format!("{name}.g"), &[c], 1);
        let beta = self.param(&format!("{name}.b"), &[c], 1);
        self.g.group_norm(x, gamma, beta, groups, 1e-5)
    }

    fn leaky(&mut self, x: NodeId) -> NodeId {
        let s = self.g.constant(0.01, DType::F32);
        let scaled = self.g.mul(x, s);
        let dims = self.g.shape(x).clone();
        self.g.binary(rlx_ir::op::BinaryOp::Max, x, scaled, dims)
    }

    fn run(&self, device: Device) -> Vec<f32> {
        self.run_with(device, CompileOptions::new())
    }

    fn run_with(&self, device: Device, opts: CompileOptions) -> Vec<f32> {
        let mut c = Session::new(device).compile_with(self.g.clone(), &opts);
        for (name, v) in &self.params {
            c.set_param(name, v);
        }
        let feeds: Vec<(&str, &[f32])> = self
            .feeds
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_slice()))
            .collect();
        c.run(&feeds).remove(0)
    }
}

/// conv 3^3 + bias, group norm, LeakyReLU; a 2x-downsampled branch upsampled
/// back and concatenated: the decoder pattern Metal reads in place; then a
/// 1^3 conv and a broadcast bias.
fn decoder(d: usize) -> Case {
    let mut c = Case::new("f16_conv3d_decoder");
    let x = c.input("x", &[1, 4, d, d, d]);
    let y = c.conv(x, "c1", 16, 3, 1);
    let y = c.bias(y, "c1.b");
    let y = c.norm(y, "n1", 4);
    let y = c.leaky(y);
    let small = c.g.interpolate3d_nearest(y, [d / 2; 3]);
    let small = c.conv(small, "c2", 16, 3, 1);
    let up = c.g.interpolate3d_nearest(small, [d; 3]);
    let shape = Shape::new(&[1, 32, d, d, d], DType::F32);
    let cat = c.g.concat(vec![up, y], 1, shape);
    let z = c.conv(cat, "c3", 16, 3, 1);
    let z = c.norm(z, "n3", 8);
    let z = c.g.relu(z);
    let z = c.conv(z, "head", 3, 1, 1);
    let z = c.bias(z, "head.b");
    c.g.set_outputs(vec![z]);
    c
}

/// The decoder's pieces one at a time, so a failure names its op.
fn piece(stage: usize, d: usize) -> Case {
    let mut c = Case::new("f16_conv3d_piece");
    let x = c.input("x", &[1, 4, d, d, d]);
    let mut y = c.conv(x, "c1", 16, 3, 1);
    if stage >= 1 {
        y = c.bias(y, "c1.b");
    }
    if stage >= 2 {
        y = c.norm(y, "n1", 4);
    }
    if stage >= 3 {
        y = c.leaky(y);
    }
    if stage == 4 {
        let small = c.g.interpolate3d_nearest(y, [d / 2; 3]);
        y = c.conv(small, "c2", 16, 3, 1);
    }
    if stage == 5 {
        let small = c.g.interpolate3d_nearest(y, [d / 2; 3]);
        y = c.g.interpolate3d_nearest(small, [d; 3]);
    }
    if stage == 6 {
        let shape = Shape::new(&[1, 32, d, d, d], DType::F32);
        let cat = c.g.concat(vec![y, y], 1, shape);
        y = c.conv(cat, "c3", 16, 3, 1);
    }
    if stage == 7 {
        let small = c.g.interpolate3d_nearest(y, [d / 2; 3]);
        let up = c.g.interpolate3d_nearest(small, [d; 3]);
        let shape = Shape::new(&[1, 32, d, d, d], DType::F32);
        let cat = c.g.concat(vec![up, y], 1, shape);
        y = c.conv(cat, "c3", 16, 3, 1);
    }
    c.g.set_outputs(vec![y]);
    c
}

/// A 5^3 filter: past one K-tile, the general `conv3d_gemm_h`.
fn wide_filter() -> Case {
    let mut c = Case::new("f16_conv3d_5");
    let x = c.input("x", &[1, 8, 16, 16, 16]);
    let y = c.conv(x, "c", 16, 5, 1);
    c.g.set_outputs(vec![y]);
    c
}

/// Grouped, and a volume too small to tile: the scalar `conv3d_h`.
fn grouped() -> Case {
    let mut c = Case::new("f16_conv3d_grouped");
    let x = c.input("x", &[1, 8, 4, 4, 4]);
    let y = c.conv(x, "c", 8, 3, 2);
    let y = c.norm(y, "n", 2);
    c.g.set_outputs(vec![y]);
    c
}

fn check(label: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "{label}");
    let peak = want.iter().fold(0f32, |a, b| a.max(b.abs()));
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| {
            if a.is_finite() {
                (a - b).abs()
            } else {
                f32::INFINITY
            }
        })
        .fold(0f32, f32::max);
    let rel = worst / peak.max(1e-30);
    let bad = got
        .iter()
        .zip(want)
        .filter(|(a, b)| !((*a - *b).abs() <= 1e-3 * peak))
        .count();
    eprintln!(
        "{label}: {} values, rel {rel:.3e}, {bad} off by > 1e-3 of the range",
        got.len()
    );
    rel
}

#[test]
fn conv3d_networks_match_cpu() {
    const NAMES: [&str; 8] = [
        "conv",
        "+bias",
        "+norm",
        "+leaky",
        "down+conv",
        "down+up",
        "concat+conv",
        "up+concat+conv",
    ];
    let mut failed = Vec::new();
    for d in [16, 48, 96] {
        let cases = (0..8)
            .map(|s| (format!("{} {d}^3", NAMES[s]), piece(s, d)))
            .chain([(format!("decoder {d}^3"), decoder(d))]);
        for (label, case) in cases {
            let rel = check(&label, &case.run(Device::Cuda), &case.run(Device::Cpu));
            if !(rel < 1e-4) {
                failed.push(label);
            }
        }
    }
    for (label, case) in [("5^3 filter", wide_filter()), ("grouped 4^3", grouped())] {
        if !(check(label, &case.run(Device::Cuda), &case.run(Device::Cpu)) < 1e-4) {
            failed.push(label.to_string());
        }
    }
    assert!(failed.is_empty(), "CUDA differs from the CPU: {failed:?}");
}

/// Under `AlwaysF16` the 3-D convolutions run in f16 on tensor cores (half
/// input, filter and output, f32 accumulation): f16's precision, ~1e-3.
#[test]
fn conv3d_networks_f16_match_cpu() {
    let mut failed = Vec::new();
    for d in [16, 48] {
        for (label, case) in [
            (format!("decoder {d}^3"), decoder(d)),
            (format!("conv {d}^3"), piece(0, d)),
        ] {
            let f16 = case.run_with(
                Device::Cuda,
                CompileOptions::new().policy(PrecisionPolicy::AlwaysF16),
            );
            let rel = check(&format!("{label} f16"), &f16, &case.run(Device::Cpu));
            if !(rel < 1e-2) {
                failed.push(label);
            }
        }
    }
    assert!(failed.is_empty(), "f16 differs from the CPU: {failed:?}");
}
