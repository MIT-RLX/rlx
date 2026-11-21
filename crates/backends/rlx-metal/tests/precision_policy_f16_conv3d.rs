// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `PrecisionPolicy::AlwaysF16` on 3-D convolution networks, against a CPU
//! f32 reference.
//!
//! Metal keeps `Conv3d` and `GroupNorm` at f16 under the policy
//! (`AutoMixedPrecision::f16_conv3d`), running the half kernels:
//! `conv3d_gemm_c_h` (filters past 32 taps in 32-tap chunks), `conv3d_h` (grouped, or too small to tile), `group_norm_h` and
//! `binary_broadcast_h`. Each graph below reaches one of them, with the
//! epilogues and in-place reads the f32 kernels have: a bias, a LeakyReLU,
//! a channel concat and a nearest upsample read in place. An f16 result
//! must agree with f32 to f16's precision; a kernel reading half bytes as
//! float, or the reverse, gives garbage, which is what this pins.

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

    fn run(&self, device: Device, policy: Option<PrecisionPolicy>) -> Vec<f32> {
        let opts = match policy {
            Some(p) => CompileOptions::new().policy(p),
            None => CompileOptions::new(),
        };
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

/// The decoder's pieces one at a time, so a failure names its op. (The
/// downsampling `Interpolate3d` was one: a host op reading half as float.)
fn piece(stage: usize) -> Case {
    let d = 16;
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

/// Group norm then SiLU (folded into the norm's store), and a group norm
/// read twice, by a ReLU and a residual add (not folded: its own output is
/// live).
fn norm_epilogues() -> Case {
    let mut c = Case::new("f16_group_norm_epilogues");
    let x = c.input("x", &[1, 16, 24, 24, 24]);
    let y = c.norm(x, "n1", 4);
    let y = c.g.silu(y);
    let z = c.norm(y, "n2", 8);
    let r = c.g.relu(z);
    let z = c.g.add(r, z);
    c.g.set_outputs(vec![z]);
    c
}

/// Transposed convolutions: kernel 2 stride 2 (the GEMM kernel,
/// `conv_transpose3d_k2s2*`) into a norm, and kernel 3 stride 2 padded (the
/// gather kernel).
fn deconv(d: usize) -> Case {
    let mut c = Case::new("f16_conv_transpose3d");
    let x = c.input("x", &[1, 48, d, d, d]);
    let w = c.param("up", &[48, 24, 2, 2, 2], 48 * 8);
    let y =
        c.g.conv_transpose3d(x, w, [2; 3], [0; 3], [1; 3], [0; 3], 1);
    let y = c.norm(y, "n", 4);
    let w3 = c.param("up3", &[24, 8, 3, 3, 3], 24 * 27);
    let z =
        c.g.conv_transpose3d(y, w3, [2; 3], [1; 3], [1; 3], [1; 3], 1);
    c.g.set_outputs(vec![z]);
    c
}

/// SwinUNETR's window attention: q·kᵀ for every window and head (batched
/// matmul), a relative-position bias broadcast over windows, a shift mask
/// broadcast over heads, softmax, ·v. 49-token windows, so nothing is a
/// multiple of a tile.
fn window_attention() -> Case {
    let (w, h, t, d) = (8usize, 3usize, 49usize, 16usize);
    let mut c = Case::new("f16_window_attention");
    let q = c.input("q", &[w, h, t, d]);
    let k = c.input("k", &[w, h, d, t]);
    let v = c.input("v", &[w, h, t, d]);
    let bias = c.param("bias", &[1, h, t, t], 1);
    let mask = c.param("mask", &[1, w, 1, t, t], 1);
    let s = c.g.mm(q, k);
    let s = c.g.add(s, bias);
    let s =
        c.g.reshape_(s, vec![1, w as i64, h as i64, t as i64, t as i64]);
    let s = c.g.add(s, mask);
    let s =
        c.g.reshape_(s, vec![w as i64, h as i64, t as i64, t as i64]);
    let p = c.g.sm(s, -1);
    let y = c.g.mm(p, v);
    c.g.set_outputs(vec![y]);
    c
}

/// Concats: along the last axis, even widths (half copied as 4-byte words)
/// and an odd one (element copy), as SwinUNETR's patch merging; and along a
/// middle axis with one outer row (one contiguous block a segment).
fn concats() -> Case {
    let mut c = Case::new("f16_concats");
    let a = c.input("a", &[2, 5, 6, 8]);
    let b = c.input("b", &[2, 5, 6, 8]);
    let e = c.input("e", &[2, 5, 6, 3]);
    let ab =
        c.g.concat(vec![a, b], 3, Shape::new(&[2, 5, 6, 16], DType::F32));
    let abe =
        c.g.concat(vec![ab, e], 3, Shape::new(&[2, 5, 6, 19], DType::F32));
    let x = c.input("x", &[1, 3, 4, 4]);
    let y = c.input("y", &[1, 5, 4, 4]);
    let xy =
        c.g.concat(vec![x, y], 1, Shape::new(&[1, 8, 4, 4], DType::F32));
    let xy = c.g.reshape_(xy, vec![2, 1, 4, 16]);
    let xy = c.g.reshape_(xy, vec![128]);
    let abe = c.g.reshape_(abe, vec![2 * 5 * 6 * 19]);
    let out = c.g.concat(
        vec![abe, xy],
        0,
        Shape::new(&[2 * 5 * 6 * 19 + 128], DType::F32),
    );
    let out = c.g.relu(out);
    c.g.set_outputs(vec![out]);
    c
}

/// A filter past one 32-tap K-tile (5^3, 7^3): `conv3d_gemm_c_h` in chunks.
fn wide_filter(k: usize) -> Case {
    let mut c = Case::new("f16_conv3d_wide");
    let x = c.input("x", &[1, 8, 16, 16, 16]);
    let y = c.conv(x, "c", 16, k, 1);
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

/// A 2-D conv (kept f32) and a per-channel bias over few channels: a
/// segmentation head on a rank-4 tensor.
fn conv2d_head(c_out: usize) -> Case {
    let mut c = Case::new("f16_conv2d_head");
    let x = c.input("x", &[1, 16, 64, 48]);
    let w = c.param("w", &[c_out, 16, 3, 3], 16 * 9);
    let y = c.g.conv2d(x, w, [3, 3], [1, 1], [1, 1], [1, 1], 1);
    let b = c.param("b", &[c_out], 1);
    let b = c.g.reshape_(b, vec![1, c_out as i64, 1, 1]);
    let y = c.g.add(y, b);
    c.g.set_outputs(vec![y]);
    c
}

/// A channel argmax as a one-hot, `relu(1 - |argmax - k|)`: SynthSeg's
/// cascade between its U-Nets.
fn argmax_onehot() -> Case {
    let mut c = Case::new("f16_argmax_onehot");
    let x = c.input("x", &[1, 4, 8, 8, 8]);
    let y = c.conv(x, "c", 6, 3, 1);
    let idx =
        c.g.argmax(y, 1, true, Shape::new(&[1, 1, 8, 8, 8], DType::F32));
    let ids = c.param("ids", &[1, 6, 1, 1, 1], 1);
    c.params.last_mut().unwrap().1 = (0..6).map(|k| k as f32).collect();
    let d = c.g.sub(idx, ids);
    let d = c.g.abs(d);
    let one = c.g.constant(1.0, DType::F32);
    let d = c.g.sub(one, d);
    let y = c.g.relu(d);
    c.g.set_outputs(vec![y]);
    c
}

fn check(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}");
    let nonfinite = got.iter().filter(|v| !v.is_finite()).count();
    let peak = want.iter().fold(0f32, |a, b| a.max(b.abs()));
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let rel = worst / peak.max(1e-30);
    eprintln!(
        "{label}: {} values, max|Δ| {worst:.3e}, rel {rel:.3e}",
        got.len()
    );
    assert_eq!(nonfinite, 0, "{label}: {nonfinite} non-finite outputs");
    // f16 keeps ~3 decimal digits; through two norms and four convs the
    // error stays under 1% of the output's range. A wrong-width read is O(1).
    assert!(rel < 1e-2, "{label}: relative error {rel:.3e}");
}

#[test]
fn f16_conv3d_networks_match_f32() {
    let pieces = (0..8).map(|s| {
        (
            [
                "conv",
                "+bias",
                "+norm",
                "+leaky",
                "down+conv",
                "down+up",
                "concat+conv",
                "up+concat+conv",
            ][s],
            piece(s),
        )
    });
    for (label, case) in pieces.chain([
        ("decoder 16^3", decoder(16)),
        ("decoder 32^3", decoder(32)),
        ("norm epilogues", norm_epilogues()),
        ("window attention", window_attention()),
        ("concats", concats()),
        ("deconv 6^3", deconv(6)),
        ("deconv 12^3", deconv(12)),
        ("5^3 filter", wide_filter(5)),
        ("7^3 filter", wide_filter(7)),
        ("grouped 4^3", grouped()),
        ("conv2d head 2", conv2d_head(2)),
        ("conv2d head 16", conv2d_head(16)),
        ("argmax one-hot", argmax_onehot()),
    ]) {
        let want = case.run(Device::Cpu, None);
        let f32_metal = case.run(Device::Metal, None);
        check(&format!("{label} metal f32"), &f32_metal, &want);
        let f16 = case.run(Device::Metal, Some(PrecisionPolicy::AlwaysF16));
        check(&format!("{label} metal f16"), &f16, &want);
        let mixed = case.run(Device::Metal, Some(PrecisionPolicy::AutoMixed));
        check(&format!("{label} metal mixed"), &mixed, &want);
    }
}
