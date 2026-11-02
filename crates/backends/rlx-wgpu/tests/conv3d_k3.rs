// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **`conv3d_k3_c8`**: a 2-D 3×3 conv over each slice of the depth (kernel
//! `[1, 3, 3]`, stride 1, padding `[0, 1, 1]`), its weights in workgroup
//! memory. Against a direct sum: widths that leave a thread's four positions
//! short or the row one position wide, heights of one, input channels past
//! and short of the eight a chunk holds, output channels short of a tile,
//! a bias and an activation in the store. Against the eight-channel tile
//! (`RLX_WGPU_CONV3D_NO_K3=1`): a stack with its activations half-packed
//! (`RLX_WGPU_HALF_ACTIVATIONS=1`) and the same striped (the two-window
//! kernel). A PReLU folded into the loads and a `max` into the store
//! (`CONV3D_K3`) give what the separate passes give. Process-global
//! switches: one test.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// the conv, directly: `x` `[n, c_in, d, h, w]`, `w` `[c_out, c_in, 1, 3, 3]`
#[allow(clippy::too_many_arguments)]
fn reference(
    n: usize,
    c_in: usize,
    c_out: usize,
    [d, h, wd]: [usize; 3],
    x: &[f32],
    w: &[f32],
    b: &[f32],
    relu: bool,
) -> Vec<f32> {
    let plane = h * wd;
    let mut out = vec![0f32; n * c_out * d * plane];
    for nn in 0..n {
        for co in 0..c_out {
            for z in 0..d {
                for y in 0..h {
                    for xx in 0..wd {
                        let mut s = 0f64;
                        for ci in 0..c_in {
                            for ky in 0..3 {
                                for kx in 0..3 {
                                    let (iy, ix) = (y + ky, xx + kx);
                                    if iy < 1 || ix < 1 || iy > h || ix > wd {
                                        continue;
                                    }
                                    let v =
                                        x[((nn * c_in + ci) * d + z) * plane + (iy - 1) * wd + ix
                                            - 1];
                                    s += f64::from(v)
                                        * f64::from(w[(co * c_in + ci) * 9 + ky * 3 + kx]);
                                }
                            }
                        }
                        let v = s as f32 + b[co];
                        out[((nn * c_out + co) * d + z) * plane + y * wd + xx] =
                            if relu { v.max(0.0) } else { v };
                    }
                }
            }
        }
    }
    out
}

fn ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 2654435761 + seed * 40503) % 1013) as f32 / 1013.0) * 2.0 - 1.0)
        .collect()
}

fn sh(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

/// `FusedConvBiasAct` with kernel `[1, 3, 3]`, padding `[0, 1, 1]`
fn conv(
    g: &mut Graph,
    x: rlx_ir::NodeId,
    w: rlx_ir::NodeId,
    b: rlx_ir::NodeId,
    out: &[usize],
    relu: bool,
) -> rlx_ir::NodeId {
    g.add_node(
        Op::FusedConvBiasAct {
            kernel_size: vec![1, 3, 3],
            stride: vec![1; 3],
            padding: vec![0, 1, 1],
            dilation: vec![1; 3],
            groups: 1,
            activation: relu.then_some(rlx_ir::op::Activation::Relu),
            has_residual: false,
        },
        vec![x, w, b],
        sh(out),
    )
}

/// one conv, `[n, c_in, d, h, w]` → `c_out`, against the CPU
fn case(n: usize, c_in: usize, c_out: usize, [d, h, w]: [usize; 3], relu: bool) {
    let x = ramp(n * c_in * d * h * w, 3);
    let wt = ramp(c_out * c_in * 9, 7);
    let b: Vec<f32> = (0..c_out).map(|i| 0.25 * i as f32 - 0.5).collect();
    let build = || {
        let mut g = Graph::new("k3");
        let xi = g.input("x", sh(&[n, c_in, d, h, w]));
        let wi = g.input("w", sh(&[c_out, c_in, 1, 3, 3]));
        let bi = g.input("b", sh(&[c_out]));
        let y = conv(&mut g, xi, wi, bi, &[n, c_out, d, h, w], relu);
        g.set_outputs(vec![y]);
        g
    };
    let ins: [(&str, &[f32]); 3] = [("x", &x), ("w", &wt), ("b", &b)];
    let want = reference(n, c_in, c_out, [d, h, w], &x, &wt, &b, relu);
    let got = WgpuExecutable::compile(build()).run(&ins).remove(0);
    let tol = 1e-4 * (c_in * 9) as f32;
    let label = format!("n{n} c {c_in}->{c_out} [{d}, {h}, {w}] relu {relu}");
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (a, e)) in got.iter().zip(&want).enumerate() {
        assert!((a - e).abs() <= tol, "{label} [{i}]: wgpu {a} cpu {e}");
    }
}

#[test]
fn k3_matches_the_cpu_and_the_eight_channel_tile() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // widths short of a thread's four, a row of one, heights of one
    case(1, 3, 5, [2, 4, 7], false);
    case(1, 2, 8, [3, 1, 1], false);
    case(1, 4, 6, [1, 5, 2], true);
    case(1, 9, 9, [2, 3, 5], false);
    // input channels past one chunk of eight and short of the next; output
    // channels a tile and one, FastSurfer's 71 (one short of nine tiles)
    case(1, 17, 17, [3, 6, 9], true);
    case(1, 7, 71, [2, 8, 12], false);
    case(2, 71, 13, [2, 5, 6], false);
    // past one workgroup of 64 threads: positions of a channel tile spread
    // over workgroups, the channel tiles innermost
    case(1, 8, 24, [4, 16, 40], false);

    // a stack of three, half-packed and striped, against the eight-channel tile
    let (c, dims) = (10, [8, 48, 60]);
    let vol = [1, c, dims[0], dims[1], dims[2]];
    let build = || {
        let mut g = Graph::new("k3_stack");
        let mut y = g.input("x", sh(&[1, 1, dims[0], dims[1], dims[2]]));
        for l in 0..3 {
            let w = g.input(
                format!("w{l}"),
                sh(&[c, if l == 0 { 1 } else { c }, 1, 3, 3]),
            );
            let b = g.input(format!("b{l}"), sh(&[c]));
            y = conv(&mut g, y, w, b, &vol, true);
        }
        g.set_outputs(vec![y]);
        g
    };
    let mut data: Vec<(String, Vec<f32>)> = vec![("x".into(), ramp(dims.iter().product(), 1))];
    for l in 0..3 {
        data.push((
            format!("w{l}"),
            ramp(c * if l == 0 { 1 } else { c } * 9, 2 + l),
        ));
        data.push((format!("b{l}"), ramp(c, 7 + l)));
    }
    let ins: Vec<(&str, &[f32])> = data
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let run = || WgpuExecutable::compile(build()).run(&ins).remove(0);
    let close = |a: &[f32], b: &[f32], tol: f32, what: &str| {
        assert_eq!(a.len(), b.len(), "{what}");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= tol * (1.0 + y.abs()),
                "{what} [{i}]: {x} vs {y}"
            );
        }
    };
    let mut want = data[0].1.clone();
    for l in 0..3 {
        let c_in = if l == 0 { 1 } else { c };
        want = reference(
            1,
            c_in,
            c,
            dims,
            &want,
            &data[1 + 2 * l].1,
            &data[2 + 2 * l].1,
            true,
        );
    }
    let k3 = run();
    close(&k3, &want, 1e-4, "k3 vs direct");
    rlx_ir::env::set("RLX_WGPU_HALF_ACTIVATIONS", "1");
    let half = run();
    rlx_ir::env::set("RLX_WGPU_CONV3D_NO_K3", "1");
    let half_c8 = run();
    rlx_ir::env::set("RLX_WGPU_CONV3D_NO_K3", "0");
    // three layers each rounded to halves: both tiles ~2% off the f32 sum
    // here, and ~0.3% off each other (the same halves in, sums in another
    // order, a rounding apart now and then)
    close(&half, &half_c8, 5e-3, "half k3 vs half c8");
    close(&half, &want, 3e-2, "half k3 vs f32");
    // ~0.9 MiB f32 tensors (0.45 packed) in 2 MiB stripes: two-window convs
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "2");
    rlx_ir::env::set("RLX_WGPU_SHARD_STAGE_MIB", "1");
    rlx_ir::env::set("RLX_WGPU_ALLOW_SHARD", "1");
    let striped_half = run();
    rlx_ir::env::set("RLX_WGPU_HALF_ACTIVATIONS", "0");
    let striped = run();
    rlx_ir::env::clear_overrides();
    assert_eq!(
        striped_half, half,
        "striped half-packed = whole half-packed"
    );
    assert_eq!(striped, k3, "striped = whole");

    // a PReLU read by 3×3 convs only is folded into their loads
    // (`CONV3D_K3`): the same values as the PReLU pass; one also read
    // by something else stays
    for (alpha, shared) in [(0.25f32, false), (-0.5, false), (1.5, false), (0.25, true)] {
        let (c, dims) = (9, [3, 10, 14]);
        let vol = [1, c, dims[0], dims[1], dims[2]];
        let build = || {
            let mut g = Graph::new("k3_prelu");
            let x = g.input("x", sh(&[1, 2, dims[0], dims[1], dims[2]]));
            let w0 = g.input("w0", sh(&[c, 2, 1, 3, 3]));
            let b0 = g.input("b0", sh(&[c]));
            let y = conv(&mut g, x, w0, b0, &vol, false);
            let p = g.add_node(
                Op::Custom {
                    name: "rlx_wgpu.prelu".into(),
                    num_inputs: 1,
                    attrs: alpha.to_bits().to_le_bytes().to_vec(),
                },
                vec![y],
                sh(&vol),
            );
            let mut outs = Vec::new();
            for l in 1..3 {
                let w = g.input(format!("w{l}"), sh(&[c, c, 1, 3, 3]));
                let b = g.input(format!("b{l}"), sh(&[c]));
                outs.push(conv(&mut g, p, w, b, &vol, l == 2));
            }
            let mut z = g.binary(rlx_ir::op::BinaryOp::Add, outs[0], outs[1], sh(&vol));
            if shared {
                z = g.binary(rlx_ir::op::BinaryOp::Add, z, p, sh(&vol));
            }
            g.set_outputs(vec![z]);
            g
        };
        let mut data: Vec<(String, Vec<f32>)> = vec![
            ("x".into(), ramp(2 * dims.iter().product::<usize>(), 4)),
            ("w0".into(), ramp(c * 2 * 9, 5)),
            ("b0".into(), ramp(c, 6)),
        ];
        for l in 1..3 {
            data.push((format!("w{l}"), ramp(c * c * 9, 8 + l)));
            data.push((format!("b{l}"), ramp(c, 10 + l)));
        }
        let ins: Vec<(&str, &[f32])> = data
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        let folded = WgpuExecutable::compile(build()).run(&ins).remove(0);
        rlx_ir::env::set("RLX_WGPU_NO_FOLD_CONV_PRELU", "1");
        let plain = WgpuExecutable::compile(build()).run(&ins).remove(0);
        rlx_ir::env::clear_overrides();
        assert_eq!(
            folded, plain,
            "α {alpha} shared {shared}: folded = PReLU pass"
        );
        // and the sum, directly
        let y = reference(1, 2, c, dims, &data[0].1, &data[1].1, &data[2].1, false);
        let p: Vec<f32> = y
            .iter()
            .map(|&v| if v >= 0.0 { v } else { alpha * v })
            .collect();
        let z1 = reference(1, c, c, dims, &p, &data[3].1, &data[4].1, false);
        let z2 = reference(1, c, c, dims, &p, &data[5].1, &data[6].1, true);
        let want: Vec<f32> = (0..p.len())
            .map(|i| z1[i] + z2[i] + if shared { p[i] } else { 0.0 })
            .collect();
        close(
            &folded,
            &want,
            1e-4,
            &format!("α {alpha} shared {shared} vs direct"),
        );
    }

    // FastSurfer's competitive dense block: prelu → conv → max with what came
    // before, three times; then a `max` whose other operand is made after
    // the conv, and one against a broadcast scalar (not folded)
    let c = 12;
    let a = 0.3f32;
    let build = |dims: [usize; 3]| {
        let vol = [1, c, dims[0], dims[1], dims[2]];
        let mut g = Graph::new("k3_dense");
        let x = g.input("x", sh(&vol));
        let mut prev = x;
        for l in 0..3 {
            let w = g.input(format!("w{l}"), sh(&[c, c, 1, 3, 3]));
            let b = g.input(format!("b{l}"), sh(&[c]));
            let p = g.prelu_scalar(prev, a);
            let y = conv(&mut g, p, w, b, &vol, false);
            prev = if l == 1 {
                g.binary(rlx_ir::op::BinaryOp::Max, prev, y, sh(&vol))
            } else {
                g.binary(rlx_ir::op::BinaryOp::Max, y, prev, sh(&vol))
            };
        }
        let w = g.input("w3", sh(&[c, c, 1, 3, 3]));
        let b = g.input("b3", sh(&[c]));
        let y = conv(&mut g, prev, w, b, &vol, false);
        let late = g.activation(rlx_ir::op::Activation::Abs, prev, sh(&vol));
        let y = g.binary(rlx_ir::op::BinaryOp::Max, y, late, sh(&vol));
        let w = g.input("w4", sh(&[c, c, 1, 3, 3]));
        let b = g.input("b4", sh(&[c]));
        let y = conv(&mut g, y, w, b, &vol, false);
        let half = g.add_node(
            Op::Constant {
                data: 0.5f32.to_le_bytes().to_vec(),
            },
            vec![],
            sh(&[1]),
        );
        let y = g.binary(rlx_ir::op::BinaryOp::Max, y, half, sh(&vol));
        g.set_outputs(vec![y]);
        g
    };
    let inputs = |dims: [usize; 3]| {
        let n: usize = dims.iter().product::<usize>() * c;
        let mut data: Vec<(String, Vec<f32>)> = vec![("x".into(), ramp(n, 21))];
        for l in 0..5 {
            data.push((
                format!("w{l}"),
                ramp(c * c * 9, 22 + l).iter().map(|v| v * 0.3).collect(),
            ));
            data.push((format!("b{l}"), ramp(c, 30 + l)));
        }
        data
    };
    let count = |g: &Graph| {
        let u = rlx_wgpu::unfuse::unfuse(g.clone());
        let maxes = u
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Binary(rlx_ir::op::BinaryOp::Max)))
            .count();
        let folded = u
            .nodes()
            .iter()
            .filter(|n| rlx_wgpu::unfuse::conv3d_max_residual(&n.op))
            .count();
        (maxes, folded)
    };
    let dims = [2, 9, 13];
    let data = inputs(dims);
    let ins: Vec<(&str, &[f32])> = data
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    // the three dense maxes and the late one fold; the broadcast one stays
    // (each PReLU's own `max(x, 0)` is gone into its fold)
    assert_eq!(count(&build(dims)), (1, 4), "maxes left, maxes folded");
    let folded = WgpuExecutable::compile(build(dims)).run(&ins).remove(0);
    rlx_ir::env::set("RLX_WGPU_NO_FOLD_CONV_MAX", "1");
    assert_eq!(count(&build(dims)).1, 0, "no fold");
    let plain = WgpuExecutable::compile(build(dims)).run(&ins).remove(0);
    rlx_ir::env::clear_overrides();
    assert_eq!(folded, plain, "max folded = max pass");
    // directly
    let mut prev = data[0].1.clone();
    for l in 0..3 {
        let p: Vec<f32> = prev
            .iter()
            .map(|&v| if v >= 0.0 { v } else { a * v })
            .collect();
        let y = reference(
            1,
            c,
            c,
            dims,
            &p,
            &data[1 + 2 * l].1,
            &data[2 + 2 * l].1,
            false,
        );
        prev = y.iter().zip(&prev).map(|(y, p)| y.max(*p)).collect();
    }
    let y = reference(1, c, c, dims, &prev, &data[7].1, &data[8].1, false);
    let y: Vec<f32> = y.iter().zip(&prev).map(|(y, p)| y.max(p.abs())).collect();
    let y = reference(1, c, c, dims, &y, &data[9].1, &data[10].1, false);
    let want: Vec<f32> = y.iter().map(|v| v.max(0.5)).collect();
    close(&folded, &want, 1e-4, "dense block vs direct");

    // past one binding: compiled again without the folds — the same values
    // but for rounding (the 1.1 MB tensors outgrow a 2 MiB stripe's usable
    // part, so the convs run another way). It caught a two-window conv
    // staging its weights over the end of its own input: such a tensor gets
    // a stripe of its own, reserve and all (`two_window_ok`; up to 4.8 off)
    let dims = [8, 48, 60];
    let data = inputs(dims);
    let ins: Vec<(&str, &[f32])> = data
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let whole = WgpuExecutable::compile(build(dims)).run(&ins).remove(0);
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "2");
    rlx_ir::env::set("RLX_WGPU_SHARD_STAGE_MIB", "1");
    rlx_ir::env::set("RLX_WGPU_ALLOW_SHARD", "1");
    let striped = WgpuExecutable::compile(build(dims)).run(&ins).remove(0);
    rlx_ir::env::clear_overrides();
    close(&striped, &whole, 1e-4, "striped dense block vs whole");
}
