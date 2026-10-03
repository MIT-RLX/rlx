// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Depthwise (grouped) conv backward w.r.t. the weight, CPU vs Metal.
//!
//! At `N == 1` this used to take a per-group im2col+GEMM path whose answer
//! depended on what happened to be in the arena: a one-shot run agreed with
//! finite differences, but after other compiles had dirtied memory the same
//! shapes came back 30-50% off. That is how the Qwen3.5 `ssm_conv1d` weight
//! gradient broke on Metal while every other gradient was exact. `N == 1` now
//! takes the same two-pass kernels as `N > 1`, which write every element they
//! read back.
//!
//! The sweep runs many shapes in ONE process on purpose — that history is what
//! made the old path fail, and a single-shape test passed right through it.
use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn run(device: Device, n: usize, c: usize, h: usize, k: usize, groups: usize) -> Vec<Vec<f32>> {
    let f = DType::F32;
    let mut g = Graph::new("dwconv");
    let x = g.input("x", Shape::new(&[n, c, h, 1], f));
    let w = g.input("w", Shape::new(&[c, c / groups, k, 1], f));
    let y = g.add_node(
        Op::Conv {
            kernel_size: vec![k, 1],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups,
        },
        vec![x, w],
        Shape::new(&[n, c, h - k + 1, 1], f),
    );
    let flat = g.reshape_(y, vec![-1]);
    let loss = g.mean(flat, vec![0], false);
    g.set_outputs(vec![loss]);
    let bwd = grad_with_loss_wrt(
        &g,
        &[Wrt::Leaf("x".into()), Wrt::Leaf("w".into())],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let xv: Vec<f32> = (0..n * c * h)
        .map(|i| 0.2 * ((i % 13) as f32 - 6.0))
        .collect();
    let wv: Vec<f32> = (0..c * (c / groups) * k)
        .map(|i| 0.1 * ((i % 7) as f32 - 3.0))
        .collect();
    let mut s = Session::new(device).compile(bwd);
    s.run(&[("x", &xv[..]), ("w", &wv[..]), ("d_output", &[1.0f32][..])])
}

/// The GDN shape: the conv's operands are `Reshape(Transpose(...))` views, and
/// the cotangent arrives through a transpose too.
fn run_views(device: Device, c: usize, h: usize, k: usize) -> Vec<Vec<f32>> {
    let f = DType::F32;
    let mut g = Graph::new("dwconv_views");
    // [1, H, C] -> transpose -> [1, C, H] -> reshape -> [1, C, H, 1]
    let xr = g.input("x", Shape::new(&[1, h, c], f));
    let xt = g.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![xr],
        Shape::new(&[1, c, h], f),
    );
    let x = g.reshape_(xt, vec![1, c as i64, h as i64, 1]);
    let w = g.input("w", Shape::new(&[c, 1, k, 1], f));
    let ho = h - k + 1;
    let y = g.add_node(
        Op::Conv {
            kernel_size: vec![k, 1],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: c,
        },
        vec![x, w],
        Shape::new(&[1, c, ho, 1], f),
    );
    // Back out through reshape + transpose, so the cotangent reaching the conv
    // is itself a `Reshape(Transpose(..))`.
    let y3 = g.reshape_(y, vec![1, c as i64, ho as i64]);
    let yt = g.add_node(
        Op::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![y3],
        Shape::new(&[1, ho, c], f),
    );
    let act = g.activation(rlx_ir::op::Activation::Silu, yt, Shape::new(&[1, ho, c], f));
    let flat = g.reshape_(act, vec![-1]);
    let loss = g.mean(flat, vec![0], false);
    g.set_outputs(vec![loss]);
    let bwd = grad_with_loss_wrt(
        &g,
        &[Wrt::Leaf("x".into()), Wrt::Leaf("w".into())],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let xv: Vec<f32> = (0..h * c).map(|i| 0.2 * ((i % 13) as f32 - 6.0)).collect();
    let wv: Vec<f32> = (0..c * k).map(|i| 0.1 * ((i % 7) as f32 - 3.0)).collect();
    let mut s = Session::new(device).compile(bwd);
    s.run(&[("x", &xv[..]), ("w", &wv[..]), ("d_output", &[1.0f32][..])])
}

#[test]
fn depthwise_conv_backward_weight_through_views_matches_cpu() {
    let mut bad = Vec::new();
    let mut cs: Vec<usize> = vec![];
    for c in [4usize, 8, 12, 16, 20, 24, 32, 48, 64] {
        cs.push(c);
    }
    let mut combos: Vec<(usize, usize, usize)> = vec![];
    for c in cs {
        for h in [7usize, 10, 20, 67] {
            combos.push((c, h, 4));
        }
    }
    for (c, h, k) in combos {
        let cpu = run_views(Device::Cpu, c, h, k);
        let met = run_views(Device::Metal, c, h, k);
        for (which, i) in [("dx", 1usize), ("dw", 2)] {
            let (a, b) = (&cpu[i], &met[i]);
            let err = a
                .iter()
                .zip(b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            let sc = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
            let flag = b.iter().any(|v| !v.is_finite()) || err / sc > 1e-3;
            println!(
                "  views C{c} H{h} K{k} {which}: rel={:.2e} |cpu|max={sc:.3e}{}",
                err / sc,
                if flag { "  <== BAD" } else { "" }
            );
            if flag {
                bad.push(format!("C{c}H{h}K{k}/{which}"));
                if std::env::var_os("RLX_SHOW_VALS").is_some() {
                    println!("      cpu  {:?}", &a[..a.len().min(12)]);
                    println!("      metal{:?}", &b[..b.len().min(12)]);
                    let per_group: Vec<usize> = (0..c)
                        .filter(|g| {
                            let lo = g * k;
                            (lo..lo + k).any(|t| (a[t] - b[t]).abs() > 1e-4 * sc)
                        })
                        .collect();
                    println!("      groups wrong: {:?} of {c}", per_group);
                }
            }
        }
    }
    println!("BAD: {bad:?}");
    assert!(
        bad.is_empty(),
        "metal conv backward through views disagrees"
    );
}

#[test]
fn depthwise_conv_backward_weight_matches_cpu() {
    let mut bad = Vec::new();
    for (n, c, h, k, groups) in [
        (1usize, 24usize, 7usize, 4usize, 24usize),
        (1, 24, 67, 4, 24),
        (1, 8, 10, 4, 8),
        (1, 8, 10, 4, 1),
        (1, 8, 10, 2, 4),
    ] {
        let cpu = run(Device::Cpu, n, c, h, k, groups);
        let met = run(Device::Metal, n, c, h, k, groups);
        for (which, i) in [("dx", 1usize), ("dw", 2)] {
            let (a, b) = (&cpu[i], &met[i]);
            let err = a
                .iter()
                .zip(b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            let sc = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-12);
            let flag = b.iter().any(|v| !v.is_finite()) || err / sc > 1e-3;
            println!(
                "  N{n} C{c} H{h} K{k} G{groups} {which}: rel={:.2e} |cpu|max={sc:.3e} n={}{}",
                err / sc,
                a.len(),
                if flag { "  <== BAD" } else { "" }
            );
            if flag {
                bad.push(format!("C{c}H{h}K{k}G{groups}/{which}"));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "metal grouped conv backward disagrees: {bad:#?}"
    );
}

/// Who is right? Finite differences over the same graph decide it.
#[test]
fn fd_says_which_device_is_right() {
    let (c, h, k) = (8usize, 10usize, 4usize);
    let f = DType::F32;
    let build_loss = |wv: &[f32], xv: &[f32], device: Device| -> f32 {
        let mut g = Graph::new("fwd");
        let xr = g.input("x", Shape::new(&[1, h, c], f));
        let xt = g.add_node(
            Op::Transpose {
                perm: vec![0, 2, 1],
            },
            vec![xr],
            Shape::new(&[1, c, h], f),
        );
        let x = g.reshape_(xt, vec![1, c as i64, h as i64, 1]);
        let w = g.input("w", Shape::new(&[c, 1, k, 1], f));
        let ho = h - k + 1;
        let y = g.add_node(
            Op::Conv {
                kernel_size: vec![k, 1],
                stride: vec![1, 1],
                padding: vec![0, 0],
                dilation: vec![1, 1],
                groups: c,
            },
            vec![x, w],
            Shape::new(&[1, c, ho, 1], f),
        );
        let y3 = g.reshape_(y, vec![1, c as i64, ho as i64]);
        let yt = g.add_node(
            Op::Transpose {
                perm: vec![0, 2, 1],
            },
            vec![y3],
            Shape::new(&[1, ho, c], f),
        );
        let act = g.activation(rlx_ir::op::Activation::Silu, yt, Shape::new(&[1, ho, c], f));
        let flat = g.reshape_(act, vec![-1]);
        let loss = g.mean(flat, vec![0], false);
        g.set_outputs(vec![loss]);
        let mut s = Session::new(device).compile(g);
        s.run(&[("x", xv), ("w", wv)])[0][0]
    };
    let xv: Vec<f32> = (0..h * c).map(|i| 0.2 * ((i % 13) as f32 - 6.0)).collect();
    let wv: Vec<f32> = (0..c * k).map(|i| 0.1 * ((i % 7) as f32 - 3.0)).collect();
    let cpu = run_views(Device::Cpu, c, h, k);
    let met = run_views(Device::Metal, c, h, k);
    let eps = 1e-3f32;
    println!("  idx      fd        cpu_dw     metal_dw");
    let mut cpu_err = 0f32;
    let mut met_err = 0f32;
    for idx in [0usize, 1, 2, 3, 4, 7, 12, 20] {
        let mut wp = wv.clone();
        wp[idx] += eps;
        let mut wm = wv.clone();
        wm[idx] -= eps;
        // Evaluate the loss on the CPU: the forward is bit-identical across
        // devices here, so this is a device-neutral oracle.
        let fd =
            (build_loss(&wp, &xv, Device::Cpu) - build_loss(&wm, &xv, Device::Cpu)) / (2.0 * eps);
        println!(
            "  {idx:>3}  {fd:>11.3e}  {:>11.3e}  {:>11.3e}",
            cpu[2][idx], met[2][idx]
        );
        cpu_err = cpu_err.max((fd - cpu[2][idx]).abs());
        met_err = met_err.max((fd - met[2][idx]).abs());
    }
    println!("  max |fd - cpu| = {cpu_err:.3e}   max |fd - metal| = {met_err:.3e}");
    assert!(
        cpu_err < 1e-4,
        "the CPU gradient disagrees with finite differences by {cpu_err:.3e} — \
         the oracle says CPU is the wrong one"
    );
}
