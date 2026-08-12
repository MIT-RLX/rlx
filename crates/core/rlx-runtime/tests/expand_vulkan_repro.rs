// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Isolates the wgpu-on-Vulkan `Pad(Replicate)` mismatch to a single primitive.
//!
//! Regression guard for two fixed wgpu-on-Vulkan bugs. `Op::Pad` is decomposed
//! for wgpu into `full`/`narrow`/`reverse`/`expand`/`concat`. On native Vulkan (Linux/NVIDIA) `pad_wgpu_matches_cpu` fails for
//! `dims=[2,3,4] pads=[[1,1],[1,2],[0,3]]` in `Replicate` mode — output element
//! `(1,2,4)` reads source `j=1` where it should read the clamped edge `j=3`.
//!
//! `Constant` and `Reflect` pass on that same shape, and `Reflect` shares
//! narrow/reverse/concat with `Replicate`. The one primitive only `Replicate`
//! uses is `expand` — broadcasting the size-1 edge slice across the pad width.
//! So these check `narrow` and `expand` on their own, in exactly the shapes the
//! decomposition produces, to say which stage is wrong rather than inferring it
//! from the composed result.

#![cfg(any(feature = "gpu", all(target_os = "macos", feature = "metal")))]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn x_data(n: usize) -> Vec<f32> {
    // Same generator as `pad_parity`, so indices map to recognisable values.
    (0..n).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect()
}

/// `narrow(x, axis, start, len)` — the edge slice `Replicate` copies.
fn run_narrow(device: Device, dims: &[usize], axis: usize, start: usize, len: usize) -> Vec<f32> {
    let n: usize = dims.iter().product();
    let mut g = Graph::new("narrow");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    let y = g.narrow_(inp, axis, start, len);
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x_data(n).as_slice())])
        .pop()
        .unwrap()
}

/// `expand(narrow(x, axis, start, 1), axis -> width)` — the broadcast that only
/// `Replicate` performs.
fn run_narrow_expand(
    device: Device,
    dims: &[usize],
    axis: usize,
    start: usize,
    width: usize,
) -> Vec<f32> {
    let n: usize = dims.iter().product();
    let mut g = Graph::new("narrow_expand");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    let edge = g.narrow_(inp, axis, start, 1);
    let mut out_dims: Vec<usize> = dims.to_vec();
    out_dims[axis] = width;
    let target: Vec<i64> = out_dims.iter().map(|&d| d as i64).collect();
    let shape = Shape::new(&out_dims, DType::F32);
    let y = g.add_node(
        Op::Expand {
            target_shape: target,
        },
        vec![edge],
        shape,
    );
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x_data(n).as_slice())])
        .pop()
        .unwrap()
}

fn device() -> Option<(Device, &'static str)> {
    #[cfg(feature = "gpu")]
    {
        return Some((Device::Gpu, "wgpu"));
    }
    #[cfg(all(target_os = "macos", feature = "metal", not(feature = "gpu")))]
    {
        return Some((Device::Metal, "metal"));
    }
    #[allow(unreachable_code)]
    None
}

/// The exact edge slice `Replicate` takes for the failing case: last index of
/// the trailing axis of `[2,3,4]`.
#[test]
fn narrow_edge_slice_matches_cpu() {
    let Some((dev, label)) = device() else {
        return;
    };
    let dims = [2usize, 3, 4];
    for (axis, start) in [(2usize, 3usize), (1, 2), (0, 1)] {
        assert_eq!(
            run_narrow(dev, &dims, axis, start, 1),
            run_narrow(Device::Cpu, &dims, axis, start, 1),
            "{label} narrow dims={dims:?} axis={axis} start={start} len=1"
        );
    }
}

/// Broadcasting that edge across the pad width — the step unique to `Replicate`.
#[test]
fn expand_of_edge_slice_matches_cpu() {
    let Some((dev, label)) = device() else {
        return;
    };
    let dims = [2usize, 3, 4];
    // Widths are the `post` pads from the failing case.
    for (axis, start, width) in [(2usize, 3usize, 3usize), (1, 2, 2), (0, 1, 1)] {
        assert_eq!(
            run_narrow_expand(dev, &dims, axis, start, width),
            run_narrow_expand(Device::Cpu, &dims, axis, start, width),
            "{label} expand(narrow) dims={dims:?} axis={axis} start={start} width={width}"
        );
    }
}

/// Which combination of padded axes triggers the mismatch.
///
/// `narrow` and `expand` are correct in isolation (above), so the fault is in
/// the composition: `lower_pad` pads one axis at a time, and every axis after
/// the first operates on a tensor that is itself the output of a `concat`.
/// Padding axes one at a time isolates whether a single axis is wrong or only
/// the stacked case is.
#[test]
fn replicate_pad_axis_combinations_match_cpu() {
    use rlx_ir::PadMode;
    let Some((dev, label)) = device() else {
        return;
    };
    let dims = [2usize, 3, 4];
    let n: usize = dims.iter().product();
    let x = x_data(n);

    let run_pad = |device: Device, pads: Vec<[usize; 2]>| -> Vec<f32> {
        let mut g = Graph::new("pad");
        let inp = g.input("x", Shape::new(&dims, DType::F32));
        let y = g.pad_(inp, pads, PadMode::Replicate);
        g.set_outputs(vec![y]);
        Session::new(device)
            .compile(g)
            .run(&[("x", x.as_slice())])
            .pop()
            .unwrap()
    };

    let cases: Vec<(&str, Vec<[usize; 2]>)> = vec![
        ("axis2 only", vec![[0, 0], [0, 0], [0, 3]]),
        ("axis1 only", vec![[0, 0], [1, 2], [0, 0]]),
        ("axis0 only", vec![[1, 1], [0, 0], [0, 0]]),
        ("axis1+2", vec![[0, 0], [1, 2], [0, 3]]),
        ("axis0+2", vec![[1, 1], [0, 0], [0, 3]]),
        ("axis0+1", vec![[1, 1], [1, 2], [0, 0]]),
        ("all three", vec![[1, 1], [1, 2], [0, 3]]),
    ];

    let mut failed = Vec::new();
    for (name, pads) in cases {
        let got = run_pad(dev, pads.clone());
        let want = run_pad(Device::Cpu, pads.clone());
        if got != want {
            let first = got
                .iter()
                .zip(&want)
                .position(|(a, b)| a != b)
                .unwrap_or(usize::MAX);
            failed.push(format!("{name} (first diff at flat {first})"));
        }
    }
    assert!(
        failed.is_empty(),
        "{label} Replicate mismatches: {}",
        failed.join(", ")
    );
}
