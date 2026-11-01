// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Half-packed activations** (`RLX_WGPU_HALF_ACTIVATIONS=1`,
//! `unfuse::pack_half_activations`): a MeshNet-shaped stack — convs with
//! folded activations, an instance norm on a reshaped volume, the fused
//! argmax head — stores its conv and norm activations as f16, two to an
//! arena word. The arena shrinks; the features stay within f16 rounding of
//! the f32 run and the labels nearly all agree; striped (two-window
//! kernels) gives exactly what the packed whole run gives, and one-workgroup
//! norms (another order of sums) come within a half's rounding. Process-global switches: one test.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

fn ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + seed * 13) % 23) as f32 - 11.0) / 17.0)
        .collect()
}

#[test]
fn packed_halves_match_f32_closely_and_shrink_the_arena() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let (c, dims) = (6, [24, 40, 40]);
    let per: usize = dims.iter().product();
    let sh = |d: &[usize]| Shape::new(d, DType::F32);
    let vol = sh(&[1, c, dims[0], dims[1], dims[2]]);
    // one channel in, as a MeshNet's scan
    let scan = sh(&[1, 1, dims[0], dims[1], dims[2]]);
    let flat = sh(&[1, c, per, 1]);
    let build_with = |features: bool| {
        let mut g = Graph::new("half");
        let x = g.input("x", scan.clone());
        let mut y = x;
        for l in 0..4 {
            let w = g.input(
                format!("w{l}"),
                sh(&[c, if l == 0 { 1 } else { c }, 3, 3, 3]),
            );
            let b = g.input(format!("b{l}"), sh(&[c]));
            y = g.add_node(
                Op::FusedConvBiasAct {
                    kernel_size: vec![3; 3],
                    stride: vec![1; 3],
                    padding: vec![l + 1; 3],
                    dilation: vec![l + 1; 3],
                    groups: 1,
                    activation: None,
                    has_residual: false,
                },
                vec![y, w, b],
                vol.clone(),
            );
            if l % 2 == 1 {
                let r = g.reshape(y, vec![1, c as i64, per as i64, 1], flat.clone());
                let (gm, bt) = (
                    g.input(format!("gamma{l}"), sh(&[c])),
                    g.input(format!("beta{l}"), sh(&[c])),
                );
                let n = g.group_norm(r, gm, bt, c, 1e-5);
                y = g.reshape(
                    n,
                    vec![1, c as i64, dims[0] as i64, dims[1] as i64, dims[2] as i64],
                    vol.clone(),
                );
            }
            y = g.activation(rlx_ir::op::Activation::Gelu, y, vol.clone());
        }
        let wh = g.input("wh", sh(&[c, c, 1, 1, 1]));
        let z = g.conv3d(y, wh, [1; 3], [0; 3], [1; 3], 1);
        let l = g.argmax(z, 1, false, sh(&[1, dims[0], dims[1], dims[2]]));
        // the last features too (an output, so they stay f32)
        g.set_outputs(if features { vec![l, y] } else { vec![l] });
        g
    };
    let mut data: Vec<(String, Vec<f32>)> = vec![("x".into(), ramp(per, 1))];
    for l in 0..4 {
        data.push((
            format!("w{l}"),
            ramp(c * if l == 0 { 1 } else { c } * 27, 2 + l),
        ));
        data.push((format!("b{l}"), ramp(c, 7 + l)));
        data.push((
            format!("gamma{l}"),
            (0..c).map(|i| 1.0 + 0.1 * i as f32).collect(),
        ));
        data.push((format!("beta{l}"), ramp(c, 11 + l)));
    }
    data.push(("wh".into(), ramp(c * c, 12)));
    let build = || build_with(true);
    let ins: Vec<(&str, &[f32])> = data
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let run = || WgpuExecutable::compile(build()).run(&ins);

    let full = run();
    let full_bytes = rlx_wgpu::planned_arena_bytes(build_with(false));
    rlx_ir::env::set("RLX_WGPU_HALF_ACTIVATIONS", "1");
    let half = run();
    let half_bytes = rlx_wgpu::planned_arena_bytes(build_with(false));
    rlx_ir::env::set("RLX_WGPU_GROUP_NORM_ONE_WG", "1");
    let one_wg = run();
    rlx_ir::env::set("RLX_WGPU_GROUP_NORM_ONE_WG", "0");
    // ~0.9 MiB f32 tensors (0.45 packed) in 2 MiB stripes, 1 MiB staging
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "2");
    rlx_ir::env::set("RLX_WGPU_SHARD_STAGE_MIB", "1");
    rlx_ir::env::set("RLX_WGPU_ALLOW_SHARD", "1");
    let striped = run();
    assert_eq!(striped, run(), "striped runs agree");
    assert!(
        !WgpuExecutable::compile(build()).blocks_on_host(),
        "striped graph blocks on the host"
    );
    rlx_ir::env::clear_overrides();

    // the activations (not the 64 MiB scratch) about halve, over the labels
    // alone (an output's producers stay f32)
    let scratch = 64usize << 20;
    eprintln!("  arena {full_bytes} → {half_bytes} bytes");
    assert!(
        (half_bytes - scratch) * 10 < (full_bytes - scratch) * 7,
        "{full_bytes} → {half_bytes}"
    );
    let (mut worst, mut scale) = (0f32, 0f32);
    for (a, b) in half[1].iter().zip(&full[1]) {
        worst = worst.max((a - b).abs());
        scale = scale.max(b.abs());
    }
    eprintln!("  features: max |Δ| {worst:e} of max |x| {scale:e}");
    assert!(worst <= 2e-2 * scale.max(1.0), "{worst} of {scale}");
    let differ = half[0].iter().zip(&full[0]).filter(|(a, b)| a != b).count();
    eprintln!("  labels: {differ} of {} differ", full[0].len());
    assert!(differ * 100 < full[0].len(), "{differ} labels differ");
    // one workgroup a norm sums in another order than the split: close
    let d1 = one_wg[1]
        .iter()
        .zip(&half[1])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("  one-workgroup norms vs split: max |Δ| {d1:e}");
    assert!(d1 <= 2e-2 * scale.max(1.0), "{d1}");
    assert_eq!(striped, half, "striped vs whole, both packed");
}
