// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Convs and norms on a striped arena run on the GPU, with the same
//! numbers.** An arena past one storage binding is striped; an op whose
//! input and output lie in different stripes used to run on the host. The
//! conv3d and group-norm kernels now bind the output as its own window
//! (`kernels::two_window_kernel`), so a MeshNet-shaped stack — convs with
//! folded activations, an instance norm, the fused argmax head — gives
//! exactly what it gives unstriped. Process-global switches: one test.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

fn ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + seed * 13) % 23) as f32 - 11.0) / 17.0)
        .collect()
}

#[test]
fn striped_convs_and_norms_match_unstriped() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let (c, dims) = (6, [24, 40, 40]);
    let per: usize = dims.iter().product();
    let sh = |d: &[usize]| Shape::new(d, DType::F32);
    let vol = sh(&[1, c, dims[0], dims[1], dims[2]]);
    let flat = sh(&[1, c, per, 1]);
    let build = || {
        let mut g = Graph::new("two_window");
        let x = g.input("x", vol.clone());
        let mut y = x;
        for l in 0..3 {
            let w = g.input(format!("w{l}"), sh(&[c, c, 3, 3, 3]));
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
            if l == 1 {
                // an instance norm over the volume, as MindGrab's
                let r = g.reshape(y, vec![1, c as i64, per as i64, 1], flat.clone());
                let (gm, bt) = (g.input("gamma", sh(&[c])), g.input("beta", sh(&[c])));
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
        g.set_outputs(vec![l, y]);
        g
    };
    let mut data: Vec<(String, Vec<f32>)> = vec![("x".into(), ramp(c * per, 1))];
    for l in 0..3 {
        data.push((format!("w{l}"), ramp(c * c * 27, 2 + l)));
        data.push((format!("b{l}"), ramp(c, 7 + l)));
    }
    data.push((
        "gamma".into(),
        (0..c).map(|i| 1.0 + 0.1 * i as f32).collect(),
    ));
    data.push(("beta".into(), ramp(c, 11)));
    data.push(("wh".into(), ramp(c * c, 12)));
    let ins: Vec<(&str, &[f32])> = data
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    let run = || WgpuExecutable::compile(build()).run(&ins);

    let whole = run();
    // ~0.9 MiB tensors in 2 MiB stripes (1 MiB of each for staging): no two
    // share a stripe with room to spare, so every op's input and output split
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "2");
    rlx_ir::env::set("RLX_WGPU_SHARD_STAGE_MIB", "1");
    rlx_ir::env::set("RLX_WGPU_ALLOW_SHARD", "1");
    rlx_ir::env::set("RLX_WGPU_STRIPE_VERBOSE", "1");
    let striped = run();
    // nothing waits on the host (the browser refuses a graph that would):
    // the weights staged between stripes are GPU copies too
    assert!(
        !WgpuExecutable::compile(build()).blocks_on_host(),
        "striped graph blocks on the host"
    );
    rlx_ir::env::set("RLX_WGPU_NO_TWO_WINDOW", "1");
    let hosted = run();
    rlx_ir::env::clear_overrides();

    assert_eq!(striped[0], whole[0], "labels: striped vs whole");
    assert_eq!(striped[1], whole[1], "features: striped vs whole");
    // and the host path it replaces is close, not equal (other sums)
    let worst = hosted[1]
        .iter()
        .zip(&whole[1])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("  hosted vs whole: max |Δ| {worst:e}");
    assert!(worst < 1e-3);
}
