// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The activation folds still give the right answer when the arena is
//! striped.** A striped arena (one larger than a storage binding: 1 GiB on
//! some adapters) runs ops whose operands span two stripes on the host, by
//! replaying the op through rlx-cpu. The folded forms (a rank-5
//! `FusedConvBiasAct` with an activation, `rlx_wgpu.conv3d_leaky`,
//! `.group_norm_act`, `.conv3d_argmax`) exist only in this backend's kernels:
//! the host ran the first as a 2-D conv, and brainchop's subcortical model at
//! a 1 GiB binding came back with a third of its voxels mislabelled.
//! `compile` now notices and compiles again without the folds.
//!
//! A tiny stripe (`RLX_WGPU_SHARD_CAP_MIB`) stands in for the small binding.
//! The switches are process-global, so this file is ONE test.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

fn ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + seed * 13) % 23) as f32 - 11.0) / 17.0)
        .collect()
}

#[test]
fn folded_nodes_are_right_on_a_striped_arena() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let (c, k, dims) = (6, 3, [24, 40, 40]);
    let per = dims.iter().product::<usize>();
    let sh = |d: &[usize]| Shape::new(d, DType::F32);
    let vol = sh(&[1, c, dims[0], dims[1], dims[2]]);
    let conv = |g: &mut Graph, x, w, b| {
        g.add_node(
            Op::FusedConvBiasAct {
                kernel_size: vec![k; 3],
                stride: vec![1; 3],
                padding: vec![1; 3],
                dilation: vec![1; 3],
                groups: 1,
                activation: None,
                has_residual: false,
            },
            vec![x, w, b],
            vol.clone(),
        )
    };
    // conv → ELU → conv → ELU → 1×1×1 conv → argmax: two activation folds
    // and the argmax fold, each tensor ~3.7 MiB
    let build = || {
        let mut g = Graph::new("striped_folds");
        let x = g.input("x", sh(&[1, c, dims[0], dims[1], dims[2]]));
        let (w0, b0) = (g.input("w0", sh(&[c, c, k, k, k])), g.input("b0", sh(&[c])));
        let (w1, b1) = (g.input("w1", sh(&[c, c, k, k, k])), g.input("b1", sh(&[c])));
        let w2 = g.input("w2", sh(&[c, c, 1, 1, 1]));
        let y = conv(&mut g, x, w0, b0);
        let y = g.activation(rlx_ir::op::Activation::Elu, y, vol.clone());
        let y = conv(&mut g, y, w1, b1);
        let y = g.activation(rlx_ir::op::Activation::Elu, y, vol.clone());
        let y = g.conv3d(y, w2, [1; 3], [0; 3], [1; 3], 1);
        let l = g.argmax(y, 1, false, sh(&[1, dims[0], dims[1], dims[2]]));
        g.set_outputs(vec![l]);
        g
    };
    let x = ramp(c * per, 1);
    let (w0, w1, w2) = (ramp(c * c * 27, 2), ramp(c * c * 27, 3), ramp(c * c, 4));
    let (b0, b1) = (ramp(c, 5), ramp(c, 6));
    let ins: Vec<(&str, &[f32])> = vec![
        ("x", &x),
        ("w0", &w0),
        ("b0", &b0),
        ("w1", &w1),
        ("b1", &b1),
        ("w2", &w2),
    ];
    let run = || WgpuExecutable::compile(build()).run(&ins).remove(0);

    // whole: folded, on the GPU
    let whole = run();
    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "4");
    rlx_ir::env::set("RLX_WGPU_ALLOW_SHARD", "1");
    let striped = run();
    // striped with the folds off from the start: the plain ops, hosted
    for f in ["CONV_ACT", "NORM_ACT", "CONV_ARGMAX"] {
        rlx_ir::env::set(format!("RLX_WGPU_NO_FOLD_{f}"), "1");
    }
    let plain = run();
    rlx_ir::env::clear_overrides();

    assert_eq!(striped, plain, "striped with folds vs without");
    // the host's sums are not the GPU's, so a near tie may differ
    let differ = striped.iter().zip(&whole).filter(|(a, b)| a != b).count();
    eprintln!("  striped vs whole: {differ} of {per} labels differ");
    assert!(
        differ * 1000 <= per,
        "{differ} of {per} labels differ from the unstriped run"
    );
}
