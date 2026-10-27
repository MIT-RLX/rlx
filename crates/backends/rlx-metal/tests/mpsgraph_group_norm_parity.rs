// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::GroupNorm` in the full-graph MPSGraph lowering.
//!
//! GroupNorm had no MPSGraph arm, so ANY graph containing one fell back to the
//! per-op thunk path (EEG-DINO's patch-embedding convs: GroupNorm with 5
//! groups). The lowering composes it from reshape + two-pass mean / variance +
//! per-channel γ, β — the same math as the `group_norm` thunk. Compared here
//! against CPU with the MPSGraph path forced, at rank 3 and rank 4.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // Offset + scale so a missing mean subtraction or variance is visible.
            ((s >> 40) as f32 / 8_388_608.0) * 3.0 + 0.7
        })
        .collect()
}

fn case(dims: &[usize], groups: usize) {
    let numel: usize = dims.iter().product();
    let c = dims[1];
    let x = noise(numel, 1);
    let gamma = noise(c, 2);
    let beta = noise(c, 3);
    let build = || {
        let mut g = Graph::new("group_norm");
        let xi = g.input("x", Shape::new(dims, F));
        let gi = g.input("gamma", Shape::new(&[c], F));
        let bi = g.input("beta", Shape::new(&[c], F));
        // Feed it through an op so the GroupNorm input is a computed tensor.
        let x2 = g.add(xi, xi);
        let y = g.group_norm(x2, gi, bi, groups, 1e-5);
        g.set_outputs(vec![y]);
        g
    };
    let inputs: [(&str, &[f32]); 3] = [("x", &x), ("gamma", &gamma), ("beta", &beta)];
    let cpu = Session::new(Device::Cpu).compile(build()).run(&inputs);
    let metal = {
        let _force = rlx_ir::env::scoped("RLX_MPSGRAPH_FORCE", "1");
        Session::new(Device::Metal).compile(build()).run(&inputs)
    };
    let (cpu, metal) = (&cpu[0], &metal[0]);
    assert_eq!(cpu.len(), metal.len());
    let scale = cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
    let worst = cpu
        .iter()
        .zip(metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst / scale < 1e-5,
        "group_norm {dims:?} groups={groups}: rel err {:.3e}",
        worst / scale
    );
}

#[test]
fn mpsgraph_group_norm_matches_cpu() {
    if rlx_ir::env::skip_unless_device(
        "metal",
        cfg!(target_os = "macos"),
        rlx_metal::is_available(),
    ) {
        return;
    }
    case(&[2, 10, 190, 9], 5); // EEG-DINO patch-embedding conv stage
    case(&[3, 12, 7], 4); // rank 3
    case(&[1, 6, 4, 5], 1); // one group == LayerNorm over C·H·W
}
