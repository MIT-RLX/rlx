// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use rlx_umap::{knn_forward_packed, knn_indices_and_distances, register};

#[test]
fn knn_runs_via_session() {
    register();

    let n = 6;
    let k = 3;
    let pairwise: Vec<f32> = (0..n)
        .flat_map(|i| {
            (0..n).map(move |j| {
                if i == j {
                    0.0
                } else {
                    ((i as i32 - j as i32).unsigned_abs() as f32) + 0.25
                }
            })
        })
        .collect();

    let mut expected_packed = vec![0f32; n * 2 * k];
    knn_forward_packed(&pairwise, n, k, &mut expected_packed);

    let build = || {
        let mut g = Graph::new("umap_knn");
        let pw = g.input("pairwise", Shape::new(&[n, n], DType::F32));
        let (idx, dist) = knn_indices_and_distances(&mut g, pw, k as u32);
        g.set_outputs(vec![idx, dist]);
        g
    };

    let mut exe = Session::new(Device::Cpu).compile(build());
    let outs = exe.run(&[("pairwise", &pairwise)]);
    assert_eq!(outs.len(), 2);

    let idx_out = &outs[0];
    let dist_out = &outs[1];
    assert_eq!(idx_out.len(), n * k);
    assert_eq!(dist_out.len(), n * k);

    for i in 0..n {
        for slot in 0..k {
            let exp_i = expected_packed[i * 2 * k + slot];
            let exp_d = expected_packed[i * 2 * k + k + slot];
            assert!(
                (idx_out[i * k + slot] - exp_i).abs() < 1e-5,
                "row {i} slot {slot} idx"
            );
            assert!(
                (dist_out[i * k + slot] - exp_d).abs() < 1e-5,
                "row {i} slot {slot} dist"
            );
        }
    }
}
