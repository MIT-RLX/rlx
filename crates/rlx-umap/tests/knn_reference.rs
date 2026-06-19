// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

use rlx_umap::{knn_backward_pairwise, knn_forward_packed};

#[test]
fn reference_knn_matches_bruteforce() {
    let n = 5;
    let k = 2;
    let pairwise: Vec<f32> = (0..n)
        .flat_map(|i| {
            (0..n).map(move |j| {
                if i == j {
                    0.0
                } else {
                    (i as f32 - j as f32).abs() + 0.1
                }
            })
        })
        .collect();

    let mut packed = vec![0f32; n * 2 * k];
    knn_forward_packed(&pairwise, n, k, &mut packed);

    for i in 0..n {
        let mut neighbors: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (pairwise[i * n + j], j))
            .collect();
        neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for slot in 0..k {
            let idx_base = i * 2 * k;
            assert_eq!(packed[idx_base + slot] as usize, neighbors[slot].1);
            assert!((packed[idx_base + k + slot] - neighbors[slot].0).abs() < 1e-6);
        }
    }
}

#[test]
fn reference_backward_is_finite() {
    let n = 4;
    let k = 2;
    let pairwise: Vec<f32> = (0..n * n)
        .map(|i| ((i % n) as f32 - (i / n) as f32).abs() + 0.5)
        .collect();
    let mut packed = vec![0f32; n * 2 * k];
    knn_forward_packed(&pairwise, n, k, &mut packed);

    let d_dist: Vec<f32> = (0..n * k).map(|i| 0.01 * (i as f32 + 1.0)).collect();
    let mut d_pairwise = vec![0f32; n * n];
    knn_backward_pairwise(&pairwise, &d_dist, n, k, &mut d_pairwise);
    assert!(d_pairwise.iter().all(|v| v.is_finite()));
}
