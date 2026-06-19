// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Reference pairwise distance matrices (row-major `[n, n]`).

const EPS: f32 = 1e-12;

/// Full Euclidean distance matrix: `sqrt(max(0, ||x_i - x_j||^2))`.
pub fn euclidean_pairwise_reference(x: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * n];
    let mut sq_norms = vec![0f32; n];
    for i in 0..n {
        let mut s = 0f32;
        for f in 0..d {
            let v = x[i * d + f];
            s += v * v;
        }
        sq_norms[i] = s;
    }
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0f32;
            for f in 0..d {
                dot += x[i * d + f] * x[j * d + f];
            }
            let sq = (sq_norms[i] + sq_norms[j] - 2.0 * dot).max(0.0);
            out[i * n + j] = sq.sqrt();
        }
    }
    out
}

/// Cosine **distance** `1 - cos(θ)`, clamped to `[0, 2]`. Diagonal is `0`.
pub fn cosine_pairwise_reference(x: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut norms = vec![EPS; n];
    for i in 0..n {
        let mut s = 0f32;
        for f in 0..d {
            let v = x[i * d + f];
            s += v * v;
        }
        norms[i] = s.sqrt().max(EPS);
    }
    let mut out = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                out[i * n + j] = 0.0;
                continue;
            }
            let mut dot = 0f32;
            for f in 0..d {
                dot += x[i * d + f] * x[j * d + f];
            }
            let sim = (dot / (norms[i] * norms[j])).clamp(-1.0, 1.0);
            out[i * n + j] = (1.0 - sim).max(0.0);
        }
    }
    out
}
