// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Reference k-NN from a row-major `[n, n]` pairwise distance matrix.
//!
//! Shared by `rlx-umap` and GPU host delegates (`rlx-wgpu`, `rlx-cuda`, …).
//!
//! Tuned for UMAP workloads:
//! - **Small `n`**: tight serial row loop (low fixed cost).
//! - **Large `n`**: Rayon row parallelism when `n * n` is big enough to amortize.

use rayon::prelude::*;

const INFINITY: f32 = f32::MAX;
/// Parallelize rows when the pairwise matrix has at least this many elements.
const PARALLEL_MIN_ELEMS: usize = 64 * 64;

/// Forward k-NN: for each row, select `k` smallest off-diagonal distances.
///
/// `pairwise` is row-major `n × n`. Writes `packed` row-major `n × (2k)`:
/// `[idx_0, …, idx_{k-1}, dist_0, …, dist_{k-1}]` per row (indices stored as f32).
pub fn knn_forward_packed(pairwise: &[f32], n: usize, k: usize, packed: &mut [f32]) {
    assert_eq!(pairwise.len(), n * n);
    assert_eq!(packed.len(), n * 2 * k);
    assert!(k < n, "k ({k}) must be strictly less than n ({n})");

    if n * n >= PARALLEL_MIN_ELEMS {
        packed
            .par_chunks_mut(2 * k)
            .enumerate()
            .for_each(|(row, out_row)| {
                knn_row(&pairwise[row * n..(row + 1) * n], row, n, k, out_row);
            });
    } else {
        for row in 0..n {
            let base = row * 2 * k;
            knn_row(
                &pairwise[row * n..(row + 1) * n],
                row,
                n,
                k,
                &mut packed[base..base + 2 * k],
            );
        }
    }
}

/// One row of k-NN (insertion sort into k slots; `k` is small in UMAP).
#[inline]
fn knn_row(row_pw: &[f32], row: usize, n: usize, k: usize, out: &mut [f32]) {
    let (idx_slice, dist_slice) = out.split_at_mut(k);
    for slot in idx_slice.iter_mut() {
        *slot = n as f32;
    }
    for slot in dist_slice.iter_mut() {
        *slot = INFINITY;
    }

    let mut worst = INFINITY;
    for (col, &dist) in row_pw.iter().enumerate() {
        if col == row || dist >= worst {
            continue;
        }
        if dist < dist_slice[k - 1] {
            let mut slot = k - 1;
            while slot > 0 && dist < dist_slice[slot - 1] {
                dist_slice[slot] = dist_slice[slot - 1];
                idx_slice[slot] = idx_slice[slot - 1];
                slot -= 1;
            }
            dist_slice[slot] = dist;
            idx_slice[slot] = col as f32;
            worst = dist_slice[k - 1];
        }
    }
}

/// Backward: scatter `d_dist [n, k]` into `d_pairwise [n, n]`.
pub fn knn_backward_pairwise(
    pairwise: &[f32],
    d_dist: &[f32],
    n: usize,
    k: usize,
    d_pairwise: &mut [f32],
) {
    assert_eq!(pairwise.len(), n * n);
    assert_eq!(d_dist.len(), n * k);
    assert_eq!(d_pairwise.len(), n * n);

    let eps = 1e-8f32;
    d_pairwise.fill(0.0);

    if n * n >= PARALLEL_MIN_ELEMS {
        d_pairwise
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(row, d_row)| {
                let mut scratch_idx = vec![0f32; k];
                let mut scratch_dist = vec![INFINITY; k];
                knn_backward_row(
                    pairwise,
                    d_dist,
                    row,
                    n,
                    k,
                    eps,
                    d_row,
                    &mut scratch_idx,
                    &mut scratch_dist,
                );
            });
    } else {
        let mut scratch_idx = vec![0f32; k];
        let mut scratch_dist = vec![INFINITY; k];
        for row in 0..n {
            knn_backward_row(
                pairwise,
                d_dist,
                row,
                n,
                k,
                eps,
                &mut d_pairwise[row * n..(row + 1) * n],
                &mut scratch_idx,
                &mut scratch_dist,
            );
        }
    }
}

fn knn_backward_row(
    pairwise: &[f32],
    d_dist: &[f32],
    row: usize,
    n: usize,
    k: usize,
    eps: f32,
    d_row: &mut [f32],
    scratch_idx: &mut [f32],
    scratch_dist: &mut [f32],
) {
    for slot in scratch_idx.iter_mut() {
        *slot = n as f32;
    }
    for slot in scratch_dist.iter_mut() {
        *slot = INFINITY;
    }

    let row_off = row * n;
    for col in 0..n {
        if row == col {
            continue;
        }
        let dist = pairwise[row_off + col];
        if dist < scratch_dist[k - 1] {
            let mut slot = k - 1;
            while slot > 0 && dist < scratch_dist[slot - 1] {
                scratch_dist[slot] = scratch_dist[slot - 1];
                scratch_idx[slot] = scratch_idx[slot - 1];
                slot -= 1;
            }
            scratch_dist[slot] = dist;
            scratch_idx[slot] = col as f32;
        }
    }

    for slot in 0..k {
        let grad_value = d_dist[row * k + slot];
        if grad_value == 0.0 {
            continue;
        }
        let dist = scratch_dist[slot].max(eps);
        let neighbor_col = scratch_idx[slot] as usize;
        if neighbor_col < n {
            d_row[neighbor_col] += grad_value / dist;
        }
    }
}
