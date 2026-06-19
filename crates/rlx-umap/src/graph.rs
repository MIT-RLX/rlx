// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Graph builders for UMAP pairwise distances and k-NN.
//!
//! Pairwise builders use rank-2 `[n, n]` matmul tiling instead of
//! `[n, 1]` + `[1, n]` broadcast adds, which are incorrect on Metal/wgpu today.
//!
//! Constants are **O(n)** vectors (outer-product matmul) instead of **O(n²)**
//! dense matrices — faster compile and less host upload for large `n`.

use rlx_ir::{DType, Graph, NodeId, Op, Shape, infer::GraphExt};

use crate::knn_attrs::KnnAttrs;
use crate::ops::UMAP_KNN;

const EPS: f32 = 1e-12;

fn scalar_f32(g: &mut Graph, value: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: value.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1, 1], DType::F32),
    )
}

/// Length-`n` vector constant (rank 1).
fn vector_constant(g: &mut Graph, n: usize, value: f32) -> NodeId {
    let data: Vec<u8> = std::iter::repeat_n(value, n)
        .flat_map(f32::to_le_bytes)
        .collect();
    g.add_node(Op::Constant { data }, vec![], Shape::new(&[n], DType::F32))
}

/// `[n, 1]` matrix filled with `value`.
fn col_vector_constant(g: &mut Graph, n: usize, value: f32) -> NodeId {
    let data: Vec<u8> = std::iter::repeat_n(value, n)
        .flat_map(f32::to_le_bytes)
        .collect();
    g.add_node(
        Op::Constant { data },
        vec![],
        Shape::new(&[n, 1], DType::F32),
    )
}

/// `[1, n]` matrix filled with `value`.
fn row_vector_constant(g: &mut Graph, n: usize, value: f32) -> NodeId {
    let data: Vec<u8> = std::iter::repeat_n(value, n)
        .flat_map(f32::to_le_bytes)
        .collect();
    g.add_node(
        Op::Constant { data },
        vec![],
        Shape::new(&[1, n], DType::F32),
    )
}

/// Broadcast a length-`n` vector to `[n, n]` via matmul: `v[i] + v[j]` at `(i, j)`.
fn outer_sum_from_vector(g: &mut Graph, v: NodeId, n: usize) -> NodeId {
    let col = g.reshape_(v, vec![n as i64, 1]);
    let row_ones = row_vector_constant(g, n, 1.0);
    let col_tile = g.mm(col, row_ones);
    let row = g.reshape_(v, vec![1, n as i64]);
    let col_ones = col_vector_constant(g, n, 1.0);
    let row_tile = g.mm(col_ones, row);
    g.add(col_tile, row_tile)
}

/// Outer product `v[i] * v[j]` as `[n, n]` via matmul.
fn outer_mul_from_vector(g: &mut Graph, v: NodeId, n: usize) -> NodeId {
    let col = g.reshape_(v, vec![n as i64, 1]);
    let row = g.reshape_(v, vec![1, n as i64]);
    g.mm(col, row)
}

/// `[n, n]` matrix of `1.0` without storing `n²` constants.
fn ones_matrix(g: &mut Graph, n: usize) -> NodeId {
    let ones = vector_constant(g, n, 1.0);
    outer_mul_from_vector(g, ones, n)
}

/// Euclidean pairwise distances via `||x_i||^2 + ||x_j||^2 - 2 x_i·x_j`.
pub fn pairwise_euclidean_graph(g: &mut Graph, x: NodeId, n: usize) -> NodeId {
    let x2 = g.mul(x, x);
    let sq_norms = g.sum(x2, vec![1], false);
    let xt = g.transpose_(x, vec![1, 0]);
    let cross = g.mm(x, xt);
    let neg2 = scalar_f32(g, -2.0);
    let neg2_cross = g.mul(neg2, cross);
    let outer = outer_sum_from_vector(g, sq_norms, n);
    let sq_dists = g.add(outer, neg2_cross);
    g.sqrt(sq_dists)
}

/// Cosine distance `1 - cos(θ)` for all pairs (matches [`crate::pairwise::cosine_pairwise_reference`]).
pub fn pairwise_cosine_graph(g: &mut Graph, x: NodeId, n: usize) -> NodeId {
    let x2 = g.mul(x, x);
    let sq_norms = g.sum(x2, vec![1], false);
    let norms = g.sqrt(sq_norms);
    let xt = g.transpose_(x, vec![1, 0]);
    let cross = g.mm(x, xt);
    let prod = outer_mul_from_vector(g, norms, n);
    let eps = scalar_f32(g, EPS);
    let denom = g.add(prod, eps);
    let sim = g.div(cross, denom);
    let ones = ones_matrix(g, n);
    let dist = g.sub(ones, sim);
    g.relu(dist)
}

/// Cosine pairwise + `umap.knn` → packed `[n, 2k]`.
pub fn cosine_knn_packed_graph(g: &mut Graph, x: NodeId, n: usize, k: u32) -> NodeId {
    let pw = pairwise_cosine_graph(g, x, n);
    knn_graph(g, pw, k)
}

/// Cosine pairwise + k-NN → `(indices [n,k], distances [n,k])`.
pub fn cosine_knn_graph(g: &mut Graph, x: NodeId, n: usize, k: u32) -> (NodeId, NodeId) {
    let packed = cosine_knn_packed_graph(g, x, n, k);
    split_knn_packed(g, packed, k)
}

/// Emit `umap.knn` on a square pairwise distance matrix.
pub fn knn_graph(g: &mut Graph, pairwise: NodeId, k: u32) -> NodeId {
    g.custom_op(UMAP_KNN, KnnAttrs { k }.encode(), vec![pairwise])
}

/// Split packed `[n, 2k]` into `(indices [n, k], distances [n, k])`.
pub fn split_knn_packed(g: &mut Graph, packed: NodeId, k: u32) -> (NodeId, NodeId) {
    let k = k as usize;
    let indices = g.narrow_(packed, 1, 0, k);
    let distances = g.narrow_(packed, 1, k, k);
    (indices, distances)
}

/// `knn` + split — convenience when both tensors are needed.
pub fn knn_indices_and_distances(g: &mut Graph, pairwise: NodeId, k: u32) -> (NodeId, NodeId) {
    let packed = knn_graph(g, pairwise, k);
    split_knn_packed(g, packed, k)
}
