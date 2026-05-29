// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Packed `[n, 2k]` layout helpers (`umap.knn` output).

/// Unpack `packed [n, 2k]` row-major into flat `(indices [n,k], distances [n,k])`.
pub fn unpack_knn_packed(packed: &[f32], n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(packed.len(), n * 2 * k);
    let mut idx = vec![0f32; n * k];
    let mut dist = vec![0f32; n * k];
    for i in 0..n {
        for s in 0..k {
            let j = i * k + s;
            idx[j] = packed[i * 2 * k + s];
            dist[j] = packed[i * 2 * k + k + s];
        }
    }
    (idx, dist)
}
