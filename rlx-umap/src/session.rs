// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Session helpers for backends with custom-op limitations.

#![cfg(feature = "bench")]

/// Cosine pairwise on MLX, k-NN on CPU (100% parity with the reference k-NN).
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub fn cosine_knn_mlx(
    data: &[f32],
    n: usize,
    d: usize,
    k: u32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    use crate::config::Metric;
    crate::encoder::knn::knn_mlx_hybrid(data, n, d, k as usize, &Metric::Cosine)
}
