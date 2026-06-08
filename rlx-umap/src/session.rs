// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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
