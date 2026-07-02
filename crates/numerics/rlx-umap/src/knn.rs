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

//! Reference k-NN — implemented in [`rlx_cpu::umap_knn`] for sharing with GPU hosts.

pub use rlx_cpu::umap_knn::{knn_backward_pairwise, knn_forward_packed};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_forward_skips_self() {
        let n = 4;
        let k = 2;
        let mut pairwise = vec![0f32; n * n];
        for i in 0..n {
            pairwise[i * n + i] = 0.0;
            for j in 0..n {
                if i != j {
                    pairwise[i * n + j] = (i as f32 - j as f32).abs() + 1.0;
                }
            }
        }
        let mut packed = vec![0f32; n * 2 * k];
        knn_forward_packed(&pairwise, n, k, &mut packed);
        for i in 0..n {
            let idx_base = i * 2 * k;
            for slot in 0..k {
                let idx = packed[idx_base + slot] as usize;
                assert_ne!(idx, i, "row {i} slot {slot}");
            }
        }
    }
}
