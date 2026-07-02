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

//! Host-side `Op::Custom("umap.knn")` for wgpu arenas (small `n` only).

use crate::buffer::Arena;

/// Prefer the in-GPU `umap_knn.wgsl` kernel at or above this point count.
pub const UMAP_KNN_GPU_MIN_N: usize = 256;

pub fn run_umap_knn(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pairwise_byte_off: usize,
    out_byte_off: usize,
    n: usize,
    k: usize,
) {
    let pw_bytes = n * n * 4;
    let pw_host = arena.read_bytes_range(device, queue, pairwise_byte_off, pw_bytes);
    let pairwise: Vec<f32> = bytemuck::cast_slice(&pw_host).to_vec();
    let mut packed = vec![0f32; n * 2 * k];
    rlx_cpu::umap_knn::knn_forward_packed(&pairwise, n, k, &mut packed);
    arena.write_bytes_range(queue, out_byte_off, bytemuck::cast_slice(&packed));
}
