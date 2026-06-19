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

//! Host-side `Op::Custom("umap.knn")` for CUDA arenas.

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub fn run_umap_knn(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    _arena_size_bytes: usize,
    pairwise_f32_off: usize,
    out_f32_off: usize,
    n: usize,
    k: usize,
) {
    stream
        .synchronize()
        .expect("rlx-cuda: umap.knn pre-sync failed");

    let pw_len = n * n;
    let out_len = n * 2 * k;
    let mut pairwise = vec![0f32; pw_len];
    stream
        .memcpy_dtoh(
            &buffer.slice(pairwise_f32_off..pairwise_f32_off + pw_len),
            &mut pairwise,
        )
        .expect("rlx-cuda: umap.knn pairwise dtoh failed");

    let mut packed = vec![0f32; out_len];
    rlx_cpu::umap_knn::knn_forward_packed(&pairwise, n, k, &mut packed);

    stream
        .memcpy_htod(
            &packed,
            &mut buffer.slice_mut(out_f32_off..out_f32_off + out_len),
        )
        .expect("rlx-cuda: umap.knn output htod failed");
}
