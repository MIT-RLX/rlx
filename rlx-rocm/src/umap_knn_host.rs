// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Host-side `Op::Custom("umap.knn")` for ROCm arenas.

use crate::device::RocmContext;
use crate::hip::HipBuffer;

pub fn run_umap_knn(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
    _arena_size_bytes: usize,
    pairwise_f32_off: usize,
    out_f32_off: usize,
    n: usize,
    k: usize,
) {
    let rt = &ctx.runtime;
    let pw_len = n * n;
    let out_len = n * 2 * k;
    let pw_bytes = pw_len * 4;
    let out_bytes = out_len * 4;
    let pw_ptr = buffer.ptr + (pairwise_f32_off as u64) * 4;
    let out_ptr = buffer.ptr + (out_f32_off as u64) * 4;

    let mut pairwise = vec![0f32; pw_len];
    let mut packed = vec![0f32; out_len];

    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
        let _ = (rt.hip_memcpy_dtoh)(pairwise.as_mut_ptr() as *mut _, pw_ptr, pw_bytes);
    }

    rlx_cpu::umap_knn::knn_forward_packed(&pairwise, n, k, &mut packed);

    unsafe {
        let _ = (rt.hip_memcpy_htod)(out_ptr, packed.as_ptr() as *const _, out_bytes);
    }
}
