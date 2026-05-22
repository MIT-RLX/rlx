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

//! GPU sgemm via custom MSL kernel.
//!
//! Initial impl uses our tiled MSL kernel (see kernels.rs::sgemm_tiled).
//! Future: bridge MPSMatrixMultiplication for Apple's optimized matmul
//! when matrices are large enough to amortize objc bridging cost.

use crate::cost::{SgemmVariant, hw_model};
use crate::device::metal_device;
use crate::kernels::kernels;
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

/// C = A @ B via custom MSL kernel. Issues set_pipeline+dispatch on a shared
/// compute encoder; caller is responsible for encoder lifecycle.
pub fn metal_sgemm(
    enc: &ComputeCommandEncoderRef,
    arena: &Buffer,
    a_off: usize,
    b_off: usize,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let kk = kernels();
    let m_u = m as u32;
    let k_u = k as u32;
    let n_u = n as u32;
    enc.set_buffer(0, Some(arena), a_off as u64);
    enc.set_buffer(1, Some(arena), b_off as u64);
    enc.set_buffer(2, Some(arena), c_off as u64);
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &m_u as *const _ as *const _,
    );
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &k_u as *const _ as *const _,
    );
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &n_u as *const _ as *const _,
    );

    match hw_model().pick_sgemm(m, k, n) {
        SgemmVariant::Mps => {
            // Should never be hit on the in-encoder path: encode_and_run splits
            // the encoder around MPS dispatches and routes to the cmd-buffer
            // variant directly. Fall back to simd_4x4 if a caller bypasses it.
            enc.set_compute_pipeline_state(&kk.sgemm_simd_4x4);
            let tg_count = MTLSize {
                width: n.div_ceil(32) as u64,
                height: m.div_ceil(32) as u64,
                depth: 1,
            };
            enc.dispatch_thread_groups(
                tg_count,
                MTLSize {
                    width: 512,
                    height: 1,
                    depth: 1,
                },
            );
        }
        SgemmVariant::Simd4x4 => {
            enc.set_compute_pipeline_state(&kk.sgemm_simd_4x4);
            let tg_count = MTLSize {
                width: n.div_ceil(32) as u64,
                height: m.div_ceil(32) as u64,
                depth: 1,
            };
            enc.dispatch_thread_groups(
                tg_count,
                MTLSize {
                    width: 512,
                    height: 1,
                    depth: 1,
                },
            );
        }
        SgemmVariant::Simd => {
            enc.set_compute_pipeline_state(&kk.sgemm_simd);
            let tg_count = MTLSize {
                width: n.div_ceil(8) as u64,
                height: m.div_ceil(8) as u64,
                depth: 1,
            };
            enc.dispatch_thread_groups(
                tg_count,
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        SgemmVariant::SimdPadded => {
            enc.set_compute_pipeline_state(&kk.sgemm_simd_padded);
            let tg_count = MTLSize {
                width: n.div_ceil(8) as u64,
                height: m.div_ceil(8) as u64,
                depth: 1,
            };
            enc.dispatch_thread_groups(
                tg_count,
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        SgemmVariant::Tiled => {
            enc.set_compute_pipeline_state(&kk.sgemm_tiled);
            let grid_w = n.div_ceil(16) * 16;
            let grid_h = m.div_ceil(16) * 16;
            let grid = MTLSize {
                width: grid_w as u64,
                height: grid_h as u64,
                depth: 1,
            };
            enc.dispatch_threads(
                grid,
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            );
        }
        SgemmVariant::Naive => {
            enc.set_compute_pipeline_state(&kk.sgemm);
            let grid = MTLSize {
                width: n as u64,
                height: m as u64,
                depth: 1,
            };
            let tg_w = 16u64.min(n as u64);
            let tg_h = 16u64.min(m as u64);
            enc.dispatch_threads(
                grid,
                MTLSize {
                    width: tg_w,
                    height: tg_h,
                    depth: 1,
                },
            );
        }
    }
}

/// Activation kind passed to fused matmul kernels.
#[repr(u32)]
#[derive(Copy, Clone)]
pub enum FusedAct {
    None = 0,
    Gelu = 1,
    Silu = 2,
}

/// C = A @ B + bias [+ activation], dispatched as a single MSL kernel.
/// Halves kernel count compared to separate sgemm + bias_add + activation.
pub fn metal_sgemm_bias(
    enc: &ComputeCommandEncoderRef,
    arena: &Buffer,
    a_off: usize,
    b_off: usize,
    bias_off: usize,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
    act: FusedAct,
) {
    let kk = kernels();

    let m_u = m as u32;
    let k_u = k as u32;
    let n_u = n as u32;
    let act_u = act as u32;

    match hw_model().pick_sgemm(m, k, n) {
        SgemmVariant::Simd4x4 => {
            enc.set_buffer(0, Some(arena), a_off as u64);
            enc.set_buffer(1, Some(arena), b_off as u64);
            enc.set_buffer(2, Some(arena), bias_off as u64);
            enc.set_buffer(3, Some(arena), c_off as u64);
            enc.set_bytes(4, 4, &m_u as *const _ as *const _);
            enc.set_bytes(5, 4, &k_u as *const _ as *const _);
            enc.set_bytes(6, 4, &n_u as *const _ as *const _);
            enc.set_bytes(7, 4, &act_u as *const _ as *const _);
            enc.set_compute_pipeline_state(&kk.sgemm_simd_4x4_bias);
            let tg_count = MTLSize {
                width: n.div_ceil(32) as u64,
                height: m.div_ceil(32) as u64,
                depth: 1,
            };
            enc.dispatch_thread_groups(
                tg_count,
                MTLSize {
                    width: 512,
                    height: 1,
                    depth: 1,
                },
            );
        }
        SgemmVariant::Simd => {
            enc.set_buffer(0, Some(arena), a_off as u64);
            enc.set_buffer(1, Some(arena), b_off as u64);
            enc.set_buffer(2, Some(arena), bias_off as u64);
            enc.set_buffer(3, Some(arena), c_off as u64);
            enc.set_bytes(4, 4, &m_u as *const _ as *const _);
            enc.set_bytes(5, 4, &k_u as *const _ as *const _);
            enc.set_bytes(6, 4, &n_u as *const _ as *const _);
            enc.set_bytes(7, 4, &act_u as *const _ as *const _);
            enc.set_compute_pipeline_state(&kk.sgemm_simd_bias);
            let tg_count = MTLSize {
                width: n.div_ceil(8) as u64,
                height: m.div_ceil(8) as u64,
                depth: 1,
            };
            enc.dispatch_thread_groups(
                tg_count,
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        SgemmVariant::SimdPadded => {
            enc.set_buffer(0, Some(arena), a_off as u64);
            enc.set_buffer(1, Some(arena), b_off as u64);
            enc.set_buffer(2, Some(arena), bias_off as u64);
            enc.set_buffer(3, Some(arena), c_off as u64);
            enc.set_bytes(4, 4, &m_u as *const _ as *const _);
            enc.set_bytes(5, 4, &k_u as *const _ as *const _);
            enc.set_bytes(6, 4, &n_u as *const _ as *const _);
            enc.set_bytes(7, 4, &act_u as *const _ as *const _);
            enc.set_compute_pipeline_state(&kk.sgemm_simd_padded_bias);
            let tg_count = MTLSize {
                width: n.div_ceil(8) as u64,
                height: m.div_ceil(8) as u64,
                depth: 1,
            };
            enc.dispatch_thread_groups(
                tg_count,
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
        }
        // Tiled / Naive variants don't have bias-fused versions yet.
        // Fall back to plain sgemm + separate bias_add (and activation) on the
        // same encoder.
        _ => {
            metal_sgemm(enc, arena, a_off, b_off, c_off, m, k, n);
            enc.set_compute_pipeline_state(&kk.bias_add);
            enc.set_buffer(0, Some(arena), c_off as u64);
            enc.set_buffer(1, Some(arena), bias_off as u64);
            enc.set_bytes(2, 4, &m_u as *const _ as *const _);
            enc.set_bytes(3, 4, &n_u as *const _ as *const _);
            let grid = MTLSize {
                width: n as u64,
                height: m as u64,
                depth: 1,
            };
            let tg = MTLSize {
                width: 16u64.min(n as u64),
                height: 16u64.min(m as u64),
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);

            if !matches!(act, FusedAct::None) {
                let pipeline = match act {
                    FusedAct::Gelu => &kk.gelu_inplace,
                    FusedAct::Silu => &kk.silu_inplace,
                    FusedAct::None => unreachable!(),
                };
                enc.set_compute_pipeline_state(pipeline);
                enc.set_buffer(0, Some(arena), c_off as u64);
                let len = (m * n) as u32;
                enc.set_bytes(1, 4, &len as *const _ as *const _);
                let tg_w = pipeline.thread_execution_width().min(len as u64);
                enc.dispatch_threads(
                    MTLSize {
                        width: len as u64,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: tg_w,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }
    }
}

/// Half-precision matmul (no bias). Uses simdgroup_half8x8 tensor units.
/// Requires M%32==K%32==N%32==0 for the tiled variant.
/// TODO: padded f16 variants for arbitrary dims; currently undefined behavior
/// for misaligned shapes (writes past output buffer).
pub fn metal_hgemm(
    enc: &ComputeCommandEncoderRef,
    arena: &Buffer,
    a_off: usize,
    b_off: usize,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let kk = kernels();
    let m_u = m as u32;
    let k_u = k as u32;
    let n_u = n as u32;
    enc.set_buffer(0, Some(arena), a_off as u64);
    enc.set_buffer(1, Some(arena), b_off as u64);
    enc.set_buffer(2, Some(arena), c_off as u64);
    enc.set_bytes(3, 4, &m_u as *const _ as *const _);
    enc.set_bytes(4, 4, &k_u as *const _ as *const _);
    enc.set_bytes(5, 4, &n_u as *const _ as *const _);
    enc.set_compute_pipeline_state(&kk.hgemm_simd_4x4);
    let tg_count = MTLSize {
        width: n.div_ceil(32) as u64,
        height: m.div_ceil(32) as u64,
        depth: 1,
    };
    enc.dispatch_thread_groups(
        tg_count,
        MTLSize {
            width: 512,
            height: 1,
            depth: 1,
        },
    );
}

/// Half-precision matmul + bias + activation fused.
pub fn metal_hgemm_bias(
    enc: &ComputeCommandEncoderRef,
    arena: &Buffer,
    a_off: usize,
    b_off: usize,
    bias_off: usize,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
    act: FusedAct,
) {
    let kk = kernels();
    let m_u = m as u32;
    let k_u = k as u32;
    let n_u = n as u32;
    let act_u = act as u32;
    enc.set_buffer(0, Some(arena), a_off as u64);
    enc.set_buffer(1, Some(arena), b_off as u64);
    enc.set_buffer(2, Some(arena), bias_off as u64);
    enc.set_buffer(3, Some(arena), c_off as u64);
    enc.set_bytes(4, 4, &m_u as *const _ as *const _);
    enc.set_bytes(5, 4, &k_u as *const _ as *const _);
    enc.set_bytes(6, 4, &n_u as *const _ as *const _);
    enc.set_bytes(7, 4, &act_u as *const _ as *const _);
    enc.set_compute_pipeline_state(&kk.hgemm_simd_4x4_bias);
    let tg_count = MTLSize {
        width: n.div_ceil(32) as u64,
        height: m.div_ceil(32) as u64,
        depth: 1,
    };
    enc.dispatch_thread_groups(
        tg_count,
        MTLSize {
            width: 512,
            height: 1,
            depth: 1,
        },
    );
}

/// Helper: create a new command buffer from the global queue.
pub fn new_command_buffer() -> metal::CommandBuffer {
    let dev = metal_device().expect("Metal device required");
    dev.queue.new_command_buffer().to_owned()
}
