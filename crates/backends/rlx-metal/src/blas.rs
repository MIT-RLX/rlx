// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU sgemm via custom MSL kernel.
//!
//! Initial impl uses our tiled MSL kernel (see kernels.rs::sgemm_tiled).
//! Future: bridge MPSMatrixMultiplication for Apple's optimized matmul
//! when matrices are large enough to amortize objc bridging cost.

use crate::cost::{SgemmVariant, hw_model};
use crate::device::metal_device;
use crate::kernels::kernels;
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

fn dispatch_sgemm_variant(enc: &ComputeCommandEncoderRef, m: usize, k: usize, n: usize) {
    let kk = kernels();
    match hw_model().pick_sgemm(m, k, n) {
        SgemmVariant::Mps => {
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
        SgemmVariant::Simd64 | SgemmVariant::Simd64SplitK => {
            // 64×64 tile, 8 simdgroups (256 threads). pick_sgemm guarantees 64/8-alignment.
            // SplitK reaches here only from call sites that can't pre-zero C (it needs
            // atomic accumulate); the non-split kernel is correct, just not split.
            enc.set_compute_pipeline_state(&kk.sgemm_simd64);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: (n / 64) as u64,
                    height: (m / 64) as u64,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 8,
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

/// C = A @ B via custom MSL kernel (separate GPU buffers).
pub fn encode_sgemm_buffers(
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    m: usize,
    k: usize,
    n: usize,
) {
    let m_u = m as u32;
    let k_u = k as u32;
    let n_u = n as u32;
    enc.set_buffer(0, Some(a), 0);
    enc.set_buffer(1, Some(b), 0);
    enc.set_buffer(2, Some(c), 0);
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
    dispatch_sgemm_variant(enc, m, k, n);
}

/// Standalone `C = A @ B` on shared GPU buffers (custom op / host sync point).
pub fn buffers_sgemm_sync(a: &Buffer, b: &Buffer, c: &Buffer, m: usize, k: usize, n: usize) {
    let Some(dev) = metal_device() else {
        return;
    };
    let cmd = dev.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_sgemm_buffers(enc, a, b, c, m, k, n);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// C = A @ B via custom MSL kernel. Issues set_pipeline+dispatch on a shared
/// compute encoder; caller is responsible for encoder lifecycle.
pub fn metal_sgemm_bufs(
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    a_off: usize,
    b: &Buffer,
    b_off: usize,
    c: &Buffer,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let m_u = m as u32;
    let k_u = k as u32;
    let n_u = n as u32;
    enc.set_buffer(0, Some(a), a_off as u64);
    enc.set_buffer(1, Some(b), b_off as u64);
    enc.set_buffer(2, Some(c), c_off as u64);
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
    // Split-K needs a pre-zeroed C (atomic accumulate) + a 3D grid + the split
    // count — handle it here where we hold the C buffer; other variants dispatch
    // straight through. pick_sgemm only returns SplitK when ksplits >= 4.
    if matches!(hw_model().pick_sgemm(m, k, n), SgemmVariant::Simd64SplitK) {
        let kk = kernels();
        let s = hw_model().ksplits(m, k, n).max(1);
        // zero C[c_off .. c_off + m*n]
        let cn = (m * n) as u32;
        enc.set_compute_pipeline_state(&kk.zero_f32);
        enc.set_buffer(0, Some(c), c_off as u64);
        enc.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &cn as *const _ as *const _,
        );
        enc.dispatch_threads(
            MTLSize {
                width: (m * n) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        // split-K accumulate (A/B/C/M/K/N already bound at 0..5; add Ksplits at 6)
        enc.set_buffer(0, Some(a), a_off as u64);
        enc.set_buffer(1, Some(b), b_off as u64);
        enc.set_buffer(2, Some(c), c_off as u64);
        enc.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &s as *const _ as *const _,
        );
        enc.set_compute_pipeline_state(&kk.sgemm_simd64_splitk);
        enc.dispatch_thread_groups(
            MTLSize {
                width: (n / 64) as u64,
                height: (m / 64) as u64,
                depth: s as u64,
            },
            MTLSize {
                width: 32,
                height: 8,
                depth: 1,
            },
        );
        return;
    }
    dispatch_sgemm_variant(enc, m, k, n);
}

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
    metal_sgemm_bufs(enc, arena, a_off, arena, b_off, arena, c_off, m, k, n);
}

/// `C = A·B + R` with the full `[m,n]` residual `R` folded into the matmul's
/// store — one dispatch instead of a matmul plus a separate elementwise-add.
/// Only the `SimdPadded` variant has a residual-epilogue kernel (the shape the
/// batch-1 decode projections hit: `m=1, k≥256, n≥256, k%8==0`). For any other
/// picked variant this writes `C = A·B` only and returns `false` so the caller
/// adds `R` itself — correctness is preserved everywhere; the launch saving is
/// taken on the decode hot path. Returns `true` iff `R` was applied in-kernel.
#[must_use]
pub fn metal_sgemm_residual_bufs(
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    a_off: usize,
    b: &Buffer,
    b_off: usize,
    c: &Buffer,
    c_off: usize,
    r: &Buffer,
    r_off: usize,
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    if !matches!(hw_model().pick_sgemm(m, k, n), SgemmVariant::SimdPadded) {
        metal_sgemm_bufs(enc, a, a_off, b, b_off, c, c_off, m, k, n);
        return false;
    }
    let kk = kernels();
    let (m_u, k_u, n_u) = (m as u32, k as u32, n as u32);
    enc.set_compute_pipeline_state(&kk.sgemm_simd_padded_residual);
    enc.set_buffer(0, Some(a), a_off as u64);
    enc.set_buffer(1, Some(b), b_off as u64);
    enc.set_buffer(2, Some(c), c_off as u64);
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
    enc.set_buffer(6, Some(r), r_off as u64);
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
    true
}

/// C = A_f32 @ B_f16 → C_f32. Loads half weights in-kernel (no full-matrix cast).
pub fn metal_sgemm_f16w_bufs(
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    a_off: usize,
    b: &Buffer,
    b_off: usize,
    c: &Buffer,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let kk = kernels();
    let m_u = m as u32;
    let k_u = k as u32;
    let n_u = n as u32;
    enc.set_buffer(0, Some(a), a_off as u64);
    enc.set_buffer(1, Some(b), b_off as u64);
    enc.set_buffer(2, Some(c), c_off as u64);
    enc.set_bytes(3, 4, &m_u as *const _ as *const _);
    enc.set_bytes(4, 4, &k_u as *const _ as *const _);
    enc.set_bytes(5, 4, &n_u as *const _ as *const _);

    // M=1 decode GEMV: K-split kernel launches KSPLIT× more threads to saturate
    // memory on the small-N projections (24.4→33→45.7 tok/s across f32→f16→this
    // on qwen3-0.6B, token-identical). Default on the F16 decode path; opt out
    // with `RLX_METAL_GEMV_SPLITK=0`.
    // half2 loads need an even N (2 columns/thread, aligned); all transformer
    // projection widths are even. Odd N falls through to the scalar kernel.
    if m == 1
        && n >= 64
        && n.is_multiple_of(2)
        && rlx_ir::env::var("RLX_METAL_GEMV_SPLITK").as_deref() != Some("0")
    {
        const KSPLIT: u64 = 32;
        enc.set_compute_pipeline_state(&kk.gemv_f16w_splitk);
        // Buffers 0-2 already bound above; this kernel takes K,N at 3,4 (no M).
        enc.set_bytes(3, 4, &k_u as *const _ as *const _);
        enc.set_bytes(4, 4, &n_u as *const _ as *const _);
        // 64 columns per threadgroup (32 threads × 2 cols via half2).
        let tg_count = MTLSize {
            width: (n as u64).div_ceil(64),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 32,
            height: KSPLIT,
            depth: 1,
        };
        enc.dispatch_thread_groups(tg_count, tg);
        return;
    }
    // Prefer small-M column kernel for CFG decode (M=2): shares B loads across
    // rows. Padded simdgroup zeros 6/8 A rows and is slower for skinny M.
    if m <= 4 && n >= 64 {
        enc.set_compute_pipeline_state(&kk.sgemm_f16w_small_m);
        let grid = MTLSize {
            width: n as u64,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 64u64.min(n as u64).max(1),
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        return;
    }
    // Prefer padded simd for large decode Linears; naive for tiny dims.
    let use_padded = matches!(
        hw_model().pick_sgemm(m, k, n),
        SgemmVariant::SimdPadded
            | SgemmVariant::Simd
            | SgemmVariant::Simd4x4
            | SgemmVariant::Mps
            | SgemmVariant::Tiled
    ) && k >= 256
        && n >= 256;
    if use_padded {
        enc.set_compute_pipeline_state(&kk.sgemm_simd_padded_f16w);
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
    } else {
        enc.set_compute_pipeline_state(&kk.sgemm_f16w);
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

pub fn metal_sgemm_f16w(
    enc: &ComputeCommandEncoderRef,
    arena: &Buffer,
    a_off: usize,
    b_off: usize,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    metal_sgemm_f16w_bufs(enc, arena, a_off, arena, b_off, arena, c_off, m, k, n);
}

/// C = A_f16 @ B_f32 → C_f32, f32 accumulate. Mirror of `metal_sgemm_f16w_bufs`
/// for the A operand: reads the 2-byte half A in-kernel instead of letting the
/// plain f32 sgemm reinterpret it as f32. Used by the mixed-precision training
/// backward, where grad matmuls take an f16 activation and an f32 upstream grad.
pub fn metal_sgemm_f16a_bufs(
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    a_off: usize,
    b: &Buffer,
    b_off: usize,
    c: &Buffer,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let kk = kernels();
    let (m_u, k_u, n_u) = (m as u32, k as u32, n as u32);
    enc.set_buffer(0, Some(a), a_off as u64);
    enc.set_buffer(1, Some(b), b_off as u64);
    enc.set_buffer(2, Some(c), c_off as u64);
    enc.set_bytes(3, 4, &m_u as *const _ as *const _);
    enc.set_bytes(4, 4, &k_u as *const _ as *const _);
    enc.set_bytes(5, 4, &n_u as *const _ as *const _);
    enc.set_compute_pipeline_state(&kk.sgemm_f16a);
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
    metal_hgemm_bufs(enc, arena, a_off, arena, b_off, arena, c_off, m, k, n);
}

pub fn metal_hgemm_bufs(
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    a_off: usize,
    b: &Buffer,
    b_off: usize,
    c: &Buffer,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    let kk = kernels();
    let m_u = m as u32;
    let k_u = k as u32;
    let n_u = n as u32;
    enc.set_buffer(0, Some(a), a_off as u64);
    enc.set_buffer(1, Some(b), b_off as u64);
    enc.set_buffer(2, Some(c), c_off as u64);
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
