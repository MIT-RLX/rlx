// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RLX CPU backend — executes optimized IR graphs on CPU.
//!
//! Takes a fused + memory-planned IR graph and executes it using:
//! - BLAS (Accelerate/MKL/OpenBLAS) for matmul
//! - NEON/AVX SIMD kernels for element-wise ops
//! - Persistent Rayon thread pool for parallelism
//! - Arena allocator for zero per-call allocation

pub mod arena;
pub mod asm_check;
pub mod attention_bwd;
pub mod autotune;
pub mod blas;
pub mod calibrate;
pub mod config;
pub mod conv3d_bwd;
pub mod conv_bwd;
pub mod conv_fwd;
pub mod cost;
pub mod dequant_cache;
pub mod dispatch;
pub mod executor;
pub mod expand;
pub use expand::{expand_cpu_nop_fused, prepare_graph_for_thunks};
pub mod gdn;
pub mod gguf_matmul;
pub mod gguf_scheme;
pub mod iir; // IIR biquad zero-phase Butterworth filtering (filter-bank front-end)
pub mod spd; // SPD-manifold Riemannian kernels (logm/expm/sqrtm, AIRM distance, Karcher mean, batched eigh)
pub mod spd_kernels; // CpuKernel impls for the core SPDNet ops (BiMap/ReEig/LogEig/SpdBatchNorm)
pub use gguf_scheme::quant_scheme_for_ggml;
pub mod im2col;
pub mod intrinsics;
pub mod kernel_config;
pub mod kernels;
pub mod llada2_gate;
pub mod lm_head;
pub mod moe_residency;
pub mod moe_topk_capture;
pub mod ms_deform_attn;
pub mod naive;
pub mod onnx_control_flow;
pub mod onnx_indexing;
pub mod onnx_ref;
pub mod op_registry;
pub mod supported_ops;
pub use supported_ops::SUPPORTED_OPS;
pub mod pool;
pub mod splat;
pub mod thunk;
pub mod tile;
pub mod training_bwd;
pub mod umap_knn;
pub mod vmath;
