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
pub mod cost;
pub mod dequant_cache;
pub mod dispatch;
pub mod executor;
pub mod gdn;
pub mod gguf_matmul;
pub mod intrinsics;
pub mod kernel_config;
pub mod kernels;
pub mod llada2_gate;
pub mod lm_head;
pub mod moe_residency;
pub mod moe_topk_capture;
pub mod naive;
pub mod op_registry;
pub mod pool;
pub mod splat;
pub mod thunk;
pub mod tile;
pub mod training_bwd;
pub mod umap_knn;
