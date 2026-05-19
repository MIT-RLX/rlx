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

//! CUDA C++ kernel sources + NVRTC compilation cache.
//!
//! Each `.cu` file is embedded as a `&'static str`, compiled to PTX
//! via NVRTC the first time it's needed, then loaded into a `cuModule`
//! and cached in a `OnceLock` for the rest of the process. Pure
//! NVRTC — no nvcc at workspace build time.

use std::sync::Arc;
use std::sync::OnceLock;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};

pub const BINARY_CU: &str = include_str!("binary.cu");
pub const FUSED_BINARY_UNARY_CU: &str = include_str!("fused_binary_unary.cu");
pub const CAST_F32_TO_HALF_CU: &str = include_str!("cast_f32_to_half.cu");
pub const UNARY_CU: &str = include_str!("unary.cu");
pub const COPY_CU: &str = include_str!("copy.cu");
pub const MATMUL_CU: &str = include_str!("matmul.cu");
pub const MATMUL_EPILOGUE_CU: &str = include_str!("matmul_epilogue.cu");
pub const MATMUL_WMMA_CU: &str = include_str!("matmul_wmma.cu");
pub const COMPARE_CU: &str = include_str!("compare.cu");
pub const WHERE_CU: &str = include_str!("where_select.cu");
pub const REDUCE_CU: &str = include_str!("reduce.cu");
pub const SOFTMAX_CU: &str = include_str!("softmax.cu");
pub const LAYERNORM_CU: &str = include_str!("layernorm.cu");
pub const FUSED_RESIDUAL_LN_CU: &str = include_str!("fused_residual_ln.cu");
pub const GATHER_CU: &str = include_str!("gather.cu");
pub const NARROW_CU: &str = include_str!("narrow.cu");
pub const CONCAT_CU: &str = include_str!("concat.cu");
pub const TRANSPOSE_CU: &str = include_str!("transpose.cu");
pub const EXPAND_CU: &str = include_str!("expand.cu");
pub const ATTENTION_CU: &str = include_str!("attention.cu");
pub const ARGMAX_CU: &str = include_str!("argmax.cu");
pub const ROPE_CU: &str = include_str!("rope.cu");
pub const CUMSUM_CU: &str = include_str!("cumsum.cu");
pub const TOPK_CU: &str = include_str!("topk.cu");
pub const GROUPED_MATMUL_CU: &str = include_str!("grouped_matmul.cu");
pub const SCATTER_ADD_CU: &str = include_str!("scatter_add.cu");
pub const DEQUANT_MATMUL_CU: &str = include_str!("dequant_matmul.cu");
pub const SAMPLE_CU: &str = include_str!("sample.cu");
pub const SELECTIVE_SCAN_CU: &str = include_str!("selective_scan.cu");
pub const POOL1D_CU: &str = include_str!("pool1d.cu");
pub const POOL2D_CU: &str = include_str!("pool2d.cu");
pub const POOL3D_CU: &str = include_str!("pool3d.cu");
pub const CONV1D_CU: &str = include_str!("conv1d.cu");
pub const CONV2D_CU: &str = include_str!("conv2d.cu");
pub const CONV3D_CU: &str = include_str!("conv3d.cu");
pub const ELEMENTWISE_REGION_CU: &str = include_str!("elementwise_region.cu");

/// One compiled NVRTC module + the function handle we use from it.
pub struct CudaKernel {
    pub module: Arc<CudaModule>,
    pub function: CudaFunction,
}

/// Persistent PTX disk cache directory. Resolved once at startup from
/// `RLX_CUDA_PTX_CACHE` (explicit override) or `XDG_CACHE_HOME` /
/// `~/.cache`, namespaced by the cuda toolkit version baked into the
/// crate. Returning `None` disables caching (still works, just slower
/// cold-start).
fn ptx_cache_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Ok(p) = std::env::var("RLX_CUDA_PTX_CACHE") {
        return Some(PathBuf::from(p));
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cache"))
        })?;
    Some(base.join("rlx-cuda").join("ptx-cuda-12060"))
}

/// FNV-1a 64-bit. Cheap and deterministic; collision-resistance is
/// good enough for filename hashing where source mismatch is the only
/// failure mode (we re-compile on cache miss, so no correctness risk).
fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn compile(ctx: &Arc<CudaContext>, src: &str, entry: &str) -> CudaKernel {
    // Try the disk cache first. The cache key folds the kernel entry
    // name into the source hash so different entry-points sharing a
    // .cu file (scatter_add_zero / scatter_add_acc) get distinct
    // cache slots.
    let cache_path =
        ptx_cache_dir().map(|d| d.join(format!("{}-{:016x}.ptx", entry, fnv1a64(src))));

    let ptx = if let Some(ref p) = cache_path {
        if let Ok(cached) = std::fs::read_to_string(p) {
            cudarc::nvrtc::Ptx::from_src(cached)
        } else {
            let fresh = cudarc::nvrtc::compile_ptx(src)
                .unwrap_or_else(|e| panic!("rlx-cuda: NVRTC compile failed for {entry}: {e}"));
            // Best-effort write to the cache. Atomic via tmp + rename
            // so a crash mid-write doesn't poison the cache.
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let tmp = p.with_extension("ptx.tmp");
            if std::fs::write(&tmp, fresh.to_src()).is_ok() {
                let _ = std::fs::rename(&tmp, p);
            }
            fresh
        }
    } else {
        cudarc::nvrtc::compile_ptx(src)
            .unwrap_or_else(|e| panic!("rlx-cuda: NVRTC compile failed for {entry}: {e}"))
    };

    let module = ctx
        .load_module(ptx)
        .unwrap_or_else(|e| panic!("rlx-cuda: load_module failed for {entry}: {e}"));
    let function = module
        .load_function(entry)
        .unwrap_or_else(|e| panic!("rlx-cuda: load_function {entry}: {e}"));
    CudaKernel { module, function }
}

macro_rules! kernel_cache {
    ($static_name:ident, $fn_name:ident, $src:expr, $entry:expr) => {
        static $static_name: OnceLock<CudaKernel> = OnceLock::new();
        pub fn $fn_name(ctx: &Arc<CudaContext>) -> &'static CudaKernel {
            $static_name.get_or_init(|| compile(ctx, $src, $entry))
        }
    };
}

kernel_cache!(BINARY, binary_kernel, BINARY_CU, "binary");
kernel_cache!(
    FUSED_BINARY_UNARY,
    fused_binary_unary_kernel,
    FUSED_BINARY_UNARY_CU,
    "fused_binary_unary"
);
kernel_cache!(
    CAST_F32_TO_HALF,
    cast_f32_to_half_kernel,
    CAST_F32_TO_HALF_CU,
    "cast_f32_to_half"
);
kernel_cache!(UNARY, unary_kernel, UNARY_CU, "unary");
kernel_cache!(COPY, copy_kernel, COPY_CU, "copy");
kernel_cache!(MATMUL, matmul_kernel, MATMUL_CU, "matmul");
kernel_cache!(
    MATMUL_EPILOGUE,
    matmul_epilogue_kernel,
    MATMUL_EPILOGUE_CU,
    "matmul_epilogue"
);
kernel_cache!(
    MATMUL_WMMA,
    matmul_wmma_kernel,
    MATMUL_WMMA_CU,
    "matmul_wmma"
);
kernel_cache!(COMPARE, compare_kernel, COMPARE_CU, "compare");
kernel_cache!(WHEREK, where_kernel, WHERE_CU, "where_select");
kernel_cache!(REDUCE, reduce_kernel, REDUCE_CU, "reduce");
kernel_cache!(SOFTMAX, softmax_kernel, SOFTMAX_CU, "softmax");
kernel_cache!(LAYERNORM, layernorm_kernel, LAYERNORM_CU, "rlx_norm");
kernel_cache!(
    FUSED_RESIDUAL_LN,
    fused_residual_ln_kernel,
    FUSED_RESIDUAL_LN_CU,
    "fused_residual_ln"
);
kernel_cache!(GATHER, gather_kernel, GATHER_CU, "gather");
kernel_cache!(NARROW, narrow_kernel, NARROW_CU, "narrow");
kernel_cache!(CONCAT, concat_kernel, CONCAT_CU, "concat");
kernel_cache!(TRANSPOSE, transpose_kernel, TRANSPOSE_CU, "transpose");
kernel_cache!(EXPAND, expand_kernel, EXPAND_CU, "expand");
kernel_cache!(ATTENTION, attention_kernel, ATTENTION_CU, "attention");
kernel_cache!(ARGMAX, argmax_kernel, ARGMAX_CU, "argmax");
kernel_cache!(ROPE, rope_kernel, ROPE_CU, "rope");
kernel_cache!(CUMSUM, cumsum_kernel, CUMSUM_CU, "cumsum");
kernel_cache!(TOPK, topk_kernel, TOPK_CU, "topk");
kernel_cache!(
    GROUPED_MATMUL,
    grouped_matmul_kernel,
    GROUPED_MATMUL_CU,
    "grouped_matmul"
);
kernel_cache!(
    SCATTER_ADD_ZERO,
    scatter_add_zero_kernel,
    SCATTER_ADD_CU,
    "scatter_add_zero"
);
kernel_cache!(
    SCATTER_ADD_ACC,
    scatter_add_acc_kernel,
    SCATTER_ADD_CU,
    "scatter_add_acc"
);
kernel_cache!(
    DEQUANT_MATMUL,
    dequant_matmul_kernel,
    DEQUANT_MATMUL_CU,
    "dequant_matmul"
);
kernel_cache!(SAMPLE, sample_kernel, SAMPLE_CU, "sample");
kernel_cache!(
    SELECTIVE_SCAN,
    selective_scan_kernel,
    SELECTIVE_SCAN_CU,
    "selective_scan"
);
kernel_cache!(POOL1D, pool1d_kernel, POOL1D_CU, "pool1d");
kernel_cache!(POOL2D, pool2d_kernel, POOL2D_CU, "pool2d");
kernel_cache!(POOL3D, pool3d_kernel, POOL3D_CU, "pool3d");
kernel_cache!(CONV1D, conv1d_kernel, CONV1D_CU, "conv1d");
kernel_cache!(CONV2D, conv2d_kernel, CONV2D_CU, "conv2d");
kernel_cache!(CONV3D, conv3d_kernel, CONV3D_CU, "conv3d");
kernel_cache!(
    ELEMENTWISE_REGION,
    elementwise_region_kernel,
    ELEMENTWISE_REGION_CU,
    "elementwise_region"
);

/// Dispatch grid for a 1-D workload of `n` threads with workgroup
/// size `block_x`. CUDA's per-grid-dim limit is 2^31-1 on the X axis,
/// so the 2-D fallback wgpu requires isn't needed here.
pub fn dispatch_grid_1d(n: u32, block_x: u32) -> (u32, u32) {
    (n.div_ceil(block_x), block_x)
}
