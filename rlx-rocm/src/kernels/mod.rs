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

//! HIP kernel sources + hipRTC compile cache.
//!
//! Mirror of `rlx-cuda::kernels`. Each `.cu` source compiles via
//! hipRTC the first time it's needed, then the resulting `.hsaco`
//! binary lives in a `OnceLock<HipKernel>` for the rest of the
//! process. Persistent disk cache under
//! `$RLX_ROCM_HSACO_CACHE` / `$XDG_CACHE_HOME/rlx-rocm/hsaco-rocm`
//! follows the same shape as rlx-cuda's PTX cache.

mod sources;
pub use sources::*;

use std::sync::Arc;
use std::sync::OnceLock;

use crate::device::RocmContext;
use crate::hip::HipKernel;

/// Disk cache directory for compiled `.hsaco` blobs. Returns `None`
/// to disable caching.
fn hsaco_cache_dir() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Ok(p) = std::env::var("RLX_ROCM_HSACO_CACHE") {
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
    Some(base.join("rlx-rocm").join("hsaco-rocm"))
}

fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn compile(ctx: &Arc<RocmContext>, src: &str, entry: &str) -> HipKernel {
    let cache_path =
        hsaco_cache_dir().map(|d| d.join(format!("{}-{:016x}.hsaco", entry, fnv1a64(src))));

    let hsaco: Vec<u8> = if let Some(ref p) = cache_path {
        if let Ok(bytes) = std::fs::read(p) {
            bytes
        } else {
            let fresh = ctx
                .runtime
                .hiprtc_compile_to_hsaco(src, entry)
                .unwrap_or_else(|e| panic!("rlx-rocm: hipRTC compile failed for {entry}: {e}"));
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let tmp = p.with_extension("hsaco.tmp");
            if std::fs::write(&tmp, &fresh).is_ok() {
                let _ = std::fs::rename(&tmp, p);
            }
            fresh
        }
    } else {
        ctx.runtime
            .hiprtc_compile_to_hsaco(src, entry)
            .unwrap_or_else(|e| panic!("rlx-rocm: hipRTC compile failed for {entry}: {e}"))
    };

    HipKernel::from_hsaco(&ctx.runtime, &hsaco, entry)
        .unwrap_or_else(|e| panic!("rlx-rocm: hipModuleLoadData failed for {entry}: {e}"))
}

macro_rules! kernel_cache {
    ($static_name:ident, $fn_name:ident, $src:expr, $entry:expr) => {
        static $static_name: OnceLock<HipKernel> = OnceLock::new();
        pub fn $fn_name(ctx: &Arc<RocmContext>) -> &'static HipKernel {
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
    MATMUL_MFMA,
    matmul_mfma_kernel,
    MATMUL_MFMA_CU,
    "matmul_mfma"
);
kernel_cache!(COMPARE, compare_kernel, COMPARE_CU, "compare");
kernel_cache!(WHEREK, where_kernel, WHERE_CU, "where_select");
kernel_cache!(REDUCE, reduce_kernel, REDUCE_CU, "reduce");
kernel_cache!(SOFTMAX, softmax_kernel, SOFTMAX_CU, "softmax");
kernel_cache!(LAYERNORM, layernorm_kernel, LAYERNORM_CU, "norm");
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

pub fn dispatch_grid_1d(n: u32, block_x: u32) -> (u32, u32) {
    (n.div_ceil(block_x), block_x)
}

/// AOT pre-warm: force-compile every kernel up-front. Mirrors
/// `rlx-cuda::backend::prewarm_all`.
pub fn prewarm_all(ctx: &Arc<RocmContext>) {
    let _ = binary_kernel(ctx);
    let _ = fused_binary_unary_kernel(ctx);
    let _ = unary_kernel(ctx);
    let _ = copy_kernel(ctx);
    let _ = matmul_kernel(ctx);
    let _ = matmul_epilogue_kernel(ctx);
    let _ = compare_kernel(ctx);
    let _ = where_kernel(ctx);
    let _ = reduce_kernel(ctx);
    let _ = softmax_kernel(ctx);
    let _ = layernorm_kernel(ctx);
    let _ = fused_residual_ln_kernel(ctx);
    let _ = gather_kernel(ctx);
    let _ = narrow_kernel(ctx);
    let _ = concat_kernel(ctx);
    let _ = transpose_kernel(ctx);
    let _ = expand_kernel(ctx);
    let _ = attention_kernel(ctx);
    let _ = argmax_kernel(ctx);
    let _ = rope_kernel(ctx);
    let _ = cumsum_kernel(ctx);
    let _ = topk_kernel(ctx);
    let _ = grouped_matmul_kernel(ctx);
    let _ = scatter_add_zero_kernel(ctx);
    let _ = scatter_add_acc_kernel(ctx);
    let _ = dequant_matmul_kernel(ctx);
    let _ = sample_kernel(ctx);
    let _ = selective_scan_kernel(ctx);
    let _ = pool1d_kernel(ctx);
    let _ = pool2d_kernel(ctx);
    let _ = pool3d_kernel(ctx);
    let _ = conv1d_kernel(ctx);
    let _ = conv2d_kernel(ctx);
    let _ = conv3d_kernel(ctx);
    let _ = elementwise_region_kernel(ctx);
}
