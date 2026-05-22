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

//! Kernel sources for the HIP backend.
//!
//! HIP and CUDA C++ are syntactically identical for the kernel
//! constructs we use (`__global__`, `__device__`, `__shared__`,
//! `__syncthreads`, `extern "C"`). The `.cu` files in
//! `rlx-cuda/src/kernels/` work unchanged under hipRTC. We pull
//! them in via relative `include_str!` rather than copy/forking
//! so optimization improvements in `rlx-cuda` automatically flow
//! to `rlx-rocm`.
//!
//! When the hipRTC compile + cuModule cache lands, mirror
//! `rlx-cuda::kernels::compile` (NVRTC → hipRTC), the
//! `kernel_cache!` macro, and the persistent PTX disk cache (HIP
//! has its own .hsaco binary format; same hash-by-source pattern).

pub const BINARY_CU: &str = include_str!("../../../rlx-cuda/src/kernels/binary.cu");
pub const FUSED_BINARY_UNARY_CU: &str =
    include_str!("../../../rlx-cuda/src/kernels/fused_binary_unary.cu");
pub const CAST_F32_TO_HALF_CU: &str =
    include_str!("../../../rlx-cuda/src/kernels/cast_f32_to_half.cu");
pub const UNARY_CU: &str = include_str!("../../../rlx-cuda/src/kernels/unary.cu");
pub const COPY_CU: &str = include_str!("../../../rlx-cuda/src/kernels/copy.cu");
pub const MATMUL_CU: &str = include_str!("../../../rlx-cuda/src/kernels/matmul.cu");
pub const MATMUL_EPILOGUE_CU: &str =
    include_str!("../../../rlx-cuda/src/kernels/matmul_epilogue.cu");
/// AMD-specific matrix-core matmul via rocWMMA. Local to rlx-rocm —
/// rlx-cuda has its own `matmul_wmma.cu` using nvcuda::wmma.
/// Opt-in at runtime via `RLX_ROCM_MFMA=1`.
pub const MATMUL_MFMA_CU: &str = include_str!("matmul_mfma.cu");
// matmul_wmma.cu is intentionally excluded: it uses NVIDIA's
// `nvcuda::wmma` namespace which has no direct AMD equivalent.
// AMD provides matrix-core intrinsics via different headers
// (`__builtin_amdgcn_mfma_*` on CDNA, WMMA intrinsics on RDNA3+).
// Will land as a separate `matmul_mfma.cu` when hipRTC dispatch
// is wired and we know which arch to target.
pub const COMPARE_CU: &str = include_str!("../../../rlx-cuda/src/kernels/compare.cu");
pub const WHERE_CU: &str = include_str!("../../../rlx-cuda/src/kernels/where_select.cu");
pub const REDUCE_CU: &str = include_str!("../../../rlx-cuda/src/kernels/reduce.cu");
pub const SOFTMAX_CU: &str = include_str!("../../../rlx-cuda/src/kernels/softmax.cu");
pub const LAYERNORM_CU: &str = include_str!("../../../rlx-cuda/src/kernels/layernorm.cu");
pub const FUSED_RESIDUAL_LN_CU: &str =
    include_str!("../../../rlx-cuda/src/kernels/fused_residual_ln.cu");
pub const GATHER_CU: &str = include_str!("../../../rlx-cuda/src/kernels/gather.cu");
pub const GATHER_AXIS_CU: &str = include_str!("../../../rlx-cuda/src/kernels/gather_axis.cu");
pub const NARROW_CU: &str = include_str!("../../../rlx-cuda/src/kernels/narrow.cu");
pub const CONCAT_CU: &str = include_str!("../../../rlx-cuda/src/kernels/concat.cu");
pub const TRANSPOSE_CU: &str = include_str!("../../../rlx-cuda/src/kernels/transpose.cu");
pub const EXPAND_CU: &str = include_str!("../../../rlx-cuda/src/kernels/expand.cu");
pub const ATTENTION_CU: &str = include_str!("../../../rlx-cuda/src/kernels/attention.cu");
pub const ATTENTION_BWD_CU: &str = include_str!("../../../rlx-cuda/src/kernels/attention_bwd.cu");
pub const ARGMAX_CU: &str = include_str!("../../../rlx-cuda/src/kernels/argmax.cu");
pub const ROPE_CU: &str = include_str!("../../../rlx-cuda/src/kernels/rope.cu");
pub const CUMSUM_CU: &str = include_str!("../../../rlx-cuda/src/kernels/cumsum.cu");
pub const TOPK_CU: &str = include_str!("../../../rlx-cuda/src/kernels/topk.cu");
pub const GROUPED_MATMUL_CU: &str = include_str!("../../../rlx-cuda/src/kernels/grouped_matmul.cu");
pub const SCATTER_ADD_CU: &str = include_str!("../../../rlx-cuda/src/kernels/scatter_add.cu");
pub const DEQUANT_MATMUL_CU: &str = include_str!("../../../rlx-cuda/src/kernels/dequant_matmul.cu");
pub const SAMPLE_CU: &str = include_str!("../../../rlx-cuda/src/kernels/sample.cu");
pub const SELECTIVE_SCAN_CU: &str = include_str!("../../../rlx-cuda/src/kernels/selective_scan.cu");
pub const POOL1D_CU: &str = include_str!("../../../rlx-cuda/src/kernels/pool1d.cu");
pub const POOL2D_CU: &str = include_str!("../../../rlx-cuda/src/kernels/pool2d.cu");
pub const POOL3D_CU: &str = include_str!("../../../rlx-cuda/src/kernels/pool3d.cu");
pub const CONV1D_CU: &str = include_str!("../../../rlx-cuda/src/kernels/conv1d.cu");
pub const CONV2D_CU: &str = include_str!("../../../rlx-cuda/src/kernels/conv2d.cu");
pub const CONV3D_CU: &str = include_str!("../../../rlx-cuda/src/kernels/conv3d.cu");
pub const ELEMENTWISE_REGION_CU: &str =
    include_str!("../../../rlx-cuda/src/kernels/elementwise_region.cu");
pub const GAUSSIAN_SPLAT_RASTERIZE_CU: &str =
    include_str!("../../../rlx-cuda/src/kernels/gaussian_splat_rasterize.cu");

/// Total number of kernel entry points the HIP backend will need to
/// compile (= rlx-cuda's count minus matmul_wmma which doesn't port
/// directly to AMD's matrix-core intrinsics). 33 today
/// (32 base + ElementwiseRegion landed for PLAN L2).
pub const KERNEL_COUNT: usize = 33;
