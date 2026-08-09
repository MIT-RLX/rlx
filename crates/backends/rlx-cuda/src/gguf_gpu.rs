// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU GGUF dequant + cuBLAS matmul for `Op::DequantMatMul` and grouped MoE
//! `Op::DequantGroupedMatMul`.
//!
//! Flow: launch `dequant_gguf` into arena scratch, then `C = X @ W^T` via
//! the shared `matmul_bt` kernel (GGUF row-major `[n,k]` weights).
//!
//! See [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use rlx_ir::{Graph, Op};
use std::sync::Arc;

use crate::gguf_host::scheme_from_id;
use crate::kernels::{
    dequant_gguf_kernel, dequant_matmul_gguf_kernel, dequant_matmul_gguf_q1_gemv_kernel,
    dequant_matmul_gguf_q4k_gemv_kernel, matmul_bt_kernel, matmul_bt_tma_kernel,
};

fn slab_bytes_for(scheme: rlx_ir::quant::QuantScheme, k: usize, n: usize) -> usize {
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    (k * n) / block_elems * block_bytes
}

pub fn gguf_fused_m1_env_disabled() -> bool {
    matches!(
        rlx_ir::env::var("RLX_CUDA_GGUF_FUSED_M1")
            .or_else(|| rlx_ir::env::var("ORPHEUS_CUDA_GGUF_FUSED_M1"))
            .as_deref(),
        Some("0") | Some("false") | Some("off")
    )
}

/// Max f32 scratch for dequantized weights `[n, k]` across all GGUF ops.
pub fn dequant_gguf_scratch_bytes(graph: &Graph) -> usize {
    let fused_disabled = gguf_fused_m1_env_disabled();
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::DequantMatMul { scheme } = &node.op
            && scheme.is_gguf()
        {
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            let total = node.shape.num_elements().unwrap();
            let m = total / n.max(1);
            let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
            let k = x_total / m.max(1);
            // Decode GEMV (m==1, fused, scratch-free — see backend.rs) needs no
            // f32 dequant slab. Skipping it avoids reserving the multi-GiB
            // tied-embed dequant scratch (Gemma 4 12B lm_head: 3840*262144*4 =
            // 4 GiB) that would push the arena past 16 GiB VRAM and OOM.
            // When `RLX_CUDA_GGUF_FUSED_M1=0`, keep planning scratch so the
            // dequant+matmul fallback stays on-device (or host if scratch is 0).
            if m == 1
                && !fused_disabled
                && gguf_fused_gemv_m1_supported(crate::gguf_host::gguf_scheme_id(*scheme), m, k)
            {
                continue;
            }
            max = max.max(k * n * std::mem::size_of::<f32>());
        }
        // MxFp4x2 decodes into the same f32 [n,k] scratch before matmul_bt.
        if let Op::DequantMatMul { scheme } = &node.op
            && scheme.mxfp4x2_config().is_some()
        {
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            let total = node.shape.num_elements().unwrap();
            let m = total / n.max(1);
            let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
            let k = x_total / m.max(1);
            max = max.max(k * n * std::mem::size_of::<f32>());
        }
        if let Op::DequantGroupedMatMul { scheme } = &node.op {
            let in_shape = &graph.node(node.inputs[0]).shape;
            let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
            let k = in_shape.dim(in_shape.rank() - 1).unwrap_static();
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            // dequant slab + packed input + packed output (bytes).
            max = max.max(k * n * 4 + m * k * 4 + m * n * 4);
            let _ = scheme;
        }
    }
    max
}

/// Fused on-device GEMV for decode (`m == 1`) on Q4_K / Q6_K / Q1_0 —
/// matches rlx-vulkan `dequant_matmul` and rlx-cpu `gguf_matmul_bt`.
pub fn gguf_fused_gemv_m1_supported(scheme_id: u32, m: usize, k: usize) -> bool {
    if m != 1 {
        return false;
    }
    match scheme_id {
        // Q4_K / Q6_K: 256-elem super-blocks
        0 | 2 => k.is_multiple_of(256),
        // Q1_0 (Bonsai): 128-elem blocks
        24 => k.is_multiple_of(128),
        _ => false,
    }
}

/// Launch fused GGUF GEMV (`m == 1`) — one thread per output column for
/// Q4_K/Q6_K; cooperative block-per-row for Q1_0.
pub fn run_dequant_matmul_gguf_gemv_m1(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    n: usize,
    k: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
) {
    debug_assert!(gguf_fused_gemv_m1_supported(scheme_id, 1, k));
    let x_off = x_byte_off / 4;
    let out_off = out_byte_off / 4;

    // Q1_0: one CTA per output row — threads split the k/128 weight blocks.
    // Cuts the serial-per-row decode that dominated Bonsai-27B tok/s.
    if scheme_id == 24 {
        let kernel = dequant_matmul_gguf_q1_gemv_kernel(ctx);
        let block = 128u32;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_u = n as u64;
        let k_u = k as u64;
        let x_u = x_off as u64;
        let w_u = w_byte_off as u64;
        let out_u = out_off as u64;
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&n_u)
            .arg(&k_u)
            .arg(&x_u)
            .arg(&w_u)
            .arg(&out_u);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: dequant_matmul_gguf_q1_gemv launch failed");
        }
        return;
    }

    // Q4_K: OPT-IN cooperative block-per-row GEMV (`RLX_CUDA_Q4K_GEMV_COOP=1`).
    // Coalesced weight loads (contiguous super-blocks per row) + fused dequant
    // (no local slab) + Neumaier reduce — vs the default one-thread-per-row
    // scalar path whose 32 lanes each stride a whole row. Left opt-in because
    // (a) it's Neumaier-compensated so NOT bit-identical to the scalar path
    // (validated coherent only on LFM2.5-Q4_K so far — this kernel is shared by
    // every CUDA GGUF model), and (b) at small k (few super-blocks/row) the
    // coalescing win is modest. Enable once validated across more models.
    if scheme_id == 0 && rlx_ir::env::flag("RLX_CUDA_Q4K_GEMV_COOP") {
        let kernel = dequant_matmul_gguf_q4k_gemv_kernel(ctx);
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (128u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_u = n as u64;
        let k_u = k as u64;
        let x_u = x_off as u64;
        let w_u = w_byte_off as u64;
        let out_u = out_off as u64;
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&n_u)
            .arg(&k_u)
            .arg(&x_u)
            .arg(&w_u)
            .arg(&out_u);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: dequant_matmul_gguf_q4k_gemv launch failed");
        }
        return;
    }

    let kernel = dequant_matmul_gguf_kernel(ctx);
    let (grid, block) = crate::kernels::dispatch_grid_1d(n as u32, 64);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let x_off_u = x_off as u32;
    let out_off_u = out_off as u32;
    let w_off = w_byte_off as u32;
    let n_u = n as u32;
    let k_u = k as u32;
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&n_u)
        .arg(&k_u)
        .arg(&x_off_u)
        .arg(&w_off)
        .arg(&out_off_u)
        .arg(&scheme_id);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: dequant_matmul_gguf launch failed");
    }
}

/// Launch `matmul_bt`: `C[m,n] = A[m,k] @ W[n,k]^T` with row-major arena offsets.
fn run_matmul_bt(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
    x_byte_off: usize,
    w_byte_off: usize,
    out_byte_off: usize,
    precise_default: bool,
) {
    let kernel = matmul_bt_kernel(ctx);
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(32) as u32, m.div_ceil(32) as u32, 1),
        block_dim: (8, 8, 1),
        shared_mem_bytes: 0,
    };
    let m_u = m as u32;
    let k_u = k as u32;
    let n_u = n as u32;
    // 64-bit element offsets: the packed 27B arena is >4 GB, so a u32 offset
    // truncates and the sgemm reads/writes the wrong rows (garbage logits).
    // Kernel params are `unsigned long long` (see matmul_bt.cu).
    let a_off = (x_byte_off / 4) as u64;
    let b_off = (w_byte_off / 4) as u64;
    let c_off = (out_byte_off / 4) as u64;
    // Double-single (compensated) dot-product accumulation — see matmul_bt.cu.
    // On by default for strict-parity runs; opt-in elsewhere via
    // RLX_CUDA_MATMUL_PRECISE. Improves the k-reduction precision so a coarse
    // 1-bit (Q1_0) model's near-tie argmaxes track the reference.
    // Double-single (compensated) dot-product accumulation — see matmul_bt.cu.
    // Always on for Q1_0 (`precise_default`) unless RLX_CUDA_MATMUL_PRECISE=0.
    // Also via RLX_CUDA_PARITY / RLX_CUDA_NO_TF32. Improves near-tie argmaxes for 1-bit.
    let precise_off = matches!(
        rlx_ir::env::var("RLX_CUDA_MATMUL_PRECISE").as_deref(),
        Some("0") | Some("false") | Some("off")
    );
    let precise: u32 = if precise_off {
        0
    } else {
        (precise_default
            || rlx_ir::env::flag("RLX_CUDA_PARITY")
            || rlx_ir::env::flag("RLX_CUDA_NO_TF32")
            || rlx_ir::env::flag("RLX_CUDA_MATMUL_PRECISE")) as u32
    };

    // Opt-in Hopper TMA NT GEMM (`RLX_CUDA_TMA` on sm_90). Only when compensated
    // accumulation isn't requested (the TMA kernel does plain FMA) and the tile
    // is eligible; otherwise fall through to `matmul_bt`. Inert off sm_90.
    if precise == 0 && crate::backend::tma_arch(ctx).is_some() {
        let base = {
            let (p, _guard) = buffer.device_ptr(stream);
            p
        };
        if let Some((a_map, w_map)) = crate::backend::build_tma_nt_maps(
            base,
            m as u32,
            k as u32,
            n as u32,
            x_byte_off as u64,
            w_byte_off as u64,
        ) {
            let tma = matmul_bt_tma_kernel(ctx);
            let cfg = LaunchConfig {
                grid_dim: (n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let c_off_u = (out_byte_off / 4) as u64;
            // matmul_bt_tma(a_map, w_map, arena, M, K, N, c_off): the two
            // tensor-maps are the leading grid-constant args.
            let mut launcher = stream.launch_builder(&tma.function);
            launcher
                .arg(&a_map)
                .arg(&w_map)
                .arg(&mut *buffer)
                .arg(&m_u)
                .arg(&k_u)
                .arg(&n_u)
                .arg(&c_off_u);
            unsafe {
                launcher
                    .launch(cfg)
                    .expect("rlx-cuda: matmul_bt_tma launch failed");
            }
            return;
        }
    }

    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&m_u)
        .arg(&k_u)
        .arg(&n_u)
        .arg(&a_off)
        .arg(&b_off)
        .arg(&c_off)
        .arg(&precise);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: matmul_bt launch failed");
    }
}

/// MxFp4x2 two-level residual E2M1 `DequantMatMul` on CUDA: decode the packed
/// `[plane0|plane1]` weight + `[s0|s1]` scales into f32 `[n,k]` scratch (the
/// `matmul_bt` row-major convention, via `mxfp4x2_dequant_nk`), then
/// `C = X @ Wᵀ`. Twin of `rlx_rocm::gguf_gpu::run_dequant_matmul_mxfp4x2_gpu`
/// (which uses hipBLAS sgemm(N,N) + the `[k,n]` `mxfp4x2_dequant` kernel).
#[allow(clippy::too_many_arguments)]
pub fn run_dequant_matmul_mxfp4x2_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
    group: usize,
    x_byte_off: usize,
    w_byte_off: usize,
    scale_byte_off: usize,
    scratch_byte_off: usize,
    out_byte_off: usize,
) {
    let total = (k * n) as u32;
    let threads = 256u32.min(total).max(1);
    let grid = total.div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let kernel = crate::kernels::mxfp4x2_dequant_nk_kernel(ctx);
    let w_off_u64 = w_byte_off as u64;
    let s_f32_off = (scale_byte_off / 4) as u64;
    let dst_f32_off = (scratch_byte_off / 4) as u64;
    let (k_u, n_u, g_u) = (k as u32, n as u32, group as u32);
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&w_off_u64)
        .arg(&s_f32_off)
        .arg(&dst_f32_off)
        .arg(&k_u)
        .arg(&n_u)
        .arg(&g_u);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: mxfp4x2_dequant_nk launch failed");
    }

    run_matmul_bt(
        ctx,
        stream,
        buffer,
        m,
        k,
        n,
        x_byte_off,
        scratch_byte_off,
        out_byte_off,
        false,
    );
}

/// Launch `dequant_gguf` into arena scratch, then `C = X @ W^T` via `matmul_bt`.
pub fn run_dequant_matmul_gguf_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    scratch_byte_off: usize,
    out_byte_off: usize,
) {
    let scheme = scheme_from_id(scheme_id);
    let block_elems = scheme.gguf_block_size() as usize;
    let total = k * n;
    let num_blocks = total / block_elems.max(1);
    if num_blocks == 0 {
        panic!(
            "rlx-cuda: dequant_gguf num_blocks=0 (m={m}, k={k}, n={n}, scheme_id={scheme_id}, block_elems={block_elems})"
        );
    }

    let kernel = dequant_gguf_kernel(ctx);
    let threads = 256u32.min(num_blocks as u32).max(1);
    let grid = num_blocks.div_ceil(threads as usize) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    // 64-bit arena offsets — see matmul_bt.cu / dequant_gguf.cu. The >4 GB
    // packed-27B arena overflows u32 → wrong weight slot → garbage.
    let dst_f32_off = (scratch_byte_off / 4) as u64;
    let w_off_u64 = w_byte_off as u64;
    let nb_u32 = num_blocks as u32;
    // Materialise the IQ grid LUT on this context once (cached). Bound
    // as a kernel arg unconditionally — non-IQ schemes ignore the pointer.
    let lut = crate::iq_grid::cuda_iq_grid_buffer(ctx, stream);
    use cudarc::driver::DevicePtr;
    let (lut_ptr, _lut_rec) = lut.device_ptr(stream);
    let mut launcher = stream.launch_builder(&kernel.function);
    launcher
        .arg(&mut *buffer)
        .arg(&w_off_u64)
        .arg(&dst_f32_off)
        .arg(&scheme_id)
        .arg(&nb_u32)
        .arg(&lut_ptr);
    unsafe {
        launcher
            .launch(cfg)
            .expect("rlx-cuda: dequant_gguf launch failed");
    }

    // Scheme 24 = GgufQ1_0 — auto-enable compensated matmul by default.
    // Also auto-enable for large K, where the naive f32 k-reduction loses
    // several mantissa bits that the dequant just produced: opt in by setting
    // RLX_CUDA_MATMUL_PRECISE_MIN_K to the K above which compensation kicks in
    // for every GGUF scheme. Default unset ⇒ no perf change (compensation
    // stays opt-in for non-Q1_0 via RLX_CUDA_MATMUL_PRECISE / PARITY / NO_TF32,
    // which run_matmul_bt already honors regardless of `precise_default`).
    let precise_min_k =
        rlx_ir::env::var("RLX_CUDA_MATMUL_PRECISE_MIN_K").and_then(|v| v.parse::<usize>().ok());
    let precise_default = scheme_id == 24 || precise_min_k.is_some_and(|t| k >= t);
    let precise_off = matches!(
        rlx_ir::env::var("RLX_CUDA_MATMUL_PRECISE").as_deref(),
        Some("0") | Some("false") | Some("off")
    );
    run_matmul_bt(
        ctx,
        stream,
        buffer,
        m,
        k,
        n,
        x_byte_off,
        scratch_byte_off,
        out_byte_off,
        precise_default && !precise_off,
    );
}

/// GPU dequant + grouped matmul for MoE packed expert stacks.
///
/// Scratch layout at `scratch_byte_off` (f32 bytes):
///   `[0 .. k*n)`: dequantized expert slab
///   `[k*n .. k*n+m*k)`: sorted token inputs
///   `[k*n+m*k .. k*n+m*k+m*n)`: sorted outputs before unpermute
pub fn run_dequant_grouped_matmul_gguf_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme_id: u32,
    x_byte_off: usize,
    w_byte_off: usize,
    idx_byte_off: usize,
    scratch_byte_off: usize,
    out_byte_off: usize,
) {
    let scheme = scheme_from_id(scheme_id);
    let slab_bytes = slab_bytes_for(scheme, k, n);
    let num_blocks = (k * n) / scheme.gguf_block_size() as usize;

    stream
        .synchronize()
        .expect("rlx-cuda: grouped gguf pre-sync failed");

    let x_f32_off = x_byte_off / 4;
    let mut x_host = vec![0f32; m * k];
    stream
        .memcpy_dtoh(&buffer.slice(x_f32_off..x_f32_off + m * k), &mut x_host)
        .expect("rlx-cuda: grouped gguf x dtoh failed");

    let idx_f32_off = idx_byte_off / 4;
    let mut idx_host = vec![0f32; m];
    stream
        .memcpy_dtoh(&buffer.slice(idx_f32_off..idx_f32_off + m), &mut idx_host)
        .expect("rlx-cuda: grouped gguf idx dtoh failed");

    let (packed_in, original_pos, offsets) =
        rlx_cpu::gguf_matmul::grouped_moe_sort_plan(&x_host, &idx_host, m, k, num_experts);

    let dequant_off = scratch_byte_off;
    let pack_in_off = scratch_byte_off + k * n * 4;
    let pack_out_off = scratch_byte_off + (k * n + m * k) * 4;

    stream
        .memcpy_htod(
            &packed_in,
            &mut buffer.slice_mut(pack_in_off / 4..pack_in_off / 4 + m * k),
        )
        .expect("rlx-cuda: grouped gguf pack_in htod failed");

    let kernel = dequant_gguf_kernel(ctx);
    let block = 256u32.min(num_blocks as u32).max(1);
    let grid = num_blocks.div_ceil(block as usize) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let dst_f32_off = (dequant_off / 4) as u64; // 64-bit: >4 GB arena (see dequant_gguf.cu)
    let nb_u32 = num_blocks as u32;

    let lut = crate::iq_grid::cuda_iq_grid_buffer(ctx, stream);
    use cudarc::driver::DevicePtr;
    let (lut_ptr, _lut_rec) = lut.device_ptr(stream);
    for e in 0..num_experts {
        let count = offsets[e + 1] - offsets[e];
        if count == 0 {
            continue;
        }
        let w_off = w_byte_off + e * slab_bytes;
        let w_off_u64 = w_off as u64; // 64-bit: >4 GB arena (see dequant_gguf.cu)
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&w_off_u64)
            .arg(&dst_f32_off)
            .arg(&scheme_id)
            .arg(&nb_u32)
            .arg(&lut_ptr);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: grouped dequant_gguf launch failed");
        }

        let in_start = offsets[e];
        run_matmul_bt(
            ctx,
            stream,
            buffer,
            count,
            k,
            n,
            pack_in_off + in_start * k * 4,
            dequant_off,
            pack_out_off + in_start * n * 4,
            scheme_id == 24,
        );
    }

    let mut packed_out = vec![0f32; m * n];
    stream
        .memcpy_dtoh(
            &buffer.slice(pack_out_off / 4..pack_out_off / 4 + m * n),
            &mut packed_out,
        )
        .expect("rlx-cuda: grouped gguf pack_out dtoh failed");

    let mut out_host = vec![0f32; m * n];
    rlx_cpu::gguf_matmul::grouped_moe_unpermute_out(
        &packed_out,
        &original_pos,
        &mut out_host,
        m,
        n,
    );

    let out_f32_off = out_byte_off / 4;
    stream
        .memcpy_htod(
            &out_host,
            &mut buffer.slice_mut(out_f32_off..out_f32_off + m * n),
        )
        .expect("rlx-cuda: grouped gguf out htod failed");
}
