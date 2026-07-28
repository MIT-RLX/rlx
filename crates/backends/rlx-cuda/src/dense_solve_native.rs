// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native CUDA dense linear solve for `Op::DenseSolve` / `Op::BatchedDenseSolve`.
//!
//! - **DenseSolve** — cuSOLVER `Sgetrf` + `Sgetrs` (single system).
//! - **BatchedDenseSolve** — cuBLAS `SgetrfBatched` + `SgetrsBatched`.
//!
//! RLX stores `A` / `b` **row-major**. Column-major LAPACK sees that memory as
//! `Aᵀ`, so we factor in place and solve with `CUBLAS_OP_T` (equivalent to the
//! CPU path's explicit transpose + `OP_N`). For `nrhs > 1` we transpose `B` to
//! column-major before `getrs` and back afterward.
//!
//! F32 only (GPU arena). F64 / other dtypes stay on `Step::HostOp`.

use cudarc::cublas::sys as cbs;
use cudarc::cusolver::sys as cs;
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use std::cell::RefCell;
use std::sync::Arc;

const OP_T_SOLVER: cs::cublasOperation_t = cs::cublasOperation_t::CUBLAS_OP_T;
const OP_T_BLAS: cbs::cublasOperation_t = cbs::cublasOperation_t::CUBLAS_OP_T;

/// Batched row-major ↔ column-major transpose for `B` panels of shape `[n, nrhs]`.
const TRANSPOSE_SRC: &str = r#"
extern "C" __global__ void dense_solve_transpose_batched(
    const float* __restrict__ in,
    float* __restrict__ out,
    int n, int nrhs, int batch, int to_col_major)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long per = (long)n * nrhs;
    if (idx >= (long)batch * per) return;
    int b = (int)(idx / per);
    int r = (int)(idx % per);
    long base = (long)b * per;
    if (to_col_major) {
        int i = r / nrhs;
        int j = r % nrhs;
        out[base + (long)i + (long)j * n] = in[idx];
    } else {
        int i = r % n;
        int j = r / n;
        out[base + (long)i * nrhs + j] = in[idx];
    }
}
"#;

struct SolverCtx {
    handle: cs::cusolverDnHandle_t,
    transpose: crate::kernels::CudaKernel,
}

thread_local! {
    static SOLVER: RefCell<Option<SolverCtx>> = const { RefCell::new(None) };
}

fn ensure_solver(ctx: &Arc<CudaContext>) {
    SOLVER.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            let handle = cudarc::cusolver::result::dn_create().expect("cusolverDnCreate");
            let transpose =
                crate::kernels::compile(ctx, TRANSPOSE_SRC, "dense_solve_transpose_batched");
            *opt = Some(SolverCtx { handle, transpose });
        }
    });
}

fn launch_transpose_batched(
    stream: &Arc<CudaStream>,
    kernel: &crate::kernels::CudaKernel,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    n: i32,
    nrhs: i32,
    batch: i32,
    to_col_major: i32,
) {
    let total = (batch as u32) * (n as u32) * (nrhs as u32);
    let block = 256u32;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(block), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut lb = stream.launch_builder(&kernel.function);
    lb.arg(src)
        .arg(dst)
        .arg(&n)
        .arg(&nrhs)
        .arg(&batch)
        .arg(&to_col_major);
    unsafe {
        lb.launch(cfg)
            .expect("dense_solve_transpose_batched launch");
    }
}

fn check_info(stream: &Arc<CudaStream>, d_info: &CudaSlice<i32>, label: &str) {
    let mut host = vec![0i32; 1];
    stream
        .memcpy_dtoh(d_info, &mut host)
        .expect("dense_solve info D2H");
    if host[0] != 0 {
        panic!("{label}: singular or invalid (info={})", host[0]);
    }
}

fn check_info_batch(stream: &Arc<CudaStream>, d_info: &CudaSlice<i32>, batch: usize, label: &str) {
    let mut host = vec![0i32; batch];
    stream
        .memcpy_dtoh(d_info, &mut host)
        .expect("dense_solve batch info D2H");
    for (i, &v) in host.iter().enumerate() {
        if v != 0 {
            panic!("{label}: batch[{i}] singular or invalid (info={v})");
        }
    }
}

/// Single-system `A[n,n] x = b[n]` or `b[n,nrhs]` via cuSOLVER getrf+getrs.
#[allow(clippy::too_many_arguments)]
pub fn run_dense(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    a_off: usize,
    b_off: usize,
    x_off: usize,
    n: usize,
    nrhs: usize,
) {
    debug_assert!(n > 0 && nrhs > 0);
    ensure_solver(ctx);
    SOLVER.with(|cell| {
        let borrow = cell.borrow();
        let c = borrow.as_ref().unwrap();
        unsafe {
            cudarc::cusolver::result::dn_set_stream(
                c.handle,
                stream.cu_stream() as *mut cs::CUstream_st,
            )
            .expect("cusolverDnSetStream");
        }

        let nn = n * n;
        let bn = n * nrhs;
        let mut d_a = stream.alloc_zeros::<f32>(nn).unwrap();
        stream
            .memcpy_dtod(&buffer.slice(a_off..a_off + nn), &mut d_a)
            .unwrap();

        let mut d_b_work = stream.alloc_zeros::<f32>(bn).unwrap();
        if nrhs == 1 {
            stream
                .memcpy_dtod(&buffer.slice(b_off..b_off + bn), &mut d_b_work)
                .unwrap();
        } else {
            let mut d_b_rm = stream.alloc_zeros::<f32>(bn).unwrap();
            stream
                .memcpy_dtod(&buffer.slice(b_off..b_off + bn), &mut d_b_rm)
                .unwrap();
            launch_transpose_batched(
                stream,
                &c.transpose,
                &d_b_rm,
                &mut d_b_work,
                n as i32,
                nrhs as i32,
                1,
                1,
            );
        }

        let mut d_ipiv = stream.alloc_zeros::<i32>(n).unwrap();
        let mut d_info = stream.alloc_zeros::<i32>(1).unwrap();

        let mut lwork = 0i32;
        unsafe {
            let (ap, _g) = d_a.device_ptr_mut(stream);
            cs::cusolverDnSgetrf_bufferSize(
                c.handle,
                n as i32,
                n as i32,
                ap as *mut f32,
                n as i32,
                &mut lwork,
            )
            .result()
            .expect("Sgetrf_bufferSize");
        }
        let mut d_work = stream.alloc_zeros::<f32>(lwork.max(1) as usize).unwrap();
        unsafe {
            let (ap, _g1) = d_a.device_ptr_mut(stream);
            let (wp, _g2) = d_work.device_ptr_mut(stream);
            let (ip, _g3) = d_ipiv.device_ptr_mut(stream);
            let (inf, _g4) = d_info.device_ptr_mut(stream);
            cs::cusolverDnSgetrf(
                c.handle,
                n as i32,
                n as i32,
                ap as *mut f32,
                n as i32,
                wp as *mut f32,
                ip as *mut i32,
                inf as *mut i32,
            )
            .result()
            .expect("Sgetrf");
        }
        check_info(stream, &d_info, "DenseSolveNative Sgetrf");

        unsafe {
            let (ap, _g1) = d_a.device_ptr(stream);
            let (ip, _g2) = d_ipiv.device_ptr(stream);
            let (bp, _g3) = d_b_work.device_ptr_mut(stream);
            let (inf, _g4) = d_info.device_ptr_mut(stream);
            cs::cusolverDnSgetrs(
                c.handle,
                OP_T_SOLVER,
                n as i32,
                nrhs as i32,
                ap as *const f32,
                n as i32,
                ip as *const i32,
                bp as *mut f32,
                n as i32,
                inf as *mut i32,
            )
            .result()
            .expect("Sgetrs");
        }
        check_info(stream, &d_info, "DenseSolveNative Sgetrs");

        if nrhs == 1 {
            stream
                .memcpy_dtod(&d_b_work, &mut buffer.slice_mut(x_off..x_off + bn))
                .unwrap();
        } else {
            let mut d_out = stream.alloc_zeros::<f32>(bn).unwrap();
            launch_transpose_batched(
                stream,
                &c.transpose,
                &d_b_work,
                &mut d_out,
                n as i32,
                nrhs as i32,
                1,
                0,
            );
            stream
                .memcpy_dtod(&d_out, &mut buffer.slice_mut(x_off..x_off + bn))
                .unwrap();
        }
    });
}

/// Batched `A[B,n,n] X = b[B,n]` / `b[B,n,K]` via cuBLAS getrfBatched/getrsBatched.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::not_unsafe_ptr_arg_deref)] // cublas handle + device ptr FFI
pub fn run_batched(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    blas_handle: cbs::cublasHandle_t,
    buffer: &mut CudaSlice<f32>,
    a_off: usize,
    b_off: usize,
    x_off: usize,
    batch: usize,
    n: usize,
    nrhs: usize,
) {
    debug_assert!(batch > 0 && n > 0 && nrhs > 0);
    ensure_solver(ctx);
    unsafe {
        cudarc::cublas::result::set_stream(blas_handle, stream.cu_stream() as _)
            .expect("cublasSetStream");
    }

    SOLVER.with(|cell| {
        let borrow = cell.borrow();
        let c = borrow.as_ref().unwrap();

        let a_stride = n * n;
        let b_stride = n * nrhs;
        let mut d_a = stream.alloc_zeros::<f32>(batch * a_stride).unwrap();
        stream
            .memcpy_dtod(&buffer.slice(a_off..a_off + batch * a_stride), &mut d_a)
            .unwrap();

        let mut d_b_work = stream.alloc_zeros::<f32>(batch * b_stride).unwrap();
        if nrhs == 1 {
            stream
                .memcpy_dtod(
                    &buffer.slice(b_off..b_off + batch * b_stride),
                    &mut d_b_work,
                )
                .unwrap();
        } else {
            let mut d_b_rm = stream.alloc_zeros::<f32>(batch * b_stride).unwrap();
            stream
                .memcpy_dtod(&buffer.slice(b_off..b_off + batch * b_stride), &mut d_b_rm)
                .unwrap();
            launch_transpose_batched(
                stream,
                &c.transpose,
                &d_b_rm,
                &mut d_b_work,
                n as i32,
                nrhs as i32,
                batch as i32,
                1,
            );
        }

        let mut d_ipiv = stream.alloc_zeros::<i32>(batch * n).unwrap();
        let mut d_info = stream.alloc_zeros::<i32>(batch).unwrap();

        let a_ptrs: Vec<u64>;
        let b_ptrs: Vec<u64>;
        {
            let (a_base, _ga) = d_a.device_ptr_mut(stream);
            let (b_base, _gb) = d_b_work.device_ptr_mut(stream);
            a_ptrs = (0..batch)
                .map(|i| a_base + (i * a_stride * 4) as u64)
                .collect();
            b_ptrs = (0..batch)
                .map(|i| b_base + (i * b_stride * 4) as u64)
                .collect();
        }
        let mut d_a_ptrs = stream.alloc_zeros::<u64>(batch).unwrap();
        let mut d_b_ptrs = stream.alloc_zeros::<u64>(batch).unwrap();
        stream.memcpy_htod(&a_ptrs, &mut d_a_ptrs).unwrap();
        stream.memcpy_htod(&b_ptrs, &mut d_b_ptrs).unwrap();

        unsafe {
            let (ap, _g1) = d_a_ptrs.device_ptr(stream);
            let (ip, _g2) = d_ipiv.device_ptr_mut(stream);
            let (inf, _g3) = d_info.device_ptr_mut(stream);
            cbs::cublasSgetrfBatched(
                blas_handle,
                n as i32,
                ap as *const *mut f32,
                n as i32,
                ip as *mut i32,
                inf as *mut i32,
                batch as i32,
            )
            .result()
            .expect("cublasSgetrfBatched");
        }
        check_info_batch(
            stream,
            &d_info,
            batch,
            "BatchedDenseSolveNative SgetrfBatched",
        );

        let mut h_info = 0i32;
        unsafe {
            let (ap, _g1) = d_a_ptrs.device_ptr(stream);
            let (ip, _g2) = d_ipiv.device_ptr(stream);
            let (bp, _g3) = d_b_ptrs.device_ptr(stream);
            cbs::cublasSgetrsBatched(
                blas_handle,
                OP_T_BLAS,
                n as i32,
                nrhs as i32,
                ap as *const *const f32,
                n as i32,
                ip as *const i32,
                bp as *const *mut f32,
                n as i32,
                &mut h_info,
                batch as i32,
            )
            .result()
            .expect("cublasSgetrsBatched");
        }
        if h_info != 0 {
            panic!("BatchedDenseSolveNative SgetrsBatched: info={h_info}");
        }

        if nrhs == 1 {
            stream
                .memcpy_dtod(
                    &d_b_work,
                    &mut buffer.slice_mut(x_off..x_off + batch * b_stride),
                )
                .unwrap();
        } else {
            let mut d_out = stream.alloc_zeros::<f32>(batch * b_stride).unwrap();
            launch_transpose_batched(
                stream,
                &c.transpose,
                &d_b_work,
                &mut d_out,
                n as i32,
                nrhs as i32,
                batch as i32,
                0,
            );
            stream
                .memcpy_dtod(
                    &d_out,
                    &mut buffer.slice_mut(x_off..x_off + batch * b_stride),
                )
                .unwrap();
        }
    });
}
