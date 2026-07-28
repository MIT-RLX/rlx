// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native ROCm dense linear solve for `Op::DenseSolve` / `Op::BatchedDenseSolve`.
//!
//! - **DenseSolve** — hipSOLVER `Sgetrf` + `Sgetrs`.
//! - **BatchedDenseSolve** — hipBLAS `SgetrfBatched` + `SgetrsBatched`.
//!
//! Layout matches [`crate`] CUDA twin: row-major `A` is seen as `Aᵀ` by
//! column-major LAPACK, so we solve with `OP_T`. `nrhs > 1` transposes `B`
//! to/from column-major. F32 only; missing libraries keep HostOp at compile.

use std::cell::RefCell;
use std::sync::Arc;

use crate::device::RocmContext;
use crate::hip::{HipBuffer, HipKernel, HipStream};
use crate::hipblas::{HipblasContext, HipblasOperation, batched_lu_available};
use crate::hipsolver::{HipsolverContext, HipsolverOperation, runtime as load_solver};

const OP_T: HipsolverOperation = HipsolverOperation::T;
const OP_T_BLAS: HipblasOperation = HipblasOperation::T;

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

struct SolverTls {
    solver: HipsolverContext,
}

thread_local! {
    static SOLVER: RefCell<Option<SolverTls>> = const { RefCell::new(None) };
    static TRANSPOSE: RefCell<Option<HipKernel>> = const { RefCell::new(None) };
}

fn with_transpose<R>(rocm: &Arc<RocmContext>, f: impl FnOnce(&HipKernel) -> R) -> R {
    TRANSPOSE.with(|cell| {
        {
            let mut opt = cell.borrow_mut();
            if opt.is_none() {
                *opt = Some(crate::kernels::compile(
                    rocm,
                    TRANSPOSE_SRC,
                    "dense_solve_transpose_batched",
                ));
            }
        }
        let borrow = cell.borrow();
        f(borrow.as_ref().unwrap())
    })
}

fn ensure_solver(stream: HipStream) -> Option<()> {
    SOLVER.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            let rt = load_solver()?;
            let solver = HipsolverContext::new(&rt, stream)?;
            *opt = Some(SolverTls { solver });
        }
        Some(())
    })
}

/// True when hipSOLVER exposes getrf/getrs.
pub fn is_available() -> bool {
    crate::hipsolver::is_available()
}

fn launch_transpose(
    kernel: &HipKernel,
    stream: HipStream,
    src: &HipBuffer<f32>,
    dst: &HipBuffer<f32>,
    n: i32,
    nrhs: i32,
    batch: i32,
    to_col_major: i32,
) {
    let total = (batch as u32) * (n as u32) * (nrhs as u32);
    let block = 256u32;
    let grid = (total.div_ceil(block), 1u32, 1u32);
    let block_dim = (block, 1u32, 1u32);
    let mut in_ptr = src.ptr;
    let mut out_ptr = dst.ptr;
    let mut n_i = n;
    let mut nrhs_i = nrhs;
    let mut batch_i = batch;
    let mut flag = to_col_major;
    crate::launch_kernel!(
        kernel,
        stream,
        grid,
        block_dim,
        [
            &mut in_ptr,
            &mut out_ptr,
            &mut n_i,
            &mut nrhs_i,
            &mut batch_i,
            &mut flag
        ]
    );
}

fn check_info(d_info: &HipBuffer<i32>, label: &str) {
    let mut host = vec![0i32; 1];
    d_info
        .copy_to_host(&mut host)
        .expect("dense_solve info D2H");
    if host[0] != 0 {
        panic!("{label}: singular or invalid (info={})", host[0]);
    }
}

fn check_info_batch(d_info: &HipBuffer<i32>, batch: usize, label: &str) {
    let mut host = vec![0i32; batch];
    d_info
        .copy_to_host(&mut host)
        .expect("dense_solve batch info D2H");
    for (i, &v) in host.iter().enumerate() {
        if v != 0 {
            panic!("{label}: batch[{i}] singular or invalid (info={v})");
        }
    }
}

/// Single-system dense solve via hipSOLVER getrf+getrs.
#[allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]
pub fn run_dense(
    rocm: &Arc<RocmContext>,
    stream: HipStream,
    buffer: &HipBuffer<f32>,
    a_off: usize,
    b_off: usize,
    x_off: usize,
    n: usize,
    nrhs: usize,
) {
    debug_assert!(n > 0 && nrhs > 0);
    ensure_solver(stream).expect("rlx-rocm: hipSOLVER required for DenseSolveNative");

    SOLVER.with(|cell| {
        let borrow = cell.borrow();
        let c = borrow.as_ref().unwrap();
        unsafe {
            c.solver.set_stream(stream).expect("hipsolverSetStream");
        }
        let rt = &rocm.runtime;
        let nn = n * n;
        let bn = n * nrhs;

        let d_a = HipBuffer::<f32>::alloc_zeros(rt, nn).expect("dense_solve A");
        unsafe {
            (rt.hip_memcpy_dtod)(d_a.ptr, buffer.ptr + (a_off as u64) * 4, nn * 4)
                .ok()
                .expect("memcpy A");
        }

        let d_b_work = HipBuffer::<f32>::alloc_zeros(rt, bn).expect("dense_solve B");
        if nrhs == 1 {
            unsafe {
                (rt.hip_memcpy_dtod)(d_b_work.ptr, buffer.ptr + (b_off as u64) * 4, bn * 4)
                    .ok()
                    .expect("memcpy B");
            }
        } else {
            let d_b_rm = HipBuffer::<f32>::alloc_zeros(rt, bn).expect("dense_solve B rm");
            unsafe {
                (rt.hip_memcpy_dtod)(d_b_rm.ptr, buffer.ptr + (b_off as u64) * 4, bn * 4)
                    .ok()
                    .expect("memcpy B rm");
            }
            with_transpose(rocm, |k| {
                launch_transpose(k, stream, &d_b_rm, &d_b_work, n as i32, nrhs as i32, 1, 1);
            });
            let _ = d_b_rm;
        }

        let d_ipiv = HipBuffer::<i32>::alloc_zeros(rt, n).expect("ipiv");
        let d_info = HipBuffer::<i32>::alloc_zeros(rt, 1).expect("info");

        let mut lwork = 0i32;
        unsafe {
            (c.solver.runtime.sgetrf_buffer_size)(
                c.solver.handle,
                n as i32,
                n as i32,
                d_a.ptr as *mut f32,
                n as i32,
                &mut lwork,
            )
            .ok()
            .expect("hipsolverSgetrf_bufferSize");
        }
        let d_work = HipBuffer::<f32>::alloc_zeros(rt, lwork.max(1) as usize).expect("work");
        unsafe {
            (c.solver.runtime.sgetrf)(
                c.solver.handle,
                n as i32,
                n as i32,
                d_a.ptr as *mut f32,
                n as i32,
                d_work.ptr as *mut f32,
                lwork,
                d_ipiv.ptr as *mut i32,
                d_info.ptr as *mut i32,
            )
            .ok()
            .expect("hipsolverSgetrf");
        }
        check_info(&d_info, "DenseSolveNative Sgetrf");

        let mut lwork_rs = 0i32;
        unsafe {
            (c.solver.runtime.sgetrs_buffer_size)(
                c.solver.handle,
                OP_T,
                n as i32,
                nrhs as i32,
                d_a.ptr as *mut f32,
                n as i32,
                d_ipiv.ptr as *mut i32,
                d_b_work.ptr as *mut f32,
                n as i32,
                &mut lwork_rs,
            )
            .ok()
            .expect("hipsolverSgetrs_bufferSize");
        }
        let d_work_rs =
            HipBuffer::<f32>::alloc_zeros(rt, lwork_rs.max(1) as usize).expect("getrs work");
        unsafe {
            (c.solver.runtime.sgetrs)(
                c.solver.handle,
                OP_T,
                n as i32,
                nrhs as i32,
                d_a.ptr as *mut f32,
                n as i32,
                d_ipiv.ptr as *mut i32,
                d_b_work.ptr as *mut f32,
                n as i32,
                d_work_rs.ptr as *mut f32,
                lwork_rs,
                d_info.ptr as *mut i32,
            )
            .ok()
            .expect("hipsolverSgetrs");
        }
        check_info(&d_info, "DenseSolveNative Sgetrs");

        if nrhs == 1 {
            unsafe {
                (rt.hip_memcpy_dtod)(buffer.ptr + (x_off as u64) * 4, d_b_work.ptr, bn * 4)
                    .ok()
                    .expect("memcpy X");
            }
        } else {
            let d_out = HipBuffer::<f32>::alloc_zeros(rt, bn).expect("X out");
            with_transpose(rocm, |k| {
                launch_transpose(k, stream, &d_b_work, &d_out, n as i32, nrhs as i32, 1, 0);
            });
            unsafe {
                (rt.hip_memcpy_dtod)(buffer.ptr + (x_off as u64) * 4, d_out.ptr, bn * 4)
                    .ok()
                    .expect("memcpy X");
            }
            let _ = d_out;
        }
        let _ = (&d_a, &d_b_work, &d_ipiv, &d_info, &d_work, &d_work_rs);
    });
}

/// Batched dense solve via hipBLAS getrfBatched/getrsBatched.
#[allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]
pub fn run_batched(
    rocm: &Arc<RocmContext>,
    stream: HipStream,
    blas: &HipblasContext,
    buffer: &HipBuffer<f32>,
    a_off: usize,
    b_off: usize,
    x_off: usize,
    batch: usize,
    n: usize,
    nrhs: usize,
) {
    debug_assert!(batch > 0 && n > 0 && nrhs > 0);
    if !batched_lu_available(&blas.runtime) {
        panic!("rlx-rocm: hipblasSgetrfBatched/SgetrsBatched unavailable");
    }
    unsafe {
        blas.set_stream(stream).expect("hipblasSetStream");
    }

    let rt = &rocm.runtime;
    let a_stride = n * n;
    let b_stride = n * nrhs;
    let d_a = HipBuffer::<f32>::alloc_zeros(rt, batch * a_stride).expect("batched A");
    unsafe {
        (rt.hip_memcpy_dtod)(
            d_a.ptr,
            buffer.ptr + (a_off as u64) * 4,
            batch * a_stride * 4,
        )
        .ok()
        .expect("memcpy A");
    }

    let d_b_work = HipBuffer::<f32>::alloc_zeros(rt, batch * b_stride).expect("batched B");
    if nrhs == 1 {
        unsafe {
            (rt.hip_memcpy_dtod)(
                d_b_work.ptr,
                buffer.ptr + (b_off as u64) * 4,
                batch * b_stride * 4,
            )
            .ok()
            .expect("memcpy B");
        }
    } else {
        let d_b_rm = HipBuffer::<f32>::alloc_zeros(rt, batch * b_stride).expect("B rm");
        unsafe {
            (rt.hip_memcpy_dtod)(
                d_b_rm.ptr,
                buffer.ptr + (b_off as u64) * 4,
                batch * b_stride * 4,
            )
            .ok()
            .expect("memcpy B rm");
        }
        with_transpose(rocm, |k| {
            launch_transpose(
                k,
                stream,
                &d_b_rm,
                &d_b_work,
                n as i32,
                nrhs as i32,
                batch as i32,
                1,
            );
        });
        let _ = d_b_rm;
    }

    let d_ipiv = HipBuffer::<i32>::alloc_zeros(rt, batch * n).expect("ipiv");
    let d_info = HipBuffer::<i32>::alloc_zeros(rt, batch).expect("info");

    let a_ptrs: Vec<u64> = (0..batch)
        .map(|i| d_a.ptr + (i * a_stride * 4) as u64)
        .collect();
    let b_ptrs: Vec<u64> = (0..batch)
        .map(|i| d_b_work.ptr + (i * b_stride * 4) as u64)
        .collect();
    let mut d_a_ptrs = HipBuffer::<u64>::alloc_zeros(rt, batch).expect("A ptrs");
    let mut d_b_ptrs = HipBuffer::<u64>::alloc_zeros(rt, batch).expect("B ptrs");
    d_a_ptrs.copy_from_host(&a_ptrs).expect("A ptrs H2D");
    d_b_ptrs.copy_from_host(&b_ptrs).expect("B ptrs H2D");

    let sgetrf = blas.runtime.sgetrf_batched.unwrap();
    let sgetrs = blas.runtime.sgetrs_batched.unwrap();
    unsafe {
        sgetrf(
            blas.handle,
            n as i32,
            d_a_ptrs.ptr as *const *mut f32,
            n as i32,
            d_ipiv.ptr as *mut i32,
            d_info.ptr as *mut i32,
            batch as i32,
        )
        .ok()
        .expect("hipblasSgetrfBatched");
    }
    check_info_batch(&d_info, batch, "BatchedDenseSolveNative SgetrfBatched");

    let mut h_info = 0i32;
    unsafe {
        sgetrs(
            blas.handle,
            OP_T_BLAS,
            n as i32,
            nrhs as i32,
            d_a_ptrs.ptr as *const *const f32,
            n as i32,
            d_ipiv.ptr as *const i32,
            d_b_ptrs.ptr as *const *mut f32,
            n as i32,
            &mut h_info,
            batch as i32,
        )
        .ok()
        .expect("hipblasSgetrsBatched");
    }
    if h_info != 0 {
        panic!("BatchedDenseSolveNative SgetrsBatched: info={h_info}");
    }

    if nrhs == 1 {
        unsafe {
            (rt.hip_memcpy_dtod)(
                buffer.ptr + (x_off as u64) * 4,
                d_b_work.ptr,
                batch * b_stride * 4,
            )
            .ok()
            .expect("memcpy X");
        }
    } else {
        let d_out = HipBuffer::<f32>::alloc_zeros(rt, batch * b_stride).expect("X out");
        with_transpose(rocm, |k| {
            launch_transpose(
                k,
                stream,
                &d_b_work,
                &d_out,
                n as i32,
                nrhs as i32,
                batch as i32,
                0,
            );
        });
        unsafe {
            (rt.hip_memcpy_dtod)(
                buffer.ptr + (x_off as u64) * 4,
                d_out.ptr,
                batch * b_stride * 4,
            )
            .ok()
            .expect("memcpy X");
        }
        let _ = d_out;
    }
    let _ = (&d_a, &d_b_work, &d_ipiv, &d_info, &d_a_ptrs, &d_b_ptrs);
}
