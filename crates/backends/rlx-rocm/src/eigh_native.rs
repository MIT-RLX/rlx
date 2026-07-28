// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native ROCm symmetric eigendecomposition via hipSOLVER **`SsyevjBatched`** —
//! the on-device forward path for `Op::Eigh` / `Op::EighBatch`, replacing the
//! F64 CPU host-fallback (`Step::SpdHost`, which does D2H → LAPACK → H2D).
//!
//! Mirrors [`rlx_cuda::eigh_native`]: hipSOLVER's batched Jacobi runs the
//! cyclic sweeps for the whole batch in parallel (one block per matrix).
//! Constraint: batched syevj supports **n ≤ 32**. Larger `n` (or missing
//! `libhipsolver`) stays on `Step::SpdHost`.
//!
//! Output matches [`rlx_cpu::spd::eigh_packed`]: per matrix `[λ (n) ∥ U (n²)]`,
//! `λ` ascending, `U` row-major with column `j` = eigenvector `j`
//! (`A = U diag(λ) Uᵀ`). hipSOLVER writes eigenvectors **column-major**, so the
//! `eigh_assemble` kernel transposes them into `U` and interleaves `λ`.

use std::cell::RefCell;
use std::sync::Arc;

use crate::device::RocmContext;
use crate::hip::{HipBuffer, HipKernel, HipStream};
use crate::hipsolver::{
    HipsolverContext, HipsolverEigMode, HipsolverFillMode, runtime as load_solver,
};

const VEC: HipsolverEigMode = HipsolverEigMode::Vector;
const LO: HipsolverFillMode = HipsolverFillMode::Lower;

/// hipSOLVER's batched Jacobi supports only `n ≤ 32` (same tile as cuSOLVER).
pub const MAX_N: usize = 32;

/// Transpose hipSOLVER's column-major eigenvectors into rlx's packed `[λ ∥ U]`
/// (row-major U, column j = eigenvector j) and interleave the eigenvalues.
const ASSEMBLE_SRC: &str = r#"
extern "C" __global__ void eigh_assemble(
    const float* __restrict__ eigvec, // [batch,n,n] col-major: eigvec[b*n*n + i + j*n] = comp i of eigvec j
    const float* __restrict__ eigval, // [batch,n] ascending
    float* __restrict__ out,          // [batch, n*n+n] packed: [λ(n) ∥ U(n²) row-major]
    int n, int batch)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long per = (long)n * n + n;
    if (idx >= (long)batch * per) return;
    int b = (int)(idx / per);
    int r = (int)(idx % per);
    if (r < n) {
        out[idx] = eigval[(long)b * n + r];                 // λ
    } else {
        int u = r - n;                                      // row-major U[i][j], 0..n²
        int i = u / n, j = u % n;
        out[idx] = eigvec[(long)b * n * n + i + (long)j * n]; // U[i][j] = comp i of eigvec j
    }
}
"#;

struct EighCtx {
    solver: HipsolverContext,
    kernel: HipKernel,
}

thread_local! {
    static CTX: RefCell<Option<EighCtx>> = const { RefCell::new(None) };
}

fn ensure_ctx(rocm: &Arc<RocmContext>, stream: HipStream) -> Option<()> {
    CTX.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            let rt = load_solver()?;
            let solver = HipsolverContext::new(&rt, stream)?;
            let kernel = crate::kernels::compile(rocm, ASSEMBLE_SRC, "eigh_assemble");
            *opt = Some(EighCtx { solver, kernel });
        }
        Some(())
    })
}

/// True when hipSOLVER can be loaded (compile-time gate for `Step::EighNative`).
pub fn is_available() -> bool {
    crate::hipsolver::is_available()
}

/// Run a native batched symmetric eigendecomposition on the arena.
/// `in_off` / `out_off` are f32-element offsets: input `A [batch·n·n]`, output
/// packed `[batch·(n·n+n)]`. `A` is copied to scratch first (syevj overwrites
/// its input), so the arena input slot is left intact.
// `stream` is an opaque HIP stream handle passed through to hipSOLVER, not a
// memory pointer we dereference — the lint is a false positive here.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn run(
    rocm: &Arc<RocmContext>,
    stream: HipStream,
    buffer: &HipBuffer<f32>,
    in_off: usize,
    out_off: usize,
    n: usize,
    batch: usize,
) {
    debug_assert!(n <= MAX_N && n > 0 && batch > 0);
    ensure_ctx(rocm, stream).expect("rlx-rocm: hipSOLVER required for EighNative");

    CTX.with(|cell| {
        let borrow = cell.borrow();
        let c = borrow.as_ref().unwrap();
        unsafe {
            c.solver.set_stream(stream).expect("hipsolverSetStream");
        }

        let rt = &rocm.runtime;
        let nn = batch * n * n;
        let d_a = HipBuffer::<f32>::alloc_zeros(rt, nn.max(1)).expect("eigh scratch A");
        let src = buffer.ptr + (in_off as u64) * 4;
        unsafe {
            (rt.hip_memcpy_dtod)(d_a.ptr, src, nn * 4)
                .ok()
                .expect("eigh memcpy A");
        }
        let d_w = HipBuffer::<f32>::alloc_zeros(rt, (batch * n).max(1)).expect("eigh scratch W");
        let d_info = HipBuffer::<i32>::alloc_zeros(rt, batch.max(1)).expect("eigh scratch info");

        let mut lwork = 0i32;
        unsafe {
            (c.solver.runtime.ssyevj_batched_buffer_size)(
                c.solver.handle,
                VEC,
                LO,
                n as i32,
                d_a.ptr as *mut f32,
                n as i32,
                d_w.ptr as *mut f32,
                &mut lwork,
                c.solver.params,
                batch as i32,
            )
            .ok()
            .expect("hipsolverSsyevjBatched_bufferSize");
        }
        let d_work = HipBuffer::<f32>::alloc_zeros(rt, lwork.max(1) as usize).expect("eigh work");
        unsafe {
            (c.solver.runtime.ssyevj_batched)(
                c.solver.handle,
                VEC,
                LO,
                n as i32,
                d_a.ptr as *mut f32,
                n as i32,
                d_w.ptr as *mut f32,
                d_work.ptr as *mut f32,
                lwork,
                d_info.ptr as *mut i32,
                c.solver.params,
                batch as i32,
            )
            .ok()
            .expect("hipsolverSsyevjBatched");
        }

        let per = n * n + n;
        let total = (batch * per) as u32;
        let block = 256u32;
        let grid = (total.div_ceil(block), 1u32, 1u32);
        let block_dim = (block, 1u32, 1u32);
        let mut out_ptr = buffer.ptr + (out_off as u64) * 4;
        let mut a_ptr = d_a.ptr;
        let mut w_ptr = d_w.ptr;
        let mut n_i = n as i32;
        let mut batch_i = batch as i32;
        crate::launch_kernel!(
            &c.kernel,
            stream,
            grid,
            block_dim,
            [&mut a_ptr, &mut w_ptr, &mut out_ptr, &mut n_i, &mut batch_i]
        );
        // Keep scratch buffers alive through the async launch.
        let _ = (&d_a, &d_w, &d_work, &d_info);
    });
}
