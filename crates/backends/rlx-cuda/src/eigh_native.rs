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

//! Native CUDA symmetric eigendecomposition via cuSOLVER **`SsyevjBatched`** —
//! the on-device forward path for `Op::Eigh` / `Op::EighBatch`, replacing the
//! F64 CPU host-fallback (`Step::SpdHost`, which does D2H → LAPACK → H2D).
//!
//! cuSOLVER lowers `syevjBatched` to a single templated `batch_parallel_jacobi`
//! kernel that runs the cyclic-Jacobi sweeps for the whole batch in parallel
//! (one block per matrix), so thousands of small covariances resolve in one
//! launch — ~0.5 µs/matrix at n=32 on an NVIDIA GPU vs ~7.6 µs on the rayon
//! CPU path, and no host round-trip. Runs in f32 (matching the widened SPD
//! arena; f32 Jacobi matches f64 LAPACK to cos≈1.0 for these small SPD blocks).
//!
//! Constraint: batched syevj supports **n ≤ 32** (the batched-Jacobi tile). The
//! scheduler routes larger `n` back to `Step::SpdHost`.
//!
//! Output matches [`rlx_cpu::spd::eigh_packed`]: per matrix `[λ (n) ∥ U (n²)]`,
//! `λ` ascending, `U` row-major with column `j` = eigenvector `j`
//! (`A = U diag(λ) Uᵀ`). cuSOLVER writes eigenvectors **column-major**, so the
//! `eigh_assemble` kernel transposes them into `U` and interleaves `λ`.

use cudarc::cusolver::sys as cs;
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use std::cell::RefCell;
use std::sync::Arc;

const VEC: cs::cusolverEigMode_t = cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
const LO: cs::cublasFillMode_t = cs::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;

/// cuSOLVER's batched Jacobi supports only `n ≤ 32`.
pub const MAX_N: usize = 32;

/// Transpose cuSOLVER's column-major eigenvectors into rlx's packed `[λ ∥ U]`
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
    handle: cs::cusolverDnHandle_t,
    params: cs::syevjInfo_t,
    kernel: crate::kernels::CudaKernel,
}

thread_local! {
    static CTX: RefCell<Option<EighCtx>> = const { RefCell::new(None) };
}

/// Run a native batched symmetric eigendecomposition on the arena in place.
/// `in_off` / `out_off` are f32-element offsets: input `A [batch·n·n]`, output
/// packed `[batch·(n·n+n)]`. `A` is copied to scratch first (syevj overwrites
/// its input), so the arena input slot is left intact.
#[allow(clippy::too_many_arguments)]
pub fn run(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    in_off: usize,
    out_off: usize,
    n: usize,
    batch: usize,
) {
    CTX.with(|cell| {
        {
            let mut opt = cell.borrow_mut();
            if opt.is_none() {
                let handle = cudarc::cusolver::result::dn_create().expect("cusolverDnCreate");
                let mut params: cs::syevjInfo_t = std::ptr::null_mut();
                unsafe {
                    cs::cusolverDnCreateSyevjInfo(&mut params)
                        .result()
                        .expect("cusolverDnCreateSyevjInfo");
                }
                let kernel = crate::kernels::compile(ctx, ASSEMBLE_SRC, "eigh_assemble");
                *opt = Some(EighCtx {
                    handle,
                    params,
                    kernel,
                });
            }
        }
        let borrow = cell.borrow();
        let c = borrow.as_ref().unwrap();
        unsafe {
            cudarc::cusolver::result::dn_set_stream(
                c.handle,
                stream.cu_stream() as *mut cs::CUstream_st,
            )
            .expect("cusolverDnSetStream");
        }

        // Scratch: copy A (syevj overwrites it with eigenvectors), W, info.
        let mut d_a = stream.alloc_zeros::<f32>(batch * n * n).unwrap();
        stream
            .memcpy_dtod(&buffer.slice(in_off..in_off + batch * n * n), &mut d_a)
            .unwrap();
        let mut d_w = stream.alloc_zeros::<f32>(batch * n).unwrap();
        let mut d_info = stream.alloc_zeros::<i32>(batch).unwrap();

        let mut lwork = 0i32;
        unsafe {
            let (ap, _g1) = d_a.device_ptr(stream);
            let (wp, _g2) = d_w.device_ptr(stream);
            cs::cusolverDnSsyevjBatched_bufferSize(
                c.handle,
                VEC,
                LO,
                n as i32,
                ap as *const f32,
                n as i32,
                wp as *const f32,
                &mut lwork,
                c.params,
                batch as i32,
            )
            .result()
            .expect("SsyevjBatched_bufferSize");
        }
        let mut d_work = stream.alloc_zeros::<f32>(lwork.max(1) as usize).unwrap();
        unsafe {
            let (ap, _g1) = d_a.device_ptr_mut(stream);
            let (wp, _g2) = d_w.device_ptr_mut(stream);
            let (kp, _g3) = d_work.device_ptr_mut(stream);
            let (ip, _g4) = d_info.device_ptr_mut(stream);
            cs::cusolverDnSsyevjBatched(
                c.handle,
                VEC,
                LO,
                n as i32,
                ap as *mut f32,
                n as i32,
                wp as *mut f32,
                kp as *mut f32,
                lwork,
                ip as *mut i32,
                c.params,
                batch as i32,
            )
            .result()
            .expect("SsyevjBatched");
        }

        // Assemble packed [λ ∥ U] into the arena output slot.
        let per = n * n + n;
        let total = (batch * per) as u32;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut out_view = buffer.slice_mut(out_off..out_off + batch * per);
        let (n_i, batch_i) = (n as i32, batch as i32);
        let mut lb = stream.launch_builder(&c.kernel.function);
        lb.arg(&d_a)
            .arg(&d_w)
            .arg(&mut out_view)
            .arg(&n_i)
            .arg(&batch_i);
        unsafe { lb.launch(cfg).expect("eigh_assemble launch") };
    });
}
