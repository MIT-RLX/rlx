#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

#[allow(unused_variables)]
pub(crate) fn compile_complex_norm_sq(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ComplexNormSq = &node.op else {
        unreachable!()
    };
    {
        let len: usize = (0..node.shape.rank())
            .map(|i| node.shape.dim(i).unwrap_static())
            .product();
        let src = node_offset(arena, node.inputs[0]);
        let dst = node_offset(arena, node.id);
        Thunk::ComplexNormSqF32 {
            src,
            dst,
            len: len as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_complex_norm_sq_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ComplexNormSqBackward = &node.op else {
        unreachable!()
    };
    {
        let len: usize = (0..node.shape.rank())
            .map(|i| node.shape.dim(i).unwrap_static())
            .product();
        let z = node_offset(arena, node.inputs[0]);
        let g = node_offset(arena, node.inputs[1]);
        let dz = node_offset(arena, node.id);
        Thunk::ComplexNormSqBackwardF32 {
            z,
            g,
            dz,
            len: len as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_conjugate(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Conjugate = &node.op else {
        unreachable!()
    };
    {
        let len: usize = (0..node.shape.rank())
            .map(|i| node.shape.dim(i).unwrap_static())
            .product();
        Thunk::ConjugateC64 {
            src: node_offset(arena, node.inputs[0]),
            dst: node_offset(arena, node.id),
            len: len as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_dense_solve(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::DenseSolve = &node.op else {
        unreachable!()
    };
    {
        // A: [n, n], b: [n] or [n, nrhs]. Output matches b.
        let a_shape = &graph.node(node.inputs[0]).shape;
        let n = a_shape.dim(0).unwrap_static();
        debug_assert_eq!(
            n,
            a_shape.dim(1).unwrap_static(),
            "DenseSolve: A must be square"
        );
        let b_elems = node.shape.num_elements().unwrap();
        let nrhs = b_elems / n;
        match node.shape.dtype() {
            rlx_ir::DType::F64 => Thunk::DenseSolveF64 {
                a: node_offset(arena, node.inputs[0]),
                b: node_offset(arena, node.inputs[1]),
                x: node_offset(arena, node.id),
                n: n as u32,
                nrhs: nrhs as u32,
            },
            rlx_ir::DType::F32 => Thunk::DenseSolveF32 {
                a: node_offset(arena, node.inputs[0]),
                b: node_offset(arena, node.inputs[1]),
                x: node_offset(arena, node.id),
                n: n as u32,
                nrhs: nrhs as u32,
            },
            other => panic!(
                "DenseSolve: F32 + F64 lowered; got {other:?}. \
                         Add another variant when needed."
            ),
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_batched_dense_solve(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::BatchedDenseSolve = &node.op else {
        unreachable!()
    };
    {
        // A: [B, N, N], b: [B, N] or [B, N, K]. Output matches b.
        let a_shape = &graph.node(node.inputs[0]).shape;
        assert_eq!(a_shape.rank(), 3, "BatchedDenseSolve: A rank must be 3");
        let batch = a_shape.dim(0).unwrap_static();
        let n = a_shape.dim(1).unwrap_static();
        debug_assert_eq!(
            n,
            a_shape.dim(2).unwrap_static(),
            "BatchedDenseSolve: A's last two dims must match"
        );
        let total = node.shape.num_elements().unwrap();
        let nrhs = total / (batch * n);
        match node.shape.dtype() {
            rlx_ir::DType::F32 => Thunk::BatchedDenseSolveF32 {
                a: node_offset(arena, node.inputs[0]),
                b: node_offset(arena, node.inputs[1]),
                x: node_offset(arena, node.id),
                batch: batch as u32,
                n: n as u32,
                nrhs: nrhs as u32,
            },
            rlx_ir::DType::F64 => Thunk::BatchedDenseSolveF64 {
                a: node_offset(arena, node.inputs[0]),
                b: node_offset(arena, node.inputs[1]),
                x: node_offset(arena, node.id),
                batch: batch as u32,
                n: n as u32,
                nrhs: nrhs as u32,
            },
            other => panic!("BatchedDenseSolve: F32 + F64 only, got {other:?}"),
        }
    }
}

#[inline(always)]
pub(crate) fn exec_cgemm_c64(t: &Thunk, base: *mut u8) {
    let Thunk::CgemmC64 { a, b, c, m, k, n } = t else {
        unreachable!()
    };
    unsafe {
        cgemm_c64(*a, *b, *c, *m as usize, *k as usize, *n as usize, base);
    }
}

#[inline(always)]
pub(crate) fn exec_dense_solve_f64(t: &Thunk, base: *mut u8) {
    let Thunk::DenseSolveF64 { a, b, x, n, nrhs } = t else {
        unreachable!()
    };
    {
        let (n_, nrhs_) = (*n as usize, *nrhs as usize);
        // LAPACK overwrites both A and B; clone into scratch
        // each call. Caller's A and b must be preserved for
        // VJP recompute. (Eventually: swap to a factor-once /
        // solve-many scheme; that's the symbolic-reuse story
        // and lives with the sparse path.)
        unsafe {
            let a_src = sl_f64(*a, base, n_ * n_);
            let b_src = sl_f64(*b, base, n_ * nrhs_);
            let mut a_scratch: Vec<f64> = a_src.to_vec();
            let mut x_buf: Vec<f64> = b_src.to_vec();
            let info = crate::blas::dgesv(&mut a_scratch, &mut x_buf, n_, nrhs_);
            if info != 0 {
                panic!(
                    "DenseSolveF64: dgesv reported singular matrix \
                                (info={info}, n={n_}, nrhs={nrhs_})"
                );
            }
            let dst = sl_mut_f64(*x, base, n_ * nrhs_);
            dst.copy_from_slice(&x_buf);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_dense_solve_f32(t: &Thunk, base: *mut u8) {
    let Thunk::DenseSolveF32 { a, b, x, n, nrhs } = t else {
        unreachable!()
    };
    {
        let (n_, nrhs_) = (*n as usize, *nrhs as usize);
        unsafe {
            let a_src = sl(*a, base, n_ * n_);
            let b_src = sl(*b, base, n_ * nrhs_);
            let mut a_scratch: Vec<f32> = a_src.to_vec();
            let mut x_buf: Vec<f32> = b_src.to_vec();
            let info = crate::blas::sgesv(&mut a_scratch, &mut x_buf, n_, nrhs_);
            if info != 0 {
                panic!(
                    "DenseSolveF32: sgesv reported singular matrix \
                             (info={info}, n={n_}, nrhs={nrhs_})"
                );
            }
            let dst = sl_mut(*x, base, n_ * nrhs_);
            dst.copy_from_slice(&x_buf);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_batched_dense_solve_f64(t: &Thunk, base: *mut u8) {
    let Thunk::BatchedDenseSolveF64 {
        a,
        b,
        x,
        batch,
        n,
        nrhs,
    } = t
    else {
        unreachable!()
    };
    {
        // Per slice: extract A_i and b_i, dgesv, write x_i.
        // LAPACK has no batched dgesv on Accelerate, so this
        // is a serial loop over the batch axis. cuSOLVER /
        // hipSOLVER expose `getrfBatched` / `getrsBatched` for
        // the GPU path — we'll wire that in rlx-cuda when
        // someone needs Linux+CUDA.
        let (b_, n_, nrhs_) = (*batch as usize, *n as usize, *nrhs as usize);
        let a_stride = n_ * n_;
        let b_stride = n_ * nrhs_;
        unsafe {
            let a_full = sl_f64(*a, base, b_ * a_stride);
            let b_full = sl_f64(*b, base, b_ * b_stride);
            let x_full = sl_mut_f64(*x, base, b_ * b_stride);
            for bi in 0..b_ {
                let mut a_scratch: Vec<f64> = a_full[bi * a_stride..(bi + 1) * a_stride].to_vec();
                let mut x_buf: Vec<f64> = b_full[bi * b_stride..(bi + 1) * b_stride].to_vec();
                let info = crate::blas::dgesv(&mut a_scratch, &mut x_buf, n_, nrhs_);
                if info != 0 {
                    panic!(
                        "BatchedDenseSolveF64: slice {bi} \
                                    singular (info={info}, n={n_}, nrhs={nrhs_})"
                    );
                }
                x_full[bi * b_stride..(bi + 1) * b_stride].copy_from_slice(&x_buf);
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_batched_dense_solve_f32(t: &Thunk, base: *mut u8) {
    let Thunk::BatchedDenseSolveF32 {
        a,
        b,
        x,
        batch,
        n,
        nrhs,
    } = t
    else {
        unreachable!()
    };
    {
        let (b_, n_, nrhs_) = (*batch as usize, *n as usize, *nrhs as usize);
        let a_stride = n_ * n_;
        let b_stride = n_ * nrhs_;
        unsafe {
            let a_full = sl(*a, base, b_ * a_stride);
            let b_full = sl(*b, base, b_ * b_stride);
            let x_full = sl_mut(*x, base, b_ * b_stride);
            for bi in 0..b_ {
                let mut a_scratch = a_full[bi * a_stride..(bi + 1) * a_stride].to_vec();
                let mut x_buf = b_full[bi * b_stride..(bi + 1) * b_stride].to_vec();
                let info = crate::blas::sgesv(&mut a_scratch, &mut x_buf, n_, nrhs_);
                if info != 0 {
                    panic!("BatchedDenseSolveF32: slice {bi} singular (info={info})");
                }
                x_full[bi * b_stride..(bi + 1) * b_stride].copy_from_slice(&x_buf);
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_batched_dgemm_f64(t: &Thunk, base: *mut u8) {
    let Thunk::BatchedDgemmF64 {
        a,
        b,
        c,
        batch,
        m,
        k,
        n,
    } = t
    else {
        unreachable!()
    };
    {
        let (b_, m_, k_, n_) = (*batch as usize, *m as usize, *k as usize, *n as usize);
        let a_stride = m_ * k_;
        let b_stride = k_ * n_;
        let c_stride = m_ * n_;
        unsafe {
            let a_full = sl_f64(*a, base, b_ * a_stride);
            let b_full = sl_f64(*b, base, b_ * b_stride);
            let c_full = sl_mut_f64(*c, base, b_ * c_stride);
            for bi in 0..b_ {
                let a_slice = &a_full[bi * a_stride..(bi + 1) * a_stride];
                let b_slice = &b_full[bi * b_stride..(bi + 1) * b_stride];
                let c_slice = &mut c_full[bi * c_stride..(bi + 1) * c_stride];
                crate::blas::dgemm(a_slice, b_slice, c_slice, m_, k_, n_);
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_dgemm(t: &Thunk, base: *mut u8) {
    let Thunk::Dgemm { a, b, c, m, k, n } = t else {
        unreachable!()
    };
    {
        let (m, k, n) = (*m as usize, *k as usize, *n as usize);
        unsafe {
            crate::blas::dgemm(
                sl_f64(*a, base, m * k),
                sl_f64(*b, base, k * n),
                sl_mut_f64(*c, base, m * n),
                m,
                k,
                n,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_complex_norm_sq_f32(t: &Thunk, base: *mut u8) {
    let Thunk::ComplexNormSqF32 { src, dst, len } = t else {
        unreachable!()
    };
    {
        let n = *len as usize;
        unsafe {
            let s = sl(*src, base, 2 * n);
            let d = sl_mut(*dst, base, n);
            for i in 0..n {
                let re = s[2 * i];
                let im = s[2 * i + 1];
                d[i] = re * re + im * im;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_complex_norm_sq_backward_f32(t: &Thunk, base: *mut u8) {
    let Thunk::ComplexNormSqBackwardF32 { z, g, dz, len } = t else {
        unreachable!()
    };
    {
        // Wirtinger: dz = g · z, element-wise complex
        // (g is real, z is complex).
        let n = *len as usize;
        unsafe {
            let zb = sl(*z, base, 2 * n);
            let gb = sl(*g, base, n);
            let db = sl_mut(*dz, base, 2 * n);
            for i in 0..n {
                let re = zb[2 * i];
                let im = zb[2 * i + 1];
                let gv = gb[i];
                db[2 * i] = gv * re;
                db[2 * i + 1] = gv * im;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_conjugate_c64(t: &Thunk, base: *mut u8) {
    let Thunk::ConjugateC64 { src, dst, len } = t else {
        unreachable!()
    };
    {
        let n = *len as usize;
        unsafe {
            let s = sl(*src, base, 2 * n);
            let d = sl_mut(*dst, base, 2 * n);
            for i in 0..n {
                d[2 * i] = s[2 * i];
                d[2 * i + 1] = -s[2 * i + 1];
            }
        }
    }
}

/// f32 counterpart of `execute_fft1d_f64`. Same 2N-real-block layout
/// (first N real, second N imag per row), same unnormalized
/// convention; only the element width differs. Twiddle factors are
/// computed in f64 and cast to f32 to keep large-N error closer to
/// the f64 path (the savings from f32 are in memory bandwidth, not in
/// twiddle precision).
/// Complex (C64) dense GEMM `C[m,n] = A[m,k] · B[k,n]`. Operands are
/// interleaved `[re, im]` f32; `a_off`/`b_off`/`c_off` are byte offsets
/// into `base`. Parallel over output rows (disjoint writes).
pub(crate) unsafe fn cgemm_c64(
    a_off: usize,
    b_off: usize,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
    base: *mut u8,
) {
    let bptr = base as usize;
    unsafe {
        let a = std::slice::from_raw_parts((bptr + a_off) as *const f32, 2 * m * k);
        let b = std::slice::from_raw_parts((bptr + b_off) as *const f32, 2 * k * n);
        let c_base = bptr + c_off;
        crate::pool::par_range(m, |i| {
            let crow = std::slice::from_raw_parts_mut((c_base + i * n * 8) as *mut f32, 2 * n);
            for j in 0..n {
                let mut re = 0f32;
                let mut im = 0f32;
                for l in 0..k {
                    let ar = a[2 * (i * k + l)];
                    let ai = a[2 * (i * k + l) + 1];
                    let br = b[2 * (l * n + j)];
                    let bi = b[2 * (l * n + j) + 1];
                    re += ar * br - ai * bi;
                    im += ar * bi + ai * br;
                }
                crow[2 * j] = re;
                crow[2 * j + 1] = im;
            }
        });
    }
}
