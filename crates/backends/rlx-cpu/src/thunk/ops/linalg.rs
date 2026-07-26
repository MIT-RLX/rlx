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
pub(crate) fn compile_cholesky(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Cholesky = &node.op else {
        unreachable!()
    };
    let a_shape = &graph.node(node.inputs[0]).shape;
    let n = a_shape.dim(0).unwrap_static();
    debug_assert_eq!(
        n,
        a_shape.dim(1).unwrap_static(),
        "Cholesky: A must be square"
    );
    assert_eq!(
        node.shape.dtype(),
        rlx_ir::DType::F32,
        "Cholesky: only F32 lowered; got {:?}",
        node.shape.dtype()
    );
    Thunk::CholeskyF32 {
        a: node_offset(arena, node.inputs[0]),
        l: node_offset(arena, node.id),
        n: n as u32,
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_triangular_solve(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::TriangularSolve { lower, transpose } = &node.op else {
        unreachable!()
    };
    let a_shape = &graph.node(node.inputs[0]).shape;
    let n = a_shape.dim(0).unwrap_static();
    let nrhs = node.shape.num_elements().unwrap() / n;
    assert_eq!(
        node.shape.dtype(),
        rlx_ir::DType::F32,
        "TriangularSolve: only F32 lowered; got {:?}",
        node.shape.dtype()
    );
    Thunk::TriangularSolveF32 {
        a: node_offset(arena, node.inputs[0]),
        b: node_offset(arena, node.inputs[1]),
        x: node_offset(arena, node.id),
        n: n as u32,
        nrhs: nrhs as u32,
        lower: *lower,
        transpose: *transpose,
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

pub(crate) fn exec_cholesky(t: &Thunk, base: *mut u8) {
    let Thunk::CholeskyF32 { a, l, n } = t else {
        unreachable!()
    };
    {
        let n_ = *n as usize;
        unsafe {
            let a_src = sl(*a, base, n_ * n_);
            // LAPACK `dpotrf` is f64; promote for numerical robustness. `lower`
            // yields the row-major lower-triangular L (strict upper zeroed).
            let mut buf: Vec<f64> = a_src.iter().map(|&v| v as f64).collect();
            let info = crate::blas::dpotrf(&mut buf, n_, true);
            assert_eq!(
                info, 0,
                "CholeskyF32: matrix not SPD (dpotrf info={info}, n={n_})"
            );
            let dst = sl_mut(*l, base, n_ * n_);
            for i in 0..n_ * n_ {
                dst[i] = buf[i] as f32;
            }
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_det(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let log_abs = match &node.op {
        Op::Det => false,
        Op::LogDet => true,
        _ => unreachable!(),
    };
    let a_shape = &graph.node(node.inputs[0]).shape;
    let n = a_shape.dim(0).unwrap_static();
    Thunk::DetF32 {
        a: node_offset(arena, node.inputs[0]),
        out: node_offset(arena, node.id),
        n: n as u32,
        log_abs,
    }
}

pub(crate) fn exec_det(t: &Thunk, base: *mut u8) {
    let Thunk::DetF32 { a, out, n, log_abs } = t else {
        unreachable!()
    };
    {
        let n_ = *n as usize;
        unsafe {
            let a_src = sl(*a, base, n_ * n_);
            let mut a64: Vec<f64> = a_src.iter().map(|&v| v as f64).collect();
            let (logabs, _sign, det) = crate::blas::lu_slogdet(&mut a64, n_);
            let dst = sl_mut(*out, base, 1);
            dst[0] = if *log_abs { logabs as f32 } else { det as f32 };
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_sort(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let (axis, descending, arg) = match &node.op {
        Op::Sort { axis, descending } => (*axis, *descending, false),
        Op::ArgSort { axis, descending } => (*axis, *descending, true),
        _ => unreachable!(),
    };
    let shape = &graph.node(node.inputs[0]).shape;
    let dims: Vec<usize> = (0..shape.rank())
        .map(|d| shape.dim(d).unwrap_static())
        .collect();
    let axis_dim = dims[axis];
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product();
    Thunk::SortF32 {
        src: node_offset(arena, node.inputs[0]),
        dst: node_offset(arena, node.id),
        outer: outer as u32,
        axis_dim: axis_dim as u32,
        inner: inner as u32,
        descending,
        arg,
    }
}

pub(crate) fn exec_sort(t: &Thunk, base: *mut u8) {
    let Thunk::SortF32 {
        src,
        dst,
        outer,
        axis_dim,
        inner,
        descending,
        arg,
    } = t
    else {
        unreachable!()
    };
    {
        let (outer, axis_dim, inner) = (*outer as usize, *axis_dim as usize, *inner as usize);
        let total = outer * axis_dim * inner;
        unsafe {
            let inp = sl(*src, base, total);
            let out = sl_mut(*dst, base, total);
            for o in 0..outer {
                for i in 0..inner {
                    let base_off = o * axis_dim * inner + i;
                    // Stable sort of the `axis_dim` strided elements.
                    let mut perm: Vec<usize> = (0..axis_dim).collect();
                    perm.sort_by(|&a, &b| {
                        let va = inp[base_off + a * inner];
                        let vb = inp[base_off + b * inner];
                        let ord = va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal);
                        if *descending { ord.reverse() } else { ord }
                    });
                    for (k, &r) in perm.iter().enumerate() {
                        out[base_off + k * inner] = if *arg {
                            r as f32
                        } else {
                            inp[base_off + r * inner]
                        };
                    }
                }
            }
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_svd(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let part = match &node.op {
        Op::Svd { part } => match part {
            rlx_ir::op::SvdPart::U => 0u8,
            rlx_ir::op::SvdPart::S => 1u8,
            rlx_ir::op::SvdPart::Vt => 2u8,
        },
        _ => unreachable!(),
    };
    let a_shape = &graph.node(node.inputs[0]).shape;
    Thunk::SvdF32 {
        a: node_offset(arena, node.inputs[0]),
        out: node_offset(arena, node.id),
        m: a_shape.dim(0).unwrap_static() as u32,
        n: a_shape.dim(1).unwrap_static() as u32,
        part,
    }
}

pub(crate) fn exec_svd(t: &Thunk, base: *mut u8) {
    let Thunk::SvdF32 { a, out, m, n, part } = t else {
        unreachable!()
    };
    {
        let (m_, n_) = (*m as usize, *n as usize);
        let k = m_.min(n_);
        unsafe {
            let a_src = sl(*a, base, m_ * n_);
            let mut a64: Vec<f64> = a_src.iter().map(|&v| v as f64).collect();
            let mut s = vec![0f64; k];
            let mut u = vec![0f64; m_ * k];
            let mut vt = vec![0f64; k * n_];
            let info = crate::blas::dgesdd_thin(&mut a64, m_, n_, &mut s, &mut u, &mut vt);
            assert_eq!(info, 0, "Svd: gesdd failed (info={info}, m={m_}, n={n_})");
            let (dst_len, srcv): (usize, &[f64]) = match part {
                0 => (m_ * k, &u),
                1 => (k, &s),
                _ => (k * n_, &vt),
            };
            let dst = sl_mut(*out, base, dst_len);
            for i in 0..dst_len {
                dst[i] = srcv[i] as f32;
            }
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_qr(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let part = match &node.op {
        Op::Qr { part } => match part {
            rlx_ir::op::QrPart::Q => 0u8,
            rlx_ir::op::QrPart::R => 1u8,
        },
        _ => unreachable!(),
    };
    let a_shape = &graph.node(node.inputs[0]).shape;
    Thunk::QrF32 {
        a: node_offset(arena, node.inputs[0]),
        out: node_offset(arena, node.id),
        m: a_shape.dim(0).unwrap_static() as u32,
        n: a_shape.dim(1).unwrap_static() as u32,
        part,
    }
}

pub(crate) fn exec_qr(t: &Thunk, base: *mut u8) {
    let Thunk::QrF32 { a, out, m, n, part } = t else {
        unreachable!()
    };
    {
        let (m_, n_) = (*m as usize, *n as usize);
        let k = m_.min(n_);
        unsafe {
            let a_src = sl(*a, base, m_ * n_);
            let a64: Vec<f64> = a_src.iter().map(|&v| v as f64).collect();
            let mut q = vec![0f64; m_ * k];
            let mut r = vec![0f64; k * n_];
            let info = crate::blas::qr_thin(&a64, m_, n_, &mut q, &mut r);
            assert_eq!(info, 0, "Qr: geqrf/orgqr failed (info={info})");
            let (dst_len, srcv): (usize, &[f64]) = if *part == 0 {
                (m_ * k, &q)
            } else {
                (k * n_, &r)
            };
            let dst = sl_mut(*out, base, dst_len);
            for i in 0..dst_len {
                dst[i] = srcv[i] as f32;
            }
        }
    }
}

pub(crate) fn exec_triangular_solve(t: &Thunk, base: *mut u8) {
    let Thunk::TriangularSolveF32 {
        a,
        b,
        x,
        n,
        nrhs,
        lower,
        transpose,
    } = t
    else {
        unreachable!()
    };
    {
        let (n_, nrhs_) = (*n as usize, *nrhs as usize);
        unsafe {
            let a_src = sl(*a, base, n_ * n_);
            let b_src = sl(*b, base, n_ * nrhs_);
            let a64: Vec<f64> = a_src.iter().map(|&v| v as f64).collect();
            let mut x64: Vec<f64> = b_src.iter().map(|&v| v as f64).collect();
            crate::blas::dtrsm_lower_or_upper(&a64, &mut x64, n_, nrhs_, *lower, *transpose);
            let dst = sl_mut(*x, base, n_ * nrhs_);
            for i in 0..n_ * nrhs_ {
                dst[i] = x64[i] as f32;
            }
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
