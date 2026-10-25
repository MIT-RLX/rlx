// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NumPy's `np.dot` and `np.matmul` (`@`) for `float64`, routed to BLAS exactly
//! as NumPy 2.x routes them — so a port of NumPy/SciPy/scikit-learn code gets the
//! same bits, not just the same mathematics.
//!
//! The two entry points do **not** route alike, and the routing decides the
//! result's last bits (different BLAS calls round differently):
//!
//! * `np.dot` — `cblasfuncs.c::cblas_matrixproduct`: matrix·matrix is `gemm`
//!   with `Trans` set for **F-contiguous** operands; matrix·vector is
//!   `gemv(RowMajor, NoTrans)` (or `ColMajor` for an F-ordered matrix); vector·vector
//!   is `0.0 + ddot`. A `(n, 1)` operand counts as a *column* (vector).
//! * `np.matmul` — `matmul.c.src`: matrix·matrix is `gemm` with `Trans` when an
//!   operand is only blasable transposed; matrix·column is `gemv(ColMajor, Trans)`;
//!   row·matrix is `gemv` with the operands swapped.
//! * Both use `dsyrk` (upper triangle, then mirrored) when the two operands are the
//!   **same buffer** viewed as `A` and `Aᵀ` — that is how `np.cov` and
//!   `X.T @ X` evaluate.
//!
//! The BLAS symbols are the ones NumPy links on macOS:
//! [`rlx_cpu::numpy_blas`] (`cblas_*$NEWLAPACK$ILP64`).

use rlx_cpu::numpy_blas::{self as nb, COL_MAJOR, NO_TRANS, ROW_MAJOR, TRANS};

/// A strided 2-D `float64` view, as NumPy sees an array: element `(i, j)` is
/// `data[offset + i*rs + j*cs]`. Two views of the same buffer compare equal by
/// `(data pointer, offset)` — what NumPy's `PyArray_BYTES(a) == PyArray_BYTES(b)`
/// tests.
#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    pub data: &'a [f64],
    pub offset: usize,
    pub rows: usize,
    pub cols: usize,
    /// row stride in elements
    pub rs: usize,
    /// column stride in elements
    pub cs: usize,
}

impl<'a> View<'a> {
    /// A C-contiguous `rows × cols` array.
    pub fn c(data: &'a [f64], rows: usize, cols: usize) -> Self {
        View {
            data,
            offset: 0,
            rows,
            cols,
            rs: cols,
            cs: 1,
        }
    }
    /// `a.T` — the transposed view of the same buffer.
    pub fn t(self) -> Self {
        View {
            rows: self.cols,
            cols: self.rows,
            rs: self.cs,
            cs: self.rs,
            ..self
        }
    }
    fn at(&self, i: usize, j: usize) -> f64 {
        self.data[self.offset + i * self.rs + j * self.cs]
    }
    fn ptr(&self) -> *const f64 {
        self.data[self.offset..].as_ptr()
    }
    fn slice(&self) -> &'a [f64] {
        &self.data[self.offset..]
    }
    /// `PyArray_IS_C_CONTIGUOUS`: strides `(cols, 1)`, ignoring length-1 axes.
    pub fn is_c_contiguous(&self) -> bool {
        (self.cols <= 1 || self.cs == 1) && (self.rows <= 1 || self.rs == self.cols)
    }
    /// `PyArray_IS_F_CONTIGUOUS`: strides `(1, rows)`, ignoring length-1 axes.
    pub fn is_f_contiguous(&self) -> bool {
        (self.rows <= 1 || self.rs == 1) && (self.cols <= 1 || self.cs == self.rows)
    }
    /// A C-contiguous copy (`PyArray_Copy`).
    pub fn to_c(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.rows * self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.push(self.at(i, j));
            }
        }
        out
    }
}

/// `matmul.c.src::is_blasable2d(stride1, stride2, d1, d2)` in element strides.
fn is_blasable2d(stride1: usize, stride2: usize, _d1: usize, d2: usize) -> bool {
    stride2 == 1 && stride1 >= d2
}

/// `syrk` + NumPy's mirror of the upper triangle into the lower.
fn syrk_mirror(
    order: i32,
    trans: i32,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    out: &mut [f64],
    ldc: usize,
) {
    nb::dsyrk_upper(order, trans, n, k, a, lda, out, ldc);
    for i in 0..n {
        for j in i + 1..n {
            out[j * ldc + i] = out[i * ldc + j];
        }
    }
}

/// `np.dot(a, b)` for two 2-D arrays where neither is a row/column/scalar
/// (both dimensions > 1) — the Level-3 branch of `cblas_matrixproduct`.
/// Returns the C-contiguous `a.rows × b.cols` result.
pub fn dot(a: View, b: View) -> Vec<f64> {
    assert!(
        a.rows > 1 && a.cols > 1 && b.rows > 1 && b.cols > 1,
        "dot: only the matrix·matrix (both dims > 1) branch is implemented; use dot_mv / dot_vv"
    );
    assert_eq!(
        a.cols, b.rows,
        "dot: shapes ({}, {}) and ({}, {}) not aligned",
        a.rows, a.cols, b.rows, b.cols
    );
    let (l, n, m) = (a.rows, b.cols, b.rows);
    // non-contiguous operands are copied (C order)
    let a_buf;
    let a = if !a.is_c_contiguous() && !a.is_f_contiguous() {
        a_buf = a.to_c();
        View::c(&a_buf, a.rows, a.cols)
    } else {
        a
    };
    let b_buf;
    let b = if !b.is_c_contiguous() && !b.is_f_contiguous() {
        b_buf = b.to_c();
        View::c(&b_buf, b.rows, b.cols)
    } else {
        b
    };
    let mut out = vec![0.0f64; l * n];
    let ldc = n.max(1);
    let (mut t1, mut t2) = (NO_TRANS, NO_TRANS);
    let mut lda = a.cols.max(1);
    let mut ldb = b.cols.max(1);
    if a.is_f_contiguous() {
        t1 = TRANS;
        lda = a.rows.max(1);
    }
    if b.is_f_contiguous() {
        t2 = TRANS;
        ldb = b.rows.max(1);
    }
    let same = std::ptr::eq(a.ptr(), b.ptr())
        && a.rows == b.cols
        && a.cols == b.rows
        && a.rs == b.cs
        && a.cs == b.rs;
    if same && ((t1 == TRANS) ^ (t2 == TRANS)) {
        if t1 == NO_TRANS {
            syrk_mirror(ROW_MAJOR, t1, n, m, a.slice(), lda, &mut out, ldc);
        } else {
            syrk_mirror(ROW_MAJOR, t1, n, m, b.slice(), ldb, &mut out, ldc);
        }
    } else {
        nb::dgemm(
            ROW_MAJOR,
            t1,
            t2,
            l,
            n,
            m,
            a.slice(),
            lda,
            b.slice(),
            ldb,
            &mut out,
            ldc,
        );
    }
    out
}

/// `np.dot(A, x)` for a 2-D `A` (both dims > 1) and a 1-D `x` (or an `(n, 1)` column):
/// `gemv(RowMajor, NoTrans)` for a C-ordered `A`, `ColMajor` for an F-ordered one.
pub fn dot_mv(a: View, x: &[f64], incx: usize) -> Vec<f64> {
    let a_buf;
    let a = if !(a.is_c_contiguous() || a.is_f_contiguous()) {
        a_buf = a.to_c();
        View::c(&a_buf, a.rows, a.cols)
    } else {
        a
    };
    let mut y = vec![0.0f64; a.rows];
    let (order, lda) = if a.is_c_contiguous() {
        (ROW_MAJOR, a.cols.max(1))
    } else {
        (COL_MAJOR, a.rows.max(1))
    };
    nb::dgemv(
        order,
        NO_TRANS,
        a.rows,
        a.cols,
        a.slice(),
        lda,
        x,
        incx,
        &mut y,
        1,
    );
    y
}

/// `np.dot(x, y)` for two 1-D arrays: `DOUBLE_dot` — `0.0 + Σ ddot(chunk)`.
pub fn dot_vv(x: &[f64], incx: usize, y: &[f64], incy: usize, n: usize) -> f64 {
    const CHUNK: usize = 1 << 30; // NPY_CBLAS_CHUNK
    let mut sum = 0.0f64;
    let mut done = 0;
    while done < n {
        let c = (n - done).min(CHUNK);
        sum += nb::ddot(c, &x[done * incx..], incx, &y[done * incy..], incy);
        done += c;
    }
    sum
}

/// `a @ b` (`np.matmul`) for 2-D operands, returning the C-contiguous result.
///
/// Implements `DOUBLE_matmul`'s BLAS branches: `gemv` for matrix @ column and row @
/// matrix, `gemm` / `syrk` for matrix @ matrix. (The non-BLAS fallbacks NumPy uses
/// for 1-sized or non-blasable cores are plain loops; they are reproduced by
/// copying to C order first, which NumPy also does for non-blasable inputs.)
pub fn matmul(a: View, b: View) -> Vec<f64> {
    assert_eq!(
        a.cols, b.rows,
        "matmul: shapes ({}, {}) and ({}, {}) not aligned",
        a.rows, a.cols, b.rows, b.cols
    );
    let (dm, dn, dp) = (a.rows, a.cols, b.cols);
    let mut out = vec![0.0f64; dm * dp];
    let (is1_m, is1_n) = (a.rs, a.cs);
    let (is2_n, is2_p) = (b.rs, b.cs);
    let (os_m, os_p) = (dp, 1usize);
    let i1_c = is_blasable2d(is1_m, is1_n, dm, dn);
    let i1_f = is_blasable2d(is1_n, is1_m, dn, dm);
    let i2_c = is_blasable2d(is2_n, is2_p, dn, dp);
    let i2_f = is_blasable2d(is2_p, is2_n, dp, dn);
    let (i1b, i2b) = (i1_c || i1_f, i2_c || i2_f);
    let special = dm == 1 || dn == 1 || dp == 1;
    if special {
        let vector_matrix = dm == 1 && i2b && is_blasable2d(is1_n, 1, dn, 1);
        let matrix_vector = dp == 1 && i1b && is_blasable2d(is2_n, 1, dn, 1);
        if dm == 1 && dp == 1 {
            // row @ column → DOUBLE_dot
            out[0] = dot_vv(a.slice(), is1_n, b.slice(), is2_n, dn);
        } else if vector_matrix {
            gemv_matmul(
                b.slice(),
                is2_p,
                is2_n,
                a.slice(),
                is1_n,
                &mut out,
                os_p,
                dp,
                dn,
            );
        } else if matrix_vector {
            gemv_matmul(
                a.slice(),
                is1_m,
                is1_n,
                b.slice(),
                is2_n,
                &mut out,
                os_m,
                dm,
                dn,
            );
        } else {
            // DOUBLE_matmul_inner_noblas: plain triple loop, accumulating from 0
            for m in 0..dm {
                for p in 0..dp {
                    let mut s = 0.0;
                    for n in 0..dn {
                        s += a.at(m, n) * b.at(n, p);
                    }
                    out[m * dp + p] = s;
                }
            }
        }
        return out;
    }
    // matrix @ matrix: copy non-blasable operands (C order)
    let a_buf;
    let a = if !i1b {
        a_buf = a.to_c();
        View::c(&a_buf, a.rows, a.cols)
    } else {
        a
    };
    let b_buf;
    let b = if !i2b {
        b_buf = b.to_c();
        View::c(&b_buf, b.rows, b.cols)
    } else {
        b
    };
    let (is1_m, is1_n, is2_n, is2_p) = (a.rs, a.cs, b.rs, b.cs);
    let ldc = os_m;
    let (trans1, lda) = if is_blasable2d(is1_m, is1_n, dm, dn) {
        (NO_TRANS, is1_m)
    } else {
        (TRANS, is1_n)
    };
    let (trans2, ldb) = if is_blasable2d(is2_n, is2_p, dn, dp) {
        (NO_TRANS, is2_n)
    } else {
        (TRANS, is2_p)
    };
    if std::ptr::eq(a.ptr(), b.ptr())
        && dm == dp
        && is1_m == is2_p
        && is1_n == is2_n
        && trans1 != trans2
    {
        if trans1 == NO_TRANS {
            syrk_mirror(ROW_MAJOR, trans1, dp, dn, a.slice(), lda, &mut out, ldc);
        } else {
            syrk_mirror(ROW_MAJOR, trans1, dp, dn, a.slice(), ldb, &mut out, ldc);
        }
    } else {
        nb::dgemm(
            ROW_MAJOR,
            trans1,
            trans2,
            dm,
            dp,
            dn,
            a.slice(),
            lda,
            b.slice(),
            ldb,
            &mut out,
            ldc,
        );
    }
    out
}

/// `DOUBLE_gemv` in `matmul.c.src`: `ColMajor`+`Trans` for a row-blasable matrix.
#[allow(clippy::too_many_arguments)]
fn gemv_matmul(
    ip1: &[f64],
    is1_m: usize,
    is1_n: usize,
    ip2: &[f64],
    is2_n: usize,
    op: &mut [f64],
    op_m: usize,
    m: usize,
    n: usize,
) {
    let (order, lda) = if is_blasable2d(is1_m, is1_n, m, n) {
        (COL_MAJOR, is1_m)
    } else {
        (ROW_MAJOR, is1_n)
    };
    nb::dgemv(order, TRANS, n, m, ip1, lda, ip2, is2_n, op, op_m);
}
