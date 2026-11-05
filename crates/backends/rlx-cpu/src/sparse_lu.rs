// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sparse LU of a square real matrix on Accelerate (macOS 15.5+):
//! supernodal, multithreaded, its own fill-reducing ordering and pivoting.
//! The structure is analysed once ([`SparseLu::analyze`]); each new set of
//! values is factored again on it ([`SparseLu::factor`]), then solved.
//!
//! ```ignore
//! // A in compressed columns: col_starts[n + 1], row_indices[nnz]
//! let mut lu = SparseLu::analyze(n, &col_starts, &row_indices, Pivoting::Threshold)?;
//! lu.factor(&values)?;           // values in the same order as row_indices
//! lu.solve(&mut b)?;             // b := A⁻¹ b
//! ```

use std::ffi::c_void;
use std::sync::Mutex;

/// Accelerate's sparse factorizations crash when two threads factor at once
/// (seen on macOS 26: a null read in libSparse, each thread with its own
/// factorization), so every call into it holds this lock. A call is itself
/// multithreaded (GCD).
static ACCELERATE: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    ACCELERATE.lock().unwrap_or_else(|e| e.into_inner())
}

unsafe extern "C" {
    fn rlx_splu_symbolic(n: i32, colptr: *const i64, rowidx: *const i32, kind: i32) -> *mut c_void;
    fn rlx_splu_numeric(h: *mut c_void, vals: *const f64) -> i32;
    fn rlx_splu_solve(h: *mut c_void, b: *mut f64) -> i32;
    fn rlx_splu_free(h: *mut c_void);
}

/// How the factorization pivots (`SparseFactorizationLU*`)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pivoting {
    /// Accelerate's default (`SparseFactorizationLU`)
    Default = 0,
    /// threshold partial pivoting (`SparseFactorizationLUTPP`)
    Threshold = 1,
    /// static pivoting (`SparseFactorizationLUSPP`)
    Static = 2,
    /// none: the diagonal as it is (`SparseFactorizationLUUnpivoted`)
    None = 3,
}

/// A failed analysis or factorization: Accelerate's `SparseStatus_t`
/// (`-1` the analysis failed, `-2` singular, `-3` not positive definite,
/// `-4` invalid argument, `-5` …)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseLuError(pub i32);

/// The analysis of a structure and, once factored, its factors
pub struct SparseLu {
    h: *mut c_void,
    n: usize,
    nnz: usize,
    factored: bool,
}

// SAFETY: the handle is owned by this value; Accelerate's factor objects
// may be used from any thread, one at a time (`&mut self`)
unsafe impl Send for SparseLu {}

impl SparseLu {
    /// The symbolic analysis of the n×n structure in compressed columns
    pub fn analyze(
        n: usize,
        col_starts: &[i64],
        row_indices: &[i32],
        pivoting: Pivoting,
    ) -> Result<SparseLu, SparseLuError> {
        assert_eq!(col_starts.len(), n + 1, "col_starts: n + 1 entries");
        let nnz = col_starts[n] as usize;
        assert!(
            row_indices.len() >= nnz,
            "row_indices: col_starts[n] entries"
        );
        let _g = lock();
        // SAFETY: the shim copies the arrays
        let h = unsafe {
            rlx_splu_symbolic(
                n as i32,
                col_starts.as_ptr(),
                row_indices.as_ptr(),
                pivoting as i32,
            )
        };
        if h.is_null() {
            return Err(SparseLuError(-1));
        }
        Ok(SparseLu {
            h,
            n,
            nnz,
            factored: false,
        })
    }

    /// The numeric factorization of `values` (in `row_indices`' order)
    pub fn factor(&mut self, values: &[f64]) -> Result<(), SparseLuError> {
        assert!(values.len() >= self.nnz, "values: one per structural entry");
        let _g = lock();
        // SAFETY: values outlives the call; the factors keep no pointer to it
        let st = unsafe { rlx_splu_numeric(self.h, values.as_ptr()) };
        self.factored = st == 0;
        if st == 0 {
            Ok(())
        } else {
            Err(SparseLuError(st))
        }
    }

    /// `b := A⁻¹ b` with the last factors. Accelerate factors an exactly
    /// singular matrix without complaint (a zero pivot divides to inf in
    /// the solve), so a solution that is not finite is `SparseLuError(-2)`
    /// (`SparseMatrixIsSingular`) here.
    pub fn solve(&mut self, b: &mut [f64]) -> Result<(), SparseLuError> {
        assert!(self.factored, "SparseLu::solve before a successful factor");
        assert_eq!(b.len(), self.n);
        let _g = lock();
        // SAFETY: b has n entries
        let st = unsafe { rlx_splu_solve(self.h, b.as_mut_ptr()) };
        if st != 0 {
            Err(SparseLuError(st))
        } else if b.iter().all(|x| x.is_finite()) {
            Ok(())
        } else {
            Err(SparseLuError(-2))
        }
    }

    pub fn size(&self) -> usize {
        self.n
    }
}

impl Drop for SparseLu {
    fn drop(&mut self) {
        let _g = lock();
        // SAFETY: the shim's own handle, freed once
        unsafe { rlx_splu_free(self.h) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_small_system() {
        // [4 1 0; 1 3 1; 0 1 2] x = [1 2 3]
        let col_starts = [0i64, 2, 5, 7];
        let rows = [0, 1, 0, 1, 2, 1, 2];
        let vals = [4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0];
        for p in [Pivoting::Default, Pivoting::Threshold, Pivoting::None] {
            let mut lu = SparseLu::analyze(3, &col_starts, &rows, p).unwrap();
            lu.factor(&vals).unwrap();
            let mut b = [1.0, 2.0, 3.0];
            lu.solve(&mut b).unwrap();
            let r = [
                4.0 * b[0] + b[1] - 1.0,
                b[0] + 3.0 * b[1] + b[2] - 2.0,
                b[1] + 2.0 * b[2] - 3.0,
            ];
            assert!(r.iter().all(|x| x.abs() < 1e-12), "{p:?}: {r:?}");
        }
    }

    #[test]
    fn singular_is_an_error() {
        let col_starts = [0i64, 1, 2];
        let rows = [0, 1];
        for p in [
            Pivoting::Default,
            Pivoting::Threshold,
            Pivoting::Static,
            Pivoting::None,
        ] {
            let mut lu = SparseLu::analyze(2, &col_starts, &rows, p).unwrap();
            // (Accelerate factors it; the solve finds it)
            let singular = match lu.factor(&[1.0, 0.0]) {
                Err(e) => e == SparseLuError(-2),
                Ok(()) => lu.solve(&mut [1.0, 1.0]) == Err(SparseLuError(-2)),
            };
            assert!(singular, "{p:?}");
        }
    }
}
