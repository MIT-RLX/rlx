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

//! Linear solvers for FDM stiffness systems (dense LU + sparse PCG).

use crate::equilibrium::FdmError;

/// Solve `K x = b` for `nrhs` right-hand sides stored column-major (`b[i*nrhs + c]`).
pub fn solve_columns_dense(
    k: &[f64],
    b: &[f64],
    n: usize,
    nrhs: usize,
) -> Result<Vec<f64>, FdmError> {
    if k.len() != n * n || b.len() != n * nrhs {
        return Err(FdmError::Dimension(format!(
            "K is {n}×{n}, b len {}",
            b.len()
        )));
    }
    let mut x = b.to_vec();
    lu_solve(k, n, &mut x, nrhs)?;
    Ok(x)
}

/// Solve one CSR system `A x = b` with Jacobi-preconditioned CG (jax_fdm / `rlx-sparse` style).
pub fn pcg_solve(
    values: &[f64],
    col_idx: &[i32],
    row_ptr: &[i32],
    b: &[f64],
    out: &mut [f64],
    max_iter: u32,
    tol: f64,
) -> Result<(), FdmError> {
    let n = b.len();
    if out.len() != n || row_ptr.len() != n + 1 {
        return Err(FdmError::Dimension(format!(
            "pcg: n={n}, out={}, row_ptr={}",
            out.len(),
            row_ptr.len()
        )));
    }

    let mut diag = vec![1.0f64; n];
    for r in 0..n {
        for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
            if col_idx[k] as usize == r {
                diag[r] = values[k].max(f64::MIN_POSITIVE);
                break;
            }
        }
    }
    let inv_diag: Vec<f64> = diag.iter().map(|&d| 1.0 / d).collect();

    let matvec = |x: &[f64], y: &mut [f64]| {
        for r in 0..n {
            let mut acc = 0.0;
            for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                acc += values[k] * x[col_idx[k] as usize];
            }
            y[r] = acc;
        }
    };

    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut z: Vec<f64> = r.iter().zip(&inv_diag).map(|(rv, mi)| rv * mi).collect();
    let mut p = z.clone();
    let mut ap = vec![0.0; n];
    let mut rho_old: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();

    for _ in 0..max_iter {
        let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if r_norm < tol {
            break;
        }
        matvec(&p, &mut ap);
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if pap == 0.0 {
            return Err(FdmError::SingularStiffness);
        }
        let alpha = rho_old / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
        }
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }
        for i in 0..n {
            z[i] = r[i] * inv_diag[i];
        }
        let rho_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rho_new / rho_old;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rho_old = rho_new;
    }
    out.copy_from_slice(&x);
    Ok(())
}

/// Gaussian elimination with partial pivoting (`K` may be indefinite).
pub fn lu_solve(a: &[f64], n: usize, b: &mut [f64], nrhs: usize) -> Result<(), FdmError> {
    let mut a = a.to_vec();
    let mut piv = (0..n).collect::<Vec<_>>();
    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_abs = a[piv[col] * n + col].abs();
        for row in (col + 1)..n {
            let v = a[piv[row] * n + col].abs();
            if v > pivot_abs {
                pivot_abs = v;
                pivot_row = row;
            }
        }
        if pivot_abs < 1e-14 {
            return Err(FdmError::SingularStiffness);
        }
        piv.swap(col, pivot_row);
        let p = piv[col];
        for row in (col + 1)..n {
            let pr = piv[row];
            let factor = a[pr * n + col] / a[p * n + col];
            a[pr * n + col] = 0.0;
            for k in (col + 1)..n {
                a[pr * n + k] -= factor * a[p * n + k];
            }
            for rhs in 0..nrhs {
                b[pr * nrhs + rhs] -= factor * b[p * nrhs + rhs];
            }
        }
    }
    for col in 0..nrhs {
        for i in (0..n).rev() {
            let p = piv[i];
            let mut s = b[p * nrhs + col];
            for k in (i + 1)..n {
                s -= a[p * n + k] * b[piv[k] * nrhs + col];
            }
            let diag = a[p * n + i];
            if diag.abs() < 1e-14 {
                return Err(FdmError::SingularStiffness);
            }
            b[p * nrhs + col] = s / diag;
        }
    }
    Ok(())
}
