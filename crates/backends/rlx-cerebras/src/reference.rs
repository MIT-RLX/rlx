// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rust forward pass — the parity oracle for the emitted CSL.
//!
//! The emitted `pe_program.csl` kernel computes the *exact same* nested loop
//! as [`matmul_f32`]; the host `run.py` independently checks the device output
//! against numpy. Keeping a Rust oracle here lets CI assert the lowering's
//! intended semantics without the SDK simulator, and gives the eventual
//! simulator harness a second reference to diff against.

/// Row-major `Y[m,n] = X[m,k] · W[k,n]`, f32, naive triple loop.
///
/// This is the *reference*, not a fast kernel — it mirrors the scalar CSL
/// emitted for a single PE so the two are trivially comparable. Panics on a
/// length mismatch (a codegen-side invariant, not a runtime input path).
pub fn matmul_f32(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * k, "X must be M*K");
    assert_eq!(w.len(), k * n, "W must be K*N");
    let mut y = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[i * k + kk] * w[kk * n + j];
            }
            y[i * n + j] = acc;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        // X (2x2) times identity = X.
        let x = [1.0, 2.0, 3.0, 4.0];
        let id = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(matmul_f32(&x, &id, 2, 2, 2), x);
    }

    #[test]
    fn matmul_rectangular() {
        // [1 2 3] · [[1],[0],[-1]] = [1 - 3] = [-2]; second row [4 5 6] = [-2].
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let w = [1.0, 0.0, -1.0]; // 3x1
        assert_eq!(matmul_f32(&x, &w, 2, 3, 1), vec![-2.0, -2.0]);
    }
}
