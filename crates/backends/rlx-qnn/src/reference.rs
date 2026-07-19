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

//! Rust forward pass — the parity reference for the emitted QNN model.
//!
//! The emitted `MatMul` node computes the *exact same* `in0 · in1` as
//! [`matmul_f32`]; [`linear_f32`] mirrors `in0 · in1 + in2`,
//! [`linear_relu_f32`] mirrors `relu(in0 · in1 + in2)`, and [`mlp2_f32`]
//! mirrors a two-layer `LinearRelu → Linear`. The host `verify.py`
//! independently checks the device output against numpy.

/// Row-major `out[m,n] = in0[m,k] · in1[k,n]`, f32, naive triple loop.
///
/// This is the *reference*, not a fast kernel — it mirrors the plain `MatMul`
/// semantics (`transpose_in0 = transpose_in1 = false`) so the two are trivially
/// comparable. Panics on a length mismatch (a codegen-side invariant, not a
/// runtime input path).
pub fn matmul_f32(in0: &[f32], in1: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(in0.len(), m * k, "in0 must be M*K");
    assert_eq!(in1.len(), k * n, "in1 must be K*N");
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += in0[i * k + kk] * in1[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Row-major `out[m,n] = in0[m,k] · in1[k,n] + in2[m,n]`, f32.
pub fn linear_f32(in0: &[f32], in1: &[f32], in2: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(in0.len(), m * k, "in0 must be M*K");
    assert_eq!(in1.len(), k * n, "in1 must be K*N");
    assert_eq!(in2.len(), m * n, "in2 must be M*N");
    let mm = matmul_f32(in0, in1, m, k, n);
    mm.iter().zip(in2).map(|(a, b)| a + b).collect()
}

/// Row-major `out[m,n] = max(in0[m,k] · in1[k,n] + in2[m,n], 0)`, f32.
pub fn linear_relu_f32(
    in0: &[f32],
    in1: &[f32],
    in2: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    assert_eq!(in0.len(), m * k, "in0 must be M*K");
    assert_eq!(in1.len(), k * n, "in1 must be K*N");
    assert_eq!(in2.len(), m * n, "in2 must be M*N");
    let mm = matmul_f32(in0, in1, m, k, n);
    mm.iter().zip(in2).map(|(a, b)| (a + b).max(0.0)).collect()
}

/// Two-layer MLP: `out = relu(in0·w1+b1)·w2+b2`, dims `M×K → H → N`.
pub fn mlp2_f32(
    in0: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    m: usize,
    k: usize,
    h: usize,
    n: usize,
) -> Vec<f32> {
    let hidden = linear_relu_f32(in0, w1, b1, m, k, h);
    linear_f32(&hidden, w2, b2, m, h, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        // in0 (2x2) times identity = in0.
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

    #[test]
    fn linear_matches_matmul_plus_bias() {
        let x = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let w = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let b = [0.5, -0.5, 1.0, -1.0]; // 2x2
        assert_eq!(linear_f32(&x, &w, &b, 2, 2, 2), vec![1.5, 1.5, 4.0, 3.0]);
    }

    #[test]
    fn linear_relu_matches_matmul_plus_bias() {
        let x = [0.0, 2.0]; // 2x1
        let w = [1.0]; // 1x1
        let b = [0.0, 0.0]; // 2x1
        assert_eq!(linear_relu_f32(&x, &w, &b, 2, 1, 1), vec![0.0, 2.0]);
    }

    #[test]
    fn mlp2_composes_linear_relu_then_linear() {
        let x = [1.0, -2.0]; // 1x2
        let w1 = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let b1 = [-0.5, 0.0]; // 1x2 → relu([0.5, -2]) = [0.5, 0]
        let w2 = [2.0, 3.0]; // 2x1
        let b2 = [1.0]; // 1x1
        // hidden = [0.5, 0]; out = 0.5*2 + 0*3 + 1 = 2
        assert_eq!(mlp2_f32(&x, &w1, &b1, &w2, &b2, 1, 2, 2, 1), vec![2.0]);
    }
}
