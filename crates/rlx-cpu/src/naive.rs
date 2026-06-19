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

//! Naive reference implementations — for correctness testing and benchmarking.
//!
//! These are intentionally simple, unoptimized implementations that serve as:
//! 1. Ground truth for verifying SIMD/BLAS kernels produce correct results
//! 2. Baseline for benchmarking speedups from optimized implementations
//! 3. Fallback when no BLAS is linked

/// Naive matrix multiply: C = A @ B
/// A: [m, k], B: [k, n], C: [m, n], all row-major.
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    c.fill(0.0);
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
}

/// Naive GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
pub fn gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Naive erf (Abramowitz & Stegun)
pub fn erf(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0f32 } else { -1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let y = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    sign * (1.0 - y * (-xa * xa).exp())
}

/// Naive SiLU: x * sigmoid(x)
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Naive LayerNorm
pub fn layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    h: usize,
    eps: f32,
) {
    let n = input.len() / h;
    for row in 0..n {
        let base = row * h;
        let slice = &input[base..base + h];
        let mean: f32 = slice.iter().sum::<f32>() / h as f32;
        let var: f32 = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / h as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..h {
            output[base + i] = (input[base + i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

/// Naive softmax along last dimension
pub fn softmax(data: &mut [f32], rows: usize, cols: usize) {
    for row in 0..rows {
        let base = row * cols;
        let slice = &mut data[base..base + cols];
        let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for v in slice.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in slice.iter_mut() {
            *v *= inv;
        }
    }
}

/// Naive bias add: data[row, j] += bias\[j\]
pub fn bias_add(data: &mut [f32], bias: &[f32], m: usize, n: usize) {
    for row in 0..m {
        for j in 0..n {
            data[row * n + j] += bias[j];
        }
    }
}

/// Naive matmul + bias + GELU
pub fn matmul_bias_gelu(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    matmul(a, b, c, m, k, n);
    for row in 0..m {
        for j in 0..n {
            c[row * n + j] = gelu(c[row * n + j] + bias[j]);
        }
    }
}

/// Naive SDPA: softmax(Q @ K^T * scale + mask) @ V
/// mask: per-position mask where 0.0 = attend, 1.0 = ignore.
/// Pass empty slice for no masking.
pub fn sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    mask: &[f32],
    seq: usize,
    head_dim: usize,
    scale: f32,
) {
    let mut scores = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq + j] = dot * scale;
            // Apply mask: if mask[j] == 1.0 (padding), set to -inf
            if !mask.is_empty() && mask[j] > 0.5 {
                scores[i * seq + j] = -1e9;
            }
        }
    }
    softmax(&mut scores, seq, seq);
    matmul(&scores, v, output, seq, seq, head_dim);
}

/// Backward-compat wrapper without mask.
pub fn sdpa_no_mask(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq: usize,
    head_dim: usize,
    scale: f32,
) {
    sdpa(q, k, v, output, &[], seq, head_dim, scale);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_matmul_identity() {
        let a = [1.0, 0.0, 0.0, 1.0f32];
        let b = [3.0, 4.0, 5.0, 6.0f32];
        let mut c = [0.0f32; 4];
        matmul(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, [3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn naive_gelu_values() {
        assert!((gelu(0.0)).abs() < 1e-6);
        assert!((gelu(1.0) - 0.8413).abs() < 0.01);
        assert!((gelu(-1.0) - -0.1587).abs() < 0.01);
    }

    #[test]
    fn naive_layer_norm_zero_mean() {
        let input = vec![1.0, 2.0, 3.0, 4.0f32];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let mut output = vec![0.0; 4];
        layer_norm(&input, &gamma, &beta, &mut output, 4, 1e-5);
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() < 1e-4, "LN should zero-center: sum={sum}");
    }
}
