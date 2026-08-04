// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Small dependency-free **truncated SVD** — the shared numerical primitive
//! under every tensor decomposition here: Tucker (HOSVD unfolds → SVD each
//! mode), TT (TT-SVD sweeps SVD the sequential reshape), and Monarch (rank-1
//! SVD of each block). Power iteration with explicit deflation: robust and
//! plenty accurate for a mining tool that only needs the leading `r` triplets.

/// L2 norm.
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Truncated SVD `A[rows,cols] ≈ U·diag(S)·Vᵀ` keeping the top `r` singular
/// triplets. Returns `(U[rows*r], S[r], V[cols*r])`, all column-major-in-`r`
/// (element `[i,t] = buf[i*r + t]`). Deflates a working copy after each triplet.
pub fn truncated_svd(
    a: &[f32],
    rows: usize,
    cols: usize,
    r: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let r = r.min(rows).min(cols).max(1);
    let mut work = a.to_vec();
    let mut us = vec![0f32; rows * r];
    let mut ss = vec![0f32; r];
    let mut vs = vec![0f32; cols * r];
    // Deterministic xorshift init so results are reproducible across runs.
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s as i64 as f64 / i64::MAX as f64) as f32
    };
    for t in 0..r {
        let mut v: Vec<f32> = (0..cols).map(|_| rnd()).collect();
        let n0 = norm(&v).max(1e-20);
        for x in &mut v {
            *x /= n0;
        }
        let mut u = vec![0f32; rows];
        let mut sigma = 0f32;
        for _ in 0..60 {
            // u = A·v
            for i in 0..rows {
                let row = &work[i * cols..(i + 1) * cols];
                u[i] = row.iter().zip(&v).map(|(a, b)| a * b).sum();
            }
            let un = norm(&u);
            if un < 1e-20 {
                break;
            }
            for x in &mut u {
                *x /= un;
            }
            // v = Aᵀ·u
            let mut nv = vec![0f32; cols];
            for i in 0..rows {
                let ui = u[i];
                let row = &work[i * cols..(i + 1) * cols];
                for j in 0..cols {
                    nv[j] += row[j] * ui;
                }
            }
            sigma = norm(&nv);
            if sigma < 1e-20 {
                break;
            }
            for j in 0..cols {
                v[j] = nv[j] / sigma;
            }
        }
        // Over-requesting rank past the true rank leaves a ~0 residual, and power
        // iteration on it yields a *bogus* unit vector that isn't orthogonal to
        // the real ones — which would break callers that rely on orthonormal
        // columns (Tucker's projector, TT's bond basis). Drop such triplets to
        // zero: they span nothing, so the kept columns stay an orthonormal set.
        if t > 0 && sigma < 1e-6 * ss[0].max(1e-30) {
            ss[t] = 0.0;
            continue; // us/vs already zero here; deflation is a no-op at σ≈0
        }
        for i in 0..rows {
            us[i * r + t] = u[i];
        }
        for j in 0..cols {
            vs[j * r + t] = v[j];
        }
        ss[t] = sigma;
        // Deflate: work -= σ·u·vᵀ so the next sweep finds the next triplet.
        for i in 0..rows {
            let su = sigma * u[i];
            let row = &mut work[i * cols..(i + 1) * cols];
            for j in 0..cols {
                row[j] -= su * v[j];
            }
        }
    }
    (us, ss, vs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_reconstructs_rank2() {
        // A = u1 v1ᵀ + u2 v2ᵀ, rank 2. Top-2 SVD must reconstruct it exactly.
        let (rows, cols) = (12usize, 9usize);
        let mk = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 7 + seed * 3) % 11) as f32 - 5.0)
                .collect()
        };
        let (u1, v1) = (mk(1, rows), mk(2, cols));
        let (u2, v2) = (mk(3, rows), mk(4, cols));
        let mut a = vec![0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                a[i * cols + j] = u1[i] * v1[j] + u2[i] * v2[j];
            }
        }
        let (u, s, v) = truncated_svd(&a, rows, cols, 2);
        let mut recon = vec![0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let mut acc = 0f32;
                for t in 0..2 {
                    acc += u[i * 2 + t] * s[t] * v[j * 2 + t];
                }
                recon[i * cols + j] = acc;
            }
        }
        let num: f32 = a
            .iter()
            .zip(&recon)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();
        let den: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(num / den < 1e-4, "rank-2 SVD recon err {}", num / den);
    }
}
