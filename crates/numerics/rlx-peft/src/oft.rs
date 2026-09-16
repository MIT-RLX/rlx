// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! OFT — orthogonal fine-tuning.
//!
//! Instead of adding to the weights, OFT *rotates* them: `W' = R W` with `R`
//! block-diagonal and orthogonal. Because an orthogonal map preserves norms and
//! pairwise angles, the layer's internal geometry — the relationships the
//! pretrained model learned — survives adaptation exactly. That is the property
//! OFT is chosen for, and it is what these tests assert.
//!
//! `R` is parameterised through the **Cayley transform** of a skew-symmetric
//! `Q`:
//!
//! ```text
//! R = (I − Q)(I + Q)⁻¹,   Qᵀ = −Q
//! ```
//!
//! which is orthogonal for any `Q`, so training is unconstrained — no
//! re-orthogonalisation step, no drift off the manifold. `Q = 0` gives `R = I`,
//! so an untrained OFT layer is the identity.
//!
//! Blocks are small (PEFT's default `block_size` is 8), so the inverse is a
//! local Gauss–Jordan rather than a LAPACK call: dispatch would cost more than
//! the solve.

use crate::matmul;

/// Skew-symmetrise: `Q ← (Q − Qᵀ)/2`.
///
/// PEFT does this rather than trusting the parameter to stay skew: floating
/// point accumulates a symmetric component, and any of it breaks orthogonality
/// silently — `R` stays close to orthogonal, so nothing errors, and the
/// guarantee the method exists for quietly stops holding.
pub fn skew(q: &[f64], n: usize) -> Result<Vec<f64>, String> {
    if !(q.len() == n * n) {
        return Err(format!("Q is {} values, expected {n}·{n}", q.len()));
    }
    let mut out = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = 0.5 * (q[i * n + j] - q[j * n + i]);
        }
    }
    Ok(out)
}

/// `R = (I − Q)(I + Q)⁻¹` for skew-symmetric `Q`.
pub fn cayley_orthogonal(q: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let s = skew(q, n)?;
    let mut minus = vec![0f64; n * n];
    let mut plus = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let v = s[i * n + j];
            let id = f64::from(i == j);
            minus[i * n + j] = id - v;
            plus[i * n + j] = id + v;
        }
    }
    // (I + Q) is always invertible for skew Q: its eigenvalues are 1 ± i·λ,
    // never zero. A singular result here means Q was not skew-symmetrised.
    let inv = invert(&plus, n)
        .ok_or_else(|| "(I + Q) is singular — Q is not skew-symmetric".to_string())?;
    matmul(&minus, &inv, n, n, n)
}

/// `W' = R W` with `R` block-diagonal: each block of `block_size` output rows
/// is rotated by its own orthogonal matrix.
///
/// The block structure is what keeps OFT cheap — a full `out × out` rotation
/// would cost more parameters than the weight itself.
pub fn oft_weight(
    w: &[f64],
    blocks: &[Vec<f64>],
    in_features: usize,
    out_features: usize,
    block_size: usize,
) -> Result<Vec<f64>, String> {
    if !(w.len() == out_features * in_features) {
        return Err(format!(
            "W is {} values, expected {out_features}·{in_features}",
            w.len()
        ));
    }
    if !(block_size > 0) {
        return Err(format!("block_size must be positive, got {block_size}"));
    }
    let n_blocks = out_features.div_ceil(block_size);
    if blocks.len() != n_blocks {
        return Err(format!(
            "{} blocks for {out_features} outputs at block_size {block_size}; expected {n_blocks}",
            blocks.len()
        ));
    }

    let mut out = vec![0f64; w.len()];
    for (bi, q) in blocks.iter().enumerate() {
        let start = bi * block_size;
        let size = block_size.min(out_features - start);
        if !(q.len() == size * size) {
            return Err(format!(
                "block {bi} is {} values, expected {size}·{size}",
                q.len()
            ));
        }
        let r = cayley_orthogonal(q, size)?;
        // R · W[block rows]
        for i in 0..size {
            for j in 0..in_features {
                out[(start + i) * in_features + j] = (0..size)
                    .map(|t| r[i * size + t] * w[(start + t) * in_features + j])
                    .sum();
            }
        }
    }
    Ok(out)
}

/// Gauss–Jordan inverse with partial pivoting, for the small Cayley blocks.
fn invert(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = vec![0f64; n * 2 * n];
    for i in 0..n {
        m[i * 2 * n..i * 2 * n + n].copy_from_slice(&a[i * n..(i + 1) * n]);
        m[i * 2 * n + n + i] = 1.0;
    }
    let w = 2 * n;
    let scale = a.iter().fold(0f64, |acc, v| acc.max(v.abs())).max(1.0);
    for col in 0..n {
        let (mut piv, mut best) = (col, m[col * w + col].abs());
        for r in (col + 1)..n {
            if m[r * w + col].abs() > best {
                piv = r;
                best = m[r * w + col].abs();
            }
        }
        if best < 1e-12 * scale {
            return None;
        }
        if piv != col {
            for c in 0..w {
                m.swap(col * w + c, piv * w + c);
            }
        }
        let d = m[col * w + col];
        for c in col..w {
            m[col * w + c] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * w + col];
            if f != 0.0 {
                for c in col..w {
                    m[r * w + c] -= f * m[col * w + c];
                }
            }
        }
    }
    Some(
        (0..n)
            .flat_map(|i| m[i * w + n..(i + 1) * w].to_vec())
            .collect(),
    )
}
