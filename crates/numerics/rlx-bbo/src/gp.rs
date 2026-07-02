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

//! Gaussian-process posterior with RBF and Matérn kernels.
//!
//! Self-contained: implements its own dense symmetric-PSD Cholesky for
//! the (typically ≤200×200) kernel matrix that BO sees. For larger
//! covariance matrices, swap in [`rlx-linalg::cholesky`].

#[derive(Clone, Debug)]
pub enum Kernel {
    /// Squared-exponential / RBF: `exp(−0.5·r²/ℓ²)`, `ν → ∞`.
    Rbf { length_scale: f64, variance: f64 },
    /// Matérn ν = 5/2: `(1 + √5·r/ℓ + 5·r²/(3·ℓ²))·exp(−√5·r/ℓ)`.
    Matern52 { length_scale: f64, variance: f64 },
    /// Matérn ν = 3/2: `(1 + √3·r/ℓ)·exp(−√3·r/ℓ)`.
    Matern32 { length_scale: f64, variance: f64 },
}

impl Kernel {
    pub fn variance(&self) -> f64 {
        match *self {
            Kernel::Rbf { variance, .. }
            | Kernel::Matern52 { variance, .. }
            | Kernel::Matern32 { variance, .. } => variance,
        }
    }

    pub fn k(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut sq = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = x - y;
            sq += d * d;
        }
        let r = sq.sqrt();
        match *self {
            Kernel::Rbf {
                length_scale,
                variance,
            } => variance * (-0.5 * sq / (length_scale * length_scale)).exp(),
            Kernel::Matern52 {
                length_scale,
                variance,
            } => {
                let s = 5f64.sqrt() * r / length_scale;
                variance * (1.0 + s + s * s / 3.0) * (-s).exp()
            }
            Kernel::Matern32 {
                length_scale,
                variance,
            } => {
                let s = 3f64.sqrt() * r / length_scale;
                variance * (1.0 + s) * (-s).exp()
            }
        }
    }
}

/// Fitted GP posterior. Holds the training points + the Cholesky factor
/// and α = L⁻ᵀ(L⁻¹y) for fast posterior queries.
#[derive(Clone, Debug)]
pub struct GpPosterior {
    pub x_train: Vec<Vec<f64>>,
    pub y_train: Vec<f64>,
    pub kernel: Kernel,
    pub noise: f64,
    l: Vec<Vec<f64>>, // lower-triangular Cholesky factor of K + σ²·I
    alpha: Vec<f64>,
}

impl GpPosterior {
    /// Fit a posterior. Returns `None` if the kernel matrix is not
    /// positive-definite (numerical breakdown).
    pub fn fit(
        x_train: Vec<Vec<f64>>,
        y_train: Vec<f64>,
        kernel: Kernel,
        noise: f64,
    ) -> Option<Self> {
        let n = x_train.len();
        assert_eq!(y_train.len(), n);
        let mut k_mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let mut v = kernel.k(&x_train[i], &x_train[j]);
                if i == j {
                    v += noise * noise;
                }
                k_mat[i][j] = v;
                k_mat[j][i] = v;
            }
        }
        let l = cholesky(&k_mat)?;
        // Solve L·z = y, then Lᵀ·α = z.
        let z = forward_substitute(&l, &y_train);
        let alpha = back_substitute_transpose(&l, &z);
        Some(Self {
            x_train,
            y_train,
            kernel,
            noise,
            l,
            alpha,
        })
    }

    /// Posterior mean at `x_star`.
    pub fn mean(&self, x_star: &[f64]) -> f64 {
        let k_star: Vec<f64> = self
            .x_train
            .iter()
            .map(|x| self.kernel.k(x, x_star))
            .collect();
        dot(&k_star, &self.alpha)
    }

    /// Posterior variance at `x_star`. Always ≥ 0 (clamped to defend
    /// against tiny negative roundoff).
    pub fn variance(&self, x_star: &[f64]) -> f64 {
        let k_star: Vec<f64> = self
            .x_train
            .iter()
            .map(|x| self.kernel.k(x, x_star))
            .collect();
        let v = forward_substitute(&self.l, &k_star);
        let prior = self.kernel.k(x_star, x_star);
        (prior - dot(&v, &v)).max(0.0)
    }

    /// `(mean, std-dev)` convenience.
    pub fn mean_std(&self, x_star: &[f64]) -> (f64, f64) {
        let m = self.mean(x_star);
        let s = self.variance(x_star).sqrt();
        (m, s)
    }
}

/// Lower-triangular Cholesky `L L^T = A` with no pivoting. Returns
/// `None` if the matrix is not PSD.
pub fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L z = b` for lower-triangular `L`.
pub fn forward_substitute(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * z[j];
        }
        z[i] = s / l[i][i];
    }
    z
}

/// Solve `L^T x = z` for lower-triangular `L`.
pub fn back_substitute_transpose(l: &[Vec<f64>], z: &[f64]) -> Vec<f64> {
    let n = z.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = z[i];
        for j in (i + 1)..n {
            s -= l[j][i] * x[j];
        }
        x[i] = s / l[i][i];
    }
    x
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbf_kernel_diagonal_equals_variance() {
        let k = Kernel::Rbf {
            length_scale: 1.5,
            variance: 2.0,
        };
        let x = vec![0.1, -0.3, 0.7];
        assert!((k.k(&x, &x) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn cholesky_reconstructs_psd_matrix() {
        let a = vec![
            vec![4.0, 12.0, -16.0],
            vec![12.0, 37.0, -43.0],
            vec![-16.0, -43.0, 98.0],
        ];
        let l = cholesky(&a).unwrap();
        // Reconstruct LLᵀ and compare.
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += l[i][k] * l[j][k];
                }
                assert!((s - a[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn gp_interpolates_training_points() {
        // With zero noise the posterior mean at a training point equals y.
        let x_train = vec![vec![0.0], vec![1.0], vec![2.5]];
        let y_train = vec![0.0, 1.0, -0.5];
        let kernel = Kernel::Rbf {
            length_scale: 1.0,
            variance: 1.0,
        };
        let gp = GpPosterior::fit(x_train.clone(), y_train.clone(), kernel, 1e-6).unwrap();
        for (xi, yi) in x_train.iter().zip(y_train.iter()) {
            let m = gp.mean(xi);
            assert!(
                (m - yi).abs() < 1e-3,
                "GP interpolation: got {m}, want {yi}"
            );
        }
    }

    #[test]
    fn gp_variance_zero_at_noiseless_training_points() {
        let x_train = vec![vec![0.0], vec![1.0]];
        let y_train = vec![0.5, -0.5];
        let kernel = Kernel::Rbf {
            length_scale: 0.5,
            variance: 1.0,
        };
        let gp = GpPosterior::fit(x_train.clone(), y_train, kernel, 1e-8).unwrap();
        for x in &x_train {
            let v = gp.variance(x);
            assert!(v < 1e-5, "expected ~0 variance at training point, got {v}");
        }
    }
}
