// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! PCA for warm-starting parametric UMAP embeddings (host-only, no Burn).

/// PCA projection of row-major `data` `[n, d]` → `(projected, components, mean)`.
///
/// - `projected`: `[n, n_components]`
/// - `components`: `[n_components, d]` row vectors
/// - `mean`: `[d]` column means used for centering
pub fn pca(
    data: &[f32],
    n_samples: usize,
    n_features: usize,
    n_components: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut mean = vec![0.0f64; n_features];
    for i in 0..n_samples {
        for j in 0..n_features {
            mean[j] += data[i * n_features + j] as f64;
        }
    }
    for j in 0..n_features {
        mean[j] /= n_samples as f64;
    }

    let mut centered = vec![0.0f32; n_samples * n_features];
    for i in 0..n_samples {
        for j in 0..n_features {
            centered[i * n_features + j] = data[i * n_features + j] - mean[j] as f32;
        }
    }

    let mut cov_data = vec![0.0f32; n_features * n_features];
    let denom = (n_samples.saturating_sub(1)).max(1) as f32;
    for i in 0..n_features {
        for j in 0..n_features {
            let mut s = 0.0f32;
            for row in 0..n_samples {
                s += centered[row * n_features + i] * centered[row * n_features + j];
            }
            cov_data[i * n_features + j] = s / denom;
        }
    }

    let mut components = Vec::with_capacity(n_components * n_features);
    for _comp in 0..n_components {
        let mut v = vec![0.0f32; n_features];
        for (j, vi) in v.iter_mut().enumerate() {
            *vi = if j % 2 == 0 { 1.0 } else { -1.0 };
        }
        normalize_vec(&mut v);

        for _iter in 0..100 {
            let mut w = vec![0.0f32; n_features];
            for i in 0..n_features {
                let mut s = 0.0f32;
                for j in 0..n_features {
                    s += cov_data[i * n_features + j] * v[j];
                }
                w[i] = s;
            }
            normalize_vec(&mut w);
            let dot: f32 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            v = w;
            if dot.abs() > 1.0 - 1e-8 {
                break;
            }
        }

        components.extend_from_slice(&v);

        let mut eigenvalue = 0.0f32;
        for i in 0..n_features {
            let mut s = 0.0f32;
            for j in 0..n_features {
                s += cov_data[i * n_features + j] * v[j];
            }
            eigenvalue += v[i] * s;
        }
        for i in 0..n_features {
            for j in 0..n_features {
                cov_data[i * n_features + j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    let mut projected = vec![0.0f32; n_samples * n_components];
    for i in 0..n_samples {
        for c in 0..n_components {
            let mut s = 0.0f32;
            for j in 0..n_features {
                s += centered[i * n_features + j] * components[c * n_features + j];
            }
            projected[i * n_components + c] = s;
        }
    }

    let mean_f32: Vec<f32> = mean.iter().map(|&x| x as f32).collect();
    (projected, components, mean_f32)
}

fn normalize_vec(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
