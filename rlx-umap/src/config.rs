// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! UMAP configuration (compatible with fast-umap / umap-rs style).

use std::fmt;

#[cfg(any(feature = "serde", feature = "full"))]
use serde::{Deserialize, Serialize};

/// Fit UMAP kernel parameters `a`, `b` from `min_dist` and `spread`.
///
/// Kernel: `phi(d) = 1 / (1 + a * d^(2b))`.
pub fn fit_ab(min_dist: f32, spread: f32) -> (f32, f32) {
    let n = 300;
    let x_max = 3.0 * spread;
    let xs: Vec<f32> = (0..n)
        .map(|i| (i as f32 + 0.5) / n as f32 * x_max)
        .collect();
    let ys: Vec<f32> = xs
        .iter()
        .map(|&x| {
            if x <= min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            }
        })
        .collect();

    let residual = |a: f32, b: f32| -> f32 {
        xs.iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                let pred = 1.0 / (1.0 + a * x.powf(2.0 * b));
                (pred - y) * (pred - y)
            })
            .sum()
    };

    let mut best_a = 1.0f32;
    let mut best_b = 1.0f32;
    let mut best_err = f32::INFINITY;

    for ai in 1..=80 {
        let a = ai as f32 * 0.08;
        for bi in 1..=50 {
            let b = bi as f32 * 0.06;
            let err = residual(a, b);
            if err < best_err {
                best_err = err;
                best_a = a;
                best_b = b;
            }
        }
    }

    for _ in 0..100 {
        let step_a = best_a * 0.02;
        let step_b = best_b * 0.02;
        for &da in &[-step_a, 0.0, step_a] {
            for &db in &[-step_b, 0.0, step_b] {
                let a = (best_a + da).max(1e-4);
                let b = (best_b + db).max(1e-4);
                let err = residual(a, b);
                if err < best_err {
                    best_err = err;
                    best_a = a;
                    best_b = b;
                }
            }
        }
    }

    (best_a, best_b)
}

#[cfg_attr(
    any(feature = "serde", feature = "full"),
    derive(Serialize, Deserialize)
)]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Metric {
    #[default]
    Euclidean,
    Cosine,
    Manhattan,
    Minkowski,
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Euclidean => write!(f, "Euclidean"),
            Self::Cosine => write!(f, "cosine"),
            Self::Manhattan => write!(f, "Manhattan"),
            Self::Minkowski => write!(f, "minkowski"),
        }
    }
}

#[cfg_attr(
    any(feature = "serde", feature = "full"),
    derive(Serialize, Deserialize)
)]
#[derive(Debug, Clone)]
pub enum LossReduction {
    Mean,
    Sum,
}

#[cfg_attr(
    any(feature = "serde", feature = "full"),
    derive(Serialize, Deserialize)
)]
#[derive(Debug, Clone)]
pub struct ManifoldParams {
    pub min_dist: f32,
    pub spread: f32,
}

impl Default for ManifoldParams {
    fn default() -> Self {
        Self {
            min_dist: 0.1,
            spread: 1.0,
        }
    }
}

#[cfg_attr(
    any(feature = "serde", feature = "full"),
    derive(Serialize, Deserialize)
)]
#[derive(Debug, Clone)]
pub struct GraphParams {
    pub n_neighbors: usize,
    pub metric: Metric,
    pub normalized: bool,
    pub minkowski_p: f64,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            metric: Metric::Euclidean,
            normalized: false,
            minkowski_p: 2.0,
        }
    }
}

#[cfg_attr(
    any(feature = "serde", feature = "full"),
    derive(Serialize, Deserialize)
)]
#[derive(Debug, Clone)]
pub struct OptimizationParams {
    pub n_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub penalty: f32,
    pub repulsion_strength: f32,
    pub patience: Option<i32>,
    pub loss_reduction: LossReduction,
    pub min_desired_loss: Option<f64>,
    pub timeout: Option<u64>,
    pub verbose: bool,
    pub neg_sample_rate: usize,
    pub cooldown_ms: u64,
    /// PCA warm-start before UMAP loss (requires `pca` feature at compile time).
    #[cfg_attr(
        any(feature = "serde", feature = "full"),
        serde(default = "default_pca_warmstart")
    )]
    pub pca_warmstart: bool,
}

#[cfg(any(feature = "serde", feature = "full"))]
fn default_pca_warmstart() -> bool {
    true
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            n_epochs: 100,
            batch_size: 1000,
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            penalty: 1e-5,
            repulsion_strength: 1.0,
            patience: None,
            loss_reduction: LossReduction::Mean,
            min_desired_loss: None,
            timeout: None,
            verbose: false,
            neg_sample_rate: 5,
            cooldown_ms: 0,
            pca_warmstart: true,
        }
    }
}

#[cfg_attr(
    any(feature = "serde", feature = "full"),
    derive(Serialize, Deserialize)
)]
#[derive(Debug, Clone)]
pub struct UmapConfig {
    pub n_components: usize,
    pub hidden_sizes: Vec<usize>,
    pub manifold: ManifoldParams,
    pub graph: GraphParams,
    pub optimization: OptimizationParams,
}

impl Default for UmapConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            hidden_sizes: vec![100],
            manifold: ManifoldParams::default(),
            graph: GraphParams::default(),
            optimization: OptimizationParams::default(),
        }
    }
}

/// Internal training parameters (derived from [`UmapConfig`]).
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub penalty: f32,
    pub verbose: bool,
    pub patience: Option<i32>,
    pub k_neighbors: usize,
    pub min_desired_loss: Option<f64>,
    pub timeout: Option<u64>,
    pub repulsion_strength: f32,
    pub kernel_a: f32,
    pub kernel_b: f32,
    pub neg_sample_rate: usize,
    pub cooldown_ms: u64,
    pub pca_warmstart: bool,
}

impl TrainingConfig {
    pub fn from_umap_config(config: &UmapConfig) -> Self {
        let (kernel_a, kernel_b) = fit_ab(config.manifold.min_dist, config.manifold.spread);
        Self {
            epochs: config.optimization.n_epochs,
            learning_rate: config.optimization.learning_rate,
            beta1: config.optimization.beta1,
            beta2: config.optimization.beta2,
            penalty: config.optimization.penalty,
            verbose: config.optimization.verbose,
            patience: config.optimization.patience,
            k_neighbors: config.graph.n_neighbors,
            min_desired_loss: config.optimization.min_desired_loss,
            timeout: config.optimization.timeout,
            repulsion_strength: config.optimization.repulsion_strength,
            kernel_a,
            kernel_b,
            neg_sample_rate: config.optimization.neg_sample_rate,
            cooldown_ms: config.optimization.cooldown_ms,
            pca_warmstart: config.optimization.pca_warmstart,
        }
    }
}
