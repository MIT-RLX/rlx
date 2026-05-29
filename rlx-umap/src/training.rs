// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Parametric UMAP **training** on RLX Session + autodiff (no Burn).
//!
//! | Stage | Engine |
//! |-------|--------|
//! | MLP forward | RLX `CompiledGraph` |
//! | UMAP sparse loss | RLX graph |
//! | Backward | [`rlx_autodiff::grad_with_loss`] → compiled graph |
//! | Optimizer | Host Adam (gradients from autodiff only) |
//! | Global k-NN | Precomputed once on the training device (fused pairwise + `umap.knn`) |
//!
//! Use [`fit`] or [`Umap::fit`](crate::umap::Umap::fit) for the full pipeline including
//! embedding extraction and [`FittedUmap`](crate::fitted::FittedUmap).

use crossbeam_channel::Receiver;
use rlx_driver::Device;

use crate::config::{TrainingConfig, UmapConfig};
use crate::encoder::mlp::ModelSpec;
use crate::fitted::FittedUmap;
use crate::interrupt;
pub use crate::train::{EpochProgress, MAX_POS_EDGES_PER_EPOCH, TrainResult, train_sparse};
use crate::utils::{NormStats, f32_to_f64, f64_to_f32, flatten_f64, unflatten_f64};

/// Options for [`fit`].
#[derive(Debug, Clone)]
pub struct FitOptions {
    pub device: Device,
    pub exit_rx: Option<Receiver<()>>,
}

impl FitOptions {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            exit_rx: None,
        }
    }

    pub fn with_ctrlc(mut self) -> Self {
        self.exit_rx = Some(interrupt::install_ctrlc_handler());
        self
    }
}

/// Train parametric UMAP and return a fitted model (RLX autodiff training loop).
pub fn fit(config: UmapConfig, data: Vec<Vec<f64>>, options: FitOptions) -> FittedUmap {
    crate::register();
    let (mut flat, n, d) = flatten_f64(&data);
    assert!(n >= 2, "need at least 2 samples");
    assert!(d > 0, "need at least 1 feature");

    let norm_stats = NormStats::compute(&flat, n, d);
    let spec = ModelSpec::from_config(&config, n, d);
    let train_cfg = TrainingConfig::from_umap_config(&config);
    let metric = config.graph.metric.clone();

    let result = train_sparse(
        options.device,
        &mut flat,
        n,
        d,
        &spec,
        &train_cfg,
        &metric,
        options.exit_rx,
        None,
    );

    let x = f64_to_f32(&flat);
    let mut compiled = result.compiled;
    let emb_f32 = compiled.forward_embedding(&x);
    let embedding = unflatten_f64(&f32_to_f64(&emb_f32), n, config.n_components);

    FittedUmap::new(
        config,
        result.weights,
        embedding,
        d,
        n,
        norm_stats,
        compiled,
        result.n_pos,
        result.n_neg,
    )
}

/// Train with progress callbacks (loss readback epochs only).
pub fn fit_with_progress(
    config: UmapConfig,
    data: Vec<Vec<f64>>,
    options: FitOptions,
    on_progress: impl Fn(EpochProgress) + Send + 'static,
) -> FittedUmap {
    crate::register();
    let (mut flat, n, d) = flatten_f64(&data);
    assert!(n >= 2, "need at least 2 samples");
    assert!(d > 0, "need at least 1 feature");

    let norm_stats = NormStats::compute(&flat, n, d);
    let spec = ModelSpec::from_config(&config, n, d);
    let train_cfg = TrainingConfig::from_umap_config(&config);
    let metric = config.graph.metric.clone();

    let result = train_sparse(
        options.device,
        &mut flat,
        n,
        d,
        &spec,
        &train_cfg,
        &metric,
        options.exit_rx,
        Some(&|p| on_progress(p)),
    );

    let x = f64_to_f32(&flat);
    let mut compiled = result.compiled;
    let emb_f32 = compiled.forward_embedding(&x);
    let embedding = unflatten_f64(&f32_to_f64(&emb_f32), n, config.n_components);

    FittedUmap::new(
        config,
        result.weights,
        embedding,
        d,
        n,
        norm_stats,
        compiled,
        result.n_pos,
        result.n_neg,
    )
}

/// Run training only (no `FittedUmap` wrapper). Returns compiled graphs + weights.
pub fn train_only(
    config: &UmapConfig,
    data: &mut [f64],
    n: usize,
    d: usize,
    device: Device,
    exit_rx: Option<Receiver<()>>,
) -> TrainResult {
    crate::register();
    let spec = ModelSpec::from_config(config, n, d);
    let train_cfg = TrainingConfig::from_umap_config(config);
    let metric = config.graph.metric.clone();
    train_sparse(
        device, data, n, d, &spec, &train_cfg, &metric, exit_rx, None,
    )
}
