// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Fitted parametric UMAP model.

use rlx_driver::Device;
use rlx_runtime::Session;

use crate::config::UmapConfig;
use crate::encoder::mlp::ModelSpec;
use crate::model::CompiledUmap;
use crate::serialize::{LoadedModel, SaveBundle};
use crate::utils::{NormStats, f64_to_f32, flatten_f64, unflatten_f64};
use crate::weights::WeightStore;

/// Trained parametric UMAP — embedding + inference graph.
pub struct FittedUmap {
    pub config: UmapConfig,
    pub weights: WeightStore,
    pub embedding: Vec<Vec<f64>>,
    pub num_features: usize,
    pub n_train: usize,
    norm_stats: NormStats,
    compiled: CompiledUmap,
    n_pos: usize,
    n_neg: usize,
}

impl FittedUmap {
    pub(crate) fn new(
        config: UmapConfig,
        weights: WeightStore,
        embedding: Vec<Vec<f64>>,
        num_features: usize,
        n_train: usize,
        norm_stats: NormStats,
        compiled: CompiledUmap,
        n_pos: usize,
        n_neg: usize,
    ) -> Self {
        Self {
            config,
            weights,
            embedding,
            num_features,
            n_train,
            norm_stats,
            compiled,
            n_pos,
            n_neg,
        }
    }

    fn from_loaded(loaded: LoadedModel, device: Device) -> std::io::Result<Self> {
        let config = loaded.config.clone().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "model file has no embedded config (v3 or older) — use load_with_config",
            )
        })?;
        Self::from_loaded_with_config(loaded, config, device)
    }

    fn from_loaded_with_config(
        loaded: LoadedModel,
        config: UmapConfig,
        device: Device,
    ) -> std::io::Result<Self> {
        let m = loaded.meta;
        let spec = ModelSpec::from_config(&config, m.n_train, m.n_features);
        let session = Session::new(device);
        let mut compiled = CompiledUmap::compile(&session, &spec, m.n_pos, m.n_neg);
        compiled.set_weights(&loaded.weights);
        Ok(Self {
            config,
            weights: loaded.weights,
            embedding: Vec::new(),
            num_features: m.n_features,
            n_train: m.n_train,
            norm_stats: loaded.norm,
            compiled,
            n_pos: m.n_pos,
            n_neg: m.n_neg,
        })
    }

    pub fn n_train(&self) -> usize {
        self.n_train
    }

    pub fn norm_stats(&self) -> &NormStats {
        &self.norm_stats
    }

    pub fn weights(&self) -> &WeightStore {
        &self.weights
    }

    pub fn config(&self) -> &UmapConfig {
        &self.config
    }

    pub fn embedding(&self) -> &[Vec<f64>] {
        &self.embedding
    }

    pub fn into_embedding(self) -> Vec<Vec<f64>> {
        self.embedding
    }

    /// Project new points using training normalization statistics.
    pub fn transform(&mut self, data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let (mut flat, n, d) = flatten_f64(&data);
        assert_eq!(d, self.num_features, "feature count mismatch");
        self.norm_stats.apply(&mut flat, n, d);
        let x = f64_to_f32(&flat);
        let emb = self.compiled.forward_embedding(&x);
        unflatten_f64(&crate::utils::f32_to_f64(&emb), n, self.config.n_components)
    }

    /// Save full model: weights, metadata, normalization stats, and config (v4).
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        crate::serialize::save_model(
            SaveBundle {
                weights: &self.weights,
                meta: crate::serialize::ModelMetadata {
                    n_train: self.n_train,
                    n_features: self.num_features,
                    n_pos: self.n_pos,
                    n_neg: self.n_neg,
                },
                norm: &self.norm_stats,
                config: &self.config,
            },
            path,
        )
    }

    /// Save encoder weights only (`.safetensors` / `.gguf`; shapes in metadata).
    pub fn save_weights(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let spec = ModelSpec::from_config(&self.config, self.n_train, self.num_features);
        self.weights.save(path, &spec)
    }

    /// Load a v4 model saved with [`save`](Self::save) (config embedded in file).
    pub fn load(path: impl AsRef<std::path::Path>, device: Device) -> std::io::Result<Self> {
        let loaded = crate::serialize::load_model(path)?;
        Self::from_loaded(loaded, device)
    }

    /// Load archive; use file config when present, otherwise `config`.
    pub fn load_with_config(
        path: impl AsRef<std::path::Path>,
        config: UmapConfig,
        device: Device,
    ) -> std::io::Result<Self> {
        let loaded = crate::serialize::load_model(path)?;
        let cfg = loaded.config.clone().unwrap_or(config);
        Self::from_loaded_with_config(loaded, cfg, device)
    }
}
