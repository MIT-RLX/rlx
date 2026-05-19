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

//! High-level embedding API — one-liner model loading and inference.
//!
//! ```rust,ignore
//! use rlx_models::embed::RlxEmbed;
//!
//! let mut model = RlxEmbed::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")?;
//! let embeddings = model.embed(&["Hello world", "Test passage"])?;
//! // embeddings: Vec<Vec<f32>>, one per input text
//! ```

use crate::weight_map::WeightMap;
use anyhow::Result;
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Model architecture (auto-detected from config.json).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Arch {
    Bert,
    NomicBert,
    NomicVision,
}

/// Pooling strategy.
#[derive(Debug, Clone, Copy)]
pub enum Pooling {
    Mean,
    Cls,
}

/// High-level embedding model. One struct for all architectures.
pub struct RlxEmbed {
    compiled: CompiledGraph,
    arch: Arch,
    hidden_size: usize,
    // `pooling` is captured at construction so callers can branch on
    // it later; mean/cls is currently applied at the burnembed bench
    // layer, not here.
    #[allow(dead_code)]
    pooling: Pooling,
    compiled_bs: (usize, usize),
    config_path: PathBuf,
    weights_path: String,
}

impl RlxEmbed {
    /// Load a model from a local directory containing config.json + model.safetensors.
    pub fn from_dir(dir: &Path, pooling: Pooling) -> Result<Self> {
        let config_path = dir.join("config.json");
        let weights_path = dir.join("model.safetensors");
        let wt_str = weights_path.to_str().unwrap().to_string();

        let arch = detect_arch(&config_path)?;
        let (hidden_size, compiled, _) = compile_model(arch, &config_path, &wt_str, 1, 1)?;

        Ok(Self {
            compiled,
            arch,
            hidden_size,
            pooling,
            compiled_bs: (1, 1),
            config_path,
            weights_path: wt_str,
        })
    }

    /// Load a model by HuggingFace repo ID. Downloads if not cached.
    #[cfg(feature = "hf-download")]
    pub fn from_pretrained(repo_id: &str) -> Result<Self> {
        let repo = hf_hub::api::sync::ApiBuilder::new()
            .with_progress(true)
            .build()?
            .model(repo_id.to_string());
        let config_file = repo.get("config.json")?;
        let weights_file = repo.get("model.safetensors")?;

        let arch = detect_arch(&config_file)?;
        let pooling = default_pooling(repo_id);
        let wt_str = weights_file.to_str().unwrap().to_string();
        let (hidden_size, compiled, _) = compile_model(arch, &config_file, &wt_str, 1, 1)?;

        Ok(Self {
            compiled,
            arch,
            hidden_size,
            pooling,
            compiled_bs: (1, 1),
            config_path: config_file,
            weights_path: wt_str,
        })
    }

    /// Get the embedding dimension.
    pub fn dim(&self) -> usize {
        self.hidden_size
    }

    /// Get the detected architecture.
    pub fn arch(&self) -> Arch {
        self.arch
    }

    /// Run forward pass on pre-tokenized data.
    /// Returns raw hidden states [batch * seq * hidden].
    pub fn forward(
        &mut self,
        inputs: &[(&str, &[f32])],
        batch: usize,
        seq: usize,
    ) -> Result<Vec<f32>> {
        self.ensure_compiled(batch, seq)?;
        let outputs = self.compiled.run(inputs);
        Ok(outputs.into_iter().next().unwrap_or_default())
    }

    fn ensure_compiled(&mut self, batch: usize, seq: usize) -> Result<()> {
        if self.compiled_bs == (batch, seq) {
            return Ok(());
        }
        let (_, compiled, _) =
            compile_model(self.arch, &self.config_path, &self.weights_path, batch, seq)?;
        self.compiled = compiled;
        self.compiled_bs = (batch, seq);
        Ok(())
    }
}

/// Detect architecture from config.json fields.
fn detect_arch(config_path: &Path) -> Result<Arch> {
    let data = std::fs::read_to_string(config_path)?;
    let json: serde_json::Value = serde_json::from_str(&data)?;

    // NomicVision: has img_size + patch_size
    if json.get("img_size").is_some() && json.get("patch_size").is_some() {
        return Ok(Arch::NomicVision);
    }
    // NomicBert: has rotary_emb_base or rotary_emb_fraction
    if json.get("rotary_emb_base").is_some() || json.get("rotary_emb_fraction").is_some() {
        return Ok(Arch::NomicBert);
    }
    // Default: BERT
    Ok(Arch::Bert)
}

/// Default pooling based on repo name.
#[allow(dead_code)]
fn default_pooling(repo_id: &str) -> Pooling {
    let lower = repo_id.to_lowercase();
    if lower.contains("bge") || lower.contains("nomic") {
        Pooling::Cls
    } else {
        Pooling::Mean
    }
}

/// Compile a model for the given batch/seq.
fn compile_model(
    arch: Arch,
    config_path: &Path,
    weights_path: &str,
    batch: usize,
    seq: usize,
) -> Result<(usize, CompiledGraph, HashMap<String, Vec<f32>>)> {
    let mut wm = WeightMap::from_file(weights_path)?;

    let (graph, params, hidden_size) = match arch {
        Arch::Bert => {
            let cfg = crate::BertConfig::from_file(config_path)?;
            let hs = cfg.hidden_size;
            let (g, p) = crate::build_bert_graph_sized(&cfg, &mut wm, batch, seq)?;
            (g, p, hs)
        }
        Arch::NomicBert => {
            let cfg = crate::NomicBertConfig::from_file(config_path)?;
            let hs = cfg.hidden_size;
            let (g, p) = crate::build_nomic_graph_sized(&cfg, &mut wm, batch, seq)?;
            (g, p, hs)
        }
        Arch::NomicVision => {
            let cfg = crate::NomicVisionConfig::from_file(config_path)?;
            let hs = cfg.hidden_size;
            let (g, p, _preprocess) = crate::build_vision_graph_sized(&cfg, &mut wm, batch)?;
            (g, p, hs)
        }
    };

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(graph);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }

    Ok((hidden_size, compiled, params))
}
