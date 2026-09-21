// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native loader for NVIDIA **NeMo** `.nemo` model files.
//!
//! A `.nemo` is an (uncompressed) TAR archive containing:
//!   * `model_config.yaml` — hyperparameters ([`NemoConfig`]),
//!   * `model_weights.ckpt` — a `torch.save` ZIP of the state dict, and
//!   * optional tokenizer artifacts (SentencePiece `*.model`, `vocab.txt`).
//!
//! [`NemoModel::open`] indexes the archive and the embedded checkpoint
//! without decompressing or copying the multi-gigabyte weight blob;
//! [`NemoModel::tensor`] then pulls individual tensors on demand as
//! contiguous `f32`, regardless of their on-disk dtype (fp32 / fp16 /
//! bf16 / int).
//!
//! The `torch.save` half — ZIP index, pickle, dtype decode — lives in
//! `rlx-torch-ckpt`, which this crate layers the tar + YAML wrapper on top
//! of. Its [`PtModel`] (re-exported here) loads a plain PyTorch
//! `.pt` / `.pth` / `pytorch_model.bin`: the same checkpoint ZIP without
//! the `.nemo` wrapper.
//!
//! ```no_run
//! use rlx_nemo::NemoModel;
//! let m = rlx_nemo::NemoModel::open(std::path::Path::new("model.nemo"))?;
//! let d_model = m.config().get_usize("encoder.d_model");
//! let w = m.tensor("encoder.layers.0.norm_out.weight")?; // -> NemoTensor (f32)
//! # anyhow::Ok(())
//! ```

mod arch;
mod config;

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow};

pub use arch::{
    EncoderOpts, TensorShapes, build_nemo_encoder_graph, build_nemo_probe_graph, nemo_arch_summary,
};
pub use config::NemoConfig;
// Re-exported so `.nemo` consumers need not also depend on `rlx-torch-ckpt`.
pub use rlx_torch_ckpt::{DType, PtModel, PtTensor, TensorMeta};

use rlx_torch_ckpt::archive::{
    Seekable, ZipEntry, list_tar, list_zip, prepare_seekable, read_member,
};
use rlx_torch_ckpt::{index_torch_zip, read_torch_tensor};

/// A tensor materialized from a `.nemo` checkpoint as contiguous f32.
#[derive(Debug, Clone)]
pub struct NemoTensor {
    pub name: String,
    pub shape: Vec<usize>,
    /// The on-disk dtype before conversion to f32 (for reporting/quant).
    pub dtype: DType,
    pub data: Vec<f32>,
}

/// A tokenizer file extracted from the archive (e.g. SentencePiece model).
#[derive(Debug, Clone)]
pub struct TokenizerArtifact {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// An opened `.nemo` model: config + a lazily-readable tensor index.
pub struct NemoModel {
    /// The seekable archive (a temp decompressed copy for gzip `.nemo`s,
    /// kept alive so tensor reads stay valid).
    source: Seekable,
    config: NemoConfig,
    /// Param name → tensor view metadata (from the pickle).
    tensors: BTreeMap<String, TensorMeta>,
    /// Storage key → its (absolute-offset) zip entry inside the .nemo.
    storages: HashMap<String, ZipEntry>,
    /// Tokenizer artifacts found alongside the weights.
    tokenizers: Vec<TokenizerArtifact>,
}

impl NemoModel {
    /// Open and index a `.nemo` file.
    pub fn open(path: &Path) -> Result<Self> {
        let source =
            prepare_seekable(path).with_context(|| format!("preparing {}", path.display()))?;
        let read_path = source.path().to_path_buf();
        let mut file =
            File::open(&read_path).with_context(|| format!("opening {}", read_path.display()))?;

        let members = list_tar(&mut file).context("reading .nemo tar")?;
        let find = |needle: &str| members.iter().find(|m| m.name.ends_with(needle));

        let cfg_member = find("model_config.yaml")
            .or_else(|| find(".yaml"))
            .ok_or_else(|| anyhow!("no model_config.yaml in {}", path.display()))?;
        let cfg_bytes = read_member(&mut file, cfg_member)?;
        let config = NemoConfig::from_yaml_bytes(&cfg_bytes)?;

        let ckpt_member = find("model_weights.ckpt")
            .or_else(|| find(".ckpt"))
            .ok_or_else(|| anyhow!("no model_weights.ckpt in {}", path.display()))?
            .clone();

        // Tokenizer artifacts (best-effort; many ASR models bundle SPM).
        let mut tokenizers = Vec::new();
        for m in &members {
            let lower = m.name.to_ascii_lowercase();
            let is_tok = lower.ends_with(".model")
                || lower.ends_with("vocab.txt")
                || lower.ends_with("tokenizer.json")
                || lower.contains("tokenizer");
            // Skip the weights/config we already handle.
            if is_tok && !lower.ends_with(".ckpt") && !lower.ends_with(".yaml") {
                let bytes = read_member(&mut file, m)?;
                tokenizers.push(TokenizerArtifact {
                    name: m.name.clone(),
                    bytes,
                });
            }
        }

        // Parse the embedded torch.save zip.
        let entries = list_zip(&mut file, ckpt_member.offset, ckpt_member.size)
            .context("reading model_weights.ckpt zip")?;
        let (tensors, storages) =
            index_torch_zip(&mut file, &entries).context("indexing model_weights.ckpt")?;

        Ok(Self {
            source,
            config,
            tensors,
            storages,
            tokenizers,
        })
    }

    /// The parsed `model_config.yaml`.
    pub fn config(&self) -> &NemoConfig {
        &self.config
    }

    /// All tensor names, sorted.
    pub fn names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Number of tensors in the checkpoint.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Tokenizer artifacts bundled in the archive (may be empty).
    pub fn tokenizers(&self) -> &[TokenizerArtifact] {
        &self.tokenizers
    }

    /// Shape of a tensor without reading its data.
    pub fn shape_of(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    /// Read one tensor and materialize it as contiguous f32.
    pub fn tensor(&self, name: &str) -> Result<NemoTensor> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("no tensor named {name:?}"))?;
        let data = read_torch_tensor(self.source.path(), meta, &self.storages)?;
        Ok(NemoTensor {
            name: name.to_string(),
            shape: meta.shape.clone(),
            dtype: meta.dtype,
            data,
        })
    }
}
