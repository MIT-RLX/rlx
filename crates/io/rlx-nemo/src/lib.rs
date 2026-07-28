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
//! The same `torch.save` machinery also drives [`PtModel`], a standalone
//! loader for plain PyTorch `.pt` / `.pth` / `pytorch_model.bin`
//! checkpoints (the checkpoint ZIP without the `.nemo` tar + YAML wrapper).
//!
//! ```no_run
//! use rlx_nemo::NemoModel;
//! let m = rlx_nemo::NemoModel::open(std::path::Path::new("model.nemo"))?;
//! let d_model = m.config().get_usize("encoder.d_model");
//! let w = m.tensor("encoder.layers.0.norm_out.weight")?; // -> NemoTensor (f32)
//! # anyhow::Ok(())
//! ```

mod arch;
mod archive;
mod config;
mod dtype;
mod pickle;
mod pt;
mod storage;
mod torch;

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

pub use arch::{build_nemo_probe_graph, nemo_arch_summary};
pub use config::NemoConfig;
pub use dtype::DType;
pub use pickle::TensorMeta;
pub use pt::{PtModel, PtTensor};

use archive::{
    Seekable, ZipEntry, list_tar, list_zip, prepare_seekable, read_member, read_zip_entry,
};

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

/// Index a `torch.save` ZIP: unpickle its `data.pkl` into a flat tensor
/// table and map each storage key to its `data/<key>` ZIP entry. Shared by
/// the `.nemo` loader and the standalone [`PtModel`] — both wrap the exact
/// same container, the latter without the outer tar + YAML.
pub(crate) fn index_torch_zip(
    file: &mut File,
    entries: &[ZipEntry],
) -> Result<(BTreeMap<String, TensorMeta>, HashMap<String, ZipEntry>)> {
    // Locate `<archive>/data.pkl` and the `<archive>/data/<key>` storages.
    let pkl_entry = entries
        .iter()
        .find(|e| e.name.ends_with("data.pkl"))
        .ok_or_else(|| anyhow!("no data.pkl in checkpoint zip"))?;
    let archive_prefix = pkl_entry
        .name
        .strip_suffix("data.pkl")
        .unwrap_or("")
        .to_string();
    let data_prefix = format!("{archive_prefix}data/");

    let mut storages = HashMap::new();
    for e in entries {
        if let Some(key) = e.name.strip_prefix(&data_prefix) {
            if !key.is_empty() {
                storages.insert(key.to_string(), e.clone());
            }
        }
    }

    let pkl_bytes = read_zip_entry(file, pkl_entry)?;
    let root = pickle::unpickle(&pkl_bytes).context("unpickling data.pkl")?;
    let tensors = torch::collect_state_dict(&root)?;
    Ok((tensors, storages))
}

/// Materialize one tensor's storage view as a contiguous, row-major `f32`
/// vector, given the container path and the storage-key → ZIP-entry map.
pub(crate) fn read_torch_tensor(
    path: &Path,
    meta: &TensorMeta,
    storages: &HashMap<String, ZipEntry>,
) -> Result<Vec<f32>> {
    let entry = storages
        .get(&meta.storage_key)
        .ok_or_else(|| anyhow!("missing storage {:?}", meta.storage_key))?;

    let expected = meta.dtype.size() as u64;
    if entry.size % expected != 0 {
        bail!(
            "storage {:?}: {} bytes not a multiple of dtype width {}",
            meta.storage_key,
            entry.size,
            expected
        );
    }

    let mut file = File::open(path)?;
    let raw = read_zip_entry(&mut file, entry)?;
    let storage_f32 = meta.dtype.decode_f32(&raw);
    storage::gather(meta, &storage_f32)
}
