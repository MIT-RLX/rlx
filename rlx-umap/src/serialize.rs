// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Weight and model I/O.
//!
//! **Default:** [safetensors](https://huggingface.co/docs/safetensors) (`.safetensors`) with
//! `rlx_umap.*` string metadata — see [`crate::model_io`].
//!
//! **Also:** GGUF F32 (`.gguf`, feature `io-gguf`), legacy `.ruama` v1–v4 (load only for old files).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::config::UmapConfig;
use crate::encoder::mlp::ModelSpec;
use crate::utils::NormStats;
use crate::weights::WeightStore;

const MAGIC: &[u8; 4] = b"RUMA";
const VERSION_V1: u32 = 1;
const VERSION_V2: u32 = 2;
const VERSION_V3: u32 = 3;
const VERSION_V4: u32 = 4;

/// Training layout metadata stored alongside weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelMetadata {
    pub n_train: usize,
    pub n_features: usize,
    pub n_pos: usize,
    pub n_neg: usize,
}

/// Everything needed to reconstruct a [`crate::fitted::FittedUmap`].
#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub weights: WeightStore,
    pub meta: ModelMetadata,
    pub norm: NormStats,
    pub config: Option<UmapConfig>,
}

/// View for writing a full fitted model (v4).
pub struct SaveBundle<'a> {
    pub weights: &'a WeightStore,
    pub meta: ModelMetadata,
    pub norm: &'a NormStats,
    pub config: &'a UmapConfig,
}

/// Save weights only (`.safetensors` / `.gguf` by extension).
pub fn save_weights(
    w: &WeightStore,
    spec: &ModelSpec,
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    crate::model_io::save_weights(w, spec, path)
}

/// Load encoder weights (safetensors, GGUF, or legacy `.ruama`).
pub fn load_weights(path: impl AsRef<Path>) -> std::io::Result<WeightStore> {
    crate::model_io::load_weights(path)
}

/// Save weights to legacy `.ruama` v1 (backward compatibility).
pub(crate) fn save_weights_ruama(w: &WeightStore, path: impl AsRef<Path>) -> std::io::Result<()> {
    write_bundle(path, w, None, None, None)
}

/// Load weights from a legacy `.ruama` v1 file only.
pub(crate) fn load_weights_ruama(path: impl AsRef<Path>) -> std::io::Result<WeightStore> {
    let mut file = std::fs::File::open(path.as_ref())?;
    let version = read_header(&mut file)?;
    if version == VERSION_V1 {
        let count = read_count(&mut file)?;
        return read_tensors(&mut file, count);
    }
    drop(file);
    load_bundle(path).map(|b| b.weights)
}

/// Save full model (weights + metadata + norm + config).
pub fn save_model(bundle: SaveBundle<'_>, path: impl AsRef<Path>) -> std::io::Result<()> {
    crate::model_io::save_model(bundle, path)
}

/// Load a full model (safetensors, GGUF, or legacy `.ruama`).
pub fn load_model(path: impl AsRef<Path>) -> std::io::Result<LoadedModel> {
    crate::model_io::load_model(path)
}

/// Legacy `.ruama` v2+ bundle (used by [`crate::model_io`]).
pub(crate) fn load_legacy_ruama(path: impl AsRef<Path>) -> std::io::Result<LoadedModel> {
    load_bundle(path)
}

fn write_bytes(file: &mut std::fs::File, data: &[u8]) -> std::io::Result<()> {
    file.write_all(&(data.len() as u32).to_le_bytes())?;
    file.write_all(data)?;
    Ok(())
}

fn read_bytes(file: &mut std::fs::File) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    file.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut data = vec![0u8; len];
    file.read_exact(&mut data)?;
    Ok(data)
}

fn write_f64_slice(file: &mut std::fs::File, data: &[f64]) -> std::io::Result<()> {
    file.write_all(&(data.len() as u32).to_le_bytes())?;
    for &v in data {
        file.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_f64_slice(file: &mut std::fs::File, expect: usize) -> std::io::Result<Vec<f64>> {
    let mut len_buf = [0u8; 4];
    file.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len != expect {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("expected {expect} norm values, got {len}"),
        ));
    }
    let mut out = vec![0f64; len];
    for slot in &mut out {
        let mut b = [0u8; 8];
        file.read_exact(&mut b)?;
        *slot = f64::from_le_bytes(b);
    }
    Ok(out)
}

fn write_bundle(
    path: impl AsRef<Path>,
    w: &WeightStore,
    meta: Option<ModelMetadata>,
    norm: Option<&NormStats>,
    config: Option<&UmapConfig>,
) -> std::io::Result<()> {
    let mut names: Vec<String> = w.0.keys().cloned().collect();
    names.sort();
    let mut file = std::fs::File::create(path.as_ref())?;
    file.write_all(MAGIC)?;
    let version = if config.is_some() {
        VERSION_V4
    } else if norm.is_some() {
        VERSION_V3
    } else if meta.is_some() {
        VERSION_V2
    } else {
        VERSION_V1
    };
    file.write_all(&version.to_le_bytes())?;
    if let Some(m) = meta {
        file.write_all(&(m.n_train as u32).to_le_bytes())?;
        file.write_all(&(m.n_features as u32).to_le_bytes())?;
        file.write_all(&(m.n_pos as u32).to_le_bytes())?;
        file.write_all(&(m.n_neg as u32).to_le_bytes())?;
    }
    if let Some(n) = norm {
        write_f64_slice(&mut file, &n.mean)?;
        write_f64_slice(&mut file, &n.std)?;
    }
    if let Some(cfg) = config {
        let json = serde_json::to_vec(cfg)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        write_bytes(&mut file, &json)?;
    }
    let count = names.len() as u32;
    file.write_all(&count.to_le_bytes())?;
    for name in &names {
        let data = &w.0[name];
        let name_bytes = name.as_bytes();
        file.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        file.write_all(name_bytes)?;
        file.write_all(&(data.len() as u32).to_le_bytes())?;
        for &v in data {
            file.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn read_header(file: &mut std::fs::File) -> std::io::Result<u32> {
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not an rlx-umap weight file (expected RUMA magic)",
        ));
    }
    let mut word_buf = [0u8; 4];
    file.read_exact(&mut word_buf)?;
    Ok(u32::from_le_bytes(word_buf))
}

fn load_bundle(path: impl AsRef<Path>) -> std::io::Result<LoadedModel> {
    let mut file = std::fs::File::open(path.as_ref())?;
    let version = read_header(&mut file)?;

    let (meta, norm, config, count) = match version {
        VERSION_V4 => {
            let m = read_meta(&mut file)?;
            let mean = read_f64_slice(&mut file, m.n_features)?;
            let std = read_f64_slice(&mut file, m.n_features)?;
            let json = read_bytes(&mut file)?;
            let cfg: UmapConfig = serde_json::from_slice(&json)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            let count = read_count(&mut file)?;
            (Some(m), Some(NormStats { mean, std }), Some(cfg), count)
        }
        VERSION_V3 => {
            let m = read_meta(&mut file)?;
            let mean = read_f64_slice(&mut file, m.n_features)?;
            let std = read_f64_slice(&mut file, m.n_features)?;
            let count = read_count(&mut file)?;
            (Some(m), Some(NormStats { mean, std }), None, count)
        }
        VERSION_V2 => {
            let m = read_meta(&mut file)?;
            let count = read_count(&mut file)?;
            (Some(m), None, None, count)
        }
        VERSION_V1 => {
            let count = read_count(&mut file)?;
            (None, None, None, count)
        }
        _ => {
            // Legacy: first u32 after magic was tensor count.
            let count = version as usize;
            (None, None, None, count)
        }
    };

    let meta = meta.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "file has weights only — use load_weights or re-save with save_model",
        )
    })?;

    let norm = norm.unwrap_or_else(|| NormStats {
        mean: vec![0.0; meta.n_features],
        std: vec![1.0; meta.n_features],
    });

    let weights = read_tensors(&mut file, count)?;

    Ok(LoadedModel {
        weights,
        meta,
        norm,
        config,
    })
}

fn read_tensors(file: &mut std::fs::File, count: usize) -> std::io::Result<WeightStore> {
    let mut weights = WeightStore::default();
    for _ in 0..count {
        let mut nlen_buf = [0u8; 4];
        file.read_exact(&mut nlen_buf)?;
        let nlen = u32::from_le_bytes(nlen_buf) as usize;
        let mut name_bytes = vec![0u8; nlen];
        file.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let mut dlen_buf = [0u8; 4];
        file.read_exact(&mut dlen_buf)?;
        let dlen = u32::from_le_bytes(dlen_buf) as usize;
        let mut data = vec![0f32; dlen];
        for slot in &mut data {
            let mut b = [0u8; 4];
            file.read_exact(&mut b)?;
            *slot = f32::from_le_bytes(b);
        }
        weights.0.insert(name, data);
    }
    Ok(weights)
}

fn read_meta(file: &mut std::fs::File) -> std::io::Result<ModelMetadata> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    let n_train = u32::from_le_bytes(buf) as usize;
    file.read_exact(&mut buf)?;
    let n_features = u32::from_le_bytes(buf) as usize;
    file.read_exact(&mut buf)?;
    let n_pos = u32::from_le_bytes(buf) as usize;
    file.read_exact(&mut buf)?;
    let n_neg = u32::from_le_bytes(buf) as usize;
    Ok(ModelMetadata {
        n_train,
        n_features,
        n_pos,
        n_neg,
    })
}

fn read_count(file: &mut std::fs::File) -> std::io::Result<usize> {
    let mut count_buf = [0u8; 4];
    file.read_exact(&mut count_buf)?;
    Ok(u32::from_le_bytes(count_buf) as usize)
}

/// Suggested path: `dir/model.safetensors`.
pub fn model_path(dir: impl AsRef<Path>, stem: &str) -> PathBuf {
    crate::model_io::model_path(dir, stem)
}
