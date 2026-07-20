// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! `*.rlx` binary format — magic + schema version + bincode body.
//!
//! Schema **v2** merges the optimized MIR graph with an explicit weight table
//! (model + weights in one file). Schema v1 (graph-only) is still readable.

use crate::BakeReport;
use crate::optimize::{OptimizeStats, WeightEncoding, WeightRewrite};
use anyhow::{Context, Result, bail};
use rlx_ir::{Dim, Graph, Op, Shape};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Magic bytes for a baked RLX artifact (`RLXBAKE1`).
pub const RLX_MAGIC: &[u8; 8] = b"RLXBAKE1";

/// Current on-disk schema (v2 = graph + explicit weights table).
pub const RLX_FORMAT_VERSION: u32 = 2;

/// Named tensor I/O metadata stored alongside the baked graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RlxIo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// One weight tensor in the merged artifact (final encoding after bake opts).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RlxWeight {
    pub name: String,
    pub shape: Vec<usize>,
    /// `f32` | `gguf_tq2_0` | `gguf_q8_0`
    pub encoding: String,
    pub data: Vec<u8>,
    pub note: String,
}

impl RlxWeight {
    pub fn from_rewrite(r: &WeightRewrite) -> Self {
        Self {
            name: r.name.clone(),
            shape: r.shape.clone(),
            encoding: r.encoding.as_str().to_string(),
            data: r.data.clone(),
            note: r.note.clone(),
        }
    }

    pub fn from_f32(name: impl Into<String>, shape: Vec<usize>, values: &[f32]) -> Self {
        let mut data = Vec::with_capacity(values.len() * 4);
        for &v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            name: name.into(),
            shape,
            encoding: WeightEncoding::F32.as_str().to_string(),
            data,
            note: "baked f32".into(),
        }
    }
}

/// Light metadata for a baked `*.rlx` file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RlxMeta {
    pub name: String,
    pub inputs: Vec<RlxIo>,
    pub outputs: Vec<RlxIo>,
    pub params_remaining: Vec<String>,
    pub nodes: usize,
    pub constant_bytes: usize,
    pub params_baked: usize,
    /// Number of tensors in the weight table.
    pub weight_count: usize,
    /// Total bytes across the weight table.
    pub weight_bytes: usize,
    pub skipped_zero_matmuls: usize,
    pub ternary_packed: usize,
    pub quant_packed: usize,
}

/// On-disk / in-memory baked artifact: optimized graph **plus** merged weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlxFile {
    pub meta: RlxMeta,
    pub graph: Graph,
    /// Authoritative weight table. With [`crate::MemoryMode::Compact`], this holds
/// the bytes while graph Constants are empty until `materialize_weights`.
    pub weights: Vec<RlxWeight>,
}

#[derive(Deserialize)]
struct RlxFileV1 {
    meta: RlxMetaV1,
    graph: Graph,
}

#[derive(Deserialize)]
struct RlxMetaV1 {
    name: String,
    inputs: Vec<RlxIo>,
    outputs: Vec<RlxIo>,
    params_remaining: Vec<String>,
    nodes: usize,
    constant_bytes: usize,
    params_baked: usize,
}

impl RlxFile {
    /// Build an [`RlxFile`] from a baked graph, weight rewrites, and reports.
    pub fn from_baked(
        graph: Graph,
        report: &BakeReport,
        weights: Vec<RlxWeight>,
        opt: &OptimizeStats,
    ) -> Self {
        let inputs = graph
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Input { name } => Some(io_from_shape(name, &n.shape)),
                _ => None,
            })
            .collect();
        let outputs = graph
            .outputs
            .iter()
            .map(|&id| {
                let n = graph.node(id);
                let name = n.name.clone().unwrap_or_else(|| format!("out{}", id.0));
                io_from_shape(&name, &n.shape)
            })
            .collect();
        let weight_bytes: usize = weights.iter().map(|w| w.data.len()).sum();
        let weight_count = weights.len();
        Self {
            meta: RlxMeta {
                name: graph.name.clone(),
                inputs,
                outputs,
                params_remaining: report.params_remaining.clone(),
                nodes: report.nodes_after,
                constant_bytes: report.constant_bytes,
                params_baked: report.params_baked,
                weight_count,
                weight_bytes,
                skipped_zero_matmuls: opt.skipped_zero_matmuls,
                ternary_packed: opt.ternary_packed,
                quant_packed: opt.quant_packed,
            },
            graph,
            weights,
        }
    }
}

fn io_from_shape(name: &str, shape: &Shape) -> RlxIo {
    let dims: Vec<usize> = shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
            Dim::Dynamic(_) => 0,
        })
        .collect();
    RlxIo {
        name: name.to_string(),
        shape: dims,
        dtype: format!("{:?}", shape.dtype()).to_lowercase(),
    }
}

/// Serialize `file` to a plaintext `*.rlx` path.
pub fn write_rlx(path: impl AsRef<Path>, file: &RlxFile) -> Result<()> {
    let path = path.as_ref();
    let bytes = to_bytes(file).context("serializing *.rlx")?;
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Serialize `file` to a fully encrypted `*.rlx` (ChaCha20-Poly1305 + Argon2id).
///
/// Requires the `encrypt` cargo feature.
#[cfg(feature = "encrypt")]
pub fn write_rlx_encrypted(
    path: impl AsRef<Path>,
    file: &RlxFile,
    password: &str,
) -> Result<()> {
    let path = path.as_ref();
    let plain = to_bytes(file).context("serializing *.rlx")?;
    let enc = crate::crypto::encrypt_bytes(&plain, password).context("encrypting *.rlx")?;
    std::fs::write(path, enc).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Load a plaintext `*.rlx`. Encrypted files need the `encrypt` feature and
/// [`read_rlx_with_password`].
pub fn read_rlx(path: impl AsRef<Path>) -> Result<RlxFile> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    parse_rlx_bytes(&bytes, path)
}

/// Like [`read_rlx`], but maps the file with `memmap2` instead of copying into a
/// `Vec` first (feature `mmap`). Encrypted files still need a full decrypt buffer.
#[cfg(feature = "mmap")]
pub fn read_rlx_mmap(path: impl AsRef<Path>) -> Result<RlxFile> {
    use std::fs::File;
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // Safety: read-only map of a local artifact file for the duration of parse.
    let map = unsafe { memmap2::Mmap::map(&file) }
        .with_context(|| format!("mmap {}", path.display()))?;
    parse_rlx_bytes(&map, path)
}

fn parse_rlx_bytes(bytes: &[u8], path: &Path) -> Result<RlxFile> {
    if looks_encrypted(bytes) {
        #[cfg(feature = "encrypt")]
        bail!(
            "{} is encrypted; pass a password via read_rlx_with_password / --password",
            path.display()
        );
        #[cfg(not(feature = "encrypt"))]
        bail!(
            "{} is encrypted; rebuild rlx-bake with `--features encrypt` to load it",
            path.display()
        );
    }
    from_bytes(bytes).with_context(|| format!("parsing {}", path.display()))
}

/// Load a `*.rlx` that may be plaintext or password-encrypted.
///
/// Requires the `encrypt` cargo feature.
#[cfg(feature = "encrypt")]
pub fn read_rlx_with_password(path: impl AsRef<Path>, password: &str) -> Result<RlxFile> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let plain = if crate::crypto::is_encrypted(&bytes) {
        crate::crypto::decrypt_bytes(&bytes, password)
            .with_context(|| format!("decrypting {}", path.display()))?
    } else {
        bytes
    };
    from_bytes(&plain).with_context(|| format!("parsing {}", path.display()))
}

fn looks_encrypted(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[..8] == b"RLXENC01"
}

/// Encode an [`RlxFile`] to the on-disk byte layout.
pub fn to_bytes(file: &RlxFile) -> Result<Vec<u8>> {
    let body = bincode::serialize(file).context("bincode serialize RlxFile")?;
    let mut out = Vec::with_capacity(12 + body.len());
    out.extend_from_slice(RLX_MAGIC);
    out.extend_from_slice(&RLX_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode an [`RlxFile`] from the on-disk byte layout (v1 or v2).
pub fn from_bytes(b: &[u8]) -> Result<RlxFile> {
    if b.len() < 12 {
        bail!("*.rlx too short ({} bytes)", b.len());
    }
    if &b[..8] != RLX_MAGIC {
        bail!(
            "bad *.rlx magic (expected RLXBAKE1, got {:?})",
            String::from_utf8_lossy(&b[..8])
        );
    }
    let ver = u32::from_le_bytes(b[8..12].try_into().unwrap());
    let body = &b[12..];
    match ver {
        2 => bincode::deserialize(body).context("bincode deserialize RlxFile v2"),
        1 => {
            let v1: RlxFileV1 =
                bincode::deserialize(body).context("bincode deserialize RlxFile v1")?;
            Ok(RlxFile {
                meta: RlxMeta {
                    name: v1.meta.name,
                    inputs: v1.meta.inputs,
                    outputs: v1.meta.outputs,
                    params_remaining: v1.meta.params_remaining,
                    nodes: v1.meta.nodes,
                    constant_bytes: v1.meta.constant_bytes,
                    params_baked: v1.meta.params_baked,
                    weight_count: 0,
                    weight_bytes: 0,
                    skipped_zero_matmuls: 0,
                    ternary_packed: 0,
                    quant_packed: 0,
                },
                graph: v1.graph,
                weights: Vec::new(),
            })
        }
        other => bail!("*.rlx schema version {other} unsupported (current {RLX_FORMAT_VERSION})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BakeOptions, bake};
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Shape};
    use std::collections::HashMap;

    #[test]
    fn roundtrip_bytes_with_weights() {
        let s = Shape::new(&[2], DType::F32);
        let mut g = Graph::new("rt");
        let x = g.input("x", s.clone());
        let w = g.param("w", s.clone());
        let y = g.binary(BinaryOp::Mul, x, w, s);
        g.set_outputs(vec![y]);
        let mut bindings = HashMap::new();
        bindings.insert("w".into(), vec![1.0, 2.0]);
        let (file, report) = bake(&g, &bindings, &BakeOptions::default());
        assert!(!file.weights.is_empty(), "expected weights in artifact");
        assert_eq!(file.meta.weight_count, file.weights.len());
        let bytes = to_bytes(&file).unwrap();
        let back = from_bytes(&bytes).unwrap();
        assert_eq!(back.meta.name, "rt");
        assert_eq!(back.weights.len(), file.weights.len());
        assert_eq!(back.weights[0].name, "w");
        assert_eq!(back.meta.params_remaining, report.params_remaining);
    }

    #[cfg(feature = "encrypt")]
    #[test]
    fn encrypted_roundtrip_file() {
        let s = Shape::new(&[2], DType::F32);
        let mut g = Graph::new("enc");
        let x = g.input("x", s.clone());
        let w = g.param("w", s.clone());
        let y = g.binary(BinaryOp::Mul, x, w, s);
        g.set_outputs(vec![y]);
        let mut bindings = HashMap::new();
        bindings.insert("w".into(), vec![3.0, 4.0]);
        let (file, _) = bake(&g, &bindings, &BakeOptions::default());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.rlx");
        let plain = to_bytes(&file).unwrap();
        let enc = crate::crypto::encrypt_bytes_with_params(&plain, "pw", 8, 1, 1).unwrap();
        std::fs::write(&path, &enc).unwrap();

        assert!(crate::crypto::is_encrypted(&enc));
        assert!(read_rlx(&path).is_err());
        let back = read_rlx_with_password(&path, "pw").unwrap();
        assert_eq!(back.meta.name, "enc");
        assert_eq!(back.weights.len(), file.weights.len());
        assert_eq!(back.weights[0].data, file.weights[0].data);
    }
}
