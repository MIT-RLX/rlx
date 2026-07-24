// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Flat `.rlxp` binary — hybrid hot / warm / cold data region.

use crate::manifest::Manifest;
use crate::placement::Placement;
use crate::tier::{Codec, StorageTier, decode_payload, encode_for_tier};
use crate::weights_index::{WeightEntry, WeightsIndex};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// Magic for the flat package (`RLXPFLAT`).
pub const FLAT_MAGIC: &[u8; 8] = b"RLXPFLAT";

/// Flat container format version. v2 = hybrid tiers; v1 still readable.
pub const FLAT_CONTAINER_VERSION: u32 = 2;

/// Align data region to this many bytes (SIMD / DMA friendly).
pub const DATA_ALIGN: u64 = 64;

/// Flag bit: package uses hybrid hot/warm/cold codecs.
pub const FLAG_HYBRID: u32 = 1;

/// Flag bit: TOC is bincode rather than JSON.
pub const FLAG_BINCODE_TOC: u32 = 2;

/// One tensor in the flat TOC (offsets relative to the data region).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FlatTensor {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// Index into [`FlatToc::strings`] when names are interned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name_i: Option<u32>,
    pub shape: Vec<usize>,
    pub scheme: String,
    pub layout: String,
    pub offset: u64,
    /// Stored (possibly compressed) byte length.
    pub length: u64,
    /// Uncompressed length (equals `length` for hot/raw).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_length: Option<u64>,
    #[serde(default)]
    pub tier: StorageTier,
    #[serde(default)]
    pub codec: Codec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}

impl FlatTensor {
    pub fn resolved_name<'a>(&'a self, strings: &'a [String]) -> Result<&'a str> {
        if !self.name.is_empty() {
            return Ok(&self.name);
        }
        if let Some(i) = self.name_i {
            strings
                .get(i as usize)
                .map(|s| s.as_str())
                .with_context(|| format!("name_i {i} out of strings table"))
        } else {
            bail!("flat tensor missing name and name_i")
        }
    }
}

/// Sidecar blob in the flat TOC.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FlatSidecar {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    pub offset: u64,
    pub length: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_length: Option<u64>,
    #[serde(default)]
    pub tier: StorageTier,
    #[serde(default)]
    pub codec: Codec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}

/// Table of contents after the fixed header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatToc {
    pub manifest: Manifest,
    pub graph_offset: u64,
    pub graph_length: u64,
    /// Graph stays hot/raw (mmap + bincode).
    #[serde(default)]
    pub graph_tier: StorageTier,
    #[serde(default)]
    pub graph_codec: Codec,
    pub tensors: Vec<FlatTensor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sidecars: Vec<FlatSidecar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement_offset: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement_length: Option<u64>,
    #[serde(default)]
    pub placement_tier: StorageTier,
    #[serde(default)]
    pub placement_codec: Codec,
    /// Uncompressed warm block size used when writing (informational).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warm_block_size: Option<u32>,
    /// Interned strings for tensor names (`name_i` indexes this).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub strings: Vec<String>,
}

impl FlatToc {
    /// Resolve interned tensor names into `FlatTensor::name`.
    pub fn resolve_names(&mut self) -> Result<()> {
        for t in &mut self.tensors {
            if t.name.is_empty() {
                if let Some(i) = t.name_i {
                    t.name = self
                        .strings
                        .get(i as usize)
                        .cloned()
                        .with_context(|| format!("name_i {i} out of strings table"))?;
                }
            }
        }
        Ok(())
    }
}

/// Fixed header before the TOC (24 bytes).
#[derive(Debug, Clone, Copy)]
pub struct FlatHeader {
    pub container_version: u32,
    pub flags: u32,
    pub toc_len: u64,
}

impl FlatHeader {
    pub const SIZE: usize = 8 + 4 + 4 + 8;

    pub fn encode(self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[..8].copy_from_slice(FLAT_MAGIC);
        out[8..12].copy_from_slice(&self.container_version.to_le_bytes());
        out[12..16].copy_from_slice(&self.flags.to_le_bytes());
        out[16..24].copy_from_slice(&self.toc_len.to_le_bytes());
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            bail!("flat .rlxp too short for header");
        }
        if &bytes[..8] != FLAT_MAGIC {
            bail!("not a flat RLXPFLAT package");
        }
        let container_version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let toc_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        if container_version > FLAT_CONTAINER_VERSION {
            bail!(
                "flat container version {container_version} newer than loader {FLAT_CONTAINER_VERSION}"
            );
        }
        Ok(Self {
            container_version,
            flags,
            toc_len,
        })
    }

    pub fn is_hybrid(self) -> bool {
        self.flags & FLAG_HYBRID != 0 || self.container_version >= 2
    }

    pub fn is_bincode_toc(self) -> bool {
        self.flags & FLAG_BINCODE_TOC != 0
    }
}

pub fn is_flat_magic(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[..8] == FLAT_MAGIC
}

fn align_up(n: u64, align: u64) -> u64 {
    (n + align - 1) & !(align - 1)
}

/// Wire form when `FLAG_BINCODE_TOC` is set: bincode envelope with JSON manifest
/// (avoids bincode ↔ `serde_json::Value` in `Manifest::extensions`).
/// Tensor/sidecar rows use always-present fields — bincode is not self-describing,
/// so `skip_serializing_if` on the JSON TOC types cannot be reused here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FlatTocBin {
    manifest_json: Vec<u8>,
    graph_offset: u64,
    graph_length: u64,
    graph_tier: StorageTier,
    graph_codec: Codec,
    tensors: Vec<FlatTensorBin>,
    sidecars: Vec<FlatSidecarBin>,
    placement_offset: Option<u64>,
    placement_length: Option<u64>,
    placement_tier: StorageTier,
    placement_codec: Codec,
    warm_block_size: Option<u32>,
    strings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FlatTensorBin {
    name: String,
    name_i: Option<u32>,
    shape: Vec<usize>,
    scheme: String,
    layout: String,
    offset: u64,
    length: u64,
    raw_length: Option<u64>,
    tier: StorageTier,
    codec: Codec,
    rank: Option<u32>,
    checksum: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FlatSidecarBin {
    id: String,
    media_type: Option<String>,
    offset: u64,
    length: u64,
    raw_length: Option<u64>,
    tier: StorageTier,
    codec: Codec,
    checksum: Option<String>,
}

impl From<&FlatTensor> for FlatTensorBin {
    fn from(t: &FlatTensor) -> Self {
        Self {
            name: t.name.clone(),
            name_i: t.name_i,
            shape: t.shape.clone(),
            scheme: t.scheme.clone(),
            layout: t.layout.clone(),
            offset: t.offset,
            length: t.length,
            raw_length: t.raw_length,
            tier: t.tier,
            codec: t.codec,
            rank: t.rank,
            checksum: t.checksum.clone(),
        }
    }
}

impl From<FlatTensorBin> for FlatTensor {
    fn from(t: FlatTensorBin) -> Self {
        Self {
            name: t.name,
            name_i: t.name_i,
            shape: t.shape,
            scheme: t.scheme,
            layout: t.layout,
            offset: t.offset,
            length: t.length,
            raw_length: t.raw_length,
            tier: t.tier,
            codec: t.codec,
            rank: t.rank,
            checksum: t.checksum,
        }
    }
}

impl From<&FlatSidecar> for FlatSidecarBin {
    fn from(s: &FlatSidecar) -> Self {
        Self {
            id: s.id.clone(),
            media_type: s.media_type.clone(),
            offset: s.offset,
            length: s.length,
            raw_length: s.raw_length,
            tier: s.tier,
            codec: s.codec,
            checksum: s.checksum.clone(),
        }
    }
}

impl From<FlatSidecarBin> for FlatSidecar {
    fn from(s: FlatSidecarBin) -> Self {
        Self {
            id: s.id,
            media_type: s.media_type,
            offset: s.offset,
            length: s.length,
            raw_length: s.raw_length,
            tier: s.tier,
            codec: s.codec,
            checksum: s.checksum,
        }
    }
}

impl FlatTocBin {
    pub(crate) fn from_toc(toc: &FlatToc) -> Result<Self> {
        Ok(Self {
            manifest_json: serde_json::to_vec(&toc.manifest)
                .context("manifest json for bincode TOC")?,
            graph_offset: toc.graph_offset,
            graph_length: toc.graph_length,
            graph_tier: toc.graph_tier,
            graph_codec: toc.graph_codec,
            tensors: toc.tensors.iter().map(FlatTensorBin::from).collect(),
            sidecars: toc.sidecars.iter().map(FlatSidecarBin::from).collect(),
            placement_offset: toc.placement_offset,
            placement_length: toc.placement_length,
            placement_tier: toc.placement_tier,
            placement_codec: toc.placement_codec,
            warm_block_size: toc.warm_block_size,
            strings: toc.strings.clone(),
        })
    }

    pub(crate) fn into_toc(self) -> Result<FlatToc> {
        let manifest: Manifest =
            serde_json::from_slice(&self.manifest_json).context("manifest in bincode TOC")?;
        Ok(FlatToc {
            manifest,
            graph_offset: self.graph_offset,
            graph_length: self.graph_length,
            graph_tier: self.graph_tier,
            graph_codec: self.graph_codec,
            tensors: self.tensors.into_iter().map(FlatTensor::from).collect(),
            sidecars: self.sidecars.into_iter().map(FlatSidecar::from).collect(),
            placement_offset: self.placement_offset,
            placement_length: self.placement_length,
            placement_tier: self.placement_tier,
            placement_codec: self.placement_codec,
            warm_block_size: self.warm_block_size,
            strings: self.strings,
        })
    }
}

/// Payload pieces for [`write_flat`].
pub struct FlatPayload {
    pub graph: Vec<u8>,
    pub weights: Vec<(FlatTensorMeta, Vec<u8>)>,
    pub sidecars: Vec<(String, Option<String>, Vec<u8>)>,
    pub placement: Option<Vec<u8>>,
    pub warm_block_size: u32,
    /// Compress sidecars as cold zstd (default true for hybrid).
    pub compress_sidecars: bool,
    /// Keep placement hot/raw (default true).
    pub placement_hot: bool,
    /// Prefer bincode TOC.
    pub bincode_toc: bool,
    /// Intern tensor names into TOC `strings`.
    pub intern_strings: bool,
    /// When false, `graph` should be empty (weight-only pack).
    pub include_graph: bool,
}

/// Tensor metadata without bytes.
#[derive(Debug, Clone)]
pub struct FlatTensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub scheme: String,
    pub layout: String,
    pub rank: Option<u32>,
    pub tier: StorageTier,
    pub checksum: Option<String>,
}

/// Serialize a flat (optionally hybrid) package to `path`.
pub fn write_flat(path: impl AsRef<Path>, manifest: Manifest, payload: FlatPayload) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut data = Vec::new();
    let graph_offset = 0u64;
    data.extend_from_slice(&payload.graph);
    let graph_length = payload.graph.len() as u64;

    let mut string_table: Vec<String> = Vec::new();
    let mut string_index: HashMap<String, u32> = HashMap::new();
    let mut intern = |s: &str| -> u32 {
        if let Some(&i) = string_index.get(s) {
            return i;
        }
        let i = string_table.len() as u32;
        string_table.push(s.to_string());
        string_index.insert(s.to_string(), i);
        i
    };

    let mut tensors = Vec::with_capacity(payload.weights.len());
    for (meta, raw) in &payload.weights {
        let (codec, stored) = encode_for_tier(meta.tier, raw, payload.warm_block_size)?;
        let offset = data.len() as u64;
        let length = stored.len() as u64;
        let raw_length = if codec == Codec::None {
            None
        } else {
            Some(raw.len() as u64)
        };
        data.extend_from_slice(&stored);
        let (name, name_i) = if payload.intern_strings {
            (String::new(), Some(intern(&meta.name)))
        } else {
            (meta.name.clone(), None)
        };
        tensors.push(FlatTensor {
            name,
            name_i,
            shape: meta.shape.clone(),
            scheme: meta.scheme.clone(),
            layout: meta.layout.clone(),
            offset,
            length,
            raw_length,
            tier: meta.tier,
            codec,
            rank: meta.rank,
            checksum: meta.checksum.clone(),
        });
    }

    let mut sidecars = Vec::new();
    for (id, media, raw) in &payload.sidecars {
        let (tier, codec, stored, raw_length) = if payload.compress_sidecars {
            let (c, s) = encode_for_tier(StorageTier::Cold, raw, payload.warm_block_size)?;
            (StorageTier::Cold, c, s, Some(raw.len() as u64))
        } else {
            (StorageTier::Hot, Codec::None, raw.clone(), None)
        };
        let offset = data.len() as u64;
        let length = stored.len() as u64;
        data.extend_from_slice(&stored);
        let checksum = meta_checksum_for_sidecar(raw, &payload);
        sidecars.push(FlatSidecar {
            id: id.clone(),
            media_type: media.clone(),
            offset,
            length,
            raw_length,
            tier,
            codec,
            checksum,
        });
    }

    let (placement_offset, placement_length, placement_tier, placement_codec) =
        if let Some(pl) = &payload.placement {
            let (tier, codec, stored) = if payload.placement_hot {
                (StorageTier::Hot, Codec::None, pl.clone())
            } else {
                let (c, s) = encode_for_tier(StorageTier::Cold, pl, payload.warm_block_size)?;
                (StorageTier::Cold, c, s)
            };
            let offset = data.len() as u64;
            data.extend_from_slice(&stored);
            (Some(offset), Some(stored.len() as u64), tier, codec)
        } else {
            (None, None, StorageTier::Hot, Codec::None)
        };

    let hybrid = tensors.iter().any(|t| t.tier != StorageTier::Hot)
        || sidecars.iter().any(|s| s.codec != Codec::None)
        || placement_codec != Codec::None;

    let mut manifest = manifest;
    if !payload.include_graph || graph_length == 0 {
        manifest.graph.encoding = "none".into();
        push_feat(&mut manifest.features, "weight_only");
    }
    if hybrid {
        push_feat(&mut manifest.features, "hybrid_storage");
        if tensors.iter().any(|t| t.tier == StorageTier::Warm) {
            push_feat(&mut manifest.features, "zstd_blocks_warm");
        }
        if sidecars.iter().any(|s| s.codec == Codec::Zstd)
            || placement_codec == Codec::Zstd
        {
            push_feat(&mut manifest.features, "zstd_cold");
        }
    }
    if payload.intern_strings {
        push_feat(&mut manifest.features, "toc_string_table");
    }
    if payload.bincode_toc {
        push_feat(&mut manifest.features, "toc_bincode");
    }

    let toc = FlatToc {
        manifest,
        graph_offset,
        graph_length,
        graph_tier: StorageTier::Hot,
        graph_codec: Codec::None,
        tensors,
        sidecars,
        placement_offset,
        placement_length,
        placement_tier,
        placement_codec,
        warm_block_size: Some(payload.warm_block_size),
        strings: if payload.intern_strings {
            string_table
        } else {
            Vec::new()
        },
    };
    let toc_bytes = if payload.bincode_toc {
        let wire = FlatTocBin::from_toc(&toc)?;
        bincode::serialize(&wire).context("serialize flat TOC (bincode)")?
    } else {
        serde_json::to_vec(&toc).context("serialize flat TOC")?
    };

    let mut flags = 0u32;
    if hybrid {
        flags |= FLAG_HYBRID;
    }
    if payload.bincode_toc {
        flags |= FLAG_BINCODE_TOC;
    }
    let header = FlatHeader {
        container_version: FLAT_CONTAINER_VERSION,
        flags,
        toc_len: toc_bytes.len() as u64,
    };
    let header_bytes = header.encode();
    let data_start = align_up((FlatHeader::SIZE as u64) + header.toc_len, DATA_ALIGN);
    let pad = data_start as usize - FlatHeader::SIZE - toc_bytes.len();

    let mut file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    file.write_all(&header_bytes)?;
    file.write_all(&toc_bytes)?;
    if pad > 0 {
        file.write_all(&vec![0u8; pad])?;
    }
    file.write_all(&data)?;
    Ok(())
}

fn meta_checksum_for_sidecar(raw: &[u8], payload: &FlatPayload) -> Option<String> {
    // Checksums for sidecars follow the same policy as weights: only when any
    // weight carried a checksum (write_checksums). Detect via first weight.
    if payload.weights.iter().any(|(m, _)| m.checksum.is_some()) {
        Some(crate::tier::checksum_hex(raw))
    } else {
        None
    }
}

fn push_feat(feats: &mut Vec<String>, f: &str) {
    if !feats.iter().any(|x| x == f) {
        feats.push(f.to_string());
    }
}

/// Parsed flat package views into an mmap.
pub struct FlatView<'a> {
    #[allow(dead_code)]
    pub header: FlatHeader,
    pub toc: FlatToc,
    pub data: &'a [u8],
    pub data_start: u64,
}

impl<'a> FlatView<'a> {
    pub fn parse(map: &'a [u8]) -> Result<Self> {
        let header = FlatHeader::decode(map)?;
        let toc_end = FlatHeader::SIZE
            .checked_add(header.toc_len as usize)
            .context("toc overflow")?;
        let toc_bytes = map
            .get(FlatHeader::SIZE..toc_end)
            .context("truncated flat TOC")?;
        let mut toc: FlatToc = if header.is_bincode_toc() {
            let wire: FlatTocBin =
                bincode::deserialize(toc_bytes).context("deserialize flat TOC (bincode)")?;
            wire.into_toc()?
        } else {
            serde_json::from_slice(toc_bytes).context("deserialize flat TOC")?
        };
        toc.resolve_names()?;
        toc.manifest.validate()?;
        let data_start = align_up((FlatHeader::SIZE as u64) + header.toc_len, DATA_ALIGN);
        let data = map
            .get(data_start as usize..)
            .context("truncated flat data region")?;
        Ok(Self {
            header,
            toc,
            data,
            data_start,
        })
    }

    pub fn slice(&self, offset: u64, length: u64) -> Result<&'a [u8]> {
        let start = offset as usize;
        let end = start
            .checked_add(length as usize)
            .context("flat slice overflow")?;
        self.data
            .get(start..end)
            .with_context(|| format!("flat data slice [{offset}..+{length}]"))
    }

    pub fn decode_slice(&self, offset: u64, length: u64, codec: Codec) -> Result<Vec<u8>> {
        let stored = self.slice(offset, length)?;
        decode_payload(codec, stored)
    }

    pub fn weights_index(&self) -> WeightsIndex {
        WeightsIndex {
            tensors: self
                .toc
                .tensors
                .iter()
                .map(|t| WeightEntry {
                    name: t.name.clone(),
                    shape: t.shape.clone(),
                    scheme: t.scheme.clone(),
                    layout: t.layout.clone(),
                    shard: "__flat__".into(),
                    offset: t.offset,
                    length: t.length,
                    raw_length: t.raw_length.or(Some(t.length)),
                    tier: t.tier,
                    codec: t.codec,
                    rank: t.rank,
                    checksum: t.checksum.clone(),
                })
                .collect(),
        }
    }

    pub fn placement(&self) -> Result<Option<Placement>> {
        match (self.toc.placement_offset, self.toc.placement_length) {
            (Some(off), Some(len)) => {
                let bytes = self.decode_slice(off, len, self.toc.placement_codec)?;
                Ok(Some(
                    serde_json::from_slice(&bytes).context("placement in flat pack")?,
                ))
            }
            _ => Ok(None),
        }
    }
}
