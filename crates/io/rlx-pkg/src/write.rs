// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Write flat / directory / ZIP `.rlxp` packages.

use crate::flat::{FlatPayload, FlatTensorMeta, write_flat};
use crate::manifest::{DistRef, Manifest, SidecarRef};
use crate::placement::Placement;
use crate::store_zip;
use crate::tier::{DEFAULT_WARM_BLOCK, StorageTier, checksum_hex, encode_for_tier};
use crate::weights_index::{WeightEntry, WeightsIndex};
use anyhow::{Context, Result, bail};
use rlx_ir::{Graph, Op};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// On-disk container kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContainerKind {
    /// Single-file mmap blob (default for `.rlxp`) — fastest load, least overhead.
    #[default]
    Flat,
    /// ZIP64 STORE — inspectable with `unzip`; slower/larger than flat.
    Zip,
    /// Unpacked directory tree — best for editing during development.
    Dir,
}

/// Options for [`write_package`].
///
/// Defaults favor deploy packs: strip graph weight bytes, cold-compress
/// sidecars, embed the MIR graph, xxh3 checksums, and TOC string interning.
#[derive(Debug, Clone)]
pub struct WriteOptions {
    /// Package display name (manifest `name`).
    pub name: String,
    /// Optional producer string (e.g. `"rlx-bake"`).
    pub producer: Option<String>,
    /// Capability / feature tags written into the manifest.
    pub features: Vec<String>,
    /// Flat / ZIP / directory container.
    pub container: ContainerKind,
    /// Sidecar blobs: `(id, media_type, bytes)`.
    pub sidecars: Vec<(String, String, Vec<u8>)>,
    /// Optional distribution placement metadata.
    pub placement: Option<Placement>,
    /// When true (default), clear weight `Constant` payloads in the embedded
    /// graph so bytes live only in the data region (no duplex copy on disk).
    pub strip_graph_weights: bool,
    /// Compress sidecars with cold zstd (default true).
    pub compress_sidecars: bool,
    /// Uncompressed block size for warm `zstd_blocks` (default 1 MiB).
    pub warm_block_size: u32,
    /// Embed MIR graph bytes (default true). Set false for weight-only packs.
    pub include_graph: bool,
    /// Write xxh3 checksums of uncompressed payloads into the TOC.
    pub write_checksums: bool,
    /// Prefer bincode TOC envelope (flat only; smaller/faster than JSON).
    pub bincode_toc: bool,
    /// Intern repeated name strings into TOC `strings` table (flat).
    pub intern_strings: bool,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            name: "model".into(),
            producer: None,
            features: Vec::new(),
            container: ContainerKind::Flat,
            sidecars: Vec::new(),
            placement: None,
            strip_graph_weights: true,
            compress_sidecars: true,
            warm_block_size: DEFAULT_WARM_BLOCK,
            include_graph: true,
            write_checksums: true,
            bincode_toc: false,
            intern_strings: true,
        }
    }
}

/// One weight to pack (from bake or another producer).
///
/// `scheme` is a string such as `"f32"` or `"gguf_q4_k"`. `tier` selects
/// hot mmap vs warm/cold compression ([`StorageTier`]).
#[derive(Debug, Clone)]
pub struct PackedWeight {
    pub name: String,
    pub shape: Vec<usize>,
    pub scheme: String,
    pub layout: String,
    pub data: Vec<u8>,
    pub rank: Option<u32>,
    /// Hot (mmap), warm (blocked zstd), or cold (whole zstd).
    pub tier: StorageTier,
}

impl PackedWeight {
    /// Convenience constructor with [`StorageTier::Hot`].
    pub fn hot(
        name: impl Into<String>,
        shape: Vec<usize>,
        scheme: impl Into<String>,
        layout: impl Into<String>,
        data: Vec<u8>,
    ) -> Self {
        Self {
            name: name.into(),
            shape,
            scheme: scheme.into(),
            layout: layout.into(),
            data,
            rank: None,
            tier: StorageTier::Hot,
        }
    }
}

/// Write a package to `out` (path inferred as flat / ZIP / dir unless
/// [`WriteOptions::container`] is set).
pub fn write_package(
    out: impl AsRef<Path>,
    graph: &Graph,
    weights: &[PackedWeight],
    opts: &WriteOptions,
) -> Result<()> {
    let out = out.as_ref();
    match opts.container {
        ContainerKind::Flat => write_package_flat(out, graph, weights, opts),
        ContainerKind::Zip => write_package_zip(out, graph, weights, opts),
        ContainerKind::Dir => write_package_dir(out, graph, weights, opts),
    }
}

/// Strip named weight constant payloads from a graph clone (disk size win).
pub fn graph_with_stripped_weights(graph: &Graph, weight_names: &[&str]) -> Graph {
    let mut g = graph.clone();
    for n in g.nodes_mut() {
        let Some(name) = n.name.as_deref() else {
            continue;
        };
        if !weight_names.contains(&name) {
            continue;
        }
        if let Op::Constant { data } = &mut n.op {
            data.clear();
        }
    }
    g
}

/// Write the flat mmap package (default `.rlxp`).
pub fn write_package_flat(
    path: impl AsRef<Path>,
    graph: &Graph,
    weights: &[PackedWeight],
    opts: &WriteOptions,
) -> Result<()> {
    let names: Vec<&str> = weights.iter().map(|w| w.name.as_str()).collect();
    let graph_for_disk = if opts.strip_graph_weights {
        graph_with_stripped_weights(graph, &names)
    } else {
        graph.clone()
    };

    let (mut manifest, _index, _shard, _io, _extra) =
        build_manifest_only(&graph_for_disk, weights, opts)?;
    // Flat packs do not use path-based members; clear path pointers that zip/dir use.
    manifest.weights = None;
    manifest.graph.path = "__flat__/graph".into();
    manifest.sidecars.clear();
    for (id, media, _) in &opts.sidecars {
        manifest.sidecars.push(SidecarRef {
            id: id.clone(),
            path: format!("__flat__/sidecar/{id}"),
            media_type: Some(media.clone()),
            codec: crate::tier::Codec::None, // flat uses FlatSidecar.codec
        });
    }
    if opts.placement.is_some() {
        manifest.dist = Some(DistRef {
            placement: Some("__flat__/placement".into()),
        });
    }

    let placement_bytes = opts
        .placement
        .as_ref()
        .map(serde_json::to_vec)
        .transpose()
        .context("serialize placement")?;

    let graph_bytes = if opts.include_graph {
        bincode::serialize(&graph_for_disk).context("serialize graph")?
    } else {
        Vec::new()
    };

    let payload = FlatPayload {
        graph: graph_bytes,
        weights: weights
            .iter()
            .map(|w| {
                (
                    FlatTensorMeta {
                        name: w.name.clone(),
                        shape: w.shape.clone(),
                        scheme: w.scheme.clone(),
                        layout: w.layout.clone(),
                        rank: w.rank,
                        tier: w.tier,
                        checksum: if opts.write_checksums {
                            Some(checksum_hex(&w.data))
                        } else {
                            None
                        },
                    },
                    w.data.clone(),
                )
            })
            .collect(),
        sidecars: opts
            .sidecars
            .iter()
            .map(|(id, media, data)| (id.clone(), Some(media.clone()), data.clone()))
            .collect(),
        placement: placement_bytes,
        warm_block_size: opts.warm_block_size,
        compress_sidecars: opts.compress_sidecars,
        placement_hot: true,
        bincode_toc: opts.bincode_toc,
        intern_strings: opts.intern_strings,
        include_graph: opts.include_graph,
    };
    write_flat(path, manifest, payload)
}

/// Write an unpacked directory package.
pub fn write_package_dir(
    dir: impl AsRef<Path>,
    graph: &Graph,
    weights: &[PackedWeight],
    opts: &WriteOptions,
) -> Result<()> {
    let dir = dir.as_ref();
    std::fs::create_dir_all(dir.join("graph"))
        .with_context(|| format!("mkdir {}", dir.display()))?;
    std::fs::create_dir_all(dir.join("weights/shards"))?;
    if !opts.sidecars.is_empty() {
        std::fs::create_dir_all(dir.join("sidecars"))?;
    }
    if opts.placement.is_some() {
        std::fs::create_dir_all(dir.join("dist"))?;
    }

    let names: Vec<&str> = weights.iter().map(|w| w.name.as_str()).collect();
    let graph_for_disk = if opts.strip_graph_weights {
        graph_with_stripped_weights(graph, &names)
    } else {
        graph.clone()
    };

    let (manifest, index, shard_bytes, io_json, members_extra) =
        build_members(&graph_for_disk, weights, opts)?;

    write_json(dir.join("rlx.json"), &manifest)?;
    if opts.include_graph {
        write_bytes(
            dir.join("graph/mir.bin"),
            &bincode::serialize(&graph_for_disk)?,
        )?;
    }
    write_json(dir.join("graph/io.json"), &io_json)?;
    if let Some(idx) = &index {
        write_json(dir.join("weights/index.json"), idx)?;
        write_bytes(dir.join("weights/shards/000.pack.bin"), &shard_bytes)?;
    }
    for (rel, data) in &members_extra {
        let p = dir.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        write_bytes(p, data)?;
    }
    Ok(())
}

/// Write a ZIP64 STORE archive (inspectable; larger/slower than flat).
pub fn write_package_zip(
    path: impl AsRef<Path>,
    graph: &Graph,
    weights: &[PackedWeight],
    opts: &WriteOptions,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let names: Vec<&str> = weights.iter().map(|w| w.name.as_str()).collect();
    let graph_for_disk = if opts.strip_graph_weights {
        graph_with_stripped_weights(graph, &names)
    } else {
        graph.clone()
    };
    let (manifest, index, shard_bytes, io_json, members_extra) =
        build_members(&graph_for_disk, weights, opts)?;

    let mut members: Vec<(String, Vec<u8>)> = Vec::new();
    members.push((
        "rlx.json".into(),
        serde_json::to_vec_pretty(&manifest).context("serialize rlx.json")?,
    ));
    if opts.include_graph {
        members.push((
            "graph/mir.bin".into(),
            bincode::serialize(&graph_for_disk).context("serialize graph")?,
        ));
    }
    members.push((
        "graph/io.json".into(),
        serde_json::to_vec_pretty(&io_json).context("serialize io.json")?,
    ));
    if let Some(idx) = &index {
        members.push((
            "weights/index.json".into(),
            serde_json::to_vec_pretty(idx).context("serialize weights index")?,
        ));
        members.push(("weights/shards/000.pack.bin".into(), shard_bytes));
    }
    members.extend(members_extra);

    store_zip::write_store_zip(path, &members)?;
    Ok(())
}

#[derive(Serialize)]
struct IoJson {
    inputs: Vec<IoEntry>,
    outputs: Vec<IoEntry>,
}

#[derive(Serialize)]
struct IoEntry {
    name: String,
    shape: Vec<usize>,
    dtype: String,
}

fn build_manifest_only(
    graph: &Graph,
    weights: &[PackedWeight],
    opts: &WriteOptions,
) -> Result<(
    Manifest,
    Option<WeightsIndex>,
    Vec<u8>,
    IoJson,
    Vec<(String, Vec<u8>)>,
)> {
    build_members(graph, weights, opts)
}

fn build_members(
    graph: &Graph,
    weights: &[PackedWeight],
    opts: &WriteOptions,
) -> Result<(
    Manifest,
    Option<WeightsIndex>,
    Vec<u8>,
    IoJson,
    Vec<(String, Vec<u8>)>,
)> {
    let mut manifest = Manifest::new_v1(opts.name.clone());
    manifest.producer = opts.producer.clone();
    manifest.features = opts.features.clone();
    manifest.created_unix = Some(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    );

    let mut features = manifest.features.clone();
    if weights.iter().any(|w| w.scheme.starts_with("gguf_")) {
        push_feat(&mut features, "dequant_matmul");
    }
    if weights
        .iter()
        .any(|w| w.scheme.starts_with("mlx_affine") || w.scheme.starts_with("mlx_mxfp"))
    {
        push_feat(&mut features, "dequant_matmul");
        push_feat(&mut features, "mlx_packed");
    }
    if opts.placement.is_some() {
        push_feat(&mut features, "dist_placement");
    }
    for (id, _, _) in &opts.sidecars {
        push_feat(&mut features, &format!("sidecar_{id}"));
    }
    push_feat(&mut features, "strip_graph_weights");
    if !opts.include_graph {
        push_feat(&mut features, "weight_only");
        manifest.graph.encoding = "none".into();
    }
    if weights.iter().any(|w| w.tier != StorageTier::Hot) {
        push_feat(&mut features, "hybrid_storage");
        if weights.iter().any(|w| w.tier == StorageTier::Warm) {
            push_feat(&mut features, "zstd_blocks_warm");
        }
        if weights.iter().any(|w| w.tier == StorageTier::Cold) {
            push_feat(&mut features, "zstd_cold");
        }
    }
    if opts.compress_sidecars && !opts.sidecars.is_empty() {
        push_feat(&mut features, "zstd_cold");
    }
    manifest.features = features;

    let mut extra: Vec<(String, Vec<u8>)> = Vec::new();
    for (id, media, data) in &opts.sidecars {
        let (codec, stored) = if opts.compress_sidecars {
            encode_for_tier(StorageTier::Cold, data, opts.warm_block_size)?
        } else {
            (crate::tier::Codec::None, data.clone())
        };
        let rel = if media.contains("json") {
            format!("sidecars/{id}.json")
        } else {
            format!("sidecars/{id}")
        };
        manifest.sidecars.push(SidecarRef {
            id: id.clone(),
            path: rel.clone(),
            media_type: Some(media.clone()),
            codec,
        });
        extra.push((rel, stored));
    }

    if let Some(pl) = &opts.placement {
        let rel = "dist/placement.json".to_string();
        manifest.dist = Some(DistRef {
            placement: Some(rel.clone()),
        });
        extra.push((
            rel,
            serde_json::to_vec_pretty(pl).context("serialize placement")?,
        ));
    }

    let (index, shard_bytes) = if weights.is_empty() {
        manifest.weights = None;
        (None, Vec::new())
    } else {
        let mut shard = Vec::new();
        let mut tensors = Vec::new();
        for w in weights {
            let (codec, stored) = encode_for_tier(w.tier, &w.data, opts.warm_block_size)?;
            let offset = shard.len() as u64;
            let length = stored.len() as u64;
            let raw_length = if codec == crate::tier::Codec::None {
                None
            } else {
                Some(w.data.len() as u64)
            };
            shard.extend_from_slice(&stored);
            tensors.push(WeightEntry {
                name: w.name.clone(),
                shape: w.shape.clone(),
                scheme: w.scheme.clone(),
                layout: w.layout.clone(),
                shard: "weights/shards/000.pack.bin".into(),
                offset,
                length,
                raw_length,
                tier: w.tier,
                codec,
                rank: w.rank,
                checksum: if opts.write_checksums {
                    Some(checksum_hex(&w.data))
                } else {
                    None
                },
            });
        }
        (Some(WeightsIndex { tensors }), shard)
    };

    let io_json = IoJson {
        inputs: graph
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Input { name } => Some(io_entry(name, &n.shape)),
                _ => None,
            })
            .collect(),
        outputs: graph
            .outputs
            .iter()
            .map(|&id| {
                let n = graph.node(id);
                let name = n.name.clone().unwrap_or_else(|| format!("out{}", id.0));
                io_entry(&name, &n.shape)
            })
            .collect(),
    };

    Ok((manifest, index, shard_bytes, io_json, extra))
}

fn io_entry(name: &str, shape: &rlx_ir::Shape) -> IoEntry {
    let dims: Vec<usize> = shape
        .dims()
        .iter()
        .map(|d| match d {
            rlx_ir::Dim::Static(n) => *n,
            rlx_ir::Dim::Dynamic(_) => 0,
        })
        .collect();
    IoEntry {
        name: name.to_string(),
        shape: dims,
        dtype: format!("{:?}", shape.dtype()).to_lowercase(),
    }
}

fn push_feat(feats: &mut Vec<String>, f: &str) {
    if !feats.iter().any(|x| x == f) {
        feats.push(f.to_string());
    }
}

fn write_json(path: PathBuf, v: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(v)?;
    write_bytes(path, &bytes)
}

fn write_bytes(path: PathBuf, bytes: &[u8]) -> Result<()> {
    std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))
}

/// Infer container from path / explicit override.
pub fn infer_container(path: &Path, explicit: Option<ContainerKind>) -> Result<ContainerKind> {
    if let Some(c) = explicit {
        return Ok(c);
    }
    if path.is_dir() {
        return Ok(ContainerKind::Dir);
    }
    match path.extension().and_then(|e| e.to_str()) {
        Some("rlxp") => Ok(ContainerKind::Flat),
        Some("zip") => Ok(ContainerKind::Zip),
        None => Ok(ContainerKind::Dir),
        Some(other) => {
            bail!("cannot infer package kind from .{other}; use .rlxp (flat), .zip, or a directory")
        }
    }
}

/// Back-compat alias: `true` → Zip, `false` → Dir; `None` uses path inference
/// (`.rlxp` → Flat).
pub fn infer_zip_from_path(path: &Path, explicit_zip: Option<bool>) -> Result<bool> {
    match explicit_zip {
        Some(true) => Ok(true),
        Some(false) => Ok(false),
        None => Ok(matches!(infer_container(path, None)?, ContainerKind::Zip)),
    }
}
