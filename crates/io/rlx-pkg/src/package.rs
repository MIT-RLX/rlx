// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Open and query an `.rlxp` flat / directory / ZIP package.
//!
//! # Load path
//!
//! 1. [`Package::open`] — mmap file / shards, parse TOC (JSON or bincode)
//! 2. [`Package::advise_hot_willneed`] — optional `madvise(WILLNEED)` on hot ranges
//! 3. [`Package::graph`] — deserialize MIR; fill **hot** stripped Constants
//! 4. [`Package::bind_warm_and_cold`] — optional late bind of compressed weights
//!
//! Weight-only packs (`graph.encoding = "none"`) refuse [`Package::graph`].

use crate::flat::{self, FlatView};
use crate::manifest::{Manifest, SidecarRef};
use crate::placement::Placement;
use crate::store_zip::{self, MemberRange};
use crate::tier::{
    Codec, StorageTier, decode_payload, decode_payload_parallel, decode_zstd_block_at,
};
use crate::weights_index::{WeightEntry, WeightsIndex};
use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use rlx_ir::{Graph, Op};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

/// Which weight tiers to bind into graph Constants on [`Package::graph`].
///
/// Default [`MaterializeMode::HotOnly`] keeps warm/cold out of the compile path
/// until [`Package::bind_tensor`] / [`Package::bind_warm_and_cold`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaterializeMode {
    /// Hot / mmapable tensors only (default).
    #[default]
    HotOnly,
    /// Inflate every tier into Constant payloads.
    All,
    /// Leave all weight Constants empty (caller binds later).
    None,
}

/// Where package members are served from.
#[derive(Debug)]
pub enum MemberSource {
    /// Flat `RLXPFLAT` single-file mmap.
    Flat { path: PathBuf, map: Mmap },
    /// Unpacked directory tree with per-shard mmaps.
    Dir {
        root: PathBuf,
        shard_maps: BTreeMap<String, Mmap>,
    },
    /// ZIP64 STORE archive.
    Zip {
        path: PathBuf,
        map: Mmap,
        ranges: BTreeMap<String, MemberRange>,
    },
}

/// Opened RLX package.
pub struct Package {
    source: MemberSource,
    manifest: Manifest,
    weights: Option<WeightsIndex>,
    name_index: HashMap<String, usize>,
    placement: Option<Placement>,
    /// Absolute file offset of the flat data region (Flat only).
    flat_data_start: u64,
    flat_graph: Option<(u64, u64)>,
    /// id, offset, length, codec
    flat_sidecars: Vec<(String, u64, u64, Codec)>,
}

impl Package {
    /// Open a directory, flat `.rlxp`, or ZIP archive.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            return Self::open_dir(path);
        }
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: read-only map of a local package file for the Package lifetime.
        let map =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
        if flat::is_flat_magic(&map) {
            Self::open_flat(path, map)
        } else if map.len() >= 4 && &map[..2] == b"PK" {
            Self::open_zip_mapped(path, map)
        } else {
            bail!(
                "{} is not a flat RLXPFLAT, ZIP, or package directory",
                path.display()
            )
        }
    }

    fn finish(
        source: MemberSource,
        manifest: Manifest,
        weights: Option<WeightsIndex>,
        placement: Option<Placement>,
        flat_data_start: u64,
        flat_graph: Option<(u64, u64)>,
        flat_sidecars: Vec<(String, u64, u64, Codec)>,
    ) -> Self {
        let name_index = weights
            .as_ref()
            .map(|idx| {
                idx.tensors
                    .iter()
                    .enumerate()
                    .map(|(i, t)| (t.name.clone(), i))
                    .collect()
            })
            .unwrap_or_default();
        Self {
            source,
            manifest,
            weights,
            name_index,
            placement,
            flat_data_start,
            flat_graph,
            flat_sidecars,
        }
    }

    fn open_flat(path: &Path, map: Mmap) -> Result<Self> {
        let view = FlatView::parse(&map)?;
        let weights = Some(view.weights_index());
        let placement = view.placement()?;
        let flat_sidecars = view
            .toc
            .sidecars
            .iter()
            .map(|s| (s.id.clone(), s.offset, s.length, s.codec))
            .collect();
        let flat_graph = Some((view.toc.graph_offset, view.toc.graph_length));
        let flat_data_start = view.data_start;
        let manifest = view.toc.manifest.clone();
        Ok(Self::finish(
            MemberSource::Flat {
                path: path.to_path_buf(),
                map,
            },
            manifest,
            weights,
            placement,
            flat_data_start,
            flat_graph,
            flat_sidecars,
        ))
    }

    fn open_dir(dir: &Path) -> Result<Self> {
        let manifest_path = dir.join("rlx.json");
        let bytes = std::fs::read(&manifest_path)
            .with_context(|| format!("reading {}", manifest_path.display()))?;
        let manifest: Manifest = serde_json::from_slice(&bytes).context("parsing rlx.json")?;
        manifest.validate()?;
        let weights = load_weights_index_dir(dir, &manifest)?;
        let placement = load_placement_dir(dir, &manifest)?;
        let mut shard_maps = BTreeMap::new();
        if let Some(idx) = &weights {
            for t in &idx.tensors {
                if shard_maps.contains_key(&t.shard) {
                    continue;
                }
                let p = dir.join(&t.shard);
                let f = File::open(&p).with_context(|| format!("open shard {}", p.display()))?;
                // SAFETY: read-only map of a package shard for the Package lifetime.
                let map = unsafe { Mmap::map(&f) }
                    .with_context(|| format!("mmap shard {}", p.display()))?;
                shard_maps.insert(t.shard.clone(), map);
            }
        }
        Ok(Self::finish(
            MemberSource::Dir {
                root: dir.to_path_buf(),
                shard_maps,
            },
            manifest,
            weights,
            placement,
            0,
            None,
            Vec::new(),
        ))
    }

    fn open_zip_mapped(path: &Path, map: Mmap) -> Result<Self> {
        let ranges = store_zip::resolve_store_ranges(path)?;
        let manifest_bytes = store_zip::read_member_from_map(&map, &ranges, "rlx.json")
            .context("rlx.json in package")?;
        let manifest: Manifest =
            serde_json::from_slice(manifest_bytes).context("parsing rlx.json")?;
        manifest.validate()?;
        let weights = load_weights_index_zip(&map, &ranges, &manifest)?;
        let placement = load_placement_zip(&map, &ranges, &manifest)?;
        Ok(Self::finish(
            MemberSource::Zip {
                path: path.to_path_buf(),
                map,
                ranges,
            },
            manifest,
            weights,
            placement,
            0,
            None,
            Vec::new(),
        ))
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn weights_index(&self) -> Option<&WeightsIndex> {
        self.weights.as_ref()
    }

    pub fn placement(&self) -> Option<&Placement> {
        self.placement.as_ref()
    }

    /// True when the package embeds a MIR graph (`encoding != "none"`).
    pub fn has_graph(&self) -> bool {
        self.manifest.graph.encoding != "none"
    }

    /// Deserialize the MIR graph and fill stripped weight Constants from the
    /// package (hot tier only by default — see [`MaterializeMode`]).
    pub fn graph(&self) -> Result<Graph> {
        self.graph_with(MaterializeMode::HotOnly)
    }

    /// Like [`Self::graph`] with an explicit materialize mode.
    pub fn graph_with(&self, mode: MaterializeMode) -> Result<Graph> {
        if !self.has_graph() {
            bail!("package has no embedded graph (weight-only / encoding=none)");
        }
        let mut g: Graph = match &self.source {
            MemberSource::Flat { map, .. } => {
                let (off, len) = self.flat_graph.context("flat graph missing")?;
                if len == 0 {
                    bail!("flat graph length is 0");
                }
                let start = (self.flat_data_start + off) as usize;
                let end = start
                    .checked_add(len as usize)
                    .context("flat graph range")?;
                let bytes = map.get(start..end).context("flat graph slice")?;
                bincode::deserialize(bytes).context("deserialize flat graph")?
            }
            MemberSource::Dir { .. } | MemberSource::Zip { .. } => {
                let bytes = self.member_bytes(&self.manifest.graph.path)?;
                bincode::deserialize(&bytes).context("deserialize graph/mir.bin")?
            }
        };
        self.materialize_graph_weights(&mut g, mode)?;
        Ok(g)
    }

    fn materialize_graph_weights(&self, g: &mut Graph, mode: MaterializeMode) -> Result<()> {
        if matches!(mode, MaterializeMode::None) {
            return Ok(());
        }
        let Some(idx) = &self.weights else {
            return Ok(());
        };
        for entry in &idx.tensors {
            let bind = match mode {
                MaterializeMode::All => true,
                MaterializeMode::HotOnly => entry.is_mmapable(),
                MaterializeMode::None => false,
            };
            if !bind {
                continue;
            }
            let bytes = if entry.is_mmapable() {
                self.tensor_mmap(&entry.name)?.to_vec()
            } else {
                self.tensor_bytes(&entry.name)?
            };
            fill_constant(g, &entry.name, bytes);
        }
        Ok(())
    }

    /// Bind one tensor (any tier) into matching empty Constants.
    pub fn bind_tensor(&self, g: &mut Graph, name: &str) -> Result<()> {
        let bytes = self.tensor_bytes(name)?;
        fill_constant(g, name, bytes);
        Ok(())
    }

    /// Inflate all non-hot tensors into the graph (after [`MaterializeMode::HotOnly`]).
    pub fn bind_warm_and_cold(&self, g: &mut Graph) -> Result<()> {
        let Some(idx) = &self.weights else {
            return Ok(());
        };
        for entry in &idx.tensors {
            if entry.is_mmapable() {
                continue;
            }
            self.bind_tensor(g, &entry.name)?;
        }
        Ok(())
    }

    /// Hint the OS to prefetch hot weight pages (`madvise` WILLNEED). Best-effort.
    pub fn advise_hot_willneed(&self) {
        let Some(idx) = &self.weights else {
            return;
        };
        for entry in &idx.tensors {
            if !entry.is_mmapable() {
                continue;
            }
            self.advise_range(entry.shard.as_str(), entry.offset, entry.length);
        }
    }

    /// Prefetch a single named tensor's stored range.
    pub fn prefetch_tensor(&self, name: &str) -> Result<()> {
        let entry = self.weight_entry(name)?;
        self.advise_range(entry.shard.as_str(), entry.offset, entry.length);
        Ok(())
    }

    fn advise_range(&self, shard: &str, offset: u64, length: u64) {
        // memmap2::Advice / advise_range are Unix-only; no-op on Windows.
        #[cfg(unix)]
        {
            let len = length as usize;
            if len == 0 {
                return;
            }
            match &self.source {
                MemberSource::Flat { map, .. } => {
                    let abs = (self.flat_data_start + offset) as usize;
                    let _ = map.advise_range(memmap2::Advice::WillNeed, abs, len);
                }
                MemberSource::Dir { shard_maps, .. } => {
                    if let Some(map) = shard_maps.get(shard) {
                        let _ = map.advise_range(memmap2::Advice::WillNeed, offset as usize, len);
                    }
                }
                MemberSource::Zip { map, ranges, .. } => {
                    if let Some(r) = ranges.get(shard) {
                        let abs = r.data_offset as usize + offset as usize;
                        let _ = map.advise_range(memmap2::Advice::WillNeed, abs, len);
                    }
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = (shard, offset, length);
        }
    }

    /// Sidecar bytes by id (decompresses cold sidecars).
    pub fn sidecar(&self, id: &str) -> Result<Vec<u8>> {
        if let MemberSource::Flat { map, .. } = &self.source {
            let (off, len, codec) = self
                .flat_sidecars
                .iter()
                .find(|(i, _, _, _)| i == id)
                .map(|(_, o, l, c)| (*o, *l, *c))
                .with_context(|| format!("sidecar id {id} not in flat TOC"))?;
            let start = (self.flat_data_start + off) as usize;
            let end = start.checked_add(len as usize).context("sidecar range")?;
            let stored = map.get(start..end).context("sidecar slice")?;
            return decode_payload(codec, stored);
        }
        let sc = self
            .manifest
            .sidecars
            .iter()
            .find(|s: &&SidecarRef| s.id == id)
            .with_context(|| format!("sidecar id {id} not in manifest"))?;
        let stored = self.member_bytes(&sc.path)?;
        decode_payload(sc.codec, &stored)
    }

    pub fn weight_entry(&self, name: &str) -> Result<&WeightEntry> {
        let idx = self
            .weights
            .as_ref()
            .context("package has no weights index")?;
        let i = self
            .name_index
            .get(name)
            .copied()
            .with_context(|| format!("weight tensor {name} not in index"))?;
        Ok(&idx.tensors[i])
    }

    /// Owned tensor bytes (decompresses warm/cold; copies hot). Uses parallel
    /// warm-block decode for large `zstd_blocks` blobs.
    pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let entry = self.weight_entry(name)?;
        let stored = self.shard_slice(&entry.shard, entry.offset, entry.length)?;
        if entry.codec == Codec::ZstdBlocks && stored.len() > (1 << 20) {
            decode_payload_parallel(entry.codec, stored)
        } else {
            decode_payload(entry.codec, stored)
        }
    }

    /// Zero-copy mmap view — **hot / raw only**.
    pub fn tensor_mmap(&self, name: &str) -> Result<&[u8]> {
        let entry = self.weight_entry(name)?;
        if !entry.is_mmapable() {
            bail!(
                "tensor {name} is {:?} / {:?}; use tensor_bytes (or tensor_warm_block)",
                entry.tier,
                entry.codec
            );
        }
        self.shard_slice(&entry.shard, entry.offset, entry.length)
    }

    /// Decode one warm block by index (0-based) without inflating the whole tensor.
    pub fn tensor_warm_block(&self, name: &str, block_index: usize) -> Result<Vec<u8>> {
        let entry = self.weight_entry(name)?;
        if entry.codec != Codec::ZstdBlocks {
            bail!("tensor {name} is not zstd_blocks (codec={:?})", entry.codec);
        }
        let stored = self.shard_slice(&entry.shard, entry.offset, entry.length)?;
        decode_zstd_block_at(stored, block_index)
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let entry = self.weight_entry(name)?;
        if entry.scheme != "f32" {
            bail!(
                "tensor {name} scheme {:?} is not f32; use tensor_bytes for packed weights",
                entry.scheme
            );
        }
        let raw = self.tensor_bytes(name)?;
        if raw.len() % 4 != 0 {
            bail!("f32 tensor {name} length {} not multiple of 4", raw.len());
        }
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn tensors_for_rank(&self, rank: u32) -> Vec<&WeightEntry> {
        let Some(idx) = &self.weights else {
            return Vec::new();
        };
        idx.tensors
            .iter()
            .filter(|t| t.rank.is_none() || t.rank == Some(rank))
            .collect()
    }

    pub fn tensors_by_tier(&self, tier: StorageTier) -> Vec<&WeightEntry> {
        let Some(idx) = &self.weights else {
            return Vec::new();
        };
        idx.tensors.iter().filter(|t| t.tier == tier).collect()
    }

    pub fn rank_root(&self, rank: u32) -> Option<String> {
        let rel = format!("dist/rank-{rank}");
        if self.member_exists(&rel) || self.member_exists(&format!("{rel}/weights/index.json")) {
            Some(rel)
        } else {
            None
        }
    }

    pub fn member_bytes(&self, rel: &str) -> Result<Vec<u8>> {
        match &self.source {
            MemberSource::Flat { .. } => {
                bail!("flat packages have no path members ({rel}); use graph()/sidecar()/tensor_*")
            }
            MemberSource::Dir { root, .. } => {
                let p = root.join(rel);
                std::fs::read(&p).with_context(|| format!("reading {}", p.display()))
            }
            MemberSource::Zip { map, ranges, .. } => {
                Ok(store_zip::read_member_from_map(map, ranges, rel)?.to_vec())
            }
        }
    }

    fn member_exists(&self, rel: &str) -> bool {
        match &self.source {
            MemberSource::Flat { .. } => false,
            MemberSource::Dir { root, .. } => root.join(rel).exists(),
            MemberSource::Zip { ranges, .. } => {
                ranges.contains_key(rel) || ranges.keys().any(|k| k.starts_with(&format!("{rel}/")))
            }
        }
    }

    fn shard_slice(&self, shard: &str, offset: u64, length: u64) -> Result<&[u8]> {
        let start = offset as usize;
        let end = start
            .checked_add(length as usize)
            .context("shard slice overflow")?;
        match &self.source {
            MemberSource::Flat { map, .. } => {
                let abs = (self.flat_data_start + offset) as usize;
                let abs_end = abs
                    .checked_add(length as usize)
                    .context("flat shard overflow")?;
                map.get(abs..abs_end)
                    .with_context(|| format!("flat slice [{offset}..+{length}]"))
            }
            MemberSource::Dir { shard_maps, .. } => {
                let map = shard_maps
                    .get(shard)
                    .with_context(|| format!("shard {shard} not mapped"))?;
                map.get(start..end)
                    .with_context(|| format!("slice {shard}[{offset}..+{length}]"))
            }
            MemberSource::Zip { map, ranges, .. } => {
                let base = store_zip::read_member_from_map(map, ranges, shard)?;
                base.get(start..end)
                    .with_context(|| format!("slice {shard}[{offset}..+{length}]"))
            }
        }
    }
}

fn fill_constant(g: &mut Graph, name: &str, bytes: Vec<u8>) {
    for n in g.nodes_mut() {
        if n.name.as_deref() != Some(name) {
            continue;
        }
        if let Op::Constant { data } = &mut n.op {
            if data.is_empty() {
                *data = bytes.clone();
            }
        }
    }
}

/// Map RLXP `scheme` strings to a host bind dtype for `set_param_typed`.
pub fn dtype_for_weight_scheme(scheme: &str) -> rlx_ir::DType {
    use rlx_ir::DType;
    if scheme == "f32" || scheme.starts_with("f32") {
        return DType::F32;
    }
    if scheme == "f16" || scheme == "bf16" {
        return DType::F16;
    }
    if scheme == "u8" || scheme.starts_with("mlx_") || scheme.starts_with("gguf_") || scheme == "i8"
    {
        return DType::U8;
    }
    DType::F32
}

impl Package {
    /// Yield `(name, bytes, dtype)` for every weight — used to
    /// `CompiledGraph::set_param_typed` after compile (Param path).
    pub fn typed_weight_bindings(&self) -> Result<Vec<(String, Vec<u8>, rlx_ir::DType)>> {
        let Some(idx) = &self.weights else {
            return Ok(Vec::new());
        };
        let mut out = Vec::with_capacity(idx.tensors.len());
        for entry in &idx.tensors {
            let bytes = self.tensor_bytes(&entry.name)?;
            out.push((
                entry.name.clone(),
                bytes,
                dtype_for_weight_scheme(&entry.scheme),
            ));
        }
        Ok(out)
    }
}

fn load_weights_index_dir(dir: &Path, manifest: &Manifest) -> Result<Option<WeightsIndex>> {
    let Some(wref) = &manifest.weights else {
        return Ok(None);
    };
    let p = dir.join(&wref.index);
    let bytes = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
    let idx: WeightsIndex = serde_json::from_slice(&bytes).context("parsing weights/index.json")?;
    Ok(Some(idx))
}

fn load_weights_index_zip(
    map: &[u8],
    ranges: &BTreeMap<String, MemberRange>,
    manifest: &Manifest,
) -> Result<Option<WeightsIndex>> {
    let Some(wref) = &manifest.weights else {
        return Ok(None);
    };
    let bytes = store_zip::read_member_from_map(map, ranges, &wref.index)?;
    let idx: WeightsIndex = serde_json::from_slice(bytes).context("parsing weights/index.json")?;
    Ok(Some(idx))
}

fn load_placement_dir(dir: &Path, manifest: &Manifest) -> Result<Option<Placement>> {
    let Some(dist) = &manifest.dist else {
        return Ok(None);
    };
    let Some(path) = &dist.placement else {
        return Ok(None);
    };
    let p = dir.join(path);
    if !p.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
    let pl: Placement = serde_json::from_slice(&bytes).context("parsing placement.json")?;
    Ok(Some(pl))
}

fn load_placement_zip(
    map: &[u8],
    ranges: &BTreeMap<String, MemberRange>,
    manifest: &Manifest,
) -> Result<Option<Placement>> {
    let Some(dist) = &manifest.dist else {
        return Ok(None);
    };
    let Some(path) = &dist.placement else {
        return Ok(None);
    };
    if !ranges.contains_key(path) {
        return Ok(None);
    }
    let bytes = store_zip::read_member_from_map(map, ranges, path)?;
    let pl: Placement = serde_json::from_slice(bytes).context("parsing placement.json")?;
    Ok(Some(pl))
}
