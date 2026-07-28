// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Storage tiers and codecs for hybrid packs (hot / warm / cold).
//!
//! | Tier | [`Codec`] | Access |
//! |------|-----------|--------|
//! | [`StorageTier::Hot`] | [`Codec::None`] | [`crate::Package::tensor_mmap`] |
//! | [`StorageTier::Warm`] | [`Codec::ZstdBlocks`] | [`crate::Package::tensor_bytes`] / [`decode_zstd_block_at`] |
//! | [`StorageTier::Cold`] | [`Codec::Zstd`] | sidecars via [`crate::Package::sidecar`] |
//!
//! Warm is for rarely touched host blobs — keep quantized LLM weights **hot**.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Access class for a blob inside a flat package.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageTier {
    /// Raw bytes in the mmap window — GEMM / dequant hot path.
    #[default]
    Hot,
    /// Block-compressed (seekable-ish); decompress blocks on demand.
    Warm,
    /// Whole-blob compression; inflate once (sidecars, reports).
    Cold,
}

/// On-disk encoding of a blob's payload bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Codec {
    /// Uncompressed (hot tier).
    #[default]
    None,
    /// Single zstd frame covering the whole blob (cold).
    Zstd,
    /// Concatenated zstd frames with a small block header (warm).
    ZstdBlocks,
}

impl StorageTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hot => "hot",
            Self::Warm => "warm",
            Self::Cold => "cold",
        }
    }
}

impl Codec {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Zstd => "zstd",
            Self::ZstdBlocks => "zstd_blocks",
        }
    }
}

/// Default uncompressed block size for warm `zstd_blocks` (1 MiB).
pub const DEFAULT_WARM_BLOCK: u32 = 1 << 20;

const ZBLK_MAGIC: &[u8; 4] = b"ZBLK";

/// Encode a warm blob: header + per-block zstd frames.
pub fn encode_zstd_blocks(raw: &[u8], block_size: u32) -> Result<Vec<u8>> {
    if block_size == 0 {
        bail!("warm block_size must be > 0");
    }
    let block_size = block_size as usize;
    let mut out = Vec::new();
    out.extend_from_slice(ZBLK_MAGIC);
    out.extend_from_slice(&(block_size as u32).to_le_bytes());
    out.extend_from_slice(&(raw.len() as u64).to_le_bytes());
    let n_blocks = raw.len().div_ceil(block_size);
    out.extend_from_slice(&(n_blocks as u32).to_le_bytes());
    for chunk in raw.chunks(block_size) {
        let comp = zstd::encode_all(chunk, 3).context("zstd compress warm block")?;
        out.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        out.extend_from_slice(&comp);
    }
    Ok(out)
}

/// Decode a warm `ZBLK` blob to owned raw bytes (sequential).
pub fn decode_zstd_blocks(comp: &[u8]) -> Result<Vec<u8>> {
    decode_zstd_blocks_impl(comp, false)
}

/// Decode warm blocks in parallel via rayon (better for large tensors).
pub fn decode_zstd_blocks_parallel(comp: &[u8]) -> Result<Vec<u8>> {
    decode_zstd_blocks_impl(comp, true)
}

fn decode_zstd_blocks_impl(comp: &[u8], parallel: bool) -> Result<Vec<u8>> {
    if comp.len() < 4 + 4 + 8 + 4 {
        bail!("warm blob too short");
    }
    if &comp[..4] != ZBLK_MAGIC {
        bail!("warm blob missing ZBLK magic");
    }
    let block_size = u32::from_le_bytes(comp[4..8].try_into().unwrap()) as usize;
    let raw_len = u64::from_le_bytes(comp[8..16].try_into().unwrap()) as usize;
    let n_blocks = u32::from_le_bytes(comp[16..20].try_into().unwrap()) as usize;
    let mut ranges = Vec::with_capacity(n_blocks);
    let mut pos = 20usize;
    for i in 0..n_blocks {
        if pos + 4 > comp.len() {
            bail!("warm blob truncated at block {i} header");
        }
        let clen = u32::from_le_bytes(comp[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + clen > comp.len() {
            bail!("warm blob truncated at block {i} data");
        }
        ranges.push((pos, pos + clen));
        pos += clen;
    }

    let pieces: Result<Vec<Vec<u8>>> = if parallel && n_blocks > 1 {
        use rayon::prelude::*;
        ranges
            .par_iter()
            .enumerate()
            .map(|(i, (a, b))| {
                let piece = zstd::decode_all(&comp[*a..*b])
                    .with_context(|| format!("zstd decompress block {i}"))?;
                if i + 1 < n_blocks && piece.len() != block_size {
                    bail!(
                        "warm block {i} decoded len {} != block_size {block_size}",
                        piece.len()
                    );
                }
                Ok(piece)
            })
            .collect()
    } else {
        ranges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| {
                let piece = zstd::decode_all(&comp[*a..*b])
                    .with_context(|| format!("zstd decompress block {i}"))?;
                if i + 1 < n_blocks && piece.len() != block_size {
                    bail!(
                        "warm block {i} decoded len {} != block_size {block_size}",
                        piece.len()
                    );
                }
                Ok(piece)
            })
            .collect()
    };
    let pieces = pieces?;
    let mut out = Vec::with_capacity(raw_len);
    for p in pieces {
        out.extend_from_slice(&p);
    }
    if out.len() != raw_len {
        bail!("warm decode length {} != declared {raw_len}", out.len());
    }
    Ok(out)
}

/// Decode one warm block by index without inflating the whole tensor.
pub fn decode_zstd_block_at(comp: &[u8], block_index: usize) -> Result<Vec<u8>> {
    if comp.len() < 20 || &comp[..4] != ZBLK_MAGIC {
        bail!("not a ZBLK warm blob");
    }
    let n_blocks = u32::from_le_bytes(comp[16..20].try_into().unwrap()) as usize;
    if block_index >= n_blocks {
        bail!("block_index {block_index} out of range ({n_blocks})");
    }
    let mut pos = 20usize;
    for i in 0..=block_index {
        if pos + 4 > comp.len() {
            bail!("warm blob truncated");
        }
        let clen = u32::from_le_bytes(comp[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + clen > comp.len() {
            bail!("warm blob truncated");
        }
        if i == block_index {
            return zstd::decode_all(&comp[pos..pos + clen]).context("zstd decompress block");
        }
        pos += clen;
    }
    unreachable!()
}

pub fn encode_zstd(raw: &[u8]) -> Result<Vec<u8>> {
    zstd::encode_all(raw, 3).context("zstd compress")
}

pub fn decode_zstd(comp: &[u8]) -> Result<Vec<u8>> {
    zstd::decode_all(comp).context("zstd decompress")
}

/// Decode payload bytes according to codec → owned raw.
pub fn decode_payload(codec: Codec, stored: &[u8]) -> Result<Vec<u8>> {
    match codec {
        Codec::None => Ok(stored.to_vec()),
        Codec::Zstd => decode_zstd(stored),
        Codec::ZstdBlocks => decode_zstd_blocks(stored),
    }
}

/// Like [`decode_payload`] but parallelizes warm block inflate.
pub fn decode_payload_parallel(codec: Codec, stored: &[u8]) -> Result<Vec<u8>> {
    match codec {
        Codec::None => Ok(stored.to_vec()),
        Codec::Zstd => decode_zstd(stored),
        Codec::ZstdBlocks => decode_zstd_blocks_parallel(stored),
    }
}

/// Encode raw bytes for a tier (picks codec).
pub fn encode_for_tier(tier: StorageTier, raw: &[u8], warm_block: u32) -> Result<(Codec, Vec<u8>)> {
    match tier {
        StorageTier::Hot => Ok((Codec::None, raw.to_vec())),
        StorageTier::Warm => Ok((Codec::ZstdBlocks, encode_zstd_blocks(raw, warm_block)?)),
        StorageTier::Cold => Ok((Codec::Zstd, encode_zstd(raw)?)),
    }
}

/// xxh3-64 hex checksum of raw (uncompressed) bytes.
pub fn checksum_hex(raw: &[u8]) -> String {
    format!("{:016x}", xxhash_rust::xxh3::xxh3_64(raw))
}
