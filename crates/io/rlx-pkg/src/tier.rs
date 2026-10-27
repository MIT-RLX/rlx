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
        let comp = z_encode_all(chunk, 3)?;
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

// The codec: pure-Rust `zstd-rs` by default (any target, wasm32 included);
// libzstd with `zstd-native` on native targets. Both read and write standard
// zstd frames, so a package written by either is read by either.
#[cfg(all(feature = "zstd-native", not(target_arch = "wasm32")))]
fn z_encode_all(data: &[u8], level: i32) -> Result<Vec<u8>> {
    zstd::encode_all(data, level).context("zstd compress")
}
#[cfg(not(all(feature = "zstd-native", not(target_arch = "wasm32"))))]
fn z_encode_all(data: &[u8], level: i32) -> Result<Vec<u8>> {
    let cfg = zstd_rs::CompressionConfig {
        level,
        ..zstd_rs::CompressionConfig::DEFAULT
    };
    let mut out = Vec::new();
    zstd_rs::Compressor::new(cfg)
        .and_then(|mut c| c.compress(data, None, &mut out))
        .map_err(|e| anyhow::anyhow!("zstd compress: {e}"))?;
    Ok(out)
}

/// Decode `data` (every frame in it), failing rather than producing more than
/// `cap` bytes: a decompression bomb is stopped at the cap, so peak memory is
/// tied to what the caller expects, not to the attacker's frame header.
#[cfg(all(feature = "zstd-native", not(target_arch = "wasm32")))]
fn z_decode_capped(data: &[u8], cap: usize) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut out = Vec::new();
    zstd::Decoder::new(data)
        .context("init zstd decoder")?
        .take((cap as u64).saturating_add(1))
        .read_to_end(&mut out)
        .context("zstd decompress")?;
    if out.len() > cap {
        bail!("zstd output exceeds {cap} bytes");
    }
    Ok(out)
}
#[cfg(not(all(feature = "zstd-native", not(target_arch = "wasm32"))))]
fn z_decode_capped(data: &[u8], cap: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    zstd_rs::Decompressor::new()
        .decompress(data, None, cap, &mut out)
        .map_err(|e| anyhow::anyhow!("zstd decompress: {e}"))?;
    Ok(out)
}

/// A cold blob: its decoded size is not recorded, so there is no cap.
fn z_decode_all(data: &[u8]) -> Result<Vec<u8>> {
    z_decode_capped(data, usize::MAX)
}

/// One warm block, which must decode to exactly `expected` bytes (and is stopped
/// at `expected` if it would produce more). Replaces `zstd::decode_all`, which
/// has no output cap.
fn zstd_decode_bounded(input: &[u8], expected: usize) -> Result<Vec<u8>> {
    let out = z_decode_capped(input, expected).context("zstd decode block")?;
    if out.len() != expected {
        bail!(
            "warm block decoded len {} != expected {expected}",
            out.len()
        );
    }
    Ok(out)
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
    // Validate the encoder's chunking invariant (see `encode_zstd_blocks`):
    // block_size > 0, n_blocks == ceil(raw_len / block_size), and every block
    // decompresses to exactly block_size except the last (the remainder). This
    // makes each block's output size DETERMINISTIC (not attacker-chosen), so the
    // bounded decode below caps peak memory at ~raw_len rather than a bomb frame.
    if raw_len == 0 {
        if n_blocks != 0 {
            bail!("warm blob: raw_len 0 but claims {n_blocks} blocks");
        }
    } else {
        if block_size == 0 {
            bail!("warm blob: block_size 0 with nonzero raw_len");
        }
        if n_blocks != raw_len.div_ceil(block_size) {
            bail!(
                "warm blob: n_blocks {n_blocks} inconsistent with raw_len {raw_len} / block_size {block_size}"
            );
        }
    }
    // Decompression-bomb guard: a legit warm blob's uncompressed size is at most
    // a small multiple of its compressed bytes; reject an implausible ratio up
    // front (before any decode) so a tiny blob can't declare a giant output.
    const MAX_WARM_DECOMPRESS_RATIO: usize = 1000;
    if raw_len > comp.len().saturating_mul(MAX_WARM_DECOMPRESS_RATIO) {
        bail!(
            "warm blob decompression ratio too high (raw_len {raw_len} vs {} compressed bytes) — possible bomb",
            comp.len()
        );
    }
    // `n_blocks` is an untrusted u32 from the blob header; each block needs at
    // least a 4-byte length prefix, so the real count can't exceed
    // `(comp.len() - 20) / 4`. Clamp the capacity hint to that (the loop below
    // still self-bounds by bailing on truncation) so a tiny blob claiming
    // ~4 billion blocks can't force a multi-GB allocation.
    let mut ranges = Vec::with_capacity(n_blocks.min(comp.len().saturating_sub(20) / 4));
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

    // Deterministic expected output length per block (from the validated
    // invariant): every block is `block_size` except the last (the remainder).
    // `block_size * (n_blocks - 1) <= raw_len` holds, so no underflow.
    let expected = |i: usize| -> usize {
        if i + 1 < n_blocks {
            block_size
        } else {
            raw_len - block_size * (n_blocks - 1)
        }
    };
    let decode = |(i, (a, b)): (usize, &(usize, usize))| {
        zstd_decode_bounded(&comp[*a..*b], expected(i))
            .with_context(|| format!("zstd decompress block {i}"))
    };
    #[cfg(feature = "parallel")]
    let pieces: Result<Vec<Vec<u8>>> = if parallel && n_blocks > 1 {
        use rayon::prelude::*;
        ranges.par_iter().enumerate().map(decode).collect()
    } else {
        ranges.iter().enumerate().map(decode).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let pieces: Result<Vec<Vec<u8>>> = {
        let _ = parallel;
        ranges.iter().enumerate().map(decode).collect()
    };
    let pieces = pieces?;
    // Size the output from the bytes we actually decoded, not the untrusted
    // `raw_len` header (a hostile blob could set `raw_len` to ~u64::MAX to
    // force a giant allocation); `raw_len` is still validated below.
    let out_len: usize = pieces.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(out_len);
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
    let block_size = u32::from_le_bytes(comp[4..8].try_into().unwrap()) as usize;
    let raw_len = u64::from_le_bytes(comp[8..16].try_into().unwrap()) as usize;
    let n_blocks = u32::from_le_bytes(comp[16..20].try_into().unwrap()) as usize;
    if block_index >= n_blocks {
        bail!("block_index {block_index} out of range ({n_blocks})");
    }
    // the block's decoded size, from the header (as `decode_zstd_blocks` checks it)
    let expected = if block_index + 1 < n_blocks {
        block_size
    } else {
        block_size
            .checked_mul(n_blocks - 1)
            .and_then(|full| raw_len.checked_sub(full))
            .context("warm blob header inconsistent")?
    };
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
            return zstd_decode_bounded(&comp[pos..pos + clen], expected);
        }
        pos += clen;
    }
    unreachable!()
}

#[cfg(feature = "write")]
pub fn encode_zstd(raw: &[u8]) -> Result<Vec<u8>> {
    z_encode_all(raw, 3)
}

pub fn decode_zstd(comp: &[u8]) -> Result<Vec<u8>> {
    z_decode_all(comp)
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

#[cfg(feature = "write")]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Warm ZBLK roundtrip still works after the bounded-decode hardening —
    /// incl. a partial trailing block (2 full blocks + a 100-byte remainder),
    /// via both the parallel and serial decode paths.
    #[test]
    fn warm_roundtrip_multiblock() {
        let raw: Vec<u8> = (0..(256 * 2 + 100)).map(|i| (i % 251) as u8).collect();
        let enc = encode_zstd_blocks(&raw, 256).unwrap();
        assert_eq!(decode_zstd_blocks_parallel(&enc).unwrap(), raw);
        assert_eq!(decode_zstd_blocks(&enc).unwrap(), raw);
    }

    /// A tampered header must be rejected WITHOUT allocating the declared size:
    /// a `raw_len` bomb (ratio/invariant guard) and an inconsistent `n_blocks`.
    #[test]
    fn warm_rejects_bomb_and_tampered_headers() {
        let raw: Vec<u8> = (0..1000u32).map(|i| (i % 7) as u8).collect();
        let enc = encode_zstd_blocks(&raw, 256).unwrap();

        // raw_len (bytes 8..16) → u64::MAX: must error, not try to allocate.
        let mut bomb = enc.clone();
        bomb[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(decode_zstd_blocks(&bomb).is_err());

        // n_blocks (bytes 16..20) inconsistent with raw_len/block_size.
        let mut bad_nb = enc.clone();
        bad_nb[16..20].copy_from_slice(&9999u32.to_le_bytes());
        assert!(decode_zstd_blocks(&bad_nb).is_err());
    }
}
