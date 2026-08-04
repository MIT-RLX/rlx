// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Block-quantized K/V cache — store decode-time history as `q8_0`,
//! `q4_0`, or `q5_0` GGUF-encoded blocks instead of f32/f16. Memory
//! saving vs f16 is roughly:
//!
//! | scheme | bits/elem | ratio vs f16 |
//! |--------|-----------|--------------|
//! | f16    | 16        | 1.0×         |
//! | q8_0   | 8.5       | 0.53×        |
//! | q5_0   | 5.5       | 0.34×        |
//! | q4_0   | 4.5       | 0.28×        |
//!
//! Trade-off: quantization adds noise to attention scores. q8_0 is
//! near-lossless for most decoder LMs; q4_0 typically costs ~0.3 ppl
//! at 4× memory savings.
//!
//! Layout per layer
//! ----------------
//!
//! Each layer's K and V buffer is a flat `Vec<u8>` of `past_len`
//! quantized rows. Every "row" is `kv_dim` f32 elements when
//! dequantized; rows are stored back-to-back. `kv_dim` must be a
//! multiple of the scheme's block size (32 for all three schemes).
//!
//! On read, callers materialize a window of rows to f32 via
//! [`dequant_rows`]. On write, freshly produced f32 K/V is quantized
//! one row at a time via [`quant_rows`] before being appended. The
//! quantization wrappers route to the `rlx_gguf::quantize` /
//! `dequant_*` kernels for parity with on-disk GGUF blocks.

use anyhow::{Result, anyhow, bail};
use rlx_gguf::{GgmlType, quantize};

/// Quantization scheme for cache rows. Restricted to the three
/// q-formats whose blocks are 32 elements wide and stable across
/// llama.cpp versions. The K-quants (Q4_K etc.) require 256-element
/// blocks, which doesn't compose cleanly with typical kv_dim values
/// (e.g. 128 head dim) so we don't expose them here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuant {
    /// `f16` — lossless storage of f32→f16 (no quantization). Kept as a
    /// useful baseline; 2 bytes per element.
    F16,
    Q8_0,
    Q4_0,
    Q5_0,
}

impl KvQuant {
    /// On-disk block size in elements.
    pub const fn block_elements(self) -> usize {
        match self {
            Self::F16 => 1,
            Self::Q8_0 | Self::Q4_0 | Self::Q5_0 => 32,
        }
    }

    /// On-disk block size in bytes.
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::Q8_0 => 2 + 32,
            Self::Q4_0 => 2 + 32 / 2,
            Self::Q5_0 => 2 + 4 + 32 / 2,
        }
    }

    fn ggml_type(self) -> Option<GgmlType> {
        match self {
            Self::F16 => None, // direct f16 path
            Self::Q8_0 => Some(GgmlType::Q8_0),
            Self::Q4_0 => Some(GgmlType::Q4_0),
            Self::Q5_0 => Some(GgmlType::Q5_0),
        }
    }

    /// Bytes required to store `n_elements` quantized.
    pub fn bytes_for(self, n_elements: usize) -> Result<usize> {
        let blk = self.block_elements();
        if !n_elements.is_multiple_of(blk) {
            bail!("{self:?}: element count {n_elements} not aligned to block size {blk}");
        }
        Ok((n_elements / blk) * self.block_bytes())
    }
}

/// One layer's quantized K/V buffers.
///
/// Rows are appended back-to-back. `past_len` is the number of *logical*
/// rows currently stored; the byte buffers carry `past_len × kv_dim`
/// elements' worth of quantized bytes.
#[derive(Debug, Clone)]
pub struct QuantizedKvLayer {
    pub k: Vec<u8>,
    pub v: Vec<u8>,
    pub past_len: usize,
    pub kv_dim: usize,
    pub scheme: KvQuant,
}

impl QuantizedKvLayer {
    pub fn new(kv_dim: usize, scheme: KvQuant) -> Result<Self> {
        let blk = scheme.block_elements();
        if !kv_dim.is_multiple_of(blk) {
            bail!("kv_dim ({kv_dim}) must be a multiple of {scheme:?} block size ({blk})");
        }
        Ok(Self {
            k: Vec::new(),
            v: Vec::new(),
            past_len: 0,
            kv_dim,
            scheme,
        })
    }

    /// Append `rows` worth of K and V (f32, row-major) — quantizing
    /// each row independently. Caller passes interleaved K then V
    /// blocks via separate slices.
    pub fn append_rows(&mut self, k_rows: &[f32], v_rows: &[f32]) -> Result<()> {
        if k_rows.len() != v_rows.len() {
            bail!(
                "append_rows: k len {} != v len {}",
                k_rows.len(),
                v_rows.len()
            );
        }
        if !k_rows.len().is_multiple_of(self.kv_dim) {
            bail!(
                "append_rows: byte count {} not aligned to kv_dim {}",
                k_rows.len(),
                self.kv_dim
            );
        }
        let n_rows = k_rows.len() / self.kv_dim;
        let k_bytes = quant_rows(k_rows, self.scheme)?;
        let v_bytes = quant_rows(v_rows, self.scheme)?;
        self.k.extend_from_slice(&k_bytes);
        self.v.extend_from_slice(&v_bytes);
        self.past_len += n_rows;
        Ok(())
    }

    /// Dequantize all stored rows back to f32 (K, V).
    pub fn read_all(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        let k = dequant_rows(&self.k, self.scheme, self.past_len * self.kv_dim)?;
        let v = dequant_rows(&self.v, self.scheme, self.past_len * self.kv_dim)?;
        Ok((k, v))
    }

    /// Dequantize `count` rows starting at row `start` (K, V). Lets a caller
    /// stream the cache a row (or small block) at a time — peak extra memory is
    /// `count × kv_dim` floats, not the whole past.
    pub fn read_rows(&self, start: usize, count: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        if start + count > self.past_len {
            bail!(
                "read_rows: [{start}, {}) out of range (past_len {})",
                start + count,
                self.past_len
            );
        }
        let blk = self.scheme.block_elements();
        let bytes_per_row = (self.kv_dim / blk) * self.scheme.block_bytes();
        let start_byte = start * bytes_per_row;
        let n = count * self.kv_dim;
        let k = dequant_rows(&self.k[start_byte..], self.scheme, n)?;
        let v = dequant_rows(&self.v[start_byte..], self.scheme, n)?;
        Ok((k, v))
    }

    /// Dequantize the last `window` rows (or all rows if past_len ≤ window).
    pub fn read_window(&self, window: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        if window >= self.past_len {
            return self.read_all();
        }
        self.read_rows(self.past_len - window, window)
    }

    /// Drop the oldest `n_rows` from this layer (sliding window).
    pub fn drop_front(&mut self, n_rows: usize) -> Result<()> {
        let n_rows = n_rows.min(self.past_len);
        if n_rows == 0 {
            return Ok(());
        }
        let blk = self.scheme.block_elements();
        let blocks_per_row = self.kv_dim / blk;
        let drop_bytes = n_rows * blocks_per_row * self.scheme.block_bytes();
        self.k.drain(..drop_bytes);
        self.v.drain(..drop_bytes);
        self.past_len -= n_rows;
        Ok(())
    }

    /// Memory used by both buffers (bytes).
    pub fn bytes(&self) -> usize {
        self.k.len() + self.v.len()
    }
}

/// Single-query attention directly over a **quantized** K/V layer, dequantizing
/// one row at a time so peak extra memory is `O(kv_dim)`, not `O(past_len ×
/// kv_dim)` — the point of quantized-KV attention vs the dequantize-the-whole-
/// cache path. GQA-aware: `n_heads` query heads share `kv_heads` K/V heads
/// (`group = n_heads / kv_heads`).
///
/// - `q`: the new token's query, `[n_heads × head_dim]`.
/// - returns the attention output `[n_heads × head_dim]`.
///
/// This is the host reference for a future first-class `QuantizedAttention` op;
/// it lets long-context decode keep the cache small without ever materializing
/// the full f32 history.
pub fn attend_quantized(
    q: &[f32],
    layer: &QuantizedKvLayer,
    n_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    let kv_dim = kv_heads * head_dim;
    if layer.kv_dim != kv_dim {
        bail!(
            "attend_quantized: layer kv_dim {} != {kv_heads}×{head_dim}",
            layer.kv_dim
        );
    }
    if q.len() != n_heads * head_dim {
        bail!(
            "attend_quantized: q len {} != {n_heads}×{head_dim}",
            q.len()
        );
    }
    let mut out = vec![0f32; n_heads * head_dim];
    let past = layer.past_len;
    if past == 0 {
        return Ok(out);
    }
    let group = (n_heads / kv_heads.max(1)).max(1);

    // Pass 1: stream K row-by-row, accumulate per-head scores.
    let mut scores = vec![0f32; n_heads * past];
    for p in 0..past {
        let (k_row, _) = layer.read_rows(p, 1)?;
        for h in 0..n_heads {
            let kvh = h / group;
            let q_h = &q[h * head_dim..(h + 1) * head_dim];
            let k_h = &k_row[kvh * head_dim..(kvh + 1) * head_dim];
            let dot: f32 = q_h.iter().zip(k_h).map(|(a, b)| a * b).sum();
            scores[h * past + p] = dot * scale;
        }
    }

    // Softmax per head over the `past` keys (numerically stable).
    for h in 0..n_heads {
        let row = &mut scores[h * past..(h + 1) * past];
        let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let mut sum = 0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum.max(1e-20);
        for v in row.iter_mut() {
            *v *= inv;
        }
    }

    // Pass 2: stream V row-by-row, accumulate weighted output.
    for p in 0..past {
        let (_, v_row) = layer.read_rows(p, 1)?;
        for h in 0..n_heads {
            let kvh = h / group;
            let w = scores[h * past + p];
            let v_h = &v_row[kvh * head_dim..(kvh + 1) * head_dim];
            let o_h = &mut out[h * head_dim..(h + 1) * head_dim];
            for (o, &vv) in o_h.iter_mut().zip(v_h) {
                *o += w * vv;
            }
        }
    }
    Ok(out)
}

/// All layers of a quantized KV cache.
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    pub layers: Vec<QuantizedKvLayer>,
}

impl QuantizedKvCache {
    pub fn new(n_layers: usize, kv_dim: usize, scheme: KvQuant) -> Result<Self> {
        let layers = (0..n_layers)
            .map(|_| QuantizedKvLayer::new(kv_dim, scheme))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn past_len(&self) -> usize {
        self.layers.first().map(|l| l.past_len).unwrap_or(0)
    }

    /// Total bytes across all layers.
    pub fn bytes(&self) -> usize {
        self.layers.iter().map(|l| l.bytes()).sum()
    }
}

// ─── quant / dequant entry points ────────────────────────────────────

fn quant_rows(values: &[f32], scheme: KvQuant) -> Result<Vec<u8>> {
    match scheme {
        KvQuant::F16 => {
            let mut out = Vec::with_capacity(values.len() * 2);
            for &v in values {
                let h = half::f16::from_f32(v);
                out.extend_from_slice(&h.to_le_bytes());
            }
            Ok(out)
        }
        scheme => {
            let ty = scheme
                .ggml_type()
                .ok_or_else(|| anyhow!("internal: missing ggml type for {scheme:?}"))?;
            Ok(quantize(values, ty)?)
        }
    }
}

fn dequant_rows(bytes: &[u8], scheme: KvQuant, n: usize) -> Result<Vec<f32>> {
    match scheme {
        KvQuant::F16 => {
            if bytes.len() < n * 2 {
                bail!("F16 dequant: {} bytes < {} expected", bytes.len(), n * 2);
            }
            let mut out = Vec::with_capacity(n);
            for chunk in bytes[..n * 2].chunks_exact(2) {
                let h = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(h.to_f32());
            }
            Ok(out)
        }
        KvQuant::Q8_0 => {
            let expected = scheme.bytes_for(n)?;
            Ok(rlx_gguf::dequant_q8_0(&bytes[..expected], n)?)
        }
        KvQuant::Q4_0 => {
            let expected = scheme.bytes_for(n)?;
            Ok(rlx_gguf::dequant_q4_0(&bytes[..expected], n)?)
        }
        KvQuant::Q5_0 => {
            // Q5_0 doesn't have a top-level `dequant_q5_0` export — call the
            // private path via dequant_f32 by building a one-shot GgufFile?
            // Cleaner: use the per-block helper exposed indirectly through
            // bytes_for + a hand-written loop. For now we use the public
            // `dequant_q8_0`-style path which is exposed; Q5_0 needs the
            // same. Until the gguf crate exposes a public `dequant_q5_0`,
            // route through the same block-by-block decoder lifted from
            // ggml-quants.c.
            decode_q5_0(bytes, n)
        }
    }
}

fn decode_q5_0(bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    const QK5_0: usize = 32;
    let blk_bytes = 2 + 4 + QK5_0 / 2;
    if !n.is_multiple_of(QK5_0) {
        bail!("Q5_0: n={n} not divisible by {QK5_0}");
    }
    let nb = n / QK5_0;
    if bytes.len() < nb * blk_bytes {
        bail!(
            "Q5_0: expected {} bytes, got {}",
            nb * blk_bytes,
            bytes.len()
        );
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..nb {
        let off = i * blk_bytes;
        let d = half::f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
        let qh = u32::from_le_bytes([
            bytes[off + 2],
            bytes[off + 3],
            bytes[off + 4],
            bytes[off + 5],
        ]);
        let qs = &bytes[off + 6..off + 6 + QK5_0 / 2];
        for j in 0..QK5_0 / 2 {
            let xh0 = (((qh >> j) & 1) as u8) << 4;
            let v0 = ((qs[j] & 0x0F) | xh0) as i32 - 16;
            out.push(d * v0 as f32);
        }
        for j in 0..QK5_0 / 2 {
            let xh1 = (((qh >> (j + 16)) & 1) as u8) << 4;
            let v1 = ((qs[j] >> 4) | xh1) as i32 - 16;
            out.push(d * v1 as f32);
        }
    }
    Ok(out)
}

// ─── mmap-backed storage (feature = "mmap-kv") ───────────────────────
//
// Memory-mapped storage trades a Vec<u8> for a file-backed (or anonymous)
// `MmapMut` so the OS pages quantized blocks in/out on demand. Two
// use cases:
//
// 1. **Long contexts** — KV history for 100k-token decode runs can
//    exceed RAM. With mmap, the kernel evicts cold pages to swap or
//    the backing file; reactivating them is a page fault, not a
//    user-space read().
//
// 2. **Zero-copy GPU upload** — paged memory is friendlier to
//    `cudaHostRegister` / Metal `MTLBuffer::newBufferWithBytesNoCopy`,
//    which can pin and DMA without an extra staging copy.
//
// Append semantics are emulated: we pre-allocate `capacity_rows`
// worth of blocks and track a write head, then `set_len` on the file
// when finalizing. For "grow as you go" the caller passes
// `capacity_rows = max_seq_len`.

#[cfg(feature = "mmap-kv")]
pub mod mmap {
    use super::*;
    use memmap2::{MmapMut, MmapOptions};
    use std::fs::OpenOptions;
    use std::path::{Path, PathBuf};

    /// File-backed quantized K/V buffer. K and V share one mapping —
    /// V follows K in the file. Disk layout: `[K bytes][V bytes]`.
    pub struct MmapKvLayer {
        pub mmap: MmapMut,
        pub past_len: usize,
        pub capacity_rows: usize,
        pub kv_dim: usize,
        pub scheme: KvQuant,
        pub bytes_per_row: usize,
        pub k_offset: usize,
        pub v_offset: usize,
        pub path: Option<PathBuf>,
        /// Retained fd for the `pread` read path (file-backed maps only).
        file: Option<std::fs::File>,
        /// Use positional `read_at` (pread) instead of mmap-slice for `read_rows`
        /// (opt-in via `RLX_KVSTORE_PREAD`; falls back to mmap for anonymous maps or
        /// non-Unix). Reads become explicit bounded syscalls into a fresh buffer —
        /// predictable latency, no reliance on the mapping, and the seam an
        /// `io_uring`-batched cold-read backend would plug into on Linux. Writes
        /// still go through the mmap; the shared page cache keeps the two coherent.
        pread: bool,
    }

    impl MmapKvLayer {
        /// File-backed mapping. Creates / truncates `path` to
        /// `2 × capacity_rows × bytes_per_row` and maps it RW.
        pub fn open<P: AsRef<Path>>(
            path: P,
            kv_dim: usize,
            scheme: KvQuant,
            capacity_rows: usize,
        ) -> Result<Self> {
            let blk = scheme.block_elements();
            if !kv_dim.is_multiple_of(blk) {
                bail!("kv_dim ({kv_dim}) must be a multiple of {scheme:?} block size ({blk})");
            }
            let bytes_per_row = (kv_dim / blk) * scheme.block_bytes();
            let total = 2 * capacity_rows * bytes_per_row;
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&path)?;
            file.set_len(total as u64)?;
            let mmap = unsafe { MmapOptions::new().len(total).map_mut(&file)? };
            Ok(Self {
                mmap,
                past_len: 0,
                capacity_rows,
                kv_dim,
                scheme,
                bytes_per_row,
                k_offset: 0,
                v_offset: capacity_rows * bytes_per_row,
                path: Some(path.as_ref().to_path_buf()),
                file: Some(file),
                pread: std::env::var_os("RLX_KVSTORE_PREAD").is_some(),
            })
        }

        /// Anonymous (private) mapping. Lives in swap-backed pages;
        /// not persisted. Use when you want OS-level paging without
        /// keeping a file around.
        pub fn anonymous(kv_dim: usize, scheme: KvQuant, capacity_rows: usize) -> Result<Self> {
            let blk = scheme.block_elements();
            if !kv_dim.is_multiple_of(blk) {
                bail!("kv_dim ({kv_dim}) must be a multiple of {scheme:?} block size ({blk})");
            }
            let bytes_per_row = (kv_dim / blk) * scheme.block_bytes();
            let total = 2 * capacity_rows * bytes_per_row;
            let mmap = MmapOptions::new().len(total).map_anon()?;
            Ok(Self {
                mmap,
                past_len: 0,
                capacity_rows,
                kv_dim,
                scheme,
                bytes_per_row,
                k_offset: 0,
                v_offset: capacity_rows * bytes_per_row,
                path: None,
                file: None,
                pread: false, // no fd for anonymous maps → always mmap-slice reads
            })
        }

        /// Append `n_rows` × `kv_dim` worth of K and V floats (one row
        /// at a time, quantized in place).
        pub fn append_rows(&mut self, k_rows: &[f32], v_rows: &[f32]) -> Result<()> {
            if k_rows.len() != v_rows.len() {
                bail!("append_rows: k/v length mismatch");
            }
            if !k_rows.len().is_multiple_of(self.kv_dim) {
                bail!("append_rows: byte count not aligned to kv_dim");
            }
            let n_rows = k_rows.len() / self.kv_dim;
            if self.past_len + n_rows > self.capacity_rows {
                bail!(
                    "append_rows: would exceed capacity ({} + {} > {})",
                    self.past_len,
                    n_rows,
                    self.capacity_rows
                );
            }
            let kb = quant_rows(k_rows, self.scheme)?;
            let vb = quant_rows(v_rows, self.scheme)?;
            let k_start = self.k_offset + self.past_len * self.bytes_per_row;
            let v_start = self.v_offset + self.past_len * self.bytes_per_row;
            self.mmap[k_start..k_start + kb.len()].copy_from_slice(&kb);
            self.mmap[v_start..v_start + vb.len()].copy_from_slice(&vb);
            self.past_len += n_rows;
            Ok(())
        }

        /// Dequantize all stored rows. Reads zero-copy from the page
        /// cache — only touched pages are faulted in.
        pub fn read_all(&self) -> Result<(Vec<f32>, Vec<f32>)> {
            let n = self.past_len * self.kv_dim;
            let k_end = self.k_offset + self.past_len * self.bytes_per_row;
            let v_end = self.v_offset + self.past_len * self.bytes_per_row;
            let k = dequant_rows(&self.mmap[self.k_offset..k_end], self.scheme, n)?;
            let v = dequant_rows(&self.mmap[self.v_offset..v_end], self.scheme, n)?;
            Ok((k, v))
        }

        /// Read the last `window` rows (page-fault-lazy on dequant).
        pub fn read_window(&self, window: usize) -> Result<(Vec<f32>, Vec<f32>)> {
            let window = window.min(self.past_len);
            let start_row = self.past_len - window;
            let n = window * self.kv_dim;
            let k_start = self.k_offset + start_row * self.bytes_per_row;
            let v_start = self.v_offset + start_row * self.bytes_per_row;
            let k_end = k_start + window * self.bytes_per_row;
            let v_end = v_start + window * self.bytes_per_row;
            let k = dequant_rows(&self.mmap[k_start..k_end], self.scheme, n)?;
            let v = dequant_rows(&self.mmap[v_start..v_end], self.scheme, n)?;
            Ok((k, v))
        }

        /// Read an arbitrary row range `[start_row, start_row+n_rows)` — the
        /// random-access primitive for block retrieval (only the touched pages
        /// fault in from disk). Rows outside `past_len` are clamped away.
        pub fn read_rows(&self, start_row: usize, n_rows: usize) -> Result<(Vec<f32>, Vec<f32>)> {
            let start_row = start_row.min(self.past_len);
            let n_rows = n_rows.min(self.past_len - start_row);
            let n = n_rows * self.kv_dim;
            let nbytes = n_rows * self.bytes_per_row;
            let k_start = self.k_offset + start_row * self.bytes_per_row;
            let v_start = self.v_offset + start_row * self.bytes_per_row;
            // Positional-read path (opt-in): bounded `pread` into fresh buffers
            // rather than slicing the mapping. Writes go through the (MAP_SHARED)
            // mmap, so the shared page cache keeps pread coherent with them.
            #[cfg(unix)]
            if self.pread {
                if let Some(f) = self.file.as_ref() {
                    use std::os::unix::fs::FileExt;
                    let mut kb = vec![0u8; nbytes];
                    let mut vb = vec![0u8; nbytes];
                    f.read_exact_at(&mut kb, k_start as u64)?;
                    f.read_exact_at(&mut vb, v_start as u64)?;
                    let k = dequant_rows(&kb, self.scheme, n)?;
                    let v = dequant_rows(&vb, self.scheme, n)?;
                    return Ok((k, v));
                }
            }
            let k_end = k_start + nbytes;
            let v_end = v_start + nbytes;
            let k = dequant_rows(&self.mmap[k_start..k_end], self.scheme, n)?;
            let v = dequant_rows(&self.mmap[v_start..v_end], self.scheme, n)?;
            Ok((k, v))
        }

        /// Prefetch an arbitrary row range (madvise WILLNEED) so the retrieved
        /// blocks' pages are warm before the synchronous read. Best-effort.
        pub fn prefetch_rows(&self, start_row: usize, n_rows: usize) {
            let start_row = start_row.min(self.past_len);
            let n_rows = n_rows.min(self.past_len - start_row);
            if n_rows == 0 {
                return;
            }
            let k_start = self.k_offset + start_row * self.bytes_per_row;
            let v_start = self.v_offset + start_row * self.bytes_per_row;
            #[cfg(unix)]
            {
                let len = n_rows * self.bytes_per_row;
                let _ = self
                    .mmap
                    .advise_range(memmap2::Advice::WillNeed, k_start, len);
                let _ = self
                    .mmap
                    .advise_range(memmap2::Advice::WillNeed, v_start, len);
            }
            #[cfg(not(unix))]
            {
                let _ = (k_start, v_start);
            }
        }

        /// Hint the kernel that we're about to read `window` rows
        /// linearly — prefetches pages into the page cache (madvise
        /// WILLNEED on supported platforms). Best-effort: failures are
        /// logged but don't propagate.
        pub fn prefetch_window(&self, window: usize) {
            let window = window.min(self.past_len);
            if window == 0 {
                return;
            }
            let start_row = self.past_len - window;
            let k_start = self.k_offset + start_row * self.bytes_per_row;
            let v_start = self.v_offset + start_row * self.bytes_per_row;
            // memmap2::Advice / advise_range are Unix-only; no-op on Windows.
            #[cfg(unix)]
            {
                let _ = self.mmap.advise_range(
                    memmap2::Advice::WillNeed,
                    k_start,
                    window * self.bytes_per_row,
                );
                let _ = self.mmap.advise_range(
                    memmap2::Advice::WillNeed,
                    v_start,
                    window * self.bytes_per_row,
                );
            }
            #[cfg(not(unix))]
            {
                let _ = (k_start, v_start);
            }
        }

        /// Advise the kernel this mapping is accessed **randomly**, disabling the
        /// sequential read-ahead that otherwise faults in neighbor pages a
        /// scattered block-retrieval read never uses. Opt-in (not baked into
        /// `open`) so sequential users — `read_window` / `read_all` on the decode
        /// KV path — keep their read-ahead. Best-effort, Unix-only.
        pub fn advise_random(&self) {
            #[cfg(unix)]
            {
                let _ = self.mmap.advise(memmap2::Advice::Random);
            }
        }

        /// Force the positional-read (`pread`) path on/off, overriding the
        /// `RLX_KVSTORE_PREAD` env default. No-op → mmap for anonymous maps (no fd).
        pub fn set_pread(&mut self, on: bool) {
            self.pread = on && self.file.is_some();
        }

        /// Persist any dirty pages to the backing file. No-op for
        /// anonymous mappings.
        pub fn flush(&self) -> Result<()> {
            self.mmap.flush()?;
            Ok(())
        }

        pub fn bytes(&self) -> usize {
            2 * self.past_len * self.bytes_per_row
        }
    }

    /// All-layer mmap-backed KV cache.
    pub struct MmapKvCache {
        pub layers: Vec<MmapKvLayer>,
    }

    impl MmapKvCache {
        /// One file per layer under `dir`, named `kv_{i}.bin`.
        pub fn open_dir<P: AsRef<Path>>(
            dir: P,
            n_layers: usize,
            kv_dim: usize,
            scheme: KvQuant,
            capacity_rows: usize,
        ) -> Result<Self> {
            let dir = dir.as_ref();
            std::fs::create_dir_all(dir)?;
            let layers = (0..n_layers)
                .map(|i| {
                    MmapKvLayer::open(
                        dir.join(format!("kv_{i}.bin")),
                        kv_dim,
                        scheme,
                        capacity_rows,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Self { layers })
        }

        pub fn anonymous(
            n_layers: usize,
            kv_dim: usize,
            scheme: KvQuant,
            capacity_rows: usize,
        ) -> Result<Self> {
            let layers = (0..n_layers)
                .map(|_| MmapKvLayer::anonymous(kv_dim, scheme, capacity_rows))
                .collect::<Result<Vec<_>>>()?;
            Ok(Self { layers })
        }

        pub fn n_layers(&self) -> usize {
            self.layers.len()
        }

        pub fn past_len(&self) -> usize {
            self.layers.first().map(|l| l.past_len).unwrap_or(0)
        }

        /// Total bytes currently in use across all layers.
        pub fn bytes(&self) -> usize {
            self.layers.iter().map(|l| l.bytes()).sum()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn anonymous_q8_0_roundtrip() {
            let kv_dim = 64;
            let mut layer = MmapKvLayer::anonymous(kv_dim, KvQuant::Q8_0, 4).unwrap();
            let data: Vec<f32> = (0..kv_dim).map(|i| (i as f32).sin()).collect();
            layer.append_rows(&data, &data).unwrap();
            let (k, v) = layer.read_all().unwrap();
            assert_eq!(k.len(), kv_dim);
            assert_eq!(v.len(), kv_dim);
            // Q8_0 is high-fidelity.
            for (a, b) in k.iter().zip(data.iter()) {
                assert!((a - b).abs() < 0.02);
            }
        }

        #[test]
        fn file_backed_persists_and_reopens() {
            let dir = tempfile::tempdir().unwrap();
            let kv_dim = 32;
            let path = dir.path().join("layer.bin");
            {
                let mut layer = MmapKvLayer::open(&path, kv_dim, KvQuant::F16, 8).unwrap();
                let data: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.5).collect();
                layer.append_rows(&data, &data).unwrap();
                layer.flush().unwrap();
            }
            // Re-open and verify K bytes are present (we know offset 0..len).
            let bytes = std::fs::read(&path).unwrap();
            assert!(!bytes.is_empty());
            assert!(bytes.iter().any(|&b| b != 0));
        }

        #[test]
        fn append_past_capacity_errors() {
            let mut l = MmapKvLayer::anonymous(32, KvQuant::Q8_0, 2).unwrap();
            let row = vec![0.5f32; 32];
            l.append_rows(&row, &row).unwrap();
            l.append_rows(&row, &row).unwrap();
            assert!(l.append_rows(&row, &row).is_err());
        }

        // Item 3: the opt-in positional-read (pread) path must return bit-identical
        // rows to the mmap-slice path (writes go through the mmap; the shared page
        // cache keeps pread coherent).
        #[cfg(unix)]
        #[test]
        fn pread_matches_mmap_read() {
            let dir = tempfile::tempdir().unwrap();
            let kv_dim = 32;
            let path = dir.path().join("layer.bin");
            let mut layer = MmapKvLayer::open(&path, kv_dim, KvQuant::Q8_0, 64).unwrap();
            let k: Vec<f32> = (0..kv_dim * 4).map(|i| (i as f32) * 0.01 - 1.0).collect();
            let v: Vec<f32> = (0..kv_dim * 4).map(|i| (i as f32) * -0.02 + 0.5).collect();
            layer.append_rows(&k, &v).unwrap();
            // mmap-slice read (default).
            layer.set_pread(false);
            let (km, vm) = layer.read_rows(1, 2).unwrap();
            // pread read of the same range.
            layer.set_pread(true);
            let (kp, vp) = layer.read_rows(1, 2).unwrap();
            assert_eq!(km.len(), kp.len());
            assert_eq!(vm.len(), vp.len());
            assert!(
                km.iter().zip(&kp).all(|(a, b)| (a - b).abs() < 1e-6),
                "K mmap != pread"
            );
            assert!(
                vm.iter().zip(&vp).all(|(a, b)| (a - b).abs() < 1e-6),
                "V mmap != pread"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    #[test]
    fn block_size_invariants() {
        assert_eq!(KvQuant::F16.block_bytes(), 2);
        assert_eq!(KvQuant::Q8_0.block_bytes(), 34);
        assert_eq!(KvQuant::Q4_0.block_bytes(), 18);
        assert_eq!(KvQuant::Q5_0.block_bytes(), 22);
    }

    #[test]
    fn f16_roundtrip_exact() {
        let kv_dim = 64;
        let mut layer = QuantizedKvLayer::new(kv_dim, KvQuant::F16).unwrap();
        let k_row: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * 0.1).collect();
        let v_row: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * 0.2).collect();
        layer.append_rows(&k_row, &v_row).unwrap();
        let (k, v) = layer.read_all().unwrap();
        for i in 0..kv_dim {
            // f16 round-trip is bounded ~1e-3 relative for small magnitudes.
            assert!((k[i] - k_row[i]).abs() < 0.01);
            assert!((v[i] - v_row[i]).abs() < 0.01);
        }
    }

    #[test]
    fn q8_0_roundtrip_high_fidelity() {
        let kv_dim = 64;
        let n_rows = 4;
        let mut layer = QuantizedKvLayer::new(kv_dim, KvQuant::Q8_0).unwrap();
        let total = n_rows * kv_dim;
        let k_data: Vec<f32> = (0..total).map(|i| (i as f32).sin()).collect();
        let v_data: Vec<f32> = (0..total).map(|i| (i as f32).cos()).collect();
        layer.append_rows(&k_data, &v_data).unwrap();
        assert_eq!(layer.past_len, n_rows);
        let (k, v) = layer.read_all().unwrap();
        assert!(cosine(&k, &k_data) > 0.999, "Q8_0 K cosine too low");
        assert!(cosine(&v, &v_data) > 0.999, "Q8_0 V cosine too low");
    }

    #[test]
    fn q4_0_roundtrip_lossy_but_close() {
        let kv_dim = 64;
        let mut layer = QuantizedKvLayer::new(kv_dim, KvQuant::Q4_0).unwrap();
        let k: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.05).tanh()).collect();
        let v: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.07).tanh()).collect();
        layer.append_rows(&k, &v).unwrap();
        let (kr, vr) = layer.read_all().unwrap();
        assert!(cosine(&kr, &k) > 0.99);
        assert!(cosine(&vr, &v) > 0.99);
    }

    #[test]
    fn q5_0_roundtrip_better_than_q4() {
        let kv_dim = 64;
        let mut q4 = QuantizedKvLayer::new(kv_dim, KvQuant::Q4_0).unwrap();
        let mut q5 = QuantizedKvLayer::new(kv_dim, KvQuant::Q5_0).unwrap();
        let k: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.1).sin() * 3.0).collect();
        let v: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.13).cos() * 3.0).collect();
        q4.append_rows(&k, &v).unwrap();
        q5.append_rows(&k, &v).unwrap();
        let (k4, _) = q4.read_all().unwrap();
        let (k5, _) = q5.read_all().unwrap();
        let cos4 = cosine(&k4, &k);
        let cos5 = cosine(&k5, &k);
        assert!(cos5 >= cos4 - 1e-3, "Q5_0 should not be worse than Q4_0");
    }

    #[test]
    fn sliding_window_drops_oldest() {
        let kv_dim = 32;
        let mut layer = QuantizedKvLayer::new(kv_dim, KvQuant::Q8_0).unwrap();
        for r in 0..5 {
            let v: Vec<f32> = (0..kv_dim).map(|i| (i + r * 100) as f32).collect();
            layer.append_rows(&v, &v).unwrap();
        }
        assert_eq!(layer.past_len, 5);
        layer.drop_front(2).unwrap();
        assert_eq!(layer.past_len, 3);
        let (k, _v) = layer.read_window(3).unwrap();
        // First kept row is original row 2 → starts with value 200.
        assert!((k[0] - 200.0).abs() < 1.0);
    }

    #[test]
    fn kv_dim_must_align_to_block_size() {
        // kv_dim=24 < 32 → not aligned for Q8_0/Q4_0/Q5_0.
        assert!(QuantizedKvLayer::new(24, KvQuant::Q8_0).is_err());
        assert!(QuantizedKvLayer::new(24, KvQuant::Q4_0).is_err());
        // f16 has block 1 so any dim works.
        assert!(QuantizedKvLayer::new(24, KvQuant::F16).is_ok());
    }

    #[test]
    fn streaming_quantized_attention_matches_full_dequant() {
        let (n_heads, kv_heads, head_dim) = (4usize, 2usize, 32usize);
        let kv_dim = kv_heads * head_dim; // 64 — multiple of 32 for Q8_0
        let past = 5usize;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut layer = QuantizedKvLayer::new(kv_dim, KvQuant::Q8_0).unwrap();
        for p in 0..past {
            let k: Vec<f32> = (0..kv_dim)
                .map(|i| ((i + p * 7) as f32 * 0.05).sin())
                .collect();
            let v: Vec<f32> = (0..kv_dim)
                .map(|i| ((i + p * 3) as f32 * 0.04).cos())
                .collect();
            layer.append_rows(&k, &v).unwrap();
        }
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| (i as f32 * 0.02).sin())
            .collect();

        let out = attend_quantized(&q, &layer, n_heads, kv_heads, head_dim, scale).unwrap();

        // Reference: dequantize the whole cache, run standard attention.
        let (k_all, v_all) = layer.read_all().unwrap();
        let group = n_heads / kv_heads;
        let mut reference = vec![0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let kvh = h / group;
            let mut sc = vec![0f32; past];
            for p in 0..past {
                let qh = &q[h * head_dim..(h + 1) * head_dim];
                let base = p * kv_dim + kvh * head_dim;
                let kh = &k_all[base..base + head_dim];
                let dot: f32 = qh.iter().zip(kh).map(|(a, b)| a * b).sum();
                sc[p] = dot * scale;
            }
            let max = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for x in sc.iter_mut() {
                *x = (*x - max).exp();
                sum += *x;
            }
            for x in sc.iter_mut() {
                *x /= sum;
            }
            for p in 0..past {
                let base = p * kv_dim + kvh * head_dim;
                let vh = &v_all[base..base + head_dim];
                for d in 0..head_dim {
                    reference[h * head_dim + d] += sc[p] * vh[d];
                }
            }
        }
        for (a, b) in out.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-4, "streaming {a} vs full-dequant {b}");
        }
    }

    #[test]
    fn cache_memory_decreases_with_quantization() {
        let kv_dim = 128;
        let n_layers = 4;
        let n_rows = 16;
        let data: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * 0.01).collect();
        let mut f16 = QuantizedKvCache::new(n_layers, kv_dim, KvQuant::F16).unwrap();
        let mut q8 = QuantizedKvCache::new(n_layers, kv_dim, KvQuant::Q8_0).unwrap();
        let mut q4 = QuantizedKvCache::new(n_layers, kv_dim, KvQuant::Q4_0).unwrap();
        for _ in 0..n_rows {
            for l in 0..n_layers {
                f16.layers[l].append_rows(&data, &data).unwrap();
                q8.layers[l].append_rows(&data, &data).unwrap();
                q4.layers[l].append_rows(&data, &data).unwrap();
            }
        }
        assert!(q8.bytes() < f16.bytes());
        assert!(q4.bytes() < q8.bytes());
    }
}
