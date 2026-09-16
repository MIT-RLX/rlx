// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Disk-backed MoE expert paging.**
//!
//! The other half of [`crate::experts`]. That module spreads experts across
//! *machines* and keeps them resident; this one keeps them on *local storage*
//! and reads only what a token fires. Which is right depends on what you have:
//! expert-parallel needs enough aggregate RAM in the cluster, paging needs
//! enough disk bandwidth on one box.
//!
//! Paging is what the planner assumes whenever it reports a stage as `paged`
//! ([`crate::cluster::Assignment::experts_paged`]), and its cost model — the
//! `per_layer_expert_active_bytes / io_mbps` term in `stage_secs` — is exactly
//! the arithmetic here: per token, per layer, read `top_k` of `n_routed`
//! experts.
//!
//! ```text
//!   router fires ids [3, 41, 190, ...]  (top_k of n_routed)
//!            │
//!   ExpertPager::gather(layer, bank, ids)
//!            │   cache hit  → memcpy from the LRU
//!            │   cache miss → pread the expert's slice of the bank
//!            ▼
//!   [top_k * bytes_per_expert] contiguous, in FIRED order
//!            │
//!   GroupedMatMul over a top_k-expert bank, not an n_routed-expert one
//! ```
//!
//! The gathered buffer is laid out exactly like the full bank it came from, so
//! it is a drop-in for a `GroupedMatMul` / `DequantGroupedMatMul` operand with
//! `num_experts = ids.len()`. Bytes stay **packed**: paging exists to avoid
//! materializing an f32 bank, so dequantizing on the way through would defeat
//! it (at 2 bits/weight, 16x).
//!
//! ## Why a byte budget rather than an entry count
//!
//! Experts are not all the same size — a `ffn_gate_exps` slice and a
//! `ffn_down_exps` slice differ, and quantization differs between banks. A
//! cache bounded by entries would hold wildly varying amounts of memory on
//! different models, which is not something a placement plan can reason about.
//! The budget here is the number the planner hands over.

use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// A registered bank, resolved to a dense handle.
///
/// Lets a caller hoist the `(layer, name)` lookup out of a per-expert loop, and
/// makes the cache key `Copy` so neither a hit nor an LRU move allocates.
///
/// Measured, not assumed: at GLM-5.3-Flash decode shape (42 layers x 3 banks x
/// 8 experts = ~1000 lookups per token) resolving by handle instead of by name
/// is **0.95x** — no faster, within noise. The map-key allocation is nothing
/// beside the bytes the gather moves, which run at memory bandwidth. Use the
/// handle because it is the clearer API and because it removes an allocation
/// from a hot path, not for throughput.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BankIx(usize);

/// Where an expert bank's bytes live, and how they are cut into experts.
///
/// `bytes_per_expert` is the bank's byte length divided by its expert count.
/// That is exact for the block-quantized formats this targets *provided* each
/// expert's rows are a whole number of quant blocks — true for GGUF K-quants,
/// whose block is 256 elements and whose `in_dim` is a multiple of it.
/// [`ExpertPager::register`] checks it rather than trusting it, because getting
/// it wrong reads a misaligned window and silently produces garbage weights.
#[derive(Debug, Clone)]
pub struct BankLocation {
    /// Checkpoint file holding the bank.
    pub path: PathBuf,
    /// Absolute byte offset of expert 0 within that file.
    pub offset: u64,
    pub num_experts: usize,
    pub bytes_per_expert: usize,
}

impl BankLocation {
    /// Absolute file offset of one expert's slice.
    fn expert_offset(&self, id: usize) -> u64 {
        self.offset + (id as u64) * (self.bytes_per_expert as u64)
    }
    /// Total bytes of the whole bank.
    pub fn total_bytes(&self) -> u64 {
        (self.num_experts as u64) * (self.bytes_per_expert as u64)
    }
}

/// What paging cost, in the terms the planner priced it in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagerStats {
    /// Expert reads served from RAM.
    pub hits: u64,
    /// Expert reads that went to disk.
    pub misses: u64,
    /// Bytes actually read from storage.
    pub bytes_read: u64,
    /// Bytes served from the cache (what the disk did *not* have to move).
    pub bytes_served: u64,
    /// Experts dropped to stay inside the budget.
    pub evictions: u64,
}

impl PagerStats {
    /// Share of expert reads served without touching disk, in `0..=1`.
    ///
    /// The number that decides whether paging is viable for a given model: a
    /// router that spreads uniformly over 288 experts has almost no reuse to
    /// exploit and this stays near zero however large the budget, whereas a
    /// peaked router turns most of the IO term into memcpy.
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 {
            return 0.0;
        }
        self.hits as f64 / n as f64
    }
}

/// One cached expert.
struct Entry {
    bytes: Arc<Vec<u8>>,
    /// Logical clock of the last touch; the key of `lru` below.
    used: u64,
}

struct Inner {
    /// Cached experts by `(bank, expert)`. The key is `Copy`, so neither a
    /// lookup nor an LRU move allocates.
    resident: HashMap<(BankIx, usize), Entry>,
    /// Touch order, oldest first. A `BTreeMap` rather than a queue so a touch
    /// is O(log n) instead of an O(n) scan for the key to move.
    lru: BTreeMap<u64, (BankIx, usize)>,
    /// Monotonic touch counter.
    clock: u64,
    /// Bytes currently held.
    bytes: u64,
    /// Open file handles, so a per-token gather does not re-`open` per expert.
    files: HashMap<PathBuf, Arc<File>>,
    stats: PagerStats,
}

/// A byte-budgeted, disk-backed cache of MoE experts.
///
/// Model-agnostic: it knows byte ranges, not architectures. A model registers
/// where its banks live and asks for the experts a token fired.
///
/// Cheap to share — `gather` takes `&self`, so several decode threads can page
/// against one cache.
pub struct ExpertPager {
    /// Registered banks, indexed by [`BankIx`].
    locs: Vec<BankLocation>,
    /// `layer -> bank name -> handle`. Nested rather than keyed by a
    /// `(usize, String)` tuple so a lookup can borrow the name instead of
    /// allocating one.
    by_name: HashMap<usize, HashMap<String, BankIx>>,
    budget_bytes: u64,
    inner: Mutex<Inner>,
}

impl ExpertPager {
    /// A pager holding at most `budget_bytes` of expert data.
    ///
    /// A budget of 0 disables caching: every read goes to disk. That is a
    /// legitimate configuration (it is what the planner's cost model assumes,
    /// being the pessimistic case) rather than an error.
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            locs: Vec::new(),
            by_name: HashMap::new(),
            budget_bytes,
            inner: Mutex::new(Inner {
                resident: HashMap::new(),
                lru: BTreeMap::new(),
                clock: 0,
                bytes: 0,
                files: HashMap::new(),
                stats: PagerStats::default(),
            }),
        }
    }

    /// Tell the pager where one bank lives.
    ///
    /// Rejects a location whose bytes do not divide evenly into its experts:
    /// that means `bytes_per_expert` was computed from a wrong expert count or
    /// a wrong tensor, and every read after the first would be misaligned —
    /// which does not fail, it returns plausible-looking garbage.
    pub fn register(
        &mut self,
        layer: usize,
        bank: impl Into<String>,
        loc: BankLocation,
    ) -> Result<BankIx> {
        if loc.num_experts == 0 {
            bail!("expert bank has no experts");
        }
        if loc.bytes_per_expert == 0 {
            bail!("expert bank has zero-length experts");
        }
        let bank = bank.into();
        if let Ok(meta) = std::fs::metadata(&loc.path) {
            let end = loc.offset + loc.total_bytes();
            if end > meta.len() {
                bail!(
                    "bank {bank} of layer {layer} runs to byte {end} but {} is only {} bytes",
                    loc.path.display(),
                    meta.len()
                );
            }
        }
        // Re-registering replaces the location. The handle stays valid so any
        // caller holding one keeps working, but the experts cached under it now
        // describe the old bytes and have to go.
        if let Some(&ix) = self.by_name.get(&layer).and_then(|m| m.get(&bank)) {
            self.locs[ix.0] = loc;
            self.forget(ix);
            return Ok(ix);
        }
        let ix = BankIx(self.locs.len());
        self.locs.push(loc);
        self.by_name.entry(layer).or_default().insert(bank, ix);
        Ok(ix)
    }

    /// Handle for a registered bank, or `None` if it was never registered.
    ///
    /// Hoist this out of a per-token loop: the handle is `Copy`, and every
    /// lookup taking one is allocation-free.
    pub fn bank_ix(&self, layer: usize, bank: &str) -> Option<BankIx> {
        self.by_name.get(&layer)?.get(bank).copied()
    }

    /// Where a bank lives.
    pub fn location(&self, ix: BankIx) -> &BankLocation {
        &self.locs[ix.0]
    }

    /// Drop every cached expert of one bank.
    fn forget(&mut self, ix: BankIx) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let doomed: Vec<(BankIx, usize)> = inner
            .resident
            .keys()
            .filter(|(b, _)| *b == ix)
            .copied()
            .collect();
        for k in doomed {
            if let Some(e) = inner.resident.remove(&k) {
                inner.bytes -= e.bytes.len() as u64;
                inner.lru.remove(&e.used);
            }
        }
    }

    /// Banks registered so far.
    pub fn bank_count(&self) -> usize {
        self.locs.len()
    }

    /// Bytes the whole registered checkpoint would occupy if held resident —
    /// what paging is avoiding.
    pub fn total_bank_bytes(&self) -> u64 {
        self.locs.iter().map(|l| l.total_bytes()).sum()
    }

    pub fn stats(&self) -> PagerStats {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).stats
    }

    /// Bytes currently cached.
    pub fn resident_bytes(&self) -> u64 {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).bytes
    }

    /// One expert's packed bytes, from cache or disk.
    ///
    /// Resolves the bank by name each call; [`Self::expert_at`] skips that.
    pub fn expert(&self, layer: usize, bank: &str, id: usize) -> Result<Arc<Vec<u8>>> {
        let ix = self
            .bank_ix(layer, bank)
            .with_context(|| format!("no expert bank registered for layer {layer} `{bank}`"))?;
        self.expert_at(ix, id)
    }

    /// One expert's packed bytes, by handle.
    pub fn expert_at(&self, ix: BankIx, id: usize) -> Result<Arc<Vec<u8>>> {
        let loc = self
            .locs
            .get(ix.0)
            .with_context(|| format!("bank handle {ix:?} is not registered"))?;
        if id >= loc.num_experts {
            bail!(
                "expert {id} out of range for bank {ix:?} ({} experts)",
                loc.num_experts
            );
        }
        let ck = (ix, id);

        // Phase 1: under the lock, look for a hit and take the file handle.
        let file = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(bytes) = inner.touch(&ck) {
                return Ok(bytes);
            }
            inner.open(&loc.path)?
        };

        // Phase 2: read WITHOUT the lock.
        //
        // Holding it across the `pread` would serialize every concurrent gather
        // on one disk read, in the one module whose whole purpose is IO
        // throughput — a pager that can only ever have one request in flight
        // gets none of the queue depth an NVMe needs to reach its rated speed,
        // and `stage_secs` prices paging at that rated speed.
        let mut buf = vec![0u8; loc.bytes_per_expert];
        read_exact_at(&file, &mut buf, loc.expert_offset(id)).with_context(|| {
            format!(
                "reading expert {id} of bank {ix:?} from {}",
                loc.path.display()
            )
        })?;

        // Phase 3: re-acquire and publish.
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.stats.misses += 1;
        inner.stats.bytes_read += buf.len() as u64;
        // Another thread may have admitted the same expert while this one read
        // it. Both copies are identical, so prefer the resident one and drop
        // this read rather than holding the expert twice.
        if let Some(bytes) = inner.touch(&ck) {
            return Ok(bytes);
        }
        let bytes = Arc::new(buf);
        inner.admit(ck, bytes.clone(), self.budget_bytes);
        Ok(bytes)
    }

    /// The fired experts' bytes, concatenated in the order given.
    ///
    /// Laid out exactly like the full bank, so the result is a `GroupedMatMul`
    /// operand with `num_experts = ids.len()`. Order is the caller's: it has to
    /// match the routing weights the caller will apply, so this must not sort
    /// or deduplicate. Repeating an id is legal and copies it twice.
    pub fn gather(&self, layer: usize, bank: &str, ids: &[usize]) -> Result<Vec<u8>> {
        let ix = self
            .bank_ix(layer, bank)
            .with_context(|| format!("no expert bank registered for layer {layer} `{bank}`"))?;
        self.gather_at(ix, ids)
    }

    /// [`Self::gather`] by handle — the name is resolved once, not per expert.
    pub fn gather_at(&self, ix: BankIx, ids: &[usize]) -> Result<Vec<u8>> {
        let per = self
            .locs
            .get(ix.0)
            .with_context(|| format!("bank handle {ix:?} is not registered"))?
            .bytes_per_expert;
        let mut out = Vec::with_capacity(ids.len() * per);
        for &id in ids {
            out.extend_from_slice(&self.expert_at(ix, id)?);
        }
        Ok(out)
    }

    /// Pull experts into the cache ahead of needing them.
    ///
    /// Useful when the router for step `t+1` is known while step `t` is still
    /// computing. Errors are swallowed: a prefetch that fails costs a later
    /// cache miss, and turning a speculative read into a hard failure would be
    /// worse than the miss.
    pub fn prefetch(&self, layer: usize, bank: &str, ids: &[usize]) {
        for &id in ids {
            let _ = self.expert(layer, bank, id);
        }
    }

    /// Drop everything cached, keeping the registrations and the statistics.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.resident.clear();
        inner.lru.clear();
        inner.bytes = 0;
    }
}

impl Inner {
    /// Serve `key` from the cache, moving it to the most-recently-used end.
    ///
    /// Returns `None` on a miss, having changed nothing.
    fn touch(&mut self, key: &(BankIx, usize)) -> Option<Arc<Vec<u8>>> {
        let old = self.resident.get(key)?.used;
        self.lru.remove(&old);
        self.clock += 1;
        let now = self.clock;
        self.lru.insert(now, *key);
        let e = self.resident.get_mut(key)?;
        e.used = now;
        let bytes = e.bytes.clone();
        self.stats.hits += 1;
        self.stats.bytes_served += bytes.len() as u64;
        Some(bytes)
    }

    fn open(&mut self, path: &Path) -> Result<Arc<File>> {
        if let Some(f) = self.files.get(path) {
            return Ok(f.clone());
        }
        let f = Arc::new(
            File::open(path).with_context(|| format!("opening checkpoint {}", path.display()))?,
        );
        self.files.insert(path.to_path_buf(), f.clone());
        Ok(f)
    }

    /// Insert an expert, evicting least-recently-used ones to stay in budget.
    fn admit(&mut self, key: (BankIx, usize), bytes: Arc<Vec<u8>>, budget: u64) {
        let len = bytes.len() as u64;
        // An expert larger than the whole budget is not cacheable; serving it
        // straight through beats evicting everything to hold one item.
        if len > budget {
            return;
        }
        while self.bytes + len > budget {
            let Some((&oldest, _)) = self.lru.iter().next() else {
                break;
            };
            let Some(victim) = self.lru.remove(&oldest) else {
                break;
            };
            if let Some(e) = self.resident.remove(&victim) {
                self.bytes -= e.bytes.len() as u64;
                self.stats.evictions += 1;
            }
        }
        self.clock += 1;
        let now = self.clock;
        self.lru.insert(now, key);
        self.bytes += len;
        self.resident.insert(key, Entry { bytes, used: now });
    }
}

/// Positional read that does not disturb a shared file cursor.
fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0usize;
        while done < buf.len() {
            let n = f.seek_read(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "short read paging an expert",
                ));
            }
            done += n;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = f.try_clone()?;
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(buf)
    }
}

/// Register every routed-expert bank in a GGUF checkpoint.
///
/// Reads only the header — the point of paging is never to load the tensor
/// data, and a 93 GB checkpoint's index is a few hundred KB.
///
/// A bank is any tensor under a `blk.{i}.` prefix whose name ends in one of
/// `bank_suffixes` (for `glm5next`: `ffn_gate_exps.weight`, `ffn_up_exps.weight`,
/// `ffn_down_exps.weight`). GGUF stores dimensions innermost-first, so the
/// expert count is the LAST entry of `shape` — `[4096, 2048, 8]` is 8 experts of
/// a 4096x2048 projection, not 4096 of anything.
///
/// Returns the number of banks registered, and the key each was registered
/// under is `(block_index, suffix_without_the_dot_weight)`.
#[cfg(feature = "gguf")]
pub fn register_gguf_expert_banks(
    pager: &mut ExpertPager,
    path: impl AsRef<Path>,
    bank_suffixes: &[&str],
) -> Result<usize> {
    let path = path.as_ref();
    let g = rlx_gguf::GgufFile::header_from_path(path)
        .with_context(|| format!("reading GGUF header of {}", path.display()))?;
    let base = g.data_offset();
    let mut n = 0usize;

    for t in g.tensors.values() {
        let Some(rest) = t.name.strip_prefix("blk.") else {
            continue;
        };
        let Some((idx, tail)) = rest.split_once('.') else {
            continue;
        };
        let Ok(layer) = idx.parse::<usize>() else {
            continue;
        };
        let Some(&suffix) = bank_suffixes.iter().find(|s| tail == **s) else {
            continue;
        };

        let num_experts = *t.shape.last().unwrap_or(&0);
        if num_experts == 0 {
            bail!("{}: expert bank with an empty shape", t.name);
        }
        let total = rlx_gguf::bytes_for_public(t.dtype, t.n_elements()).ok_or_else(|| {
            anyhow::anyhow!("{}: no byte size known for dtype {:?}", t.name, t.dtype)
        })?;
        if !total.is_multiple_of(num_experts) {
            // Block-quantized rows that do not divide evenly by the expert
            // count mean an expert's slice is not block-aligned, so every read
            // past the first would start mid-block and dequantize to noise.
            bail!(
                "{}: {total} bytes do not divide into {num_experts} experts;                  the expert slices are not block-aligned and cannot be paged                  independently",
                t.name
            );
        }
        let bank = suffix.strip_suffix(".weight").unwrap_or(suffix);
        pager.register(
            layer,
            bank,
            BankLocation {
                path: path.to_path_buf(),
                offset: base + t.offset,
                num_experts,
                bytes_per_expert: total / num_experts,
            },
        )?;
        n += 1;
    }
    Ok(n)
}

/// The routed-expert bank names a `glm5next` / DeepSeek-style GGUF uses.
pub const GGUF_MOE_BANKS: [&str; 3] = [
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A checkpoint whose expert `i` of bank `b` is the byte `(b*100 + i)`
    /// repeated — so a misread window is visible in the content, not just in a
    /// length.
    fn checkpoint(dir: &str, banks: usize, experts: usize, per: usize) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rlx-pager-{dir}"));
        let _ = std::fs::create_dir_all(&d);
        let p = d.join("ckpt.bin");
        let mut f = File::create(&p).unwrap();
        for b in 0..banks {
            for i in 0..experts {
                f.write_all(&vec![(b * 100 + i) as u8; per]).unwrap();
            }
        }
        f.flush().unwrap();
        p
    }

    fn loc(path: &Path, bank: usize, experts: usize, per: usize) -> BankLocation {
        BankLocation {
            path: path.to_path_buf(),
            offset: (bank * experts * per) as u64,
            num_experts: experts,
            bytes_per_expert: per,
        }
    }

    #[test]
    fn reads_the_right_expert_window() {
        let p = checkpoint("windows", 3, 8, 64);
        let mut pager = ExpertPager::new(1 << 20);
        for b in 0..3 {
            pager
                .register(b, format!("bank{b}"), loc(&p, b, 8, 64))
                .unwrap();
        }
        for b in 0..3 {
            for i in 0..8 {
                let got = pager.expert(b, &format!("bank{b}"), i).unwrap();
                assert_eq!(got.len(), 64);
                assert!(
                    got.iter().all(|&x| x == (b * 100 + i) as u8),
                    "layer {b} expert {i} read the wrong window: {:?}",
                    &got[..8]
                );
            }
        }
    }

    /// A gather is the fired experts back to back, in the caller's order — the
    /// layout a GroupedMatMul operand needs. Order is not the pager's to
    /// choose: it has to line up with the routing weights, so sorting or
    /// deduplicating would silently pair each token with the wrong gate.
    #[test]
    fn gather_concatenates_in_fired_order() {
        let p = checkpoint("gather", 1, 16, 32);
        let mut pager = ExpertPager::new(1 << 20);
        pager.register(0, "exps", loc(&p, 0, 16, 32)).unwrap();

        let ids = [7usize, 2, 15, 2];
        let g = pager.gather(0, "exps", &ids).unwrap();
        assert_eq!(g.len(), ids.len() * 32);
        for (slot, &id) in ids.iter().enumerate() {
            let w = &g[slot * 32..(slot + 1) * 32];
            assert!(
                w.iter().all(|&x| x == id as u8),
                "slot {slot} should hold expert {id}, holds {:?}",
                &w[..4]
            );
        }
    }

    /// The budget is a hard ceiling. Exceeding it is how a paged stage OOMs the
    /// node it was planned onto.
    #[test]
    fn the_byte_budget_is_never_exceeded() {
        let p = checkpoint("budget", 1, 64, 1024);
        let budget = 8 * 1024u64;
        let mut pager = ExpertPager::new(budget);
        pager.register(0, "exps", loc(&p, 0, 64, 1024)).unwrap();

        for i in 0..64 {
            pager.expert(0, "exps", i).unwrap();
            assert!(
                pager.resident_bytes() <= budget,
                "after {i} experts the cache holds {} bytes against a {budget} budget",
                pager.resident_bytes()
            );
        }
        assert!(pager.stats().evictions > 0, "nothing was ever evicted");
    }

    /// Eviction must drop the LEAST-RECENTLY-USED expert. Evicting anything
    /// else still respects the budget while destroying the hit rate, which is
    /// the only reason the cache exists.
    #[test]
    fn eviction_is_least_recently_used() {
        let p = checkpoint("lru", 1, 8, 100);
        // Exactly three experts fit.
        let mut pager = ExpertPager::new(300);
        pager.register(0, "exps", loc(&p, 0, 8, 100)).unwrap();

        for i in 0..3 {
            pager.expert(0, "exps", i).unwrap();
        }
        // Touch 0 and 2, leaving 1 as the oldest.
        pager.expert(0, "exps", 0).unwrap();
        pager.expert(0, "exps", 2).unwrap();

        let before = pager.stats();
        pager.expert(0, "exps", 3).unwrap(); // forces one eviction
        assert_eq!(pager.stats().evictions, before.evictions + 1);

        // 1 should be gone; 0 and 2 should still be hits.
        let mid = pager.stats();
        pager.expert(0, "exps", 0).unwrap();
        pager.expert(0, "exps", 2).unwrap();
        assert_eq!(
            pager.stats().hits,
            mid.hits + 2,
            "the recently-used experts were evicted instead of the stale one"
        );
        let mid = pager.stats();
        pager.expert(0, "exps", 1).unwrap();
        assert_eq!(
            pager.stats().misses,
            mid.misses + 1,
            "expert 1 was the least-recently-used and should have been evicted"
        );
    }

    /// Re-firing an expert must not re-read it. This is the entire economic
    /// case for the cache, and it is what `hit_rate` reports to the planner.
    #[test]
    fn a_refired_expert_is_served_from_ram() {
        let p = checkpoint("reuse", 1, 8, 256);
        let mut pager = ExpertPager::new(1 << 20);
        pager.register(0, "exps", loc(&p, 0, 8, 256)).unwrap();

        pager.gather(0, "exps", &[1, 3, 5]).unwrap();
        let after_first = pager.stats();
        assert_eq!(after_first.misses, 3);
        assert_eq!(after_first.bytes_read, 3 * 256);

        pager.gather(0, "exps", &[1, 3, 5]).unwrap();
        let s = pager.stats();
        assert_eq!(s.misses, 3, "the second token re-read from disk");
        assert_eq!(s.bytes_read, 3 * 256, "no new bytes should have been read");
        assert_eq!(s.hits, 3);
        assert!((s.hit_rate() - 0.5).abs() < 1e-9, "{}", s.hit_rate());
    }

    /// A zero budget is a valid configuration — no cache, every read from disk.
    /// It is also exactly what the planner's `stage_secs` assumes, so it must
    /// work rather than divide by zero or hold one entry anyway.
    #[test]
    fn a_zero_budget_pages_everything() {
        let p = checkpoint("nocache", 1, 4, 128);
        let mut pager = ExpertPager::new(0);
        pager.register(0, "exps", loc(&p, 0, 4, 128)).unwrap();

        for _ in 0..3 {
            pager.gather(0, "exps", &[0, 1]).unwrap();
        }
        let s = pager.stats();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 6);
        assert_eq!(pager.resident_bytes(), 0);
    }

    /// A bank whose declared extent runs past the file is a wrong expert count
    /// or a wrong tensor. Caught at registration, because the reads it produces
    /// do not fail — they return whatever bytes happen to be there.
    #[test]
    fn a_bank_that_overruns_the_file_is_rejected() {
        let p = checkpoint("overrun", 1, 4, 64);
        let mut pager = ExpertPager::new(1 << 20);
        let err = pager
            .register(
                0,
                "exps",
                BankLocation {
                    path: p.clone(),
                    offset: 0,
                    num_experts: 4,
                    bytes_per_expert: 1024, // 4 KB claimed, 256 B present
                },
            )
            .expect_err("an overrunning bank must be rejected");
        assert!(err.to_string().contains("bytes"), "{err}");
    }

    /// The cache lock must be free while a read is in flight.
    ///
    /// The read used to happen with the mutex held, which let exactly one
    /// request be in flight at a time — in the module whose only job is IO
    /// throughput, and against a cost model (`stage_secs`) that prices paging
    /// at the disk's rated speed, which an NVMe reaches only with queue depth.
    /// Nothing else here would notice: every correctness test passes perfectly
    /// well single-threaded.
    ///
    /// Testing it by comparing concurrent against serial throughput does not
    /// work, and the reason is worth recording: once the checkpoint is in the
    /// page cache a "read" is a memcpy, so there is no IO to overlap and eight
    /// threads simply contend for memory bandwidth — concurrent came out
    /// *slower* than serial with the lock already fixed.
    ///
    /// What is actually claimed is narrower and directly observable: while one
    /// thread is inside a read, another can still take the lock. A reader takes
    /// a large expert while a second thread spins on [`ExpertPager::stats`],
    /// which needs the lock; if the read held it, the spinner makes no progress
    /// for the read's whole duration. The separation between the two outcomes
    /// is several orders of magnitude, not a percentage.
    #[test]
    fn the_cache_lock_is_free_while_a_read_is_in_flight() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        const BIG: usize = 64 << 20;
        let p = checkpoint("lockfree", 1, 1, BIG);
        // Budget 0: the read path runs every time, nothing is served from cache.
        let mut pager = ExpertPager::new(0);
        pager.register(0, "exps", loc(&p, 0, 1, BIG)).unwrap();
        let pager = Arc::new(pager);
        pager.expert(0, "exps", 0).unwrap(); // warm the page cache

        let done = Arc::new(AtomicBool::new(false));
        let spins = Arc::new(AtomicUsize::new(0));
        let spinner = {
            let (pager, done, spins) = (pager.clone(), done.clone(), spins.clone());
            std::thread::spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    // Takes the same mutex the read path uses.
                    let _ = pager.stats();
                    spins.fetch_add(1, Ordering::Relaxed);
                }
            })
        };

        // Long enough that the spinner would be starved for a clearly
        // measurable stretch if the lock were held.
        for _ in 0..4 {
            pager.expert(0, "exps", 0).unwrap();
        }
        done.store(true, Ordering::Relaxed);
        spinner.join().unwrap();

        let n = spins.load(Ordering::Relaxed);
        assert!(
            n > 1_000,
            "the spinner took the lock only {n} times across four 64 MB reads — \
             it is being blocked for the duration of each read, so the cache \
             mutex is held across the IO"
        );
        let _ = std::fs::remove_dir_all(p.parent().unwrap());
    }

    /// Two threads racing on the SAME expert must not leave two copies in the
    /// cache. Both reads return identical bytes, so the bug is invisible except
    /// as a budget the pager quietly exceeds.
    #[test]
    fn a_racing_double_read_admits_one_copy() {
        let per = 1 << 20;
        let p = checkpoint("race", 1, 2, per);
        let mut pager = ExpertPager::new(8 * per as u64);
        pager.register(0, "exps", loc(&p, 0, 2, per)).unwrap();
        let pager = Arc::new(pager);

        let barrier = Arc::new(std::sync::Barrier::new(4));
        let hs: Vec<_> = (0..4)
            .map(|_| {
                let pager = pager.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    pager.expert(0, "exps", 1).unwrap()
                })
            })
            .collect();
        let got: Vec<_> = hs.into_iter().map(|h| h.join().unwrap()).collect();

        for g in &got {
            assert!(g.iter().all(|&x| x == 1u8), "a racing read got wrong bytes");
        }
        assert_eq!(
            pager.resident_bytes(),
            per as u64,
            "the same expert is cached more than once"
        );
        let _ = std::fs::remove_dir_all(p.parent().unwrap());
    }

    /// The handle path and the name path must be the same path.
    #[test]
    fn a_handle_resolves_to_the_same_bytes_as_a_name() {
        let p = checkpoint("handles", 2, 6, 48);
        let mut pager = ExpertPager::new(1 << 20);
        let ix0 = pager.register(0, "bank0", loc(&p, 0, 6, 48)).unwrap();
        let ix1 = pager.register(1, "bank1", loc(&p, 1, 6, 48)).unwrap();
        assert_ne!(ix0, ix1, "distinct banks must get distinct handles");
        assert_eq!(pager.bank_ix(0, "bank0"), Some(ix0));
        assert_eq!(pager.bank_ix(1, "bank1"), Some(ix1));
        assert_eq!(pager.bank_ix(0, "bank1"), None, "handles are per layer");

        for id in 0..6 {
            assert_eq!(
                pager.expert(0, "bank0", id).unwrap(),
                pager.expert_at(ix0, id).unwrap()
            );
        }
        assert_eq!(
            pager.gather(1, "bank1", &[3, 0, 5]).unwrap(),
            pager.gather_at(ix1, &[3, 0, 5]).unwrap()
        );
    }

    /// Re-registering a bank must drop what was cached under it.
    ///
    /// Handles stay valid across a re-registration so a caller holding one
    /// keeps working — which is exactly what makes the stale-cache hazard real:
    /// the experts cached under that handle describe the OLD file, and serving
    /// them after the location changed returns the wrong weights without any
    /// error. Reloading a checkpoint in place is the obvious way to hit it.
    #[test]
    fn re_registering_a_bank_invalidates_its_cached_experts() {
        // Two files whose expert 0 differs: bank 0 is filled with 0, bank 1
        // with 100.
        let p = checkpoint("reregister", 2, 4, 64);
        let mut pager = ExpertPager::new(1 << 20);
        let ix = pager.register(0, "exps", loc(&p, 0, 4, 64)).unwrap();

        let before = pager.expert_at(ix, 0).unwrap();
        assert!(before.iter().all(|&x| x == 0u8));
        assert!(pager.resident_bytes() > 0);

        // Point the same bank at the second region.
        let again = pager.register(0, "exps", loc(&p, 1, 4, 64)).unwrap();
        assert_eq!(again, ix, "the handle should survive re-registration");
        assert_eq!(
            pager.resident_bytes(),
            0,
            "experts cached from the old location are still held"
        );

        let after = pager.expert_at(ix, 0).unwrap();
        assert!(
            after.iter().all(|&x| x == 100u8),
            "re-registered bank still serves the old file's bytes: {:?}",
            &after[..4]
        );
    }

    #[test]
    fn an_out_of_range_expert_is_an_error_not_a_bad_read() {
        let p = checkpoint("range", 1, 4, 64);
        let mut pager = ExpertPager::new(1 << 20);
        pager.register(0, "exps", loc(&p, 0, 4, 64)).unwrap();
        let err = pager
            .expert(0, "exps", 4)
            .expect_err("expert 4 of 4 does not exist");
        assert!(err.to_string().contains("out of range"), "{err}");
    }
}
