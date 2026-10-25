// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weight loading trait — implemented by model-builder `WeightLoader` adapters.
//!
//! Besides the trait, the pieces every loader adapter otherwise rewrites:
//! [`transpose_2d`], a non-consuming [`CloningSource`] over any
//! [`TensorLookup`] store (one checkpoint backing graphs rebuilt per sequence
//! length), and [`WeightSourceExt`] combinators — [`renamed`] /
//! [`with_prefix`] for checkpoint key layouts and [`or`] to fall back to a
//! second source. All of them pass packed (quantized) weights through.
//!
//! [`renamed`]: WeightSourceExt::renamed
//! [`with_prefix`]: WeightSourceExt::with_prefix
//! [`or`]: WeightSourceExt::or

use std::collections::HashMap;

use anyhow::Result;

use crate::{GgufPackedBank, GgufPackedLinear};

/// Abstract weight source for block emission. Keeps `rlx-flow` independent of
/// safetensors / GGUF file formats.
pub trait WeightSource {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)>;

    /// Optional probe for arch-specific key layout detection.
    fn has(&self, key: &str) -> bool {
        let _ = key;
        false
    }

    /// Optional packed (quantized) weight for the linear projection `key`.
    ///
    /// When this returns `Some`, [`crate::FlowCtx::linear`] emits a fused
    /// `DequantMatMul` over the U8 quant blob instead of loading an F32 weight —
    /// the *same* flow topology then runs packed at any `m` (m=1 decode, m=N
    /// prefill) with no F32 weight ever materialized. Default: `None`, so
    /// F32-only sources are unaffected. Any model whose loader can hand out
    /// GGUF/MLX quant blobs opts into packed matmuls just by overriding this —
    /// no per-model graph surgery.
    ///
    /// May be called more than once for the same `key` (KV-tap + decoder stages
    /// both load a projection); implementations should return the same weight
    /// each time. The flow dedups the underlying graph param, so the U8 bytes
    /// are registered only once.
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        let _ = key;
        Ok(None)
    }

    /// Whether [`Self::take_packed`] would hand out a packed weight for `key`,
    /// without taking it. A block with a dense layout of its own (BERT's host-
    /// fused QKV) asks this to pick its packed branch and leave the dense one
    /// byte-for-byte as it was. Default: `false`.
    fn has_packed(&self, key: &str) -> bool {
        let _ = key;
        false
    }

    /// Optional packed (quantized) **expert bank** for the MoE weight `key`.
    ///
    /// The grouped sibling of [`Self::take_packed`]. When this returns `Some`,
    /// a builder can emit `Op::DequantGroupedMatMul` over the quant blob
    /// instead of a `GroupedMatMul` over an F32 bank — which for a fine-grained
    /// MoE is most of the model: at 288 experts a single layer's three banks are
    /// gigabytes packed and tens of gigabytes as F32.
    ///
    /// Separate from `take_packed` because a bank is 3-D and
    /// [`GgufPackedLinear`] has nowhere to put the expert count; reading one
    /// through `take_packed` would silently report `out_dim = num_experts`.
    /// Default: `None`, so F32-only and linear-only sources are unaffected.
    fn take_packed_bank(&mut self, key: &str) -> Result<Option<GgufPackedBank>> {
        let _ = key;
        Ok(None)
    }

    /// Every key this source can serve, when it can list them (files and
    /// in-memory maps can; a source computing tensors on demand may not).
    /// Default: empty. Used by [`Tracked`] to report tensors a build never
    /// read, and for "closest key" hints on a miss.
    fn keys(&self) -> Vec<String> {
        Vec::new()
    }

    /// The float dtype `key` is stored in (F32 / F16 / BF16), when the source
    /// knows. `take` always widens to f32; this is how
    /// [`crate::WeightStorage::Auto`] learns it can keep a half tensor at
    /// 2 B/elem without loss. Default `None` (treated as f32).
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        let _ = key;
        None
    }

    /// The rank-2 tensor `key` as its raw half-precision bytes (little-endian,
    /// in the dtype [`Self::stored_dtype`] reports), transposed when asked:
    /// the values [`Self::take`] returns, without the f32 round trip. When the
    /// flow keeps the tensor at that width ([`crate::WeightStorage`]) it queues
    /// these bytes in `typed_params` rather than an f32 copy in `params`,
    /// which halves the host transient and skips the widen-then-narrow.
    ///
    /// Opt-in, because whoever reads `typed_params` must then upload each
    /// entry by its dtype rather than treat every entry as a quant blob
    /// (`rlx_core::flow_util::compile_built` does). Default `None`.
    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let _ = (key, transpose);
        Ok(None)
    }
}

impl<T: WeightSource + ?Sized> WeightSource for &mut T {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        (*self).take(key, transpose)
    }
    // Forward every provided method, not just `take`: a default left in place
    // here silently answers for the wrapped source — `false` from `has` breaks
    // key-layout probing, `None` from `take_packed` turns a packed model into
    // an F32 one with no error.
    fn has(&self, key: &str) -> bool {
        (**self).has(key)
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        (**self).take_packed(key)
    }
    fn has_packed(&self, key: &str) -> bool {
        (**self).has_packed(key)
    }
    fn take_packed_bank(&mut self, key: &str) -> Result<Option<GgufPackedBank>> {
        (**self).take_packed_bank(key)
    }
    fn keys(&self) -> Vec<String> {
        (**self).keys()
    }
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        (**self).stored_dtype(key)
    }
    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        (**self).take_half(key, transpose)
    }
}

impl<T: WeightSource + ?Sized> WeightSource for Box<T> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        (**self).take(key, transpose)
    }
    fn has(&self, key: &str) -> bool {
        (**self).has(key)
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        (**self).take_packed(key)
    }
    fn has_packed(&self, key: &str) -> bool {
        (**self).has_packed(key)
    }
    fn take_packed_bank(&mut self, key: &str) -> Result<Option<GgufPackedBank>> {
        (**self).take_packed_bank(key)
    }
    fn keys(&self) -> Vec<String> {
        (**self).keys()
    }
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        (**self).stored_dtype(key)
    }
    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        (**self).take_half(key, transpose)
    }
}

/// The "missing weight" error, with up to three keys of `available` that end
/// the same way as `key` — a checkpoint whose names carry an extra or a
/// different prefix (`model.` vs `language_model.model.`) then names the fix.
pub fn missing_weight<S: AsRef<str>>(key: &str, available: &[S]) -> anyhow::Error {
    let parts: Vec<&str> = key.rsplit('.').collect();
    let shared = |cand: &str| {
        cand.rsplit('.')
            .zip(&parts)
            .take_while(|(a, b)| a == *b)
            .count()
    };
    let mut close: Vec<(usize, &str)> = available
        .iter()
        .map(|c| c.as_ref())
        .filter(|c| *c != key)
        .map(|c| (shared(c), c))
        .filter(|&(n, _)| n >= 2.min(parts.len()))
        .collect();
    close.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(b.1)));
    if close.is_empty() {
        return anyhow::anyhow!("missing weight: {key}");
    }
    let names: Vec<&str> = close.iter().take(3).map(|&(_, c)| c).collect();
    anyhow::anyhow!("missing weight: {key} (closest: {})", names.join(", "))
}

/// `[rows, cols]` row-major → `[cols, rows]`: the transpose every
/// [`WeightSource::take`] applies when asked, for linear weights stored
/// `[out, in]` and consumed `[in, out]`. Rank ≠ 2 is an error naming `key`.
pub fn transpose_2d(data: &[f32], shape: &[usize], key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let &[rows, cols] = shape else {
        return Err(anyhow::anyhow!("transpose requires rank-2 weight: {key}"));
    };
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    Ok((out, vec![cols, rows]))
}

/// In-memory weight map for tests and tooling.
#[derive(Debug, Default, Clone)]
pub struct MapWeights {
    pub tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl MapWeights {
    pub fn insert(&mut self, key: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) {
        self.tensors.insert(key.into(), (data, shape));
    }
}

impl WeightSource for MapWeights {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self
            .tensors
            .remove(key)
            .ok_or_else(|| anyhow::anyhow!("missing weight: {key}"))?;
        if !transpose {
            return Ok((data, shape));
        }
        transpose_2d(&data, &shape, key)
    }

    fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.tensors.keys().cloned().collect();
        k.sort();
        k
    }
}

/// A store that can lend a tensor without giving it up — what a
/// [`CloningSource`] reads from. Implement it for a loader's in-memory map to
/// get a non-consuming [`WeightSource`] for free.
pub trait TensorLookup {
    /// `(data, shape)` for `key`, row-major f32.
    fn get(&self, key: &str) -> Option<(&[f32], &[usize])>;

    /// Every key, when the store can list them; see [`WeightSource::keys`].
    fn keys(&self) -> Vec<String> {
        Vec::new()
    }
}

impl TensorLookup for MapWeights {
    fn get(&self, key: &str) -> Option<(&[f32], &[usize])> {
        self.tensors
            .get(key)
            .map(|(d, s)| (d.as_slice(), s.as_slice()))
    }
    fn keys(&self) -> Vec<String> {
        WeightSource::keys(self)
    }
}

impl TensorLookup for HashMap<String, (Vec<f32>, Vec<usize>)> {
    fn get(&self, key: &str) -> Option<(&[f32], &[usize])> {
        HashMap::get(self, key).map(|(d, s)| (d.as_slice(), s.as_slice()))
    }
    fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = HashMap::keys(self).cloned().collect();
        k.sort();
        k
    }
}

/// Non-consuming [`WeightSource`] over a borrowed [`TensorLookup`]: every
/// `take` copies the tensor (transposed when asked), so one store backs any
/// number of graph builds — encoder and decoder reading the same tensors, or
/// a graph recompiled per sequence length — where the default destructive
/// `take` would empty it on the first build.
#[derive(Debug, Clone, Copy)]
pub struct CloningSource<'a, L: ?Sized>(pub &'a L);

impl<L: TensorLookup + ?Sized> WeightSource for CloningSource<'_, L> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape) = self
            .0
            .get(key)
            .ok_or_else(|| missing_weight(key, &self.0.keys()))?;
        if transpose {
            transpose_2d(data, shape, key)
        } else {
            Ok((data.to_vec(), shape.to_vec()))
        }
    }

    fn has(&self, key: &str) -> bool {
        self.0.get(key).is_some()
    }

    fn keys(&self) -> Vec<String> {
        self.0.keys()
    }
}

/// [`WeightSource`] whose keys are rewritten by `map` before they reach
/// `inner` — for a checkpoint whose names differ from the ones a builder asks
/// for (a `language_model.` prefix, GGUF vs Hugging Face names, one renamed
/// tensor). Built by [`WeightSourceExt::renamed`]. An arbitrary `map` cannot
/// be inverted, so it lists no [`keys`](WeightSource::keys); [`Prefixed`] can.
#[derive(Debug, Clone)]
pub struct Renamed<S, F> {
    inner: S,
    map: F,
}

impl<S: WeightSource, F: Fn(&str) -> String> WeightSource for Renamed<S, F> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        self.inner.take(&(self.map)(key), transpose)
    }
    fn has(&self, key: &str) -> bool {
        self.inner.has(&(self.map)(key))
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        self.inner.take_packed(&(self.map)(key))
    }
    fn has_packed(&self, key: &str) -> bool {
        self.inner.has_packed(&(self.map)(key))
    }
    fn take_packed_bank(&mut self, key: &str) -> Result<Option<GgufPackedBank>> {
        self.inner.take_packed_bank(&(self.map)(key))
    }
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        self.inner.stored_dtype(&(self.map)(key))
    }
    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let k = (self.map)(key);
        self.inner.take_half(&k, transpose)
    }
}

/// [`WeightSource`] that asks `primary` first and `fallback` for what
/// `primary` lacks — adapter or LoRA-merged tensors over a base checkpoint,
/// a tensor shipped in a side file. Built by [`WeightSourceExt::or`].
///
/// F32 `take` falls back when `primary` fails. A packed weight comes from
/// `fallback` only when `primary` does not [`has`](WeightSource::has) the
/// key, so a tensor `primary` holds as F32 is never swapped for `fallback`'s
/// packed copy.
#[derive(Debug, Clone)]
pub struct Fallback<A, B> {
    primary: A,
    fallback: B,
}

impl<A: WeightSource, B: WeightSource> WeightSource for Fallback<A, B> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        match self.primary.take(key, transpose) {
            Ok(t) => Ok(t),
            Err(first) => self
                .fallback
                .take(key, transpose)
                .map_err(|second| anyhow::anyhow!("{first}; fallback: {second}")),
        }
    }
    fn has(&self, key: &str) -> bool {
        self.primary.has(key) || self.fallback.has(key)
    }
    fn keys(&self) -> Vec<String> {
        let mut k = self.primary.keys();
        k.extend(self.fallback.keys());
        k.sort();
        k.dedup();
        k
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        match self.primary.take_packed(key)? {
            Some(p) => Ok(Some(p)),
            None if !self.primary.has(key) => self.fallback.take_packed(key),
            None => Ok(None),
        }
    }
    fn has_packed(&self, key: &str) -> bool {
        self.primary.has_packed(key) || (!self.primary.has(key) && self.fallback.has_packed(key))
    }
    fn take_packed_bank(&mut self, key: &str) -> Result<Option<GgufPackedBank>> {
        match self.primary.take_packed_bank(key)? {
            Some(p) => Ok(Some(p)),
            None if !self.primary.has(key) => self.fallback.take_packed_bank(key),
            None => Ok(None),
        }
    }
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        if self.primary.has(key) {
            self.primary.stored_dtype(key)
        } else {
            self.fallback.stored_dtype(key)
        }
    }
    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        if self.primary.has(key) {
            self.primary.take_half(key, transpose)
        } else {
            self.fallback.take_half(key, transpose)
        }
    }
}

/// [`WeightSource`] that looks every key up as `prefix + key`. Unlike a
/// [`Renamed`] closure it can list its [`keys`](WeightSource::keys): the
/// inner keys under `prefix`, with the prefix stripped. Built by
/// [`WeightSourceExt::with_prefix`].
#[derive(Debug, Clone)]
pub struct Prefixed<S> {
    inner: S,
    prefix: String,
}

impl<S: WeightSource> WeightSource for Prefixed<S> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let full = format!("{}{key}", self.prefix);
        if !self.inner.has(&full) && !self.inner.keys().is_empty() {
            return Err(missing_weight(key, &self.keys()));
        }
        self.inner.take(&full, transpose)
    }
    fn has(&self, key: &str) -> bool {
        self.inner.has(&format!("{}{key}", self.prefix))
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        self.inner.take_packed(&format!("{}{key}", self.prefix))
    }
    fn has_packed(&self, key: &str) -> bool {
        self.inner.has_packed(&format!("{}{key}", self.prefix))
    }
    fn take_packed_bank(&mut self, key: &str) -> Result<Option<GgufPackedBank>> {
        self.inner
            .take_packed_bank(&format!("{}{key}", self.prefix))
    }
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        self.inner.stored_dtype(&format!("{}{key}", self.prefix))
    }
    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let k = format!("{}{key}", self.prefix);
        self.inner.take_half(&k, transpose)
    }
    fn keys(&self) -> Vec<String> {
        self.inner
            .keys()
            .into_iter()
            .filter_map(|k| k.strip_prefix(self.prefix.as_str()).map(str::to_string))
            .collect()
    }
}

/// [`WeightSource`] serving one key from another when it is absent. Built by
/// [`WeightSourceExt::alias`].
#[derive(Debug, Clone)]
pub struct Aliased<S> {
    inner: S,
    from: String,
    to: String,
}

impl<S: WeightSource> Aliased<S> {
    fn resolve<'k>(&'k self, key: &'k str) -> &'k str {
        if key == self.from && !self.inner.has(key) {
            &self.to
        } else {
            key
        }
    }
}

impl<S: WeightSource> WeightSource for Aliased<S> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let k = self.resolve(key).to_string();
        self.inner.take(&k, transpose)
    }
    fn has(&self, key: &str) -> bool {
        self.inner.has(self.resolve(key))
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        let k = self.resolve(key).to_string();
        self.inner.take_packed(&k)
    }
    fn has_packed(&self, key: &str) -> bool {
        self.inner.has_packed(self.resolve(key))
    }
    fn take_packed_bank(&mut self, key: &str) -> Result<Option<GgufPackedBank>> {
        let k = self.resolve(key).to_string();
        self.inner.take_packed_bank(&k)
    }
    fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        self.inner.stored_dtype(self.resolve(key))
    }
    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let k = self.resolve(key).to_string();
        self.inner.take_half(&k, transpose)
    }
}

/// [`WeightSource`] transforming selected F32 tensors. Built by
/// [`WeightSourceExt::map_tensors`].
pub struct Mapped<S, P, F> {
    inner: S,
    select: P,
    f: F,
}

impl<S, P, F> WeightSource for Mapped<S, P, F>
where
    S: WeightSource,
    P: Fn(&str) -> bool,
    F: FnMut(&str, Vec<f32>, Vec<usize>) -> Result<(Vec<f32>, Vec<usize>)>,
{
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        if !(self.select)(key) {
            return self.inner.take(key, transpose);
        }
        let (data, shape) = self.inner.take(key, false)?;
        let (data, shape) = (self.f)(key, data, shape)?;
        if transpose {
            transpose_2d(&data, &shape, key)
        } else {
            Ok((data, shape))
        }
    }
    fn has(&self, key: &str) -> bool {
        self.inner.has(key)
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        if (self.select)(key) {
            return Ok(None);
        }
        self.inner.take_packed(key)
    }
    fn has_packed(&self, key: &str) -> bool {
        !(self.select)(key) && self.inner.has_packed(key)
    }
    fn take_packed_bank(&mut self, key: &str) -> Result<Option<GgufPackedBank>> {
        if (self.select)(key) {
            return Ok(None);
        }
        self.inner.take_packed_bank(key)
    }
    fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }
}

/// [`WeightSource`] recording what a build reads. Built by
/// [`WeightSourceExt::tracked`].
#[derive(Debug, Clone)]
pub struct Tracked<S> {
    inner: S,
    served: std::collections::BTreeSet<String>,
    missing: std::collections::BTreeSet<String>,
}

/// What a [`Tracked`] source saw, sorted. `unused` needs a source that lists
/// its [`keys`](WeightSource::keys); it is empty otherwise.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WeightReport {
    /// Keys served (F32 or packed).
    pub served: Vec<String>,
    /// Keys asked for and not found (a probe through `has` does not count).
    pub missing: Vec<String>,
    /// Keys the source has that nothing asked for.
    pub unused: Vec<String>,
}

impl std::fmt::Display for WeightReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} weights served, {} missing, {} unused",
            self.served.len(),
            self.missing.len(),
            self.unused.len()
        )?;
        for (what, keys) in [("missing", &self.missing), ("unused", &self.unused)] {
            if !keys.is_empty() {
                let shown: Vec<&str> = keys.iter().take(8).map(String::as_str).collect();
                let more = keys.len().saturating_sub(shown.len());
                write!(f, "\n  {what}: {}", shown.join(", "))?;
                if more > 0 {
                    write!(f, " … (+{more})")?;
                }
            }
        }
        Ok(())
    }
}

impl<S: WeightSource> Tracked<S> {
    /// What the source has served so far.
    pub fn report(&self) -> WeightReport {
        let unused = self
            .inner
            .keys()
            .into_iter()
            .filter(|k| !self.served.contains(k))
            .collect();
        WeightReport {
            served: self.served.iter().cloned().collect(),
            missing: self
                .missing
                .iter()
                .filter(|k| !self.served.contains(*k))
                .cloned()
                .collect(),
            unused,
        }
    }

    /// The wrapped source.
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S: WeightSource> WeightSource for Tracked<S> {
    fn take(&mut self, key: &str, transpose: bool) -> Result<(Vec<f32>, Vec<usize>)> {
        let r = self.inner.take(key, transpose);
        if r.is_ok() {
            self.served.insert(key.to_string());
        } else {
            self.missing.insert(key.to_string());
        }
        r
    }
    fn has(&self, key: &str) -> bool {
        self.inner.has(key)
    }
    fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
        let r = self.inner.take_packed(key)?;
        if r.is_some() {
            self.served.insert(key.to_string());
        }
        Ok(r)
    }
    fn has_packed(&self, key: &str) -> bool {
        self.inner.has_packed(key)
    }
    fn take_packed_bank(&mut self, key: &str) -> Result<Option<GgufPackedBank>> {
        let r = self.inner.take_packed_bank(key)?;
        if r.is_some() {
            self.served.insert(key.to_string());
        }
        Ok(r)
    }
    fn keys(&self) -> Vec<String> {
        self.inner.keys()
    }
    fn stored_dtype(&self, key: &str) -> Option<rlx_ir::DType> {
        self.inner.stored_dtype(key)
    }
    fn take_half(&mut self, key: &str, transpose: bool) -> Result<Option<(Vec<u8>, Vec<usize>)>> {
        let r = self.inner.take_half(key, transpose)?;
        if r.is_some() {
            self.served.insert(key.to_string());
        }
        Ok(r)
    }
}

/// Combinators on any [`WeightSource`] (including `&mut dyn WeightSource`).
pub trait WeightSourceExt: WeightSource + Sized {
    /// Rewrite every key with `map` before looking it up.
    fn renamed<F: Fn(&str) -> String>(self, map: F) -> Renamed<Self, F> {
        Renamed { inner: self, map }
    }

    /// Look every key up as `prefix + key` (e.g. `"language_model."`).
    fn with_prefix(self, prefix: impl Into<String>) -> Prefixed<Self> {
        Prefixed {
            inner: self,
            prefix: prefix.into(),
        }
    }

    /// Serve `from` from `to` when the source has no `from` — a tied LM head
    /// (`lm_head.weight` → `model.embed_tokens.weight`). Chain for several.
    /// The source must read non-destructively ([`CloningSource`], a file) for
    /// `to` to be served twice.
    fn alias(self, from: impl Into<String>, to: impl Into<String>) -> Aliased<Self> {
        Aliased {
            inner: self,
            from: from.into(),
            to: to.into(),
        }
    }

    /// Rewrite the F32 tensors whose key matches `select` with `f(key, data,
    /// shape)`, in stored layout, before any transpose — a scale, a layout
    /// fix-up, a permutation. Selected keys are always served as F32 (a
    /// transform cannot apply to quantized bytes); the rest pass through,
    /// packed weights included.
    fn map_tensors<P, F>(self, select: P, f: F) -> Mapped<Self, P, F>
    where
        P: Fn(&str) -> bool,
        F: FnMut(&str, Vec<f32>, Vec<usize>) -> Result<(Vec<f32>, Vec<usize>)>,
    {
        Mapped {
            inner: self,
            select,
            f,
        }
    }

    /// Record what a build reads: [`Tracked::report`] then lists keys served,
    /// keys asked for but missing, and keys the source has that nothing read
    /// — the usual sign of a checkpoint naming mismatch.
    fn tracked(self) -> Tracked<Self> {
        Tracked {
            inner: self,
            served: Default::default(),
            missing: Default::default(),
        }
    }

    /// Ask `fallback` for whatever `self` lacks; see [`Fallback`].
    fn or<B: WeightSource>(self, fallback: B) -> Fallback<Self, B> {
        Fallback {
            primary: self,
            fallback,
        }
    }
}

impl<S: WeightSource> WeightSourceExt for S {}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::quant::QuantScheme;

    fn map() -> MapWeights {
        let mut m = MapWeights::default();
        m.insert("model.w", vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        m.insert("model.b", vec![7.], vec![1]);
        m
    }

    /// A source with one packed linear (`q`) and one F32 tensor (`f`).
    struct Packed;
    impl WeightSource for Packed {
        fn take(&mut self, key: &str, _: bool) -> Result<(Vec<f32>, Vec<usize>)> {
            match key {
                "f" => Ok((vec![9.], vec![1])),
                _ => Err(anyhow::anyhow!("missing weight: {key}")),
            }
        }
        fn has(&self, key: &str) -> bool {
            matches!(key, "q" | "f")
        }
        fn take_packed(&mut self, key: &str) -> Result<Option<GgufPackedLinear>> {
            Ok((key == "q").then(|| GgufPackedLinear {
                w_q: vec![1, 2, 3],
                scheme: QuantScheme::GgufQ8_0,
                in_dim: 32,
                out_dim: 1,
                bias: Vec::new(),
            }))
        }
    }

    #[test]
    fn mut_ref_and_dyn_forward_packed_weights() {
        // `S = &mut Packed` / `&mut dyn WeightSource`: both go through the
        // `&mut T` impl, the way an adapter wrapping a borrowed source sees it.
        fn packed_via<S: WeightSource>(mut s: S) -> bool {
            s.has("q") && s.take_packed("q").unwrap().is_some()
        }
        let mut p = Packed;
        assert!(packed_via(&mut p));
        let d: &mut dyn WeightSource = &mut p;
        assert!(packed_via(d));
    }

    #[test]
    fn cloning_source_does_not_consume() {
        let m = map();
        let mut src = CloningSource(&m);
        for _ in 0..2 {
            assert_eq!(
                src.take("model.w", true).unwrap(),
                (vec![1., 4., 2., 5., 3., 6.], vec![3, 2])
            );
        }
        assert_eq!(src.take("model.b", false).unwrap(), (vec![7.], vec![1]));
        assert!(src.has("model.w") && !src.has("w"));
        assert!(src.take("model.b", true).is_err(), "rank-1 transpose");
        assert!(src.take("nope", false).is_err());
    }

    #[test]
    fn renamed_and_prefixed_keys() {
        let m = map();
        let mut src = CloningSource(&m).with_prefix("model.");
        assert_eq!(src.take("b", false).unwrap().0, vec![7.]);
        assert!(src.has("w") && !src.has("model.w"));
        let mut packed = Packed.renamed(|k| k.trim_start_matches("blk.").to_string());
        assert!(packed.take_packed("blk.q").unwrap().is_some());
    }

    #[test]
    fn fallback_fills_gaps_but_keeps_primary_tensors() {
        let mut primary = MapWeights::default();
        primary.insert("f", vec![1.], vec![1]);
        let mut src = primary.or(Packed);
        // F32 from primary even though fallback has the key too.
        assert_eq!(src.take("f", false).unwrap().0, vec![1.]);
        // Packed from fallback only for keys primary lacks.
        assert!(src.take_packed("q").unwrap().is_some());
        assert!(src.has("q"));
        let err = src.take("nope", false).unwrap_err().to_string();
        assert!(err.contains("fallback"), "{err}");

        let mut shadowed = MapWeights::default();
        shadowed.insert("q", vec![2.], vec![1]);
        let mut src = shadowed.or(Packed);
        assert!(
            src.take_packed("q").unwrap().is_none(),
            "primary holds `q` as F32; fallback's packed copy must not replace it"
        );
        assert_eq!(src.take("q", false).unwrap().0, vec![2.]);
    }

    #[test]
    fn tracked_reports_served_missing_and_unused() {
        let m = map();
        let mut src = CloningSource(&m).tracked();
        src.take("model.w", false).unwrap();
        assert!(src.take("model.nope", false).is_err());
        let r = src.report();
        assert_eq!(r.served, ["model.w"]);
        assert_eq!(r.missing, ["model.nope"]);
        assert_eq!(r.unused, ["model.b"]);
        assert!(r.to_string().contains("unused: model.b"), "{r}");
        // Packed serves count; an F32 retry after `None` is not a miss.
        let mut p = Packed.tracked();
        p.take_packed("q").unwrap();
        assert_eq!(p.report().served, ["q"]);
    }

    #[test]
    fn alias_serves_tied_weights() {
        let m = map();
        let mut src = CloningSource(&m).alias("model.head", "model.w");
        assert!(src.has("model.head"));
        assert_eq!(src.take("model.head", false).unwrap().1, vec![2, 3]);
        assert_eq!(src.take("model.w", false).unwrap().1, vec![2, 3]);
        // A real tensor under `from` wins over the alias.
        let mut src = CloningSource(&m).alias("model.b", "model.w");
        assert_eq!(src.take("model.b", false).unwrap().0, vec![7.]);
    }

    #[test]
    fn map_tensors_transforms_before_transpose_and_forces_f32() {
        let m = map();
        let mut src = CloningSource(&m).map_tensors(
            |k| k == "model.w",
            |_, d, s| Ok((d.into_iter().map(|v| v * 10.).collect(), s)),
        );
        assert_eq!(
            src.take("model.w", true).unwrap(),
            (vec![10., 40., 20., 50., 30., 60.], vec![3, 2])
        );
        assert_eq!(src.take("model.b", false).unwrap().0, vec![7.]);
        let mut p = Packed.map_tensors(|k| k == "q", |_, d, s| Ok((d, s)));
        assert!(p.take_packed("q").unwrap().is_none(), "selected → F32 only");
        let mut p = Packed.map_tensors(|_| false, |_, d, s| Ok((d, s)));
        assert!(p.take_packed("q").unwrap().is_some());
    }

    #[test]
    fn missing_weight_names_the_closest_keys() {
        let keys = [
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.k_proj.weight",
            "vision.layers.0.mlp.weight",
        ];
        // Only the tensor ending the same way is offered, not every `.weight`.
        assert_eq!(
            missing_weight("model.layers.0.self_attn.q_proj.weight", &keys).to_string(),
            "missing weight: model.layers.0.self_attn.q_proj.weight \
             (closest: language_model.model.layers.0.self_attn.q_proj.weight)"
        );
        assert_eq!(
            missing_weight("unrelated", &keys).to_string(),
            "missing weight: unrelated"
        );
        let m = map();
        let mut boxed: Box<dyn WeightSource + '_> = Box::new(CloningSource(&m));
        let e = boxed.take("w", false).unwrap_err().to_string();
        assert!(e.contains("closest: model.w"), "{e}");
        assert_eq!(boxed.keys(), ["model.b", "model.w"]);
    }
}
