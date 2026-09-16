// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Model cost, derived from the checkpoint** — so placement does not need a
//! hand-supplied byte budget.
//!
//! [`super::placement::plan_placement`] needs to know what a layer costs. Making
//! the caller compute that is where "automatic placement" stopped being
//! automatic: every model crate grew its own arithmetic, and a wrong number
//! silently produced a plan that OOMs on the third node.
//!
//! A checkpoint already carries the answer. A GGUF's header lists every tensor's
//! name, shape and quant type — that is enough to sum exact bytes per block
//! without reading a single weight, and the same holds for a safetensors index.
//! [`ModelCost::from_tensor_index`] takes that listing and works out the layer
//! count, the per-layer resident cost, and the embedding/head extras.
//!
//! ## Routed experts are counted separately, on purpose
//!
//! For a fine-grained MoE the expert banks *are* the model — 86 of GLM-5.3-Flash's
//! 93 GB — but they behave nothing like the rest. Only `top_k` of `n_experts` are
//! touched per token, so they can live on disk and page in, while the dense
//! weights must be resident. Folding them into one number would say a 320 B model
//! needs 93 GB of RAM per node, when in practice it needs ~10 GB resident plus
//! disk and bandwidth. [`ModelCost::per_layer_expert_bytes`] and
//! [`ModelCost::per_layer_expert_active_bytes`] keep the two apart so the planner
//! can budget RAM, disk and IO independently.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// One tensor from a checkpoint's index: enough to size it, not to read it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    /// Logical dims (order does not matter here; only the rank and product do).
    pub dims: Vec<usize>,
    /// On-disk bytes, i.e. **after** quantization.
    pub bytes: u64,
}

impl TensorEntry {
    pub fn new(name: impl Into<String>, dims: Vec<usize>, bytes: u64) -> Self {
        Self {
            name: name.into(),
            dims,
            bytes,
        }
    }
    pub fn elements(&self) -> u64 {
        self.dims.iter().product::<usize>() as u64
    }
}

/// How to read a checkpoint's naming convention.
#[derive(Debug, Clone)]
pub struct CostOptions {
    /// Prefixes that introduce a numbered block: `blk.12.…`, `model.layers.12.…`.
    pub block_prefixes: Markers,
    /// Substrings marking a **routed expert bank**.
    ///
    /// Matched on the name rather than the rank, because rank alone is not a
    /// signal: MLA's per-head `attn_k_b` is also 3-D, and treating it as an
    /// expert bank would move it to the disk budget and under-count RAM.
    pub expert_markers: Markers,
    /// Substrings marking the token embedding (first stage only).
    pub embed_markers: Markers,
    /// Substrings marking the final norm / LM head (last stage only).
    pub head_markers: Markers,
    /// `num_experts_per_tok / n_routed_experts` — the fraction of an expert bank
    /// actually read per token, which is what paging bandwidth depends on.
    /// Defaults to 1.0 (assume everything is touched) so an unknown model is
    /// costed pessimistically rather than optimistically.
    pub expert_active_fraction: f64,
    /// Relative FLOPs per layer; 1.0 unless a model has non-uniform layers.
    pub per_layer_flops: f64,
    /// Names used to infer the KV cache width. See [`Self::kv`].
    pub kv: KvMarkers,
    /// Cache element dtype (`f32` default, `f16`/`bf16`, `f8`/`int8`).
    pub kv_dtype: String,
}

/// A set of substrings that identify a tensor by name.
///
/// The `names.iter().any(|p| t.name.contains(p))` idiom appeared five times
/// across this module with three different spellings; giving it a type also
/// gives it a place to hang the rank-1 lookup that the shape probes all need.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Markers(Vec<String>);

impl Markers {
    pub fn new<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self(names.into_iter().map(Into::into).collect())
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn names(&self) -> &[String] {
        &self.0
    }
    /// Does this name carry any of the markers?
    pub fn matches(&self, name: &str) -> bool {
        self.0.iter().any(|p| name.contains(p.as_str()))
    }
    /// First matching tensor in a block.
    pub fn find_in<'a>(&self, ts: &[&'a TensorEntry]) -> Option<&'a TensorEntry> {
        ts.iter().copied().find(|t| self.matches(&t.name))
    }
    /// Is any tensor in the block marked?
    pub fn any_in(&self, ts: &[&TensorEntry]) -> bool {
        ts.iter().any(|t| self.matches(&t.name))
    }
    /// Length of `t` when it matches and is rank-1 — the shape probes all want
    /// exactly this, and forgetting the rank check silently reads a matrix dim.
    pub fn rank1_len(&self, t: &TensorEntry) -> Option<usize> {
        match t.dims.as_slice() {
            [n] if self.matches(&t.name) => Some(*n),
            _ => None,
        }
    }
}

impl<S: Into<String>> FromIterator<S> for Markers {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        Self::new(iter)
    }
}

/// Tensor-name markers the KV inference keys off.
///
/// Every rule is anchored on `hidden_size`, which is read from a rank-1 norm
/// inside a block. That makes the inference **orientation-agnostic**: for a 2-D
/// projection, whichever dim is not `hidden` is the output, so it works on a
/// GGUF index (`[in, out]`) and an HF one (`[out, in]`) without being told which.
#[derive(Debug, Clone)]
pub struct KvMarkers {
    /// Rank-1 tensor inside a block whose length is `hidden_size`.
    pub hidden_probe: Markers,
    /// A recurrent / linear-attention layer. Its state does not grow with
    /// context, so it contributes no per-token cache.
    pub recurrent: Markers,
    /// Per-head count and per-head width of a recurrent state.
    pub recurrent_heads: Markers,
    pub recurrent_head_dim: Markers,
    /// Latent (MLA) attention: only the compressed latent is cached.
    pub latent_kv: Markers,
    /// Ordinary attention: the K projection's output width is the cached width,
    /// doubled for K and V.
    pub attn_k: Markers,
}

impl Default for KvMarkers {
    fn default() -> Self {
        Self {
            hidden_probe: Markers::new(["attn_norm.weight", "input_layernorm.weight"]),
            recurrent: Markers::new(["ssm_", ".mamba", "conv1d"]),
            recurrent_heads: Markers::new(["ssm_a", "A_log"]),
            recurrent_head_dim: Markers::new(["ssm_norm.weight", "o_norm.weight"]),
            latent_kv: Markers::new(["attn_kv_a_mqa", "kv_a_proj_with_mqa"]),
            attn_k: Markers::new(["attn_k.weight", "k_proj.weight"]),
        }
    }
}

impl Default for CostOptions {
    fn default() -> Self {
        Self {
            block_prefixes: Markers::new(["blk.", "model.layers.", "layers."]),
            expert_markers: Markers::new(["_exps", ".experts."]),
            embed_markers: Markers::new(["token_embd", "embed_tokens", "word_embeddings"]),
            head_markers: Markers::new(["output.weight", "output_norm", "lm_head", "model.norm"]),
            expert_active_fraction: 1.0,
            per_layer_flops: 1.0,
            kv: KvMarkers::default(),
            kv_dtype: "f32".into(),
        }
    }
}

impl CostOptions {
    /// Set the active-expert fraction from a router config.
    pub fn with_experts(mut self, experts_per_tok: usize, n_routed: usize) -> Self {
        if n_routed > 0 {
            self.expert_active_fraction = experts_per_tok as f64 / n_routed as f64;
        }
        self
    }

    /// `Some(block_index)` if `name` sits inside a numbered block.
    fn block_of(&self, name: &str) -> Option<usize> {
        for p in self.block_prefixes.names() {
            if let Some(rest) = name.strip_prefix(p.as_str())
                && let Some(idx) = rest.split('.').next()
                && let Ok(i) = idx.parse::<usize>()
            {
                return Some(i);
            }
        }
        None
    }
    fn is_expert(&self, name: &str) -> bool {
        self.expert_markers.matches(name)
    }
}

/// Bytes per cached element for a cache dtype name. Defaults to f32, matching
/// the runtime's default; an f16 cache is half the reservation, and refusing a
/// plan that would have fit is as bad as accepting one that will not.
pub fn kv_elem_bytes(dtype: &str) -> u64 {
    match dtype {
        "f16" | "bf16" | "half" => 2,
        "f8" | "fp8" | "int8" | "i8" => 1,
        _ => 4,
    }
}

/// One block's contribution to the cache, and which shape produced it.
///
/// Making the three shapes a type rather than three branches accumulating into
/// shared counters means a block's cost and its position in the stack cannot
/// drift apart: the vector is built by `map`, so alignment is structural.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKv {
    /// Linear / recurrent attention: a fixed state, no per-token growth.
    Recurrent { state_bytes: u64 },
    /// Latent (MLA): only the compressed latent is cached — the point of MLA,
    /// and ~64× smaller than expanded keys and values.
    Latent { bytes_per_token: u64 },
    /// Ordinary attention: keys and values.
    Attention { bytes_per_token: u64 },
}

impl BlockKv {
    pub fn bytes_per_token(&self) -> u64 {
        match *self {
            Self::Recurrent { .. } => 0,
            Self::Latent { bytes_per_token } | Self::Attention { bytes_per_token } => {
                bytes_per_token
            }
        }
    }
    pub fn state_bytes(&self) -> u64 {
        match *self {
            Self::Recurrent { state_bytes } => state_bytes,
            _ => 0,
        }
    }
}

/// How much detail we have about the cache.
///
/// A sum type because the two cases are genuinely different, not a vector that
/// might be empty: an inferred profile knows every block, a declared figure is a
/// flat rate with no block structure at all. Encoding that as `Vec` + "check if
/// empty" pushes the distinction into every caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvDetail {
    /// Per-block costs, in layer order.
    PerBlock(Vec<BlockKv>),
    /// One rate for every layer.
    Flat { bytes_per_layer_token: u64 },
}

impl Default for KvDetail {
    fn default() -> Self {
        Self::Flat {
            bytes_per_layer_token: 0,
        }
    }
}

/// Model width from any block's rank-1 norm.
fn infer_hidden(entries: &[TensorEntry], opts: &CostOptions) -> usize {
    entries
        .iter()
        .filter(|t| opts.block_of(&t.name).is_some())
        .filter_map(|t| opts.kv.hidden_probe.rank1_len(t))
        .max()
        .unwrap_or(0)
}

/// Classify one block's tensors into its cache shape.
///
/// `None` means "could not read this block" — which is deliberately distinct
/// from a block that reads as costing nothing. The previous accumulator-based
/// version conflated them: a recurrent block whose head dims were unreadable was
/// counted as known and costed at zero state.
fn classify_block(ts: &[&TensorEntry], m: &KvMarkers, elem: u64) -> Option<BlockKv> {
    // Anchor: the block's own rank-1 norm gives `hidden`, which is what makes
    // the output-width read orientation-agnostic.
    let hidden = ts.iter().find_map(|t| m.hidden_probe.rank1_len(t));

    // The non-hidden dim of a 2-D projection is its output width.
    let out_of = |t: &TensorEntry| -> Option<usize> {
        let h = hidden?;
        match t.dims.as_slice() {
            [a, b] if *a == h && *b != h => Some(*b),
            [a, b] if *b == h && *a != h => Some(*a),
            // Square, or neither matches: ambiguous, so decline.
            _ => None,
        }
    };

    // Recurrent first: a linear-attention block also has an `attn_k`, and
    // costing that as a KV cache would be badly wrong — it has none.
    if m.recurrent.any_in(ts) {
        let heads = ts.iter().find_map(|t| m.recurrent_heads.rank1_len(t))?;
        let head_dim = ts.iter().find_map(|t| m.recurrent_head_dim.rank1_len(t))?;
        return Some(BlockKv::Recurrent {
            state_bytes: (heads * head_dim * head_dim) as u64 * elem,
        });
    }
    if let Some(t) = m.latent_kv.find_in(ts) {
        return Some(BlockKv::Latent {
            bytes_per_token: out_of(t)? as u64 * elem,
        });
    }
    if let Some(t) = m.attn_k.find_in(ts) {
        return Some(BlockKv::Attention {
            bytes_per_token: 2 * out_of(t)? as u64 * elem,
        });
    }
    None
}

/// Infer [`KvProfile`] from tensor shapes, block by block.
///
/// Any block that cannot be classified makes the whole profile
/// [`KvSource::Unknown`]: a partial average under-estimates, which is the
/// dangerous direction for something the planner reserves RAM from.
fn infer_kv(entries: &[TensorEntry], opts: &CostOptions) -> KvProfile {
    use std::collections::BTreeMap;
    let elem = kv_elem_bytes(&opts.kv_dtype);
    let mut blocks: BTreeMap<usize, Vec<&TensorEntry>> = BTreeMap::new();
    for t in entries {
        if let Some(i) = opts.block_of(&t.name) {
            blocks.entry(i).or_default().push(t);
        }
    }
    if blocks.is_empty() {
        return KvProfile::unknown();
    }
    match blocks
        .values()
        .map(|ts| classify_block(ts, &opts.kv, elem))
        .collect::<Option<Vec<_>>>()
    {
        Some(per_block) => KvProfile::inferred(per_block),
        None => KvProfile::unknown(),
    }
}

/// Where a KV figure came from — so "unknown" is never mistaken for "zero".
///
/// A cache size of 0 is a legitimate answer (a purely recurrent model has no
/// per-token cache at all) and it is also what you get when nobody worked it
/// out. Those must not look the same to a planner that is about to reserve RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvSource {
    /// Read off the checkpoint's tensor shapes.
    Inferred,
    /// Supplied by the caller or the config.
    Declared,
    /// Could not be determined — the planner must not silently reserve nothing.
    #[default]
    Unknown,
}

/// What the attention cache costs, and how confident we are about it.
#[derive(Debug, Clone, Default)]
pub struct KvProfile {
    detail: KvDetail,
    source: KvSource,
}

impl KvProfile {
    pub fn declared(bytes_per_layer_token: u64) -> Self {
        Self {
            detail: KvDetail::Flat {
                bytes_per_layer_token,
            },
            source: KvSource::Declared,
        }
    }
    pub fn inferred(blocks: Vec<BlockKv>) -> Self {
        Self {
            detail: KvDetail::PerBlock(blocks),
            source: KvSource::Inferred,
        }
    }
    pub fn unknown() -> Self {
        Self::default()
    }

    pub fn source(&self) -> KvSource {
        self.source
    }
    pub fn is_known(&self) -> bool {
        self.source != KvSource::Unknown
    }
    /// Per-block detail, empty for a flat rate.
    pub fn blocks(&self) -> &[BlockKv] {
        match &self.detail {
            KvDetail::PerBlock(b) => b,
            KvDetail::Flat { .. } => &[],
        }
    }

    /// Mean per-token bytes across the stack.
    ///
    /// Derived rather than stored: a cached average is a second copy of the same
    /// fact that can fall out of step with the per-block detail.
    pub fn bytes_per_layer_token(&self) -> u64 {
        match &self.detail {
            KvDetail::Flat {
                bytes_per_layer_token,
            } => *bytes_per_layer_token,
            KvDetail::PerBlock(b) if !b.is_empty() => {
                b.iter().map(BlockKv::bytes_per_token).sum::<u64>() / b.len() as u64
            }
            KvDetail::PerBlock(_) => 0,
        }
    }
    /// Mean context-independent state per layer.
    pub fn state_bytes_per_layer(&self) -> u64 {
        match &self.detail {
            KvDetail::Flat { .. } => 0,
            KvDetail::PerBlock(b) if !b.is_empty() => {
                b.iter().map(BlockKv::state_bytes).sum::<u64>() / b.len() as u64
            }
            KvDetail::PerBlock(_) => 0,
        }
    }

    /// Total bytes for `layers` at `context`, using the stack average.
    pub fn bytes(&self, layers: u64, context: usize) -> u64 {
        self.bytes_per_layer_token()
            .saturating_mul(layers)
            .saturating_mul(context as u64)
            .saturating_add(self.state_bytes_per_layer().saturating_mul(layers))
    }

    /// Exact bytes for a specific contiguous layer range.
    ///
    /// Falls back to the average for a flat rate, so callers need not branch.
    pub fn bytes_for(&self, layers: std::ops::Range<usize>, context: usize) -> u64 {
        let b = self.blocks();
        if b.is_empty() {
            return self.bytes(layers.len() as u64, context);
        }
        let hi = layers.end.min(b.len());
        let lo = layers.start.min(hi);
        window_bytes(&b[lo..hi], context)
    }

    /// The most any contiguous window of `len` layers can cost.
    ///
    /// Capacity is decided before the range is, so budgeting for the worst
    /// placement means the plan fits wherever the range lands — instead of
    /// fitting on average and OOM-ing on the node that draws the
    /// attention-heavy slice.
    pub fn max_window_bytes(&self, len: usize, context: usize) -> u64 {
        let b = self.blocks();
        if b.is_empty() || len == 0 {
            return self.bytes(len as u64, context);
        }
        b.windows(len.min(b.len()))
            .map(|w| window_bytes(w, context))
            .max()
            .unwrap_or(0)
    }

    pub fn summary(&self, layers: u64, context: usize) -> String {
        let Some(kind) = (match self.source {
            KvSource::Unknown => None,
            KvSource::Inferred => Some("inferred"),
            KvSource::Declared => Some("declared"),
        }) else {
            return "KV: unknown".into();
        };
        let state = self.state_bytes_per_layer() * layers;
        format!(
            "KV: {:.2} GB for {layers} layers @ {context} tokens ({kind}{})",
            self.bytes(layers, context) as f64 / 1e9,
            if state > 0 {
                format!(", + {:.2} GB recurrent state", state as f64 / 1e9)
            } else {
                String::new()
            },
        )
    }
}

/// Cache bytes for one contiguous run of blocks.
fn window_bytes(blocks: &[BlockKv], context: usize) -> u64 {
    let per_token: u64 = blocks.iter().map(BlockKv::bytes_per_token).sum();
    let state: u64 = blocks.iter().map(BlockKv::state_bytes).sum();
    per_token
        .saturating_mul(context as u64)
        .saturating_add(state)
}

/// Resident-memory + compute cost model for a layer-stack model.
///
/// Bytes are the ACTUAL per-node footprint (packed weights as stored), so the
/// planner's budgets mean something. Build it with
/// [`ModelCost::from_tensor_index`] rather than by hand.
#[derive(Debug, Clone, Default)]
pub struct ModelCost {
    pub n_layers: usize,
    /// Average resident bytes per layer, **excluding** routed experts.
    pub per_layer_bytes: u64,
    /// Average routed-expert bytes per layer. These may live on disk.
    pub per_layer_expert_bytes: u64,
    /// Expert bytes actually read per token — `per_layer_expert_bytes ×
    /// expert_active_fraction`. Drives the paging-bandwidth estimate.
    pub per_layer_expert_active_bytes: u64,
    /// Extra bytes on the FIRST stage (token embedding).
    pub embed_bytes: u64,
    /// Extra bytes on the LAST stage (final norm + LM head).
    pub head_bytes: u64,
    /// Relative FLOPs per layer (throughput policy; 1.0 = uniform).
    pub per_layer_flops: f64,
    /// What the attention cache costs, and whether that is known.
    ///
    /// [`ModelCost::from_tensor_index`] infers it from tensor shapes where it
    /// can (see [`CostOptions::kv`]); otherwise it is
    /// [`KvSource::Unknown`] and the planner refuses to guess.
    pub kv: KvProfile,
    /// Total parameters, used to re-scale the model between precisions.
    pub params: u64,
    /// Model width, read from a rank-1 norm inside a block. 0 when unknown.
    ///
    /// Worth carrying rather than re-deriving: the worst-case KV bound needs it,
    /// and inferring it from byte counts (`sqrt(per_layer_bytes / 12)`) is badly
    /// wrong for a quantized MoE, whose per-layer bytes are dominated by expert
    /// banks at ~2 bits per weight.
    pub hidden_size: usize,
}

/// A slice of the checkpoint: how many bytes a layer touches, at that slice's
/// own average bit width.
struct Width {
    bytes: u64,
    bits_per_param: f64,
}

impl Width {
    /// `bytes` is the per-layer figure; `total_bytes`/`total_elems` set the width.
    fn of(bytes: u64, total_bytes: u64, total_elems: u64) -> Self {
        let bits_per_param = if total_elems > 0 {
            total_bytes as f64 * 8.0 / total_elems as f64
        } else {
            0.0
        };
        Self {
            bytes,
            bits_per_param,
        }
    }
    fn params(&self) -> f64 {
        if self.bits_per_param > 0.0 {
            self.bytes as f64 * 8.0 / self.bits_per_param
        } else {
            0.0
        }
    }
}

/// FLOPs one layer performs for one decoded token.
///
/// Every weight a layer touches contributes one multiply and one add, so the
/// count is `2 x params_touched`: the dense parameters plus the fraction of the
/// expert banks that token's routing actually fires. Each group is converted
/// from bytes at ITS OWN measured width, because a mixed-precision checkpoint
/// (2-bit experts, 4-bit attention) has no single meaningful bits-per-param.
///
/// This being a real FLOP count is load-bearing, not cosmetic. `stage_secs`
/// divides it by measured GFLOP/s and ADDS the result to `bytes / (MB/s)` — a
/// genuine seconds term. While this was a dimensionless 1.0 those two terms
/// were incommensurable, so what balanced compute against IO was the accident
/// of `per_layer_flops` defaulting to one, and the "s/token" the planner
/// printed was not a time.
///
/// Returns 0.0 when no width can be measured; callers then see a pure IO
/// estimate rather than a fabricated one.
fn decode_flops_per_layer(dense: Width, active_experts: Width) -> f64 {
    2.0 * (dense.params() + active_experts.params())
}

impl ModelCost {
    /// The same model at a different weight precision.
    ///
    /// Scales every byte figure by `target_bits / current_bits`, where the
    /// current width is inferred from the checkpoint (`bytes × 8 / params`).
    /// Approximate by construction — it assumes the whole model re-quantizes
    /// uniformly — but it is the right shape for answering "would this fit one
    /// step down?", which is the question a precision ladder asks.
    ///
    /// Returns `None` when the parameter count is unknown (nothing to scale by).
    pub fn at_bits(&self, target_bits: f64) -> Option<Self> {
        if self.params == 0 || target_bits <= 0.0 {
            return None;
        }
        let cur = self.total_bytes() as f64 * 8.0 / self.params as f64;
        if cur <= 0.0 {
            return None;
        }
        let r = target_bits / cur;
        let sc = |b: u64| (b as f64 * r) as u64;
        Some(Self {
            per_layer_bytes: sc(self.per_layer_bytes),
            per_layer_expert_bytes: sc(self.per_layer_expert_bytes),
            per_layer_expert_active_bytes: sc(self.per_layer_expert_active_bytes),
            embed_bytes: sc(self.embed_bytes),
            head_bytes: sc(self.head_bytes),
            // KV width and layer count do not follow weight precision.
            ..self.clone()
        })
    }

    /// KV bytes one node holds for `layers` at `context` tokens.
    pub fn kv_bytes(&self, layers: u64, context: usize) -> u64 {
        self.kv.bytes(layers, context)
    }

    /// Total bytes across the whole model, experts included.
    pub fn total_bytes(&self) -> u64 {
        self.embed_bytes
            + self.head_bytes
            + (self.per_layer_bytes + self.per_layer_expert_bytes) * self.n_layers as u64
    }

    /// Bytes that must be RESIDENT for the whole model (experts excluded).
    pub fn total_resident_bytes(&self) -> u64 {
        self.embed_bytes + self.head_bytes + self.per_layer_bytes * self.n_layers as u64
    }

    /// True when routed experts dominate — the regime where disk and IO decide
    /// placement rather than RAM.
    pub fn is_expert_dominated(&self) -> bool {
        self.per_layer_expert_bytes > self.per_layer_bytes
    }

    /// Derive the cost model from a checkpoint's tensor index.
    ///
    /// Layer count is `max(block index) + 1`; per-layer figures are averaged over
    /// the blocks actually present, so a model with a trailing MTP block or an
    /// uneven layer stack still gets a usable mean.
    pub fn from_tensor_index(entries: &[TensorEntry], opts: &CostOptions) -> Result<Self> {
        if entries.is_empty() {
            bail!("empty tensor index");
        }
        let mut max_block: Option<usize> = None;
        let (mut layer_bytes, mut expert_bytes) = (0u64, 0u64);
        // Element counts alongside the bytes: experts are routinely quantized
        // harder than the attention weights beside them (2-bit banks against
        // 4-bit projections is ordinary), so one average width across the whole
        // checkpoint converts bytes back into the wrong number of parameters.
        let (mut layer_elems, mut expert_elems) = (0u64, 0u64);
        let (mut embed, mut head) = (0u64, 0u64);

        for t in entries {
            match opts.block_of(&t.name) {
                Some(i) => {
                    max_block = Some(max_block.map_or(i, |m: usize| m.max(i)));
                    if opts.is_expert(&t.name) {
                        expert_bytes += t.bytes;
                        expert_elems += t.elements();
                    } else {
                        layer_bytes += t.bytes;
                        layer_elems += t.elements();
                    }
                }
                None => {
                    // Order matters: an embedding name is checked first because
                    // tied-head checkpoints reuse it for both roles.
                    if opts.embed_markers.matches(&t.name) {
                        embed += t.bytes;
                    } else if opts.head_markers.matches(&t.name) {
                        head += t.bytes;
                    }
                }
            }
        }

        let n_layers = max_block.map(|m| m + 1).unwrap_or(0);
        if n_layers == 0 {
            bail!(
                "no numbered blocks found — checked prefixes {:?}; pass CostOptions \
                 with the checkpoint's naming convention",
                opts.block_prefixes
            );
        }
        let n = n_layers as u64;
        let per_layer_expert_bytes = expert_bytes / n;
        let per_layer_bytes = layer_bytes / n;
        let per_layer_expert_active_bytes =
            (per_layer_expert_bytes as f64 * opts.expert_active_fraction.clamp(0.0, 1.0)) as u64;
        let params: u64 = entries.iter().map(|t| t.elements()).sum();
        Ok(Self {
            n_layers,
            per_layer_bytes,
            per_layer_expert_bytes,
            per_layer_expert_active_bytes,
            embed_bytes: embed,
            head_bytes: head,
            per_layer_flops: decode_flops_per_layer(
                Width::of(per_layer_bytes, layer_bytes, layer_elems),
                Width::of(per_layer_expert_active_bytes, expert_bytes, expert_elems),
            ) * opts.per_layer_flops,
            hidden_size: infer_hidden(entries, opts),
            kv: infer_kv(entries, opts),
            params,
        })
    }

    /// One-line summary for logs.
    pub fn summary(&self) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        if self.per_layer_expert_bytes > 0 {
            format!(
                "{} layers | {:.2} GB resident + {:.2} GB experts = {:.2} GB \
                 | per layer {:.2} GB + {:.2} GB experts ({:.2} GB active)",
                self.n_layers,
                gb(self.total_resident_bytes()),
                gb(self.per_layer_expert_bytes * self.n_layers as u64),
                gb(self.total_bytes()),
                gb(self.per_layer_bytes),
                gb(self.per_layer_expert_bytes),
                gb(self.per_layer_expert_active_bytes),
            )
        } else {
            format!(
                "{} layers | {:.2} GB | per layer {:.2} GB",
                self.n_layers,
                gb(self.total_bytes()),
                gb(self.per_layer_bytes),
            )
        }
    }
}

#[cfg(feature = "gguf")]
mod from_gguf {
    use super::*;
    use std::path::Path;

    /// Read a GGUF's tensor index — header only, no weight data.
    ///
    /// A split checkpoint may be passed as any one shard for the metadata, but
    /// the byte totals are only complete if every shard is listed; prefer
    /// [`tensor_index_multi`].
    pub fn tensor_index(path: impl AsRef<Path>) -> Result<Vec<TensorEntry>> {
        tensor_index_multi(std::slice::from_ref(&path.as_ref().to_path_buf()))
    }

    /// Tensor index across every shard of a split checkpoint.
    ///
    /// Reads headers only. A 93 GB model is indexed from a few MB, which is what
    /// makes planning against the real checkpoint cheap enough to do every run.
    pub fn tensor_index_multi(paths: &[std::path::PathBuf]) -> Result<Vec<TensorEntry>> {
        let mut out = Vec::new();
        for p in paths {
            let f = rlx_gguf::GgufFile::from_path(p)?;
            for t in f.tensors.values() {
                let bytes =
                    rlx_gguf::bytes_for_public(t.dtype, t.n_elements()).ok_or_else(|| {
                        anyhow::anyhow!("{}: unsupported ggml type {:?}", t.name, t.dtype)
                    })? as u64;
                out.push(TensorEntry::new(t.name.clone(), t.shape.clone(), bytes));
            }
        }
        Ok(out)
    }

    impl ModelCost {
        /// Derive the cost model straight from GGUF shards.
        pub fn from_gguf(paths: &[std::path::PathBuf], opts: &CostOptions) -> Result<Self> {
            Self::from_tensor_index(&tensor_index_multi(paths)?, opts)
        }
    }
}

#[cfg(feature = "gguf")]
pub use from_gguf::{tensor_index, tensor_index_multi};

#[cfg(test)]
mod tests {
    use super::*;

    /// `per_layer_flops` has to be a real FLOP count, because `stage_secs`
    /// divides it by GFLOP/s and adds the result to a `bytes / (MB/s)` wait. It
    /// used to be a dimensionless 1.0, which made the compute/IO balance an
    /// accident of that default and the printed "s/token" not a time.
    ///
    /// Checked against the arithmetic anyone can do by hand: a decode step
    /// touches `2 x active_params` FLOPs, so summing the per-layer figure over
    /// the stack must recover twice the model's active parameter count.
    #[test]
    fn per_layer_flops_is_a_real_decode_flop_count() {
        let idx = moe_index();
        // 8 of 288 experts fire per token, as GLM-5.3-Flash routes.
        let opts = CostOptions {
            expert_active_fraction: 8.0 / 288.0,
            ..Default::default()
        };
        let c = ModelCost::from_tensor_index(&idx, &opts).unwrap();

        // Active params: every dense weight in a block, plus the fired slice of
        // each block's expert banks. Derived from the index, not from the code
        // under test.
        let dense_elems: u64 = idx
            .iter()
            .filter(|t| opts.block_of(&t.name).is_some() && !opts.is_expert(&t.name))
            .map(|t| t.elements())
            .sum();
        let expert_elems: u64 = idx
            .iter()
            .filter(|t| opts.block_of(&t.name).is_some() && opts.is_expert(&t.name))
            .map(|t| t.elements())
            .sum();
        let active = dense_elems as f64 + expert_elems as f64 * opts.expert_active_fraction;
        let want = 2.0 * active;
        let got = c.per_layer_flops * c.n_layers as f64;
        assert!(
            (got - want).abs() <= want * 0.02,
            "stack FLOPs/token {got:.3e} against 2 x active params {want:.3e}"
        );
        assert!(
            c.per_layer_flops > 1.0,
            "still the dimensionless placeholder: {}",
            c.per_layer_flops
        );
    }

    /// Experts are quantized harder than the attention weights beside them, so
    /// converting bytes back to parameters at one checkpoint-wide average width
    /// mis-counts both groups. Each must use its own measured width.
    #[test]
    fn mixed_precision_groups_use_their_own_bit_width() {
        // Same shapes, but the expert banks are stored at a quarter the width.
        let mut idx = moe_index();
        for t in &mut idx {
            if t.name.contains("_exps") {
                t.bytes /= 4;
            }
        }
        let opts = CostOptions {
            expert_active_fraction: 8.0 / 288.0,
            ..Default::default()
        };
        let c = ModelCost::from_tensor_index(&idx, &opts).unwrap();

        let dense_elems: u64 = idx
            .iter()
            .filter(|t| opts.block_of(&t.name).is_some() && !opts.is_expert(&t.name))
            .map(|t| t.elements())
            .sum();
        let expert_elems: u64 = idx
            .iter()
            .filter(|t| opts.block_of(&t.name).is_some() && opts.is_expert(&t.name))
            .map(|t| t.elements())
            .sum();
        // Squeezing the expert BYTES must not change the parameter count, and so
        // must not change the FLOPs.
        let want = 2.0 * (dense_elems as f64 + expert_elems as f64 * opts.expert_active_fraction);
        let got = c.per_layer_flops * c.n_layers as f64;
        assert!(
            (got - want).abs() <= want * 0.02,
            "re-quantizing the experts moved the FLOP count to {got:.3e} from \
             {want:.3e}; a single average width was used for both groups"
        );
    }

    /// A GLM-5.3-Flash-shaped index: 3 dense layers then MoE, routed banks
    /// dwarfing everything else.
    fn moe_index() -> Vec<TensorEntry> {
        let mut v = vec![
            TensorEntry::new("token_embd.weight", vec![4096, 154880], 357_000_000),
            TensorEntry::new("output.weight", vec![4096, 154880], 357_000_000),
            TensorEntry::new("output_norm.weight", vec![4096], 16_384),
        ];
        for i in 0..4 {
            v.push(TensorEntry::new(
                format!("blk.{i}.attn_q.weight"),
                vec![4096, 8192],
                23_000_000,
            ));
            v.push(TensorEntry::new(
                format!("blk.{i}.attn_norm.weight"),
                vec![4096],
                16_384,
            ));
            if i >= 1 {
                for b in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
                    v.push(TensorEntry::new(
                        format!("blk.{i}.{b}.weight"),
                        vec![288, 2048, 4096],
                        600_000_000,
                    ));
                }
            }
        }
        v
    }

    #[test]
    fn derives_layers_and_splits_experts_from_dense() {
        let opts = CostOptions::default().with_experts(8, 288);
        let c = ModelCost::from_tensor_index(&moe_index(), &opts).unwrap();
        assert_eq!(c.n_layers, 4);
        assert_eq!(c.embed_bytes, 357_000_000);
        // head = output.weight + output_norm.weight
        assert_eq!(c.head_bytes, 357_000_000 + 16_384);
        // Dense per-layer: 4 blocks × (23 MB + 16 KB), averaged.
        assert_eq!(c.per_layer_bytes, 23_016_384);
        // Experts: 3 blocks × 3 banks × 600 MB over 4 layers.
        assert_eq!(c.per_layer_expert_bytes, 3 * 3 * 600_000_000 / 4);
        assert!(c.is_expert_dominated());
    }

    /// Only `top_k / n_experts` of a bank is read per token — the number the
    /// paging-bandwidth estimate hangs on.
    #[test]
    fn active_expert_bytes_track_the_router() {
        let all = CostOptions::default();
        let sparse = CostOptions::default().with_experts(8, 288);
        let idx = moe_index();
        let a = ModelCost::from_tensor_index(&idx, &all).unwrap();
        let s = ModelCost::from_tensor_index(&idx, &sparse).unwrap();
        assert_eq!(a.per_layer_expert_active_bytes, a.per_layer_expert_bytes);
        assert!(s.per_layer_expert_active_bytes * 30 < s.per_layer_expert_bytes);
        assert_eq!(s.per_layer_expert_bytes, a.per_layer_expert_bytes);
    }

    /// MLA's per-head `attn_k_b` is 3-D too. Classifying by rank instead of name
    /// would bill it to the disk budget and under-count RAM.
    #[test]
    fn rank_three_non_expert_tensors_stay_resident() {
        let idx = vec![
            TensorEntry::new("blk.0.attn_k_b.weight", vec![64, 512, 256], 9_000_000),
            TensorEntry::new("blk.0.attn_norm.weight", vec![4096], 16_384),
        ];
        let c = ModelCost::from_tensor_index(&idx, &CostOptions::default()).unwrap();
        assert_eq!(c.per_layer_expert_bytes, 0);
        assert_eq!(c.per_layer_bytes, 9_016_384);
        assert!(!c.is_expert_dominated());
    }

    #[test]
    fn hf_naming_works_too() {
        let idx = vec![
            TensorEntry::new("model.embed_tokens.weight", vec![32, 8], 1024),
            TensorEntry::new("model.layers.0.self_attn.q_proj.weight", vec![8, 8], 256),
            TensorEntry::new("model.layers.1.self_attn.q_proj.weight", vec![8, 8], 256),
            TensorEntry::new("model.layers.1.mlp.experts.0.up.weight", vec![8, 8], 4096),
            TensorEntry::new("lm_head.weight", vec![32, 8], 1024),
        ];
        let c = ModelCost::from_tensor_index(&idx, &CostOptions::default()).unwrap();
        assert_eq!(c.n_layers, 2);
        assert_eq!(c.embed_bytes, 1024);
        assert_eq!(c.head_bytes, 1024);
        assert_eq!(c.per_layer_expert_bytes, 2048);
    }

    /// Stepping down the precision ladder must shrink weights but leave the KV
    /// width and layer count alone — the cache does not follow weight precision.
    #[test]
    fn at_bits_scales_weights_only() {
        let idx = moe_index();
        let mut c = ModelCost::from_tensor_index(&idx, &CostOptions::default()).unwrap();
        c.kv = KvProfile::declared(4096);
        let before = c.total_bytes();
        let half = c
            .at_bits(c.total_bytes() as f64 * 8.0 / c.params as f64 / 2.0)
            .unwrap();
        assert!(
            (half.total_bytes() as f64 - before as f64 / 2.0).abs() < before as f64 * 0.01,
            "halving the width should roughly halve the bytes"
        );
        assert_eq!(half.n_layers, c.n_layers);
        assert_eq!(
            half.kv.bytes_per_layer_token(),
            c.kv.bytes_per_layer_token()
        );
    }

    #[test]
    fn at_bits_is_none_without_a_parameter_count() {
        let c = ModelCost {
            params: 0,
            ..Default::default()
        };
        assert!(c.at_bits(4.0).is_none());
    }

    #[test]
    fn kv_bytes_scale_with_layers_and_context() {
        let c = ModelCost {
            kv: KvProfile::declared(1024),
            ..Default::default()
        };
        assert_eq!(c.kv_bytes(4, 100), 4 * 100 * 1024);
        assert_eq!(c.kv_bytes(0, 100), 0);
    }

    // ── KV inference ────────────────────────────────────────────────────

    /// A GLM-5.3-Flash-shaped stack: 34 KDA layers whose state is O(1) in
    /// context, 11 latent-attention layers that cache only the 512-wide latent.
    fn hybrid_index() -> Vec<TensorEntry> {
        let (hidden, kv_lora, heads, head_dim) = (4096usize, 512usize, 64usize, 128usize);
        let mut v = Vec::new();
        for i in 0..45 {
            v.push(TensorEntry::new(
                format!("blk.{i}.attn_norm.weight"),
                vec![hidden],
                hidden as u64 * 4,
            ));
            if i % 4 == 3 {
                // latent attention: GGML dim order [in, out]
                v.push(TensorEntry::new(
                    format!("blk.{i}.attn_kv_a_mqa.weight"),
                    vec![hidden, kv_lora],
                    1,
                ));
            } else {
                v.push(TensorEntry::new(format!("blk.{i}.ssm_a"), vec![heads], 1));
                v.push(TensorEntry::new(
                    format!("blk.{i}.ssm_norm.weight"),
                    vec![head_dim],
                    1,
                ));
            }
        }
        v
    }

    /// The headline case: a hybrid model must not be charged an attention cache
    /// for its recurrent layers, and its recurrent state must not be charged per
    /// token. Getting either wrong misprices the model by ~4×.
    #[test]
    fn infers_hybrid_recurrent_plus_latent_attention() {
        let c = ModelCost::from_tensor_index(&hybrid_index(), &CostOptions::default()).unwrap();
        assert_eq!(c.kv.source(), KvSource::Inferred);
        // 11 latent layers × 512 × 4 B, averaged over 45.
        assert_eq!(c.kv.bytes_per_layer_token(), 11 * 512 * 4 / 45);
        // 34 recurrent layers × 64 × 128² × 4 B, averaged over 45.
        assert_eq!(c.kv.state_bytes_per_layer(), 34 * 64 * 128 * 128 * 4 / 45);

        // At 128 k this is ~2.95 GB of cache plus ~0.14 GB of fixed state.
        let total = c.kv.bytes(45, 131_072);
        assert!(
            (total as f64 / 1e9 - 3.09).abs() < 0.05,
            "expected ~3.09 GB, got {:.2} GB",
            total as f64 / 1e9
        );
    }

    /// Ordinary attention caches K and V, so twice the projection width.
    #[test]
    fn infers_gqa_attention_as_two_times_k_width() {
        let (hidden, kv_width) = (4096usize, 1024usize);
        let idx = vec![
            TensorEntry::new("blk.0.attn_norm.weight", vec![hidden], 1),
            TensorEntry::new("blk.0.attn_k.weight", vec![kv_width, hidden], 1),
        ];
        let c = ModelCost::from_tensor_index(&idx, &CostOptions::default()).unwrap();
        assert_eq!(c.kv.source(), KvSource::Inferred);
        assert_eq!(c.kv.bytes_per_layer_token(), 2 * kv_width as u64 * 4);
    }

    /// The inference is anchored on `hidden`, so it reads a GGUF index
    /// (`[in, out]`) and an HF one (`[out, in]`) identically.
    #[test]
    fn kv_inference_is_orientation_agnostic() {
        let (hidden, w) = (4096usize, 1024usize);
        let gguf = vec![
            TensorEntry::new("blk.0.attn_norm.weight", vec![hidden], 1),
            TensorEntry::new("blk.0.attn_k.weight", vec![hidden, w], 1),
        ];
        let hf = vec![
            TensorEntry::new("blk.0.attn_norm.weight", vec![hidden], 1),
            TensorEntry::new("blk.0.attn_k.weight", vec![w, hidden], 1),
        ];
        let a = ModelCost::from_tensor_index(&gguf, &CostOptions::default()).unwrap();
        let b = ModelCost::from_tensor_index(&hf, &CostOptions::default()).unwrap();
        assert_eq!(a.kv.bytes_per_layer_token(), b.kv.bytes_per_layer_token());
        assert_eq!(a.kv.bytes_per_layer_token(), 2 * w as u64 * 4);
    }

    /// Unrecognised attention leaves it Unknown — never a silent zero, which
    /// would read as "no cache" and reserve nothing.
    #[test]
    fn unreadable_attention_is_unknown_not_zero() {
        let idx = vec![
            TensorEntry::new("blk.0.attn_norm.weight", vec![4096], 1),
            TensorEntry::new("blk.0.mystery_attention.weight", vec![7, 9], 1),
        ];
        let c = ModelCost::from_tensor_index(&idx, &CostOptions::default()).unwrap();
        assert_eq!(c.kv.source(), KvSource::Unknown);
        assert!(!c.kv.is_known());
        assert_eq!(c.kv.bytes_per_layer_token(), 0);
    }

    /// A recurrent block whose head dims are unreadable must make the profile
    /// Unknown — not be counted as known and costed at zero state.
    ///
    /// The accumulator-based version had exactly this bug: the recurrent branch
    /// incremented `known` unconditionally and only added state when both dims
    /// parsed, so an unreadable linear-attention layer silently contributed
    /// nothing while still claiming the stack was fully understood.
    #[test]
    fn recurrent_block_with_unreadable_dims_is_unknown() {
        // `ssm_` marks it recurrent, but there is no rank-1 head-count tensor.
        let idx = vec![
            TensorEntry::new("blk.0.attn_norm.weight", vec![4096], 1),
            TensorEntry::new("blk.0.ssm_conv1d.weight", vec![8192, 1, 4], 1),
        ];
        let c = ModelCost::from_tensor_index(&idx, &CostOptions::default()).unwrap();
        assert_eq!(c.kv.source(), KvSource::Unknown);
        assert!(!c.kv.is_known());
    }

    /// Classification is order-sensitive: a linear-attention block also has an
    /// `attn_k`, and reading that as a KV cache would invent one where the layer
    /// has none.
    #[test]
    fn recurrent_wins_over_attn_k_in_the_same_block() {
        let idx = vec![
            TensorEntry::new("blk.0.attn_norm.weight", vec![4096], 1),
            TensorEntry::new("blk.0.attn_k.weight", vec![8192, 4096], 1),
            TensorEntry::new("blk.0.ssm_a", vec![64], 1),
            TensorEntry::new("blk.0.ssm_norm.weight", vec![128], 1),
        ];
        let c = ModelCost::from_tensor_index(&idx, &CostOptions::default()).unwrap();
        assert_eq!(
            c.kv.blocks(),
            [BlockKv::Recurrent {
                state_bytes: 64 * 128 * 128 * 4
            }]
        );
        assert_eq!(c.kv.bytes_per_layer_token(), 0, "no per-token cache");
    }

    /// Partial recognition is not knowledge: if one block cannot be read the
    /// average would under-estimate, which is the dangerous direction.
    #[test]
    fn partially_readable_stack_is_unknown() {
        let mut idx = hybrid_index();
        idx.retain(|t| !t.name.starts_with("blk.7."));
        idx.push(TensorEntry::new("blk.7.attn_norm.weight", vec![4096], 1));
        idx.push(TensorEntry::new("blk.7.something.weight", vec![3, 5], 1));
        let c = ModelCost::from_tensor_index(&idx, &CostOptions::default()).unwrap();
        assert_eq!(c.kv.source(), KvSource::Unknown);
    }

    /// A square projection is genuinely ambiguous (which dim is the output?), so
    /// it declines rather than guessing.
    #[test]
    fn square_projection_is_ambiguous_and_declines() {
        let idx = vec![
            TensorEntry::new("blk.0.attn_norm.weight", vec![4096], 1),
            TensorEntry::new("blk.0.attn_k.weight", vec![4096, 4096], 1),
        ];
        let c = ModelCost::from_tensor_index(&idx, &CostOptions::default()).unwrap();
        assert_eq!(c.kv.source(), KvSource::Unknown);
    }

    /// The average is not what any node pays. On the hybrid stack, layers 0..3
    /// hold no attention layer at all while 3..7 hold one — costing both at the
    /// mean over-reserves the first and under-reserves the second.
    #[test]
    fn per_range_kv_beats_the_average_on_a_hybrid_stack() {
        let c = ModelCost::from_tensor_index(&hybrid_index(), &CostOptions::default()).unwrap();
        let ctx = 4096;
        let none = c.kv.bytes_for(0..3, ctx); // blocks 0,1,2 are all recurrent
        let one = c.kv.bytes_for(3..7, ctx); // block 3 is latent attention
        let avg3 = c.kv.bytes(3, ctx);

        // No attention layer ⇒ no per-token cache, only recurrent state.
        assert_eq!(
            none,
            c.kv.blocks()[0..3]
                .iter()
                .map(BlockKv::state_bytes)
                .sum::<u64>()
        );
        // One attention layer ⇒ exactly its width × context, plus 3 states.
        assert_eq!(
            one,
            512 * 4 * ctx as u64
                + c.kv.blocks()[3..7]
                    .iter()
                    .map(BlockKv::state_bytes)
                    .sum::<u64>()
        );
        // The mean sits between the two, so it is wrong in both directions.
        assert!(
            avg3 > none,
            "average over-reserves the recurrent-only range"
        );
        assert!(
            c.kv.bytes(4, ctx) < one,
            "average under-reserves the range holding an attention layer"
        );
    }

    /// Capacity is decided before the range is, so it must budget for the WORST
    /// window of that length — otherwise the plan fits on average and dies on
    /// whichever node draws the attention-heavy slice.
    #[test]
    fn window_max_bounds_every_placement_of_a_given_length() {
        let c = ModelCost::from_tensor_index(&hybrid_index(), &CostOptions::default()).unwrap();
        let ctx = 4096;
        for len in [1usize, 3, 4, 8, 45] {
            let worst = c.kv.max_window_bytes(len, ctx);
            for start in 0..=(45 - len) {
                assert!(
                    c.kv.bytes_for(start..start + len, ctx) <= worst,
                    "window {start}..{} exceeds the bound at len {len}",
                    start + len
                );
            }
            assert!(
                worst >= c.kv.bytes(len as u64, ctx),
                "bound must cover the mean"
            );
        }
    }

    /// An f16 cache is half an f32 one; over-reserving 2× can refuse a plan that
    /// would have fit.
    #[test]
    fn cache_dtype_scales_the_reservation() {
        let f32o = CostOptions::default();
        let f16o = CostOptions {
            kv_dtype: "f16".into(),
            ..Default::default()
        };
        let a = ModelCost::from_tensor_index(&hybrid_index(), &f32o).unwrap();
        let b = ModelCost::from_tensor_index(&hybrid_index(), &f16o).unwrap();
        // Compare the exact per-block figures: the headline averages are integer
        // divisions over 45 blocks and so differ by a rounding step.
        assert_eq!(
            a.kv.blocks()[3].bytes_per_token(),
            2 * b.kv.blocks()[3].bytes_per_token()
        );
        // The recurrent state narrows too — it is a cache like any other.
        assert_eq!(
            a.kv.blocks()[0].state_bytes(),
            2 * b.kv.blocks()[0].state_bytes()
        );
        assert!(a.kv.blocks()[0].state_bytes() > 0);
        assert_eq!(kv_elem_bytes("bf16"), 2);
        assert_eq!(kv_elem_bytes("int8"), 1);
        assert_eq!(kv_elem_bytes("anything-else"), 4);
    }

    /// `hidden_size` is read from the checkpoint, not guessed from byte counts —
    /// the byte-count estimate is badly wrong for a quantized MoE.
    #[test]
    fn hidden_size_comes_from_the_checkpoint() {
        let c = ModelCost::from_tensor_index(&hybrid_index(), &CostOptions::default()).unwrap();
        assert_eq!(c.hidden_size, 4096);
        // What the old sqrt(per_layer_bytes/12) estimate would have said.
        let guessed = ((c.per_layer_bytes as f64 / 12.0).sqrt()).max(1.0) as usize;
        assert_ne!(guessed, c.hidden_size);
    }

    /// A declared figure has no per-block detail, so range queries must fall
    /// back to the average rather than returning zero.
    #[test]
    fn declared_profile_falls_back_to_the_average_for_ranges() {
        let k = KvProfile::declared(1000);
        assert_eq!(k.bytes_for(0..4, 10), 4 * 10 * 1000);
        assert_eq!(k.max_window_bytes(4, 10), 4 * 10 * 1000);
    }

    #[test]
    fn unrecognized_naming_is_an_error_not_a_zero_layer_model() {
        let idx = vec![TensorEntry::new("weird.0.w", vec![4], 16)];
        let err = ModelCost::from_tensor_index(&idx, &CostOptions::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("no numbered blocks"), "{err}");
    }
}
