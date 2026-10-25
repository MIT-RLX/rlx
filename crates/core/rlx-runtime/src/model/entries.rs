// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A package's entries: what its model block says a task method does, run
//! by rlx with no provider code.
//!
//! **`embed`** — a text encoder:
//!
//! ```text
//! entries: {embed: {
//!   functions: ["encode_32", "encode_128", "encode_512"],   // [batch, seq] each
//!   tokenizer: "tokenizer.json",          // beside the package, or "sidecar:<id>"
//!   inputs: {input_ids: "token_ids", attention_mask: "attention_mask",
//!            token_type_ids: "zeros", position_ids: "positions"},
//!   position_offset: 2,                   // RoBERTa-family positions start here
//!   pooling: "mean",                      // or "cls"
//!   normalize: true,
//! }}
//! ```
//!
//! A symbolic function (one graph for every `seq`) lists the sizes to run it
//! at: `functions: ["encode"], sizes: [{seq: 16}, {seq: 64}, {seq: 512}]`.
//! Each function, or size of one, is the encoder at one `[batch, seq]`
//! shape; a text runs in
//! the smallest that fits it (a longer one is cut to the largest, keeping its
//! last token). The first output is the hidden states `[batch, seq, dim]`,
//! pooled over the attention mask (or the first token) and L2-normalized.
//! One function is compiled at a time.
//!
//! **`generate`** — a causal language model, run one token at a time
//! through its decode functions (a prompt too, token by token):
//!
//! ```text
//! entries: {generate: {
//!   functions: ["decode_64", "decode_256", "decode_1024"],  // past capacity `upper` each
//!   tokenizer: "sidecar:tokenizer.json",
//!   chat: {chat_template: "sidecar:chat_template.jinja", bos_token: "…", eos_token: "…"},
//!   stop_ids: [151645, 151643],
//!   vocab_size: 151936,
//!   inputs: {input_ids: "token", mask: "decode_mask",
//!            rope_cos: {table: "rope_cos"}, rope_sin: {table: "rope_sin"}},
//!   state: [["past_k_0", 1], ["past_v_0", 2], …],          // input ← output index
//!   logits: 0,
//! }}
//! ```
//!
//! A step at cache length `past` runs the smallest function with
//! `upper >= past`: `token` is the id, `position` is `past`, `{table}` the
//! row `past` of a tensor the package holds, `decode_mask` 1 for the `past`
//! cached rows and the new one at `upper`. Each `state` input is the cache
//! padded to its shape; the output it names holds the cache's rows, then the
//! new row last.
//!
//! A stateful step (`rlx_ir::stateful::stateful_decode`) needs only
//! `{input_ids: "token", position: "position"}` and
//! `state_update: "replace"`: it computes its RoPE rows and mask, writes the
//! new row at `position`, and its cache output is the next step's input.
//! `upper` is then one less than a cache's rows. `chat`,
//! `generate_text` and `generate` follow from it, greedy or sampled.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use rlx_ir::{DType, Dim, Graph, Op};
use serde_json::{Value, json};

use super::rlxp::{Opened, PackageWeights, params_of};
use super::{
    ChatMessage, Model, ModelInfo, ModelRequest, Tensor, TensorSpec, TextCodec, TextEmbedding,
    lm_model, text_embedding_model,
};
use crate::{CompiledGraph, Device};

/// What a graph input is fed.
#[derive(Clone, Copy, PartialEq)]
enum Role {
    TokenIds,
    AttentionMask,
    Zeros,
    Positions,
    /// Each token's segment: 0 in a pair's first text, 1 in its second.
    TokenTypes,
}

impl Role {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "token_ids" => Self::TokenIds,
            "attention_mask" => Self::AttentionMask,
            "zeros" => Self::Zeros,
            "positions" => Self::Positions,
            "token_types" => Self::TokenTypes,
            other => bail!(
                "unknown input role '{other}' (token_ids, attention_mask, zeros, positions, \
                 token_types)"
            ),
        })
    }
}

/// One function: the encoder at one shape.
struct Bucket {
    graph: Graph,
    batch: usize,
    seq: usize,
    /// Each graph input: its name, dtype and role.
    inputs: Vec<(String, DType, Role)>,
    params: Vec<(String, DType, Vec<usize>)>,
}

/// An encoder's functions over `[batch, seq]` token inputs, run a batch
/// of texts (or pairs) at a time: what `embed` and `score` share.
struct Encoder {
    buckets: Vec<Bucket>,
    /// The bucket compiled now, and its executable.
    compiled: Option<(usize, CompiledGraph)>,
    weights: PackageWeights,
    device: Device,
    position_offset: usize,
}

pub(super) fn static_dims(g: &Graph, id: rlx_ir::NodeId) -> Result<Vec<usize>> {
    g.node(id)
        .shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Ok(*n),
            Dim::Dynamic(_) => bail!(
                "graph '{}' has a dynamic dim; embed needs static shapes",
                g.name
            ),
        })
        .collect()
}

impl Encoder {
    /// Entry `key`'s functions (`entries.<key>.functions` at its `sizes`,
    /// each input named in its `inputs`) and the package's weights.
    fn open(request: &ModelRequest, pkg: &mut Opened, entry: &Value, key: &str) -> Result<Self> {
        let roles: Vec<(String, Role)> = entry
            .get("inputs")
            .and_then(Value::as_object)
            .with_context(|| format!("entries.{key}.inputs maps each graph input to a role"))?
            .iter()
            .map(|(k, v)| Ok((k.clone(), Role::parse(v.as_str().unwrap_or_default())?)))
            .collect::<Result<_>>()?;
        let functions: Vec<&str> = entry
            .get("functions")
            .and_then(Value::as_array)
            .with_context(|| format!("entries.{key}.functions lists the encoder graphs"))?
            .iter()
            .filter_map(Value::as_str)
            .collect();
        let mut graphs = Vec::new();
        for f in &functions {
            graphs.extend(pkg.take_function(f, entry.get("sizes"))?);
        }
        let mut buckets = Vec::new();
        for graph in graphs {
            let f = graph.name.clone();
            let mut inputs = Vec::new();
            let mut shape = None;
            for n in graph.nodes() {
                let Op::Input { name } = &n.op else { continue };
                let role = roles
                    .iter()
                    .find(|(k, _)| k == name)
                    .map(|(_, r)| *r)
                    .with_context(|| {
                        format!("input '{name}' of '{f}' has no role in entries.{key}.inputs")
                    })?;
                let dims = static_dims(&graph, n.id)?;
                let [b, s] = dims[..] else {
                    bail!("input '{name}' of '{f}' is {dims:?}; {key} inputs are [batch, seq]");
                };
                shape.get_or_insert((b, s));
                inputs.push((name.clone(), n.shape.dtype(), role));
            }
            let (batch, seq) = shape.with_context(|| format!("'{f}' has no inputs"))?;
            let params = params_of(&graph);
            buckets.push(Bucket {
                graph,
                batch,
                seq,
                inputs,
                params,
            });
        }
        buckets.sort_by_key(|b| b.seq);
        if buckets.is_empty() {
            bail!("entries.{key}.functions is empty");
        }
        Ok(Self {
            buckets,
            compiled: None,
            weights: std::mem::replace(&mut pkg.weights, PackageWeights::empty()),
            device: request.device,
            position_offset: entry
                .get("position_offset")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
        })
    }

    /// The first output's dims in the shortest function.
    fn output_dims(&self) -> Result<Vec<usize>> {
        let g = &self.buckets[0].graph;
        static_dims(g, *g.outputs.first().context("the encoder has no output")?)
    }

    /// The longest sequence a function takes.
    fn longest(&self) -> usize {
        self.buckets.last().map_or(0, |b| b.seq)
    }

    /// Bucket `b`, compiled and bound.
    fn executable(&mut self, b: usize) -> Result<&mut CompiledGraph> {
        if self.compiled.as_ref().map(|(i, _)| *i) != Some(b) {
            self.compiled = None;
            let bucket = &self.buckets[b];
            let mut compiled = self.weights.compile(self.device, bucket.graph.clone());
            self.weights.bind(&mut compiled, &bucket.params)?;
            self.compiled = Some((b, compiled));
        }
        Ok(&mut self.compiled.as_mut().expect("just compiled").1)
    }

    /// Each of `texts` (ids and token types, cut to [`Self::longest`])
    /// through the shortest function that fits it, batched by function:
    /// `each(i, ids, output row)` with the first output's row of text `i`.
    fn run(
        &mut self,
        texts: Vec<(Vec<u32>, Vec<u32>)>,
        mut each: impl FnMut(usize, &[u32], &[f32]),
    ) -> Result<()> {
        let mut jobs: Vec<(usize, usize, Vec<u32>, Vec<u32>)> = texts
            .into_iter()
            .enumerate()
            .map(|(i, (ids, types))| {
                let b = self
                    .buckets
                    .iter()
                    .position(|b| b.seq >= ids.len())
                    .unwrap_or(self.buckets.len() - 1);
                (b, i, ids, types)
            })
            .collect();
        // By bucket, so each function compiles once per call.
        jobs.sort_by_key(|(b, i, _, _)| (*b, *i));
        let mut start = 0;
        while start < jobs.len() {
            let b = jobs[start].0;
            let (batch, seq) = (self.buckets[b].batch, self.buckets[b].seq);
            let end = (start..jobs.len())
                .take(batch)
                .take_while(|&j| jobs[j].0 == b)
                .last()
                .map_or(start, |j| j + 1);
            let rows = &jobs[start..end];
            let mut ids = vec![0f32; batch * seq];
            let mut mask = vec![0f32; batch * seq];
            let mut types = vec![0f32; batch * seq];
            for (r, (_, _, t, tt)) in rows.iter().enumerate() {
                for (s, &id) in t.iter().enumerate() {
                    ids[r * seq + s] = id as f32;
                    mask[r * seq + s] = 1.0;
                    types[r * seq + s] = tt.get(s).copied().unwrap_or(0) as f32;
                }
            }
            let positions: Vec<f32> = (0..batch)
                .flat_map(|_| (0..seq).map(|s| (s + self.position_offset) as f32))
                .collect();
            let zeros = vec![0f32; batch * seq];
            let feeds: Vec<(String, Vec<u8>, DType)> = self.buckets[b]
                .inputs
                .iter()
                .map(|(name, dtype, role)| {
                    let v = match role {
                        Role::TokenIds => &ids,
                        Role::AttentionMask => &mask,
                        Role::Zeros => &zeros,
                        Role::Positions => &positions,
                        Role::TokenTypes => &types,
                    };
                    let (bytes, dt) = typed(v, *dtype)?;
                    Ok((name.clone(), bytes, dt))
                })
                .collect::<Result<_>>()?;
            let out = *self.buckets[b]
                .graph
                .outputs
                .first()
                .context("the encoder has no output")?;
            let dims = static_dims(&self.buckets[b].graph, out)?;
            let exe = self.executable(b)?;
            let refs: Vec<(&str, &[u8], DType)> = feeds
                .iter()
                .map(|(n, d, t)| (n.as_str(), d.as_slice(), *t))
                .collect();
            let outs = exe.run_typed(&refs);
            let (data, dtype) = outs
                .into_iter()
                .next()
                .context("the encoder returned nothing")?;
            let out = super::Tensor::new("out", dtype, dims, data).to_f32()?;
            let row = out.len() / batch;
            for (r, (_, i, t, _)) in rows.iter().enumerate() {
                each(*i, t, &out[r * row..(r + 1) * row]);
            }
            start = end;
        }
        Ok(())
    }
}

/// `ids` cut to `longest`, keeping the last (the end-of-sequence token).
fn cut(mut ids: Vec<u32>, longest: usize) -> Vec<u32> {
    if ids.len() > longest {
        let last = *ids.last().expect("non-empty");
        ids.truncate(longest - 1);
        ids.push(last);
    }
    ids
}

struct TextEncoder {
    family: String,
    dim: usize,
    codec: Box<dyn TextCodec>,
    encoder: Encoder,
    cls: bool,
    normalize: bool,
}

/// The `embed` entry of `pkg` as a model.
pub(super) fn text_encoder(
    request: &ModelRequest,
    mut pkg: Opened,
    entry: &Value,
) -> Result<Box<dyn Model>> {
    let metadata = pkg.metadata();
    let encoder = Encoder::open(request, &mut pkg, entry, "embed")?;
    let dim = *encoder.output_dims()?.last().unwrap_or(&0);
    let tokenizer = entry
        .get("tokenizer")
        .and_then(Value::as_str)
        .context("entries.embed.tokenizer names the tokenizer.json")?;
    let codec = super::load_tokenizer(&pkg.file(tokenizer)?, &Value::Null)?;
    let info = ModelInfo {
        provider: "rlxp".into(),
        family: pkg.meta_str("family", "encoder"),
        task: "embedding".into(),
        device: request.device_label(),
        metadata: with_contracts(
            metadata,
            json!([{"name": "text", "kind": "text"}]),
            json!([{"name": "embedding", "kind": "embedding", "dtype": "f32", "shape": [dim]}]),
        ),
        ..Default::default()
    };
    let model = TextEncoder {
        family: info.family.clone(),
        dim,
        codec,
        encoder,
        cls: entry.get("pooling").and_then(Value::as_str) == Some("cls"),
        normalize: entry
            .get("normalize")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    };
    Ok(text_embedding_model(model, info))
}

/// `values` as the bytes and dtype to feed an input declared `dtype`.
///
/// I32 / U32 go in as f32 values, the way `CompiledGraph::run` feeds them
/// and every family runner does: on CPU their slot holds integers
/// f32-aliased for some kernels (Gather's indices) and as i32 for others
/// (custom ops), and the f32 route is the one gather-fed models (token
/// embeddings) are built and checked against.
pub(super) fn typed(values: &[f32], dtype: DType) -> Result<(Vec<u8>, DType)> {
    Ok(match dtype {
        DType::F32 | DType::I32 | DType::U32 => (
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            DType::F32,
        ),
        DType::I64 => (
            values
                .iter()
                .flat_map(|v| (*v as i64).to_le_bytes())
                .collect(),
            DType::I64,
        ),
        other => bail!("entries feed f32 / i32 / u32 / i64 inputs, not {other:?}"),
    })
}

impl TextEmbedding for TextEncoder {
    fn family(&self) -> &str {
        &self.family
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&mut self, texts: &[&str], _options: &Value) -> Result<Vec<Vec<f32>>> {
        let longest = self.encoder.longest();
        let texts = texts
            .iter()
            .map(|t| Ok((cut(self.codec.encode(t)?, longest), Vec::new())))
            .collect::<Result<Vec<_>>>()?;
        let mut out = vec![Vec::new(); texts.len()];
        let (dim, cls, normalize) = (self.dim, self.cls, self.normalize);
        self.encoder.run(texts, |i, t, row| {
            let mut v = if cls {
                row[..dim].to_vec()
            } else {
                let mut acc = vec![0f32; dim];
                for s in 0..t.len() {
                    for (a, x) in acc.iter_mut().zip(&row[s * dim..(s + 1) * dim]) {
                        *a += x;
                    }
                }
                let inv = 1.0 / (t.len().max(1) as f32);
                acc.iter_mut().for_each(|a| *a *= inv);
                acc
            };
            if normalize {
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-12;
                v.iter_mut().for_each(|x| *x /= norm);
            }
            out[i] = v;
        })?;
        Ok(out)
    }
}

// ── score: a cross-encoder ──────────────────────────────────

struct TextScorer {
    family: String,
    codec: Box<dyn TextCodec>,
    encoder: Encoder,
    /// The output column that is the score (a one-label head: 0).
    label: usize,
    /// `"sigmoid"`: scores in (0, 1); else the raw logit.
    activation: Option<String>,
}

/// The `score` entry of `pkg`: each (query, passage) pair through the
/// encoder whose output is `[batch, labels]` logits.
pub(super) fn text_scorer(
    request: &ModelRequest,
    mut pkg: Opened,
    entry: &Value,
) -> Result<Box<dyn Model>> {
    let metadata = pkg.metadata();
    let encoder = Encoder::open(request, &mut pkg, entry, "score")?;
    let dims = encoder.output_dims()?;
    let [_, labels] = dims[..] else {
        bail!("entries.score: the encoder's output is {dims:?}, not [batch, labels]");
    };
    let label = entry.get("label").and_then(Value::as_u64).unwrap_or(0) as usize;
    if label >= labels {
        bail!("entries.score.label {label}: the head has {labels} labels");
    }
    let tokenizer = entry
        .get("tokenizer")
        .and_then(Value::as_str)
        .context("entries.score.tokenizer names the tokenizer.json")?;
    let codec = super::load_tokenizer(&pkg.file(tokenizer)?, &Value::Null)?;
    let info = ModelInfo {
        provider: "rlxp".into(),
        family: pkg.meta_str("family", "encoder"),
        task: super::tasks::TEXT_RANKING.into(),
        device: request.device_label(),
        metadata: with_contracts(
            metadata,
            json!([{"name": "query", "kind": "text"}, {"name": "passage", "kind": "text"}]),
            json!([{"name": "scores", "kind": "scores", "dtype": "f32", "shape": [null]}]),
        ),
        ..Default::default()
    };
    let model = TextScorer {
        family: info.family.clone(),
        codec,
        encoder,
        label,
        activation: entry
            .get("activation")
            .and_then(Value::as_str)
            .map(String::from),
    };
    Ok(super::text_ranking_model(model, info))
}

impl super::TextRanking for TextScorer {
    fn family(&self) -> &str {
        &self.family
    }

    fn score(&mut self, query: &str, passages: &[&str], _options: &Value) -> Result<Vec<f32>> {
        let longest = self.encoder.longest();
        // A pair too long for the longest function loses the end of its
        // passage (the last token, the separator, stays).
        let pairs = passages
            .iter()
            .map(|p| {
                let (ids, mut types) = self.codec.encode_pair(query, p)?;
                let ids = cut(ids, longest);
                if types.len() > ids.len() {
                    let last = *types.last().expect("non-empty");
                    types.truncate(ids.len() - 1);
                    types.push(last);
                }
                Ok((ids, types))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut out = vec![0f32; passages.len()];
        let label = self.label;
        self.encoder.run(pairs, |i, _, row| out[i] = row[label])?;
        if self.activation.as_deref() == Some("sigmoid") {
            out.iter_mut().for_each(|s| *s = 1.0 / (1.0 + (-*s).exp()));
        }
        Ok(out)
    }
}

// ── generate: a causal language model ───────────────────────

/// What a decode input is fed.
enum Feed {
    Token,
    /// The token's position (`f32[1]`): the number of tokens before it.
    Position,
    /// The row each sequence of a batch starts at (`f32[batch]`): 0, but
    /// for a request that joined a running batch.
    Starts,
    DecodeMask,
    /// Row `past` of a table the package holds, `width` wide.
    TableRow(usize),
    /// The cache of `state[i]`.
    State(usize),
    /// A tensor an earlier stage computed once a request (an encoder's
    /// cross-attention K / V), by name.
    Context(String),
    /// What the RoPE position is past the row (`f32[1]`): 0, but after a
    /// prompt whose media share positions (multimodal RoPE).
    RopeOffset,
}

struct DecodeBucket {
    graph: Graph,
    upper: usize,
    /// Sequences a step decodes: the token input's element count.
    batch: usize,
    /// Each input: name, dtype, element count, what feeds it.
    inputs: Vec<(String, DType, usize, Feed)>,
    params: Vec<(String, DType, Vec<usize>)>,
}

/// A prompt-length function: the whole prompt in one run, padded to its
/// `seq`, the caches' rows and the last position's logits out.
struct PrefillBucket {
    graph: Graph,
    /// Prompts it runs at once (`prefill.batches`); 1 for `prefill` itself.
    batch: usize,
    seq: usize,
    tokens: (String, DType),
    /// Where the prompt's last token is (`len - 1`), when it takes it.
    last: Option<(String, DType)>,
    /// Inputs read from the model's context (`{context: <name>}`: an
    /// encoder's states), as `(input, value, dtype)`. A graph symbolic in
    /// their length is specialized to it when it runs.
    context: Vec<(String, String, DType)>,
    params: Vec<(String, DType, Vec<usize>)>,
}

struct Prefill {
    buckets: Vec<PrefillBucket>,
    /// The bucket compiled, at the binding its context gave it.
    compiled: Option<((usize, Vec<(u32, usize)>), CompiledGraph)>,
    /// Free the executable after each prompt (open option `prefill:
    /// "release"`): each executable holds its own copy of the weights, so
    /// keeping it beside the step's doubles them — 2 × ~4 GB for Qwen3-0.6B
    /// in f32 on a GPU — at the cost of a compile per prompt.
    release: bool,
    /// Where the prompt runs (open option `prefill_device`), when not on
    /// the model's device: its K/V rows then seed the step's caches on the
    /// model's device, as from any prefill.
    device: Option<crate::Device>,
    logits_out: usize,
    /// The output whose first rows seed each cache, in `states` order.
    state_outputs: Vec<usize>,
}

/// One cache: the state input it feeds, the output that updates it, its row
/// width.
struct State {
    input: String,
    output: usize,
    width: usize,
    kind: StateKind,
}

/// How a step's output becomes a state's next value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StateKind {
    /// A cache whose new row the step wrote at the position (a stateful
    /// step): its first `past + 1` rows.
    Rows,
    /// A cache returned with the new row last (a bucketed step): the past
    /// rows, then that row.
    Append,
    /// Recurrent state (an SSM's, a convolution's window): the whole output.
    Replace,
}

impl StateKind {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "rows" => Self::Rows,
            "append" => Self::Append,
            "replace" => Self::Replace,
            other => bail!("state kind '{other}': rows, append or replace"),
        })
    }
}

struct GraphLm {
    family: &'static str,
    vocab: usize,
    buckets: Vec<DecodeBucket>,
    compiled: Option<(usize, CompiledGraph)>,
    weights: PackageWeights,
    device: Device,
    /// Each table a feed reads rows of: its tensor name, values, row width.
    tables: Vec<(String, Vec<f32>, usize)>,
    states: Vec<State>,
    /// The symbolic step: specialized at a longer cache when the context
    /// outgrows the listed sizes (up to `max_context`), and at a batch for
    /// `generate_batch`.
    grow: Option<Grow>,
    /// The step sized for the last `generate_batch`: `(upper, batch)`, the
    /// executable, its bucket, and whether it keeps the caches (each step
    /// then copies only the new rows there, as a single sequence's does).
    batched: Option<((usize, usize), CompiledGraph, DecodeBucket, bool)>,
    /// The package's `check` function, for speculative decoding.
    check: Option<Check>,
    /// Images (or other media) in a prompt: the entry's `media`.
    media: Option<Media>,
    /// Decode speculatively when the package has a check (open option
    /// `speculative`): drafts by prompt lookup, each checked in one run.
    speculative: bool,
    /// A draft model (open option `draft`: a package with the same
    /// tokenizer) whose greedy guesses the check takes instead.
    draft: Option<Box<GraphLm>>,
    /// Continuous batching's requests ([`crate::LmRunner::serve_open`]).
    serving: Option<Serving>,
    /// How the keys hold their positions (the entry's `rope`), so a batch
    /// can move a request's rows: rotated back by the rows they moved.
    rope: Option<KeyRope>,
    /// The RoPE position less the row of the next token: set by a media
    /// prompt whose positions are not its rows (`media.mrope`), 0 after
    /// [`Self::reset`].
    rope_offset: i64,
    /// Run the prompts' shared prefix through the prefill function before a
    /// batch (open option `batch_prefill`, on by default).
    batch_prefill: bool,
    /// Decode batches and beams in a batched step (open option `batched`,
    /// on by default); off, one sequence at a time.
    batched_step: bool,
    logits_out: usize,
    /// `state_update: "replace"`: a cache output is the next cache input
    /// (a stateful step); else its rows then the new row last.
    replace_state: bool,
    /// The caches live in the executable (`bind_gpu_handle`) and each step
    /// copies only its new row there (`feed_kv_row`); `cache` is then filled
    /// only when the context moves to a longer function.
    resident: bool,
    /// The package's weight-sharing scope (`with_param_sharing_scope`):
    /// with `RLX_CUDA_SHARED_PARAMS=1`, its prompt function and decode step
    /// map one copy of the weights on a CUDA device instead of one each.
    scope: String,
    /// `cache` holds what the executable's caches should (a prefill wrote
    /// it): bind it before the next step.
    cache_fresh: bool,
    prefill: Option<Prefill>,
    /// The caches, `past` rows each, and the tokens in them.
    cache: Vec<Vec<f32>>,
    tokens: Vec<u32>,
    last_logits: Option<Vec<f32>>,
    /// The `Feed::Context` tensors of the current request.
    context: HashMap<String, Vec<f32>>,
}

/// The binding of `bucket`'s symbols its context inputs give: each one's
/// length over its static dims (one symbolic dim each).
fn context_binding(
    bucket: &PrefillBucket,
    context: &HashMap<String, Vec<f32>>,
) -> Result<Vec<(u32, usize)>> {
    let mut binding: Vec<(u32, usize)> = Vec::new();
    for (input, value, _) in &bucket.context {
        let Some(id) = bucket.graph.node_id_by_name(input) else {
            continue;
        };
        let dims = bucket.graph.node(id).shape.dims().to_vec();
        let syms: Vec<u32> = dims
            .iter()
            .filter_map(|d| match d {
                rlx_ir::Dim::Dynamic(s) => Some(*s),
                _ => None,
            })
            .collect();
        let sym = match syms.as_slice() {
            [] => continue,
            [s] => *s,
            _ => bail!(
                "prefill input '{input}' has {} symbolic dims; one is bound",
                syms.len()
            ),
        };
        let fixed: usize = dims
            .iter()
            .filter_map(|d| match d {
                rlx_ir::Dim::Static(n) => Some(*n),
                _ => None,
            })
            .product();
        let len = context.get(value).map_or(0, Vec::len);
        if fixed == 0 || !len.is_multiple_of(fixed) {
            bail!("'{value}' has {len} values, not a whole number of {fixed}");
        }
        let n = len / fixed;
        match binding.iter().find(|(s, _)| *s == sym) {
            Some((_, m)) if *m != n => bail!("one symbol bound to {m} and {n}"),
            Some(_) => {}
            None => binding.push((sym, n)),
        }
    }
    binding.sort();
    Ok(binding)
}

/// A codec with the package's stop ids.
struct Stopping {
    inner: Box<dyn TextCodec>,
    stop: Vec<u32>,
}

impl TextCodec for Stopping {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }
    fn encode_plain(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode_plain(text)
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner.decode(ids)
    }
    fn stop_ids(&self) -> Vec<u32> {
        let mut s = self.inner.stop_ids();
        s.extend(&self.stop);
        s
    }
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        self.inner.apply_chat_template(messages)
    }
    fn apply_chat_template_json(&self, messages: &[Value], options: &Value) -> Result<String> {
        self.inner.apply_chat_template_json(messages, options)
    }
}

/// The `generate` entry of `pkg` as a model.
/// The decode loop an entry describes (`functions`, `inputs`, `state`,
/// `prefill`, …): `generate`'s, and the decoder of `transcribe`. Takes the
/// package's weights.
fn graph_lm(request: &ModelRequest, pkg: &mut Opened, entry: &Value) -> Result<GraphLm> {
    let feeds = entry
        .get("inputs")
        .and_then(Value::as_object)
        .context("entries.generate.inputs says what feeds each decode input")?
        .clone();

    // No decode step, only a prompt function (`prefill`): each token runs it
    // over the whole context, as a runner with no cache does.
    let recompute = entry.get("functions").is_none() && entry.get("prefill").is_some();
    let functions: Vec<&str> = match entry.get("functions") {
        _ if recompute => Vec::new(),
        f => f
            .and_then(Value::as_array)
            .context("entries.generate.functions lists the decode graphs")?
            .iter()
            .filter_map(Value::as_str)
            .collect(),
    };
    // Each function at each size; a symbolic one is kept to grow into when
    // the entry says how far (`max_context`).
    // The open option `max_context` can lower the package's (its tables
    // reach no further), not raise it.
    let max_context = entry
        .get("max_context")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .map(|n| {
            request
                .option("max_context")
                .and_then(Value::as_u64)
                .map_or(n, |m| n.min(m as usize))
        });
    let growth = request
        .option("context_growth")
        .and_then(Value::as_u64)
        .map_or(2, |g| g.max(2) as usize);
    let mut graphs = Vec::new();
    let mut symbolic = None;
    for f in &functions {
        let graph = pkg.take_graph(f)?;
        if !graph.has_dynamic_dims() {
            graphs.push((graph, None));
            continue;
        }
        let sizes = entry
            .get("sizes")
            .and_then(Value::as_array)
            .with_context(|| format!("'{f}' is symbolic: its entry lists `sizes` to run it at"))?;
        for size in sizes {
            let (g, binding) = pkg.specialize_at(&graph, size)?;
            graphs.push((g, Some(binding)));
        }
        symbolic = Some(graph);
    }
    // The caches: as the entry pairs them, else the graph's variables
    // (`%k = var`, `return %k = %k_next`), whose outputs replace them.
    let state_from_vars = entry.get("state").is_none();
    // A state's kind: its entry's third field, else `state_kinds[name]`,
    // else what `state_update` says of every cache.
    let default_kind = if state_from_vars
        || entry.get("state_update").and_then(Value::as_str) == Some("replace")
    {
        StateKind::Rows
    } else {
        StateKind::Append
    };
    let kinds = entry.get("state_kinds").and_then(Value::as_object);
    let kind_of = |name: &str, explicit: Option<&str>| -> Result<StateKind> {
        match explicit.or_else(|| kinds.and_then(|k| k.get(name)).and_then(Value::as_str)) {
            Some(k) => StateKind::parse(k),
            None => Ok(default_kind),
        }
    };
    let state_pairs: Vec<(String, usize, StateKind)> = match entry.get("state") {
        Some(_) => entry
            .get("state")
            .and_then(Value::as_array)
            .context("entries.generate.state pairs each cache input with its output")?
            .iter()
            .map(|p| {
                let name = p.get(0).and_then(Value::as_str);
                let out = p.get(1).and_then(Value::as_u64);
                match (name, out) {
                    (Some(n), Some(o)) => Ok((
                        n.to_string(),
                        o as usize,
                        kind_of(n, p.get(2).and_then(Value::as_str))?,
                    )),
                    _ => bail!("a state entry is [input name, output index, kind?], found {p}"),
                }
            })
            .collect::<Result<_>>()?,
        // No step, no caches.
        None if recompute => Vec::new(),
        None => {
            let (g, _) = graphs
                .first()
                .context("entries.generate.functions is empty")?;
            if g.vars().is_empty() {
                bail!("entries.generate names no `state`, and its graph declares no variables");
            }
            g.vars()
                .iter()
                .map(|(input, out)| match &g.node(*input).op {
                    Op::Input { name } => Ok((name.clone(), *out, kind_of(name, None)?)),
                    _ => bail!("variable {input} of '{}' is not an input", g.name),
                })
                .collect::<Result<_>>()?
        }
    };
    let mut tables: Vec<(String, Vec<f32>, usize)> = Vec::new();
    let mut states: Vec<State> = state_pairs
        .iter()
        .map(|(n, o, kind)| State {
            input: n.clone(),
            output: *o,
            width: 0,
            kind: *kind,
        })
        .collect();

    let mut buckets = Vec::new();
    let mut largest: Option<(usize, Vec<(u32, usize)>)> = None;
    for (graph, binding) in graphs {
        let bucket = decode_bucket(graph, &feeds, &mut states, &mut tables, &mut pkg.weights)?;
        if let Some(binding) = binding
            && largest.as_ref().is_none_or(|(u, _)| bucket.upper > *u)
        {
            largest = Some((bucket.upper, binding));
        }
        buckets.push(bucket);
    }
    // Every symbol of the longest listed size, as an offset from its cache
    // length (`cap = upper + 1`): a longer size keeps the offsets.
    let batch_sym = match entry.get("batch").and_then(Value::as_str) {
        Some(name) => Some(
            pkg.dims
                .iter()
                .find(|(d, _)| d == name)
                .map(|(_, s)| *s)
                .with_context(|| {
                    format!("entries.generate.batch names `{name}`, which no dim is")
                })?,
        ),
        None => None,
    };
    let grow = match (symbolic, largest) {
        (Some(graph), Some((upper, binding))) => Some(Grow {
            graph,
            feeds: feeds.clone(),
            offsets: binding
                .into_iter()
                .filter(|(sym, _)| Some(*sym) != batch_sym)
                .map(|(sym, n)| (sym, n as i64 - upper as i64))
                .collect(),
            max_context,
            growth,
            batch: batch_sym,
        }),
        _ => None,
    };
    buckets.sort_by_key(|b| b.upper);
    if buckets.is_empty() && !recompute {
        bail!("entries.generate.functions is empty");
    }
    let family: &'static str = Box::leak(pkg.meta_str("family", "lm").into_boxed_str());
    let vocab = entry
        .get("vocab_size")
        .and_then(Value::as_u64)
        .context("entries.generate.vocab_size")? as usize;
    let media = match entry.get("media").filter(|m| !m.is_null()) {
        Some(m) => Some(Media::open(request, pkg, m)?),
        None => None,
    };
    let rope = match entry.get("rope").filter(|r| !r.is_null()) {
        Some(r) => Some(KeyRope::of(r, &pkg.weights)?),
        None => None,
    };
    let check = match entry.get("check").filter(|c| !c.is_null()) {
        Some(c) => Some(check_of(pkg, c, &mut states, &mut tables)?),
        None => None,
    };
    let prefill = match entry.get("prefill") {
        Some(p) => {
            let mut prefill = prefill_of(pkg, p, states.len())?;
            prefill.release = release_option(request, pkg);
            prefill.device = super::option(&request.options, "prefill_device")
                .and_then(Value::as_str)
                .map(crate::parse_device)
                .transpose()
                .map_err(|e| anyhow::anyhow!("prefill_device: {e}"))?;
            Some(prefill)
        }
        None => None,
    };
    let runner = GraphLm {
        family,
        vocab,
        buckets,
        compiled: None,
        resident: false,
        scope: format!("rlxp:{}", pkg.path.display()),
        cache_fresh: false,
        prefill,
        weights: std::mem::replace(&mut pkg.weights, PackageWeights::empty()),
        device: request.device,
        tables,
        grow,
        batched: None,
        check,
        media,
        speculative: request
            .option("speculative")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        draft: None,
        serving: None,
        rope,
        rope_offset: 0,
        batch_prefill: request
            .option("batch_prefill")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        batched_step: request
            .option("batched")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        cache: vec![Vec::new(); states.len()],
        states,
        logits_out: entry.get("logits").and_then(Value::as_u64).unwrap_or(0) as usize,
        replace_state: state_from_vars
            || entry.get("state_update").and_then(Value::as_str) == Some("replace"),
        tokens: Vec::new(),
        last_logits: None,
        context: HashMap::new(),
    };
    Ok(runner)
}

/// A decode function's bucket: what feeds each input, and the longest
/// cache it takes. Records each cache's row width in `states`, and takes
/// each table a feed reads out of `weights` the first time.
fn decode_bucket(
    graph: Graph,
    feeds: &serde_json::Map<String, Value>,
    states: &mut [State],
    tables: &mut Vec<(String, Vec<f32>, usize)>,
    weights: &mut PackageWeights,
) -> Result<DecodeBucket> {
    let f = graph.name.clone();
    // Sequences it decodes: one token each.
    let batch = graph
        .nodes()
        .iter()
        .find_map(|n| match &n.op {
            Op::Input { name } if feeds.get(name.as_str()) == Some(&Value::from("token")) => {
                Some(n.shape.num_elements().unwrap_or(1).max(1))
            }
            _ => None,
        })
        .unwrap_or(1);
    let mut inputs = Vec::new();
    let mut upper = None;
    for n in graph.nodes() {
        let Op::Input { name } = &n.op else { continue };
        let dims = static_dims(&graph, n.id)?;
        let numel: usize = dims.iter().product();
        let feed = if let Some(k) = states.iter().position(|s| &s.input == name) {
            states[k].width = *dims.last().unwrap_or(&0);
            Feed::State(k)
        } else {
            match feeds.get(name.as_str()) {
                Some(Value::String(r)) if r == "token" || r == "tokens" => Feed::Token,
                Some(Value::String(r)) if r == "position" => Feed::Position,
                Some(Value::String(r)) if r == "starts" => Feed::Starts,
                Some(Value::String(r)) if r == "rope_offset" => Feed::RopeOffset,
                Some(Value::String(r)) if r == "decode_mask" => {
                    upper = Some(numel / batch - 1);
                    Feed::DecodeMask
                }
                Some(Value::Object(o)) if o.contains_key("context") => Feed::Context(
                    o["context"]
                        .as_str()
                        .with_context(|| format!("input '{name}': {{context: <name>}}"))?
                        .to_string(),
                ),
                Some(Value::Object(o)) => {
                    let table = o
                        .get("table")
                        .and_then(Value::as_str)
                        .with_context(|| format!("input '{name}': {{table: <tensor>}}"))?;
                    let t = match tables.iter().position(|(n, _, _)| n == table) {
                        Some(t) => t,
                        None => {
                            tables.push((table.into(), weights.take_table(table)?, numel / batch));
                            tables.len() - 1
                        }
                    };
                    Feed::TableRow(t)
                }
                _ => bail!("input '{name}' of '{f}' has no feed in entries.generate.inputs"),
            }
        };
        inputs.push((name.clone(), n.shape.dtype(), numel, feed));
    }
    // The longest cache it takes: `upper` with a decode mask, else one
    // less than a cache input's rows (the new row is written in place).
    let upper = match upper {
        Some(u) => u,
        None => inputs
            .iter()
            .find_map(|(_, _, numel, feed)| match feed {
                Feed::State(k) if states[*k].width > 0 && states[*k].kind != StateKind::Replace => {
                    Some(numel / batch / states[*k].width - 1)
                }
                _ => None,
            })
            .or_else(|| {
                // Only recurrent state (a Mamba-style step): no length
                // bounds it.
                let recurrent = inputs.iter().all(|(_, _, _, feed)| match feed {
                    Feed::State(k) => states[*k].kind == StateKind::Replace,
                    _ => true,
                });
                (recurrent && !states.is_empty()).then_some(usize::MAX)
            })
            .with_context(|| format!("'{f}' has neither a decode_mask nor a cache input"))?,
    };
    let params = params_of(&graph);
    Ok(DecodeBucket {
        graph,
        upper,
        batch,
        inputs,
        params,
    })
}

/// A draft of up to `k` tokens after `context`: what followed the latest
/// earlier occurrence of its last 3 tokens (else 2, else 1). Empty when
/// none recurs.
fn lookup_draft(context: &[u32], k: usize) -> Vec<u32> {
    let len = context.len();
    for n in (1..=3usize).rev() {
        if len <= n {
            continue;
        }
        let tail = &context[len - n..];
        if let Some(i) = (0..len - n).rev().find(|&i| &context[i..i + n] == tail) {
            let from = i + n;
            return context[from..(from + k).min(len)].to_vec();
        }
    }
    Vec::new()
}

/// A `generate` entry's `media`: what turns an image (or other media) into
/// rows of the language model's input embedding, and the prefill that takes
/// embeddings.
///
/// ```text
/// media: {
///   embed:   {function, inputs: {input_ids: "ids"}, outputs: ["embeds"]},
///   prefill: {function, inputs: {inputs_embeds: "prompt_embeds", last: "last"},
///             outputs: ["logits", <each cache, in the step's order>],
///             sizes: [64, 256, 1024]},              // prompt lengths padded to
///   kinds: {image: {token: 151655, input: "image", stages: [...],
///                   output: "image_embeds"}},       // [rows, hidden]
/// }
/// ```
/// Its stages run from their own handle on the package (symbolic ones
/// compiled per shape).
struct Media {
    stages: Stages,
    embed: usize,
    prefill: usize,
    /// Prompt lengths the prefill is padded to (shortest that holds it).
    sizes: Vec<usize>,
    kinds: Vec<MediaKind>,
    /// Multimodal RoPE: each token's rows from its 3-D position.
    mrope: Option<Mrope>,
}

struct MediaKind {
    name: String,
    token: u32,
    /// The value its first stage reads the media from.
    input: String,
    stages: std::ops::Range<usize>,
    output: String,
    /// With `mrope`: the pixels per embedding row along each side (`grid:
    /// {patch}`), so an `[height, width, …]` medium is a `height / patch`
    /// by `width / patch` grid of rows.
    patch: Option<usize>,
    /// Further stage outputs laid out like the embedding (`[rows, width]`),
    /// each fed to the prefill as `[1, seq, width]`, zeros but at the
    /// medium's positions (Qwen3-VL's deepstack features, added after the
    /// first layers).
    rows: Vec<String>,
    /// `resize: {multiple, min_pixels, max_pixels}`: a `u8 [height, width,
    /// 3]` image resized first to the processor's size for it
    /// ([`super::image_resize::smart_resize`], antialiased bicubic), so a
    /// stage symbolic in the grid takes any image. `over_frames`: a video's
    /// pixel bounds are over all its frames (Qwen3-VL), not each frame's.
    resize: Option<(usize, usize, usize, bool)>,
    /// `frames: {patch, layout, t_step}`: a video, `u8 [frames, height,
    /// width, 3]`, run through the stages `patch` frames at a time (the last
    /// repeated to fill a group), each group a `[patch, height, width, 3]`
    /// input; `resize` then bounds the pixels of all the frames.
    frames: Option<Frames>,
}

/// How a video's frame groups take their places in the prompt.
struct Frames {
    patch: usize,
    /// `layout: "runs"`: each group its own run of placeholders, placed as
    /// an image is (Qwen3-VL, its timestamps in the text between).
    /// `"joined"`: one run of every group's rows, group `g` at time
    /// position `g · t_step` past the start (Qwen2.5-VL).
    runs: bool,
    t_step: usize,
}

/// `media.mrope: {sections, layout, cos, sin}`: multimodal RoPE (Qwen-VL).
/// A token has a position per axis, `(t, h, w)`: text the same three, a
/// medium's rows its grid's (from where the text left off), and the text
/// after a medium goes on from one past its largest. Frequency `i` of a
/// row turns by the position of its axis: `layout: "sections"` gives the
/// first `sections[0]` frequencies to `t`, the next `sections[1]` to `h`,
/// the rest to `w` (Qwen2-VL, Qwen2.5-VL); `"interleaved"` cycles `t, h,
/// w` over the first `3 · sections[1]` (Qwen3-VL). `cos` / `sin` are the
/// 1-D tables (`[rows, half]`). The prefill reads the prompt's rows as the
/// values `rope_cos` / `rope_sin` (`[seq, half]`), and the step goes on with
/// `rope_offset` (the next position less the next row).
struct Mrope {
    sections: Vec<usize>,
    interleaved: bool,
    cos: Vec<f32>,
    sin: Vec<f32>,
    half: usize,
}

impl Mrope {
    fn of(r: &Value, weights: &PackageWeights) -> Result<Self> {
        let sections: Vec<usize> = r
            .get("sections")
            .and_then(Value::as_array)
            .context("media.mrope.sections")?
            .iter()
            .map(|v| v.as_u64().map(|v| v as usize))
            .collect::<Option<_>>()
            .context("media.mrope.sections holds counts")?;
        let half: usize = sections.iter().sum();
        if sections.len() != 3 || half == 0 {
            bail!("media.mrope.sections is {sections:?}: three counts (t, h, w)");
        }
        let interleaved = match r
            .get("layout")
            .and_then(Value::as_str)
            .unwrap_or("sections")
        {
            "sections" => false,
            "interleaved" => true,
            other => bail!("media.mrope.layout '{other}': sections or interleaved"),
        };
        let table = |k: &str| -> Result<Vec<f32>> {
            let name = r
                .get(k)
                .and_then(Value::as_str)
                .with_context(|| format!("media.mrope.{k} names its table"))?;
            weights.values(name, &[0, half])
        };
        Ok(Self {
            sections,
            interleaved,
            cos: table("cos")?,
            sin: table("sin")?,
            half,
        })
    }

    /// The axis frequency `i` turns by.
    fn axis(&self, i: usize) -> usize {
        let s = &self.sections;
        if self.interleaved {
            match i % 3 {
                1 if i < 3 * s[1] => 1,
                2 if i < 3 * s[2] => 2,
                _ => 0,
            }
        } else if i < s[0] {
            0
        } else if i < s[0] + s[1] {
            1
        } else {
            2
        }
    }

    /// The cos and sin rows of each of `positions`.
    fn rows(&self, positions: &[[usize; 3]]) -> Result<(Vec<f32>, Vec<f32>)> {
        let h = self.half;
        let (mut cos, mut sin) = (
            Vec::with_capacity(positions.len() * h),
            Vec::with_capacity(positions.len() * h),
        );
        for p in positions {
            for i in 0..h {
                let at = p[self.axis(i)] * h + i;
                let (c, s) = self.cos.get(at).zip(self.sin.get(at)).with_context(|| {
                    format!("position {} is past the RoPE tables", p[self.axis(i)])
                })?;
                cos.push(*c);
                sin.push(*s);
            }
        }
        Ok((cos, sin))
    }
}

impl Media {
    fn open(request: &ModelRequest, from: &Opened, m: &Value) -> Result<Self> {
        let mut pkg = from.reopen()?;
        let embed = m
            .get("embed")
            .context("media.embed turns token ids into embeddings")?;
        let prefill = m.get("prefill").context("media.prefill takes embeddings")?;
        let mut list = vec![embed.clone(), prefill.clone()];
        let mut kinds = Vec::new();
        for (name, k) in m
            .get("kinds")
            .and_then(Value::as_object)
            .context("media.kinds names each kind of media")?
        {
            let stages = k
                .get("stages")
                .and_then(Value::as_array)
                .with_context(|| format!("media.kinds.{name}.stages"))?;
            let at = list.len();
            list.extend(stages.iter().cloned());
            kinds.push(MediaKind {
                name: name.clone(),
                token: k
                    .get("token")
                    .and_then(Value::as_u64)
                    .with_context(|| format!("media.kinds.{name}.token is its placeholder id"))?
                    as u32,
                input: k
                    .get("input")
                    .and_then(Value::as_str)
                    .unwrap_or(name)
                    .to_string(),
                stages: at..list.len(),
                output: k
                    .get("output")
                    .and_then(Value::as_str)
                    .with_context(|| format!("media.kinds.{name}.output names its embedding"))?
                    .to_string(),
                patch: k
                    .pointer("/grid/patch")
                    .and_then(Value::as_u64)
                    .map(|p| p as usize),
                rows: k
                    .get("rows")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .map(String::from)
                            .collect()
                    })
                    .unwrap_or_default(),
                resize: match k.get("resize").filter(|r| !r.is_null()) {
                    Some(r) => {
                        let n = |key: &str| {
                            r.get(key)
                                .and_then(Value::as_u64)
                                .map(|v| v as usize)
                                .with_context(|| format!("media.kinds.{name}.resize.{key}"))
                        };
                        if r.get("filter")
                            .and_then(Value::as_str)
                            .is_some_and(|f| f != "bicubic")
                        {
                            bail!("media.kinds.{name}.resize.filter: bicubic");
                        }
                        let over_frames = r
                            .get("over_frames")
                            .and_then(Value::as_bool)
                            .unwrap_or(false);
                        Some((
                            n("multiple")?,
                            n("min_pixels")?,
                            n("max_pixels")?,
                            over_frames,
                        ))
                    }
                    None => None,
                },
                frames: match k.get("frames").filter(|f| !f.is_null()) {
                    Some(f) => {
                        let patch = f
                            .get("patch")
                            .and_then(Value::as_u64)
                            .filter(|&p| p > 0)
                            .with_context(|| format!("media.kinds.{name}.frames.patch"))?
                            as usize;
                        let runs = match f.get("layout").and_then(Value::as_str) {
                            Some("runs") | None => true,
                            Some("joined") => false,
                            Some(other) => bail!(
                                "media.kinds.{name}.frames.layout is '{other}': runs or joined"
                            ),
                        };
                        let t_step = f.get("t_step").and_then(Value::as_u64).unwrap_or(1) as usize;
                        Some(Frames {
                            patch,
                            runs,
                            t_step,
                        })
                    }
                    None => None,
                },
            });
        }
        let mrope = match m.get("mrope").filter(|r| !r.is_null()) {
            Some(r) => Some(Mrope::of(r, &pkg.weights)?),
            None => None,
        };
        let mut stages = Stages::open(request, &mut pkg, &list)?;
        stages.release = release_option(request, from);
        stages.weights[0] = std::mem::replace(&mut pkg.weights, PackageWeights::empty());
        let mut sizes: Vec<usize> = prefill
            .get("sizes")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|v| v as usize))
                    .collect()
            })
            .unwrap_or_default();
        sizes.sort();
        if mrope.is_some()
            && let Some(k) = kinds.iter().find(|k| k.patch.is_none())
        {
            bail!("media.mrope needs media.kinds.{}.grid.patch", k.name);
        }
        Ok(Self {
            stages,
            embed: 0,
            prefill: 1,
            sizes,
            kinds,
            mrope,
        })
    }
}

/// `rope: {cos, sin, style, head_dim, keys}`: the RoPE tables (`[rows,
/// head_dim/2]`, `cos(p·θ)` / `sin(p·θ)`), the pairing (`"neox"`: `i` with
/// `i + head_dim/2`; `"gptj"`: `2i` with `2i + 1`), and the prefix of the
/// caches that hold keys (`past_k_`).
struct KeyRope {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half: usize,
    neox: bool,
    keys: String,
}

impl KeyRope {
    fn of(r: &Value, weights: &PackageWeights) -> Result<Self> {
        let field = |k: &str| {
            r.get(k)
                .and_then(Value::as_str)
                .with_context(|| format!("entries.generate.rope.{k}"))
        };
        let head_dim = r
            .get("head_dim")
            .and_then(Value::as_u64)
            .context("entries.generate.rope.head_dim")? as usize;
        let half = head_dim / 2;
        Ok(Self {
            cos: weights.values(field("cos")?, &[0, half])?,
            sin: weights.values(field("sin")?, &[0, half])?,
            half,
            neox: match field("style")? {
                "neox" => true,
                "gptj" => false,
                other => bail!("rope.style '{other}': neox or gptj"),
            },
            keys: field("keys")?.to_string(),
        })
    }

    /// Key `row` (heads of `2·half`) as written `delta` positions later:
    /// each pair rotated by `-delta·θ`.
    fn move_back(&self, row: &mut [f32], delta: usize) -> Result<()> {
        let (half, d) = (self.half, 2 * self.half);
        let at = delta * half;
        let (cos, sin) = (
            self.cos
                .get(at..at + half)
                .context("a move past the RoPE table")?,
            self.sin
                .get(at..at + half)
                .context("a move past the RoPE table")?,
        );
        for head in row.chunks_exact_mut(d) {
            for i in 0..half {
                let (a, b) = if self.neox {
                    (i, i + half)
                } else {
                    (2 * i, 2 * i + 1)
                };
                let (x1, x2) = (head[a], head[b]);
                head[a] = x1 * cos[i] + x2 * sin[i];
                head[b] = x2 * cos[i] - x1 * sin[i];
            }
        }
        Ok(())
    }
}

/// Requests decoded together, joining and leaving between steps: every
/// slot at the batch's position `past`, each from its own `start` row.
struct Serving {
    slots: Vec<Option<ServingSlot>>,
    past: usize,
    /// The caches when the step does not keep them (`[slots, rows, w]`).
    caches: Vec<Vec<f32>>,
}

struct ServingSlot {
    start: usize,
    /// Its prompt's tokens still to go in.
    pending: std::collections::VecDeque<u32>,
    /// The token it is fed next once its prompt is in.
    last: u32,
}

/// A package's `check` function: several tokens at once after its cache
/// (the last token, then a draft), for speculative decoding.
struct Check {
    /// One per listed size, shortest first.
    buckets: Vec<DecodeBucket>,
    tokens: usize,
    /// Each cache: the input and the output that is its next value.
    caches: Vec<(String, usize, usize)>,
    logits_out: usize,
    compiled: Option<(usize, CompiledGraph)>,
}

/// A symbolic decode function and how to size it at a longer cache or
/// another batch.
struct Grow {
    graph: Graph,
    feeds: serde_json::Map<String, Value>,
    /// Each symbol's size less the cache length, at the listed sizes (the
    /// batch symbol aside).
    offsets: Vec<(u32, i64)>,
    /// The longest context the package's tables and positions cover; with
    /// none, the step does not grow past the listed sizes.
    max_context: Option<usize>,
    /// The step's batch symbol (entry `batch`), for `generate_batch`.
    batch: Option<u32>,
    /// How many times longer each grown step's cache is (open option
    /// `context_growth`, 2 by default).
    growth: usize,
}

/// `entries.generate.check`: `{function, sizes, tokens, inputs, logits}`,
/// its caches the graph's variables.
fn check_of(
    pkg: &mut Opened,
    c: &Value,
    states: &mut [State],
    tables: &mut Vec<(String, Vec<f32>, usize)>,
) -> Result<Check> {
    let function = c
        .get("function")
        .and_then(Value::as_str)
        .context("entries.generate.check.function names the check")?;
    let feeds = c
        .get("inputs")
        .and_then(Value::as_object)
        .context("entries.generate.check.inputs says what feeds each input")?
        .clone();
    let tokens = c.get("tokens").and_then(Value::as_u64).unwrap_or(1) as usize;
    let mut buckets = Vec::new();
    let mut caches = Vec::new();
    for graph in pkg.take_function(function, c.get("sizes"))? {
        if caches.is_empty() {
            for (input, out) in graph.vars() {
                let Op::Input { name } = &graph.node(*input).op else {
                    bail!("a variable of '{function}' is not an input");
                };
                let width = graph
                    .node(*input)
                    .shape
                    .dims()
                    .last()
                    .map_or(0, |d| d.unwrap_static());
                caches.push((name.clone(), *out, width));
            }
        }
        // Its caches are the step's states by name (their widths, the
        // tables they read); its own feeds, `tokens` and `position`.
        let mut f = feeds.clone();
        for (name, _, _) in &caches {
            f.remove(name);
        }
        buckets.push(decode_bucket(graph, &f, states, tables, &mut pkg.weights)?);
    }
    buckets.sort_by_key(|b| b.upper);
    Ok(Check {
        buckets,
        tokens,
        caches,
        logits_out: c.get("logits").and_then(Value::as_u64).unwrap_or(0) as usize,
        compiled: None,
    })
}

pub(super) fn causal_lm(
    request: &ModelRequest,
    mut pkg: Opened,
    entry: &Value,
) -> Result<Box<dyn Model>> {
    let metadata = pkg.metadata();
    let mut runner = graph_lm(request, &mut pkg, entry)?;
    let entry_vocab = entry.get("vocab_size").cloned().unwrap_or(Value::Null);
    // `draft`: a smaller package's guesses for the check.
    if let Some(path) = request.option("draft").and_then(Value::as_str) {
        let mut d = Opened::open_merged(std::path::Path::new(path))?;
        d.target = request.device_label();
        let entry = super::rlxp::entry_of_kind(&d.meta, "generate")
            .with_context(|| format!("draft {path} has no generate entry"))?;
        if entry.get("vocab_size") != Some(&entry_vocab) {
            bail!("draft {path} has another vocabulary than the model");
        }
        runner.draft = Some(Box::new(graph_lm(request, &mut d, &entry)?));
    }
    let family = runner.family;
    let tokenizer = entry
        .get("tokenizer")
        .and_then(Value::as_str)
        .context("entries.generate.tokenizer names the tokenizer.json")?;
    // `chat.chat_template` names a file; the loader takes its text.
    let mut chat = entry.get("chat").cloned().unwrap_or(Value::Null);
    if let Some(reference) = chat.get("chat_template").and_then(Value::as_str) {
        let src = String::from_utf8(pkg.file(reference)?).context("chat template is not UTF-8")?;
        chat["chat_template"] = Value::String(src);
    }
    let codec = super::load_tokenizer(&pkg.file(tokenizer)?, &chat)?;
    let stop: Vec<u32> = entry
        .get("stop_ids")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|v| v as u32))
                .collect()
        })
        .unwrap_or_default();
    let info = ModelInfo {
        provider: "rlxp".into(),
        family: family.into(),
        task: "text-generation".into(),
        device: request.device_label(),
        metadata: {
            let mut m = with_contracts(
                metadata,
                json!([{"name": "text", "kind": "text"}]),
                json!([{"name": "text", "kind": "text"}]),
            );
            // Generation defaults (`generation_config.json`): the text
            // methods use them where the caller says nothing.
            if let Some(g) = entry.get("generation").filter(|g| g.is_object()) {
                m["generation"] = g.clone();
            }
            m
        },
        ..Default::default()
    };
    Ok(lm_model(
        Box::new(runner),
        Some(Box::new(Stopping { inner: codec, stop })),
        info,
    ))
}

/// Whether prompt and media executables are freed after use: `prefill:
/// "release"` says so, `"keep"` says not, and otherwise a package whose
/// params would not fit twice (`Opened::tight`) does.
fn release_option(request: &ModelRequest, pkg: &Opened) -> bool {
    match super::option(&request.options, "prefill").and_then(Value::as_str) {
        Some("release") => true,
        Some("keep") => false,
        _ => pkg.tight,
    }
}

/// `entries.generate.prefill`: `{function, sizes, inputs: {<name>:
/// "tokens" | "last"}, logits, state: [output per cache], batches:
/// [{function, batch}]}`, `batches` the same function at fixed batch sizes
/// (same sizes, inputs and outputs, each output `[batch, …]`).
fn prefill_of(pkg: &mut Opened, p: &Value, states: usize) -> Result<Prefill> {
    // The prompt function, symbolic in its length (`function` + `sizes`),
    // or one per length (`functions`, a model whose graph is not symbolic).
    let mut functions: Vec<(String, usize)> = p
        .get("function")
        .and_then(Value::as_str)
        .map(|f| (f.to_string(), 1usize))
        .into_iter()
        .collect();
    for f in p
        .get("functions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let f = f
            .as_str()
            .context("entries.generate.prefill.functions lists function names")?;
        functions.push((f.to_string(), 1));
    }
    if functions.is_empty() {
        bail!("entries.generate.prefill.function names the prompt function");
    }
    for b in p
        .get("batches")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let f = b
            .get("function")
            .and_then(Value::as_str)
            .context("entries.generate.prefill.batches[].function")?;
        let n = b
            .get("batch")
            .and_then(Value::as_u64)
            .filter(|&n| n > 0)
            .context("entries.generate.prefill.batches[].batch")?;
        functions.push((f.to_string(), n as usize));
    }
    let feeds = p
        .get("inputs")
        .and_then(Value::as_object)
        .context("entries.generate.prefill.inputs says what feeds each input")?;
    let state_outputs: Vec<usize> = p
        .get("state")
        .and_then(Value::as_array)
        .context("entries.generate.prefill.state lists the output seeding each cache")?
        .iter()
        .map(|v| v.as_u64().map(|o| o as usize))
        .collect::<Option<_>>()
        .context("entries.generate.prefill.state holds output indices")?;
    if state_outputs.len() != states {
        bail!(
            "entries.generate.prefill.state seeds {} caches; the step keeps {states}",
            state_outputs.len()
        );
    }
    let mut buckets = Vec::new();
    for (function, batch) in &functions {
        let (function, batch) = (function.as_str(), *batch);
        // A function fed from the context stays symbolic in it (an encoder's
        // length): specialized when it runs.
        let from_context = feeds.values().any(|f| f.get("context").is_some());
        let graphs = if from_context && p.get("sizes").is_none() {
            vec![pkg.take_graph(function)?]
        } else {
            pkg.take_function(function, p.get("sizes"))?
        };
        for graph in graphs {
            let (mut tokens, mut last) = (None, None);
            let mut context = Vec::new();
            for n in graph.nodes() {
                let Op::Input { name } = &n.op else { continue };
                let slot = Some((name.clone(), n.shape.dtype()));
                let feed = feeds.get(name.as_str());
                match feed.and_then(Value::as_str) {
                    Some("tokens") => tokens = slot.map(|s| (s, static_dims(&graph, n.id))),
                    Some("last") => last = slot,
                    _ => match feed.and_then(|f| f.get("context")).and_then(Value::as_str) {
                        Some(v) => context.push((name.clone(), v.to_string(), n.shape.dtype())),
                        None => bail!(
                            "input '{name}' of '{function}' has no feed in \
                             entries.generate.prefill.inputs"
                        ),
                    },
                }
            }
            let ((name, dtype), dims) =
                tokens.with_context(|| format!("'{function}' has no `tokens` input"))?;
            let seq: usize = dims?.iter().product::<usize>() / batch;
            let params = params_of(&graph);
            buckets.push(PrefillBucket {
                graph,
                batch,
                seq,
                tokens: (name, dtype),
                last,
                context,
                params,
            });
        }
    }
    buckets.sort_by_key(|b| (b.batch, b.seq));
    Ok(Prefill {
        buckets,
        compiled: None,
        release: false,
        device: None,
        logits_out: p.get("logits").and_then(Value::as_u64).unwrap_or(0) as usize,
        state_outputs,
    })
}

impl GraphLm {
    /// A step for a context of `past` tokens past every listed size: the
    /// symbolic step at `growth` times the longest cache (at least `past`,
    /// at most `max_context`). Nothing without a symbolic step, or past its reach.
    fn grow_to(&mut self, past: usize) -> Result<()> {
        let Some(max_context) = self.grow.as_ref().and_then(|g| g.max_context) else {
            return Ok(());
        };
        let growth = self.grow.as_ref().map_or(2, |g| g.growth);
        let longest = self.buckets.iter().map(|b| b.upper).max().unwrap_or(0);
        if past > max_context {
            return Ok(());
        }
        let upper = ((longest + 1) * growth - 1).max(past).min(max_context);
        let bucket = self.sized_bucket(upper, 1)?;
        if bucket.upper < past {
            bail!("a step sized for {upper} tokens holds {}", bucket.upper);
        }
        self.buckets.push(bucket);
        Ok(())
    }

    /// The symbolic step at a `upper`-token cache for `batch` sequences.
    fn sized_bucket(&mut self, upper: usize, batch: usize) -> Result<DecodeBucket> {
        let grow = self
            .grow
            .as_ref()
            .context("the package's decode step is not symbolic")?;
        let mut binding: Vec<(u32, usize)> = grow
            .offsets
            .iter()
            .map(|&(sym, offset)| (sym, (upper as i64 + offset).max(0) as usize))
            .collect();
        if let Some(sym) = grow.batch {
            binding.push((sym, batch));
        } else if batch > 1 {
            bail!("the package's decode step has no batch symbol (entry `batch`)");
        }
        let mut graph =
            rlx_ir::dynamic::specialize(&grow.graph, &rlx_ir::DimBinding::from_pairs(&binding));
        if graph.has_dynamic_dims() {
            bail!(
                "'{}' at a {upper}-token context, batch {batch}, still has unbound dims",
                grow.graph.name
            );
        }
        graph.name = format!("{}@{upper}x{batch}", grow.graph.name);
        let feeds = grow.feeds.clone();
        decode_bucket(
            graph,
            &feeds,
            &mut self.states,
            &mut self.tables,
            &mut self.weights,
        )
    }

    /// Greedy decode of `prompts` together, in lockstep: every sequence
    /// starts at position 0 and each step feeds one token of each — its
    /// prompt's while it has one, then what it generated — so all are at
    /// the same position and prompts of any lengths share one step. A
    /// sequence ends at a `stop` id (kept out) or `n_new` tokens. The caches
    /// pass through the host (`[batch, rows, width]` each).
    fn decode_together(
        &mut self,
        prompts: &[Vec<u32>],
        n_new: usize,
        stop: &[u32],
    ) -> Result<Vec<Vec<u32>>> {
        let b = prompts.len();
        let longest = prompts.iter().map(Vec::len).max().unwrap_or(0);
        // The last step feeds token `longest + n_new - 2`, at that past.
        let last_past = longest + n_new - 2;
        // The prompts first: with `prefill: "release"` their function is gone
        // before the batched step compiles.
        let (start, mut caches) = self.prefill_together(prompts)?;
        self.ensure_batched(b, last_past)?;
        if start > 0 {
            self.seed_together(&caches, start)?;
        }
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); b];
        let mut done = vec![false; b];
        for past in start.. {
            let tokens: Vec<u32> = (0..b)
                .map(|s| {
                    let p = &prompts[s];
                    match past.checked_sub(p.len()) {
                        None => p[past],
                        Some(g) => out[s].get(g).copied().unwrap_or(0),
                    }
                })
                .collect();
            let logits = self.step_together(&tokens, &vec![0.0; b], past, &mut caches)?;
            let per = logits.len() / b;
            for s in 0..b {
                if done[s] || past + 1 < prompts[s].len() {
                    continue;
                }
                let row = &logits[s * per..(s + 1) * per];
                let next = crate::lm::argmax_u32(&row[per.saturating_sub(self.vocab)..]);
                if stop.contains(&next) {
                    done[s] = true;
                    continue;
                }
                out[s].push(next);
                done[s] = out[s].len() >= n_new;
            }
            if done.iter().all(|d| *d) {
                break;
            }
        }
        // `prefill: "release"` keeps one copy of the weights on the device:
        // the batched step goes too.
        if self.prefill.as_ref().is_some_and(|p| p.release) {
            self.batched = None;
        }
        Ok(out)
    }

    /// Forget all but the first `n` tokens (a draft model's rejected
    /// guesses): the step writes over the rest. Recurrent state cannot go
    /// back, so it starts over.
    fn truncate(&mut self, n: usize) {
        if n >= self.tokens.len() {
            return;
        }
        if self.states.iter().any(|s| s.kind == StateKind::Replace) {
            self.reset();
            return;
        }
        self.tokens.truncate(n);
        if !self.resident {
            for (k, st) in self.states.iter().enumerate() {
                self.cache[k].truncate(n * st.width);
            }
        }
        self.last_logits = None;
    }

    /// As a draft model: its `k` greedy guesses after `context`, from its
    /// own cache (rolled back to where it and `context` agree).
    fn draft(&mut self, context: &[u32], k: usize) -> Result<Vec<u32>> {
        if k == 0 || context.is_empty() {
            return Ok(Vec::new());
        }
        let common = self
            .tokens
            .iter()
            .zip(context)
            .take_while(|(a, b)| a == b)
            .count();
        // At least the last token goes in again, for its logits.
        self.truncate(common.min(context.len() - 1));
        let mut logits = crate::LmRunner::predict_logits(self, context)?;
        let mut out = Vec::with_capacity(k);
        loop {
            let next = crate::lm::argmax_u32(&logits);
            out.push(next);
            if out.len() == k {
                return Ok(out);
            }
            logits = self.step(next)?;
        }
    }

    /// Greedy decoding through the package's check: each run feeds the last
    /// token and a draft of the next ones (what followed the context's
    /// latest n-gram the last time it appeared), and keeps the draft as far
    /// as the model's own picks agree, plus its pick where they part. The
    /// same tokens as one at a time, in fewer runs when drafts hold. The
    /// prompt goes in `tokens` at a time; the caches stay in the check's
    /// executable.
    fn decode_speculative(
        &mut self,
        prompt: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.reset();
        let check = self.check.as_ref().expect("checked by generate");
        let q = check.tokens.max(1);
        // The last run is at the context's last position: `p + q` rows.
        let need = prompt.len() + n_new - 1 + q;
        let b = check
            .buckets
            .iter()
            .position(|bk| bk.upper + 1 >= need)
            .with_context(|| {
                format!(
                    "the context is full: {need} rows for a check (this package checks up to {})",
                    check.buckets.last().map_or(0, |bk| bk.upper + 1)
                )
            })?;
        if self
            .check
            .as_ref()
            .unwrap()
            .compiled
            .as_ref()
            .map(|(i, _)| *i)
            != Some(b)
        {
            let check = self.check.as_mut().unwrap();
            check.compiled = None;
            let bucket = &check.buckets[b];
            let weights = &mut self.weights;
            let mut c = crate::with_param_sharing_scope(&self.scope, || {
                let mut c = weights.compile(self.device, bucket.graph.clone());
                weights.bind(&mut c, &bucket.params).map(|()| c)
            })?;
            for (input, out, width) in &check.caches {
                let numel = bucket
                    .inputs
                    .iter()
                    .find(|i| &i.0 == input)
                    .map(|i| i.2)
                    .with_context(|| format!("the check has no input '{input}'"))?;
                if *width == 0
                    || !c.bind_gpu_handle(input, &vec![0.0; numel])
                    || !c.register_kv_row_feed(input, *out)
                {
                    bail!("the backend keeps no caches for the check (speculative: false)");
                }
            }
            check.compiled = Some((b, c));
        }
        let width = self.check.as_ref().unwrap().caches[0].2;
        let vocab = self.vocab;
        let check = self.check.as_mut().unwrap();
        let mut drafter = self.draft.as_deref_mut();
        // One run of the check: `toks` (q of them) from position `p`; each
        // one's logits.
        let mut run = |toks: &[u32], p: usize| -> Result<Vec<Vec<f32>>> {
            let (b, exe) = check.compiled.as_mut().unwrap();
            let bucket = &check.buckets[*b];
            let mut feeds: Vec<(&str, Vec<f32>)> = Vec::new();
            for (name, _, _, feed) in &bucket.inputs {
                match feed {
                    Feed::Token => feeds.push((name, toks.iter().map(|&t| t as f32).collect())),
                    Feed::Position => feeds.push((name, vec![p as f32])),
                    Feed::State(_) => {}
                    _ => bail!("the check takes no '{name}' from the host"),
                }
            }
            let refs: Vec<(&str, &[f32])> = feeds.iter().map(|(n, v)| (*n, v.as_slice())).collect();
            let logits = exe
                .run_read_outputs(&refs, Some(&[check.logits_out]))
                .pop()
                .context("the check returned no logits")?;
            for i in 0..toks.len() {
                if !exe.feed_kv_row(p + i, p + i, width) {
                    bail!("the backend took the check's caches but cannot update them");
                }
            }
            let per = logits.len() / toks.len();
            Ok((0..toks.len())
                .map(|i| logits[(i + 1) * per - vocab.min(per)..(i + 1) * per].to_vec())
                .collect())
        };
        // The prompt but its last token, q at a time (the last run padded;
        // its extra rows are written over next).
        let last_at = prompt.len() - 1;
        for start in (0..last_at).step_by(q) {
            let mut toks = prompt[start..last_at.min(start + q)].to_vec();
            toks.resize(q, 0);
            run(&toks, start)?;
        }
        let mut context = prompt.to_vec();
        let mut produced = Vec::new();
        let mut p = last_at;
        loop {
            let draft = match drafter.as_deref_mut() {
                Some(d) => d.draft(&context, q - 1)?,
                None => lookup_draft(&context, q - 1),
            };
            let mut toks = vec![*context.last().expect("a prompt")];
            toks.extend(&draft);
            toks.resize(q, 0);
            let rows = run(&toks, p)?;
            let mut kept = 0;
            loop {
                let next = crate::lm::argmax_u32(&rows[kept]);
                produced.push(next);
                context.push(next);
                if !on_token(next) || produced.len() == n_new {
                    return Ok(produced);
                }
                if kept < draft.len() && next == draft[kept] {
                    kept += 1;
                } else {
                    break;
                }
            }
            p += kept + 1;
        }
    }

    /// The step sized for `b` sequences up to a `last_past`-token context,
    /// compiled (kept from the last batch when it fits) and its caches
    /// taken where it keeps them.
    fn ensure_batched(&mut self, b: usize, last_past: usize) -> Result<()> {
        // `prefill: "release"` keeps one executable's weights at a time: the
        // single-sequence step goes while a batch runs.
        if self.prefill.as_ref().is_some_and(|p| p.release) {
            self.compiled = None;
            self.reset();
        }
        let listed = self
            .buckets
            .iter()
            .filter(|bk| bk.batch == 1 && bk.upper != usize::MAX)
            .map(|bk| bk.upper);
        let upper = listed
            .clone()
            .filter(|&u| u >= last_past)
            .min()
            .or_else(|| {
                let max = self.grow.as_ref()?.max_context?;
                (max >= last_past).then(|| {
                    let mut u = listed.max().unwrap_or(1).max(1);
                    while u < last_past {
                        u = (u + 1) * 2 - 1;
                    }
                    u.min(max)
                })
            })
            .with_context(|| {
                format!("the context is full: {last_past} tokens for the longest sequence")
            })?;
        let key = (upper, b);
        if self.batched.as_ref().map(|(k, ..)| *k) != Some(key) {
            self.batched = None;
            let bucket = self.sized_bucket(upper, b)?;
            if bucket.batch != b {
                bail!("the step sized for {b} sequences takes {}", bucket.batch);
            }
            let weights = &mut self.weights;
            let mut c = crate::with_param_sharing_scope(&self.scope, || {
                let mut c = weights.compile(self.device, bucket.graph.clone());
                weights.bind(&mut c, &bucket.params).map(|()| c)
            })?;
            let resident = self.replace_state && self.bind_together(&mut c, &bucket);
            self.batched = Some((key, c, bucket, resident));
        }
        Ok(())
    }

    /// Reorder a batch's caches so sequence `i` continues sequence
    /// `rows[i]` (beam search keeping the best hypotheses): blocks of the
    /// host caches (`past` rows each), or of the executable's.
    fn reorder_together(&mut self, caches: &mut [Vec<f32>], rows: &[usize]) -> Result<()> {
        if rows.iter().enumerate().all(|(i, &r)| i == r) {
            return Ok(());
        }
        let permute = |data: &[f32], b: usize| -> Vec<f32> {
            let block = data.len() / b;
            rows.iter()
                .flat_map(|&r| data[r * block..(r + 1) * block].iter().copied())
                .collect()
        };
        let (_, exe, bucket, resident) = self.batched.as_mut().expect("compiled");
        let b = bucket.batch;
        if *resident {
            for st in &self.states {
                let data = exe
                    .read_gpu_handle(&st.input)
                    .with_context(|| format!("reading cache '{}'", st.input))?;
                if !exe.bind_gpu_handle(&st.input, &permute(&data, b)) {
                    bail!("the backend took the caches once but not again");
                }
            }
        } else {
            for c in caches.iter_mut() {
                if !c.is_empty() {
                    *c = permute(c, b);
                }
            }
        }
        Ok(())
    }

    /// Beam search after `prompt` with `beams` hypotheses as the sequences
    /// of a batched step: the prompt once (its prefill, then its last token
    /// in every sequence), then each step the caches follow the hypotheses
    /// kept ([`crate::lm::beam_search`]).
    fn decode_beams(
        &mut self,
        prompt: &[u32],
        beams: usize,
        n_new: usize,
        stop: &[u32],
        length_penalty: f32,
    ) -> Result<Vec<u32>> {
        let b = beams;
        let l = prompt.len() - 1;
        let (start, one) = self.prefill_together(&[prompt.to_vec()])?;
        self.ensure_batched(b, l + n_new)?;
        let mut caches: Vec<Vec<f32>> = one.iter().map(|c| c.repeat(b)).collect();
        if start > 0 {
            self.seed_together(&caches, start)?;
        }
        for (past, &t) in prompt.iter().enumerate().take(l).skip(start) {
            self.step_together(&vec![t; b], &vec![0.0; b], past, &mut caches)?;
        }
        let logits = self.step_together(&vec![prompt[l]; b], &vec![0.0; b], l, &mut caches)?;
        let per = logits.len() / b;
        let vocab = self.vocab.min(per);
        let row = |logits: &[f32], i: usize| logits[(i + 1) * per - vocab..(i + 1) * per].to_vec();
        let first = row(&logits, 0);
        let mut past = l + 1;
        let out =
            crate::lm::beam_search(&first, b, n_new, stop, length_penalty, |parents, tokens| {
                // Fewer hypotheses than sequences: the rest copy the first.
                let mut rows = parents.to_vec();
                rows.resize(b, parents[0]);
                let mut toks = tokens.to_vec();
                toks.resize(b, tokens[0]);
                self.reorder_together(&mut caches, &rows)?;
                let logits = self.step_together(&toks, &vec![0.0; b], past, &mut caches)?;
                past += 1;
                Ok((0..parents.len()).map(|i| row(&logits, i)).collect())
            })?;
        if self.prefill.as_ref().is_some_and(|p| p.release) {
            self.batched = None;
        }
        Ok(out)
    }

    /// The prompts' shared length less one (`l`), each sequence's first `l`
    /// tokens through the prefill function: where the lockstep starts, and
    /// the caches `[batch, l, width]` (each recurrent state whole, one per
    /// sequence). From position 0 with no caches when the package has no
    /// prefill, the prompts are that short, or no prefill size holds them.
    fn prefill_together(&mut self, prompts: &[Vec<u32>]) -> Result<(usize, Vec<Vec<f32>>)> {
        let none = (0, vec![Vec::new(); self.states.len()]);
        let shared = prompts.iter().map(Vec::len).min().unwrap_or(0);
        let l = shared.saturating_sub(1);
        if l < 2 || self.prefill.is_none() || !self.batch_prefill {
            return Ok(none);
        }
        let firsts: Vec<&[u32]> = prompts.iter().map(|p| &p[..l]).collect();
        if let Some(caches) = self.prefill_batch(&firsts)? {
            return Ok((l, caches));
        }
        let mut caches = vec![Vec::new(); self.states.len()];
        for p in prompts {
            self.reset();
            if self.prefill(&p[..l])?.is_none() {
                self.reset();
                return Ok(none);
            }
            for (k, rows) in self.cache.iter().enumerate() {
                caches[k].extend_from_slice(rows);
            }
        }
        self.reset();
        Ok((l, caches))
    }

    /// `prompts` (all one length) through the smallest batched prompt
    /// function that holds them, at once: each cache's rows, the prompts'
    /// one after another (`[prompts, len, width]`). `None` when the package
    /// has none that fits.
    fn prefill_batch(&mut self, prompts: &[&[u32]]) -> Result<Option<Vec<Vec<f32>>>> {
        let (count, n) = (prompts.len(), prompts.first().map_or(0, |p| p.len()));
        // One prompt is the plain prefill's; a recurrent state is not rows
        // of the prompt: the one-by-one path's.
        if count < 2 || self.states.iter().any(|s| s.kind == StateKind::Replace) {
            return Ok(None);
        }
        let Some(p) = &mut self.prefill else {
            return Ok(None);
        };
        let Some(b) = p
            .buckets
            .iter()
            .enumerate()
            .filter(|(_, b)| b.batch > 1 && b.batch >= count && b.seq >= n)
            .min_by_key(|(_, b)| (b.batch, b.seq))
            .map(|(i, _)| i)
        else {
            return Ok(None);
        };
        // Prompts that read the context go one by one.
        if !p.buckets[b].context.is_empty() {
            return Ok(None);
        }
        let key = (b, Vec::new());
        if p.compiled.as_ref().map(|(k, _)| k) != Some(&key) {
            p.compiled = None;
            let bucket = &p.buckets[b];
            let weights = &mut self.weights;
            let device = p.device.unwrap_or(self.device);
            let c = crate::with_param_sharing_scope(&self.scope, || {
                let mut c = weights.compile(device, bucket.graph.clone());
                weights.bind(&mut c, &bucket.params).map(|()| c)
            })?;
            p.compiled = Some((key, c));
        }
        let bucket = &p.buckets[b];
        let (batch, seq) = (bucket.batch, bucket.seq);
        // Each prompt padded to `seq`; rows past the prompts repeat the
        // first, and are not kept.
        let mut padded = vec![0f32; batch * seq];
        for r in 0..batch {
            let ids = prompts[r.min(count - 1)];
            for (s, &t) in ids.iter().enumerate() {
                padded[r * seq + s] = t as f32;
            }
        }
        let mut feeds = vec![(bucket.tokens.0.clone(), typed(&padded, bucket.tokens.1)?)];
        if let Some((name, dtype)) = &bucket.last {
            feeds.push((name.clone(), typed(&[(n - 1) as f32], *dtype)?));
        }
        let refs: Vec<(&str, &[u8], DType)> = feeds
            .iter()
            .map(|(name, (d, t))| (name.as_str(), d.as_slice(), *t))
            .collect();
        let exe = &mut p.compiled.as_mut().expect("compiled above").1;
        let mut outs = exe.run_typed(&refs);
        let mut caches = Vec::with_capacity(self.states.len());
        for (k, st) in self.states.iter().enumerate() {
            let i = p.state_outputs[k];
            let (data, dtype) = std::mem::replace(
                outs.get_mut(i)
                    .context("the prefill returned too few outputs")?,
                (Vec::new(), DType::F32),
            );
            let rows = super::Tensor::new("out", dtype, vec![], data).to_f32()?;
            let w = st.width;
            let block = rows.len() / batch;
            let mut kept = Vec::with_capacity(count * n * w);
            for r in 0..count {
                kept.extend_from_slice(&rows[r * block..r * block + n * w]);
            }
            caches.push(kept);
        }
        if p.release {
            p.compiled = None;
        }
        Ok(Some(caches))
    }

    /// Put prefilled caches (`[batch, rows, width]`) in a batched step that
    /// keeps its caches, each sequence's rows at the start of its block.
    fn seed_together(&mut self, caches: &[Vec<f32>], rows: usize) -> Result<()> {
        let Some((_, exe, bucket, true)) = self.batched.as_mut() else {
            return Ok(());
        };
        let b = bucket.batch;
        for (k, st) in self.states.iter().enumerate() {
            let w = st.width.max(1);
            let numel = bucket
                .inputs
                .iter()
                .find(|i| matches!(i.3, Feed::State(j) if j == k))
                .map(|i| i.2)
                .context("a batched step without a cache input")?;
            let block = numel / b;
            let mut data = vec![0.0; numel];
            for s in 0..b {
                let src = &caches[k][s * rows * w..(s + 1) * rows * w];
                data[s * block..s * block + src.len()].copy_from_slice(src);
            }
            if !exe.bind_gpu_handle(&st.input, &data) {
                bail!("the backend took the caches once but not again");
            }
        }
        Ok(())
    }

    /// The batch is at the end of its cache: each request's rows moved back
    /// to end where the longest one's does (its start moved with them), so
    /// the batch goes on from there.
    fn serve_rebase(&mut self) -> Result<()> {
        let s = self.serving.as_mut().expect("open");
        let past = s.past;
        let longest = s
            .slots
            .iter()
            .flatten()
            .map(|x| past - x.start)
            .max()
            .unwrap_or(0);
        let (_, exe, bucket, resident) = self.batched.as_mut().expect("compiled");
        if longest > bucket.upper {
            bail!(
                "a request is {longest} tokens, past the {}-token context",
                bucket.upper
            );
        }
        let b = s.slots.len();
        let moves: Vec<Option<(usize, usize)>> = s
            .slots
            .iter()
            .map(|x| x.as_ref().map(|x| (x.start, past - x.start)))
            .collect();
        // A key holds the rotation of the row it was written at: one moved
        // back `delta` rows is rotated back as much.
        let rope = &self.rope;
        let shift = |data: &mut [f32], rows: usize, w: usize, keys: bool| -> Result<()> {
            for (i, m) in moves.iter().enumerate() {
                let Some((start, len)) = *m else { continue };
                let to = longest - len;
                let block = &mut data[i * rows * w..(i + 1) * rows * w];
                block.copy_within(start * w..(start + len) * w, to * w);
                if keys && start > to {
                    let r = rope.as_ref().context(
                        "moving a request's keys needs the entry's `rope` (how they hold positions)",
                    )?;
                    for row in block[to * w..(to + len) * w].chunks_exact_mut(w) {
                        r.move_back(row, start - to)?;
                    }
                }
            }
            Ok(())
        };
        for (k, st) in self.states.iter().enumerate() {
            let w = st.width.max(1);
            let keys = self
                .rope
                .as_ref()
                .is_some_and(|r| st.input.starts_with(&r.keys))
                || self.rope.is_none() && st.input.contains("_k_");
            if *resident {
                let mut data = exe
                    .read_gpu_handle(&st.input)
                    .with_context(|| format!("reading cache '{}'", st.input))?;
                let rows = data.len() / b / w;
                shift(&mut data, rows, w, keys)?;
                if !exe.bind_gpu_handle(&st.input, &data) {
                    bail!("the backend took the caches once but not again");
                }
            } else {
                // The host's caches hold `past` rows a request: `longest` now.
                let data = &mut s.caches[k];
                let rows = data.len() / b / w;
                shift(data, rows, w, keys)?;
                *data = (0..b)
                    .flat_map(|i| data[i * rows * w..(i * rows + longest) * w].to_vec())
                    .collect();
            }
        }
        for x in s.slots.iter_mut().flatten() {
            x.start = longest - (past - x.start);
        }
        s.past = longest;
        Ok(())
    }

    /// Hand a batched step its caches to keep (zeros: every row a step
    /// reads it wrote first). False where the caches differ in width, are
    /// not written in place, or the backend keeps no handles.
    fn bind_together(&self, c: &mut CompiledGraph, bucket: &DecodeBucket) -> bool {
        let width = self.states.first().map_or(0, |s| s.width);
        if self.states.is_empty()
            || self
                .states
                .iter()
                .any(|s| s.width != width || width == 0 || s.kind != StateKind::Rows)
        {
            return false;
        }
        for (k, st) in self.states.iter().enumerate() {
            let Some(numel) = bucket
                .inputs
                .iter()
                .find(|i| matches!(i.3, Feed::State(j) if j == k))
                .map(|i| i.2)
            else {
                return false;
            };
            if !c.bind_gpu_handle(&st.input, &vec![0.0; numel])
                || !c.register_kv_row_feed(&st.input, st.output)
            {
                return false;
            }
        }
        true
    }

    /// One step of [`Self::decode_together`]: each sequence's token at
    /// `past`; its caches in and out (or kept by the executable). The
    /// logits, every sequence's.
    fn step_together(
        &mut self,
        tokens: &[u32],
        starts: &[f32],
        past: usize,
        caches: &mut [Vec<f32>],
    ) -> Result<Vec<f32>> {
        let b = tokens.len();
        let (_, exe, bucket, resident) =
            self.batched.as_mut().expect("compiled by decode_together");
        let upper = bucket.upper;
        if *resident {
            let mut feeds: Vec<(&str, Vec<f32>)> = Vec::new();
            let mut cap = 0;
            for (name, _, numel, feed) in &bucket.inputs {
                let values = match feed {
                    Feed::Token => tokens.iter().map(|&t| t as f32).collect(),
                    Feed::Position => vec![past as f32; *numel],
                    Feed::Starts => starts.to_vec(),
                    Feed::RopeOffset => vec![self.rope_offset as f32; *numel],
                    Feed::TableRow(t) => {
                        let (_, table, width) = &self.tables[*t];
                        table
                            .get(past * width..(past + 1) * width)
                            .with_context(|| format!("table for '{name}' has no row {past}"))?
                            .repeat(b)
                    }
                    Feed::State(k) => {
                        cap = numel / b / self.states[*k].width.max(1);
                        continue;
                    }
                    Feed::DecodeMask | Feed::Context(_) => {
                        bail!("a stateful batched step takes no '{name}'")
                    }
                };
                feeds.push((name, values));
            }
            let refs: Vec<(&str, &[f32])> = feeds.iter().map(|(n, v)| (*n, v.as_slice())).collect();
            let mut outs = exe.run_read_outputs(&refs, Some(&[self.logits_out]));
            // Each sequence's row `past` of each cache output, kept.
            let width = self.states[0].width;
            for s in 0..b {
                if !exe.feed_kv_row(s * cap + past, s * cap + past, width) {
                    bail!("the backend took the caches but cannot update them");
                }
            }
            return outs.pop().context("the step returned no logits");
        }
        let mut feeds: Vec<(String, Vec<u8>, DType)> = Vec::new();
        for (name, dtype, numel, feed) in &bucket.inputs {
            let per = numel / b;
            let values: Vec<f32> = match feed {
                Feed::Token => tokens.iter().map(|&t| t as f32).collect(),
                Feed::Position => vec![past as f32; *numel],
                Feed::Starts => starts.to_vec(),
                Feed::RopeOffset => vec![self.rope_offset as f32; *numel],
                Feed::DecodeMask => crate::attn_mask::bucket_decode_mask(past, upper).repeat(b),
                Feed::TableRow(t) => {
                    let (_, table, width) = &self.tables[*t];
                    let row = (past as i64 + self.rope_offset).max(0) as usize;
                    table
                        .get(row * width..(row + 1) * width)
                        .with_context(|| format!("table for '{name}' has no row {row}"))?
                        .repeat(b)
                }
                Feed::State(k) => {
                    let st = &self.states[*k];
                    if st.kind == StateKind::Replace {
                        let mut v = caches[*k].clone();
                        v.resize(*numel, 0.0);
                        v
                    } else {
                        // Each sequence's `past` rows, padded to its block.
                        let w = st.width.max(1);
                        let mut v = vec![0.0; *numel];
                        for s in 0..b {
                            let rows = &caches[*k][s * past * w..(s + 1) * past * w];
                            v[s * per..s * per + rows.len()].copy_from_slice(rows);
                        }
                        v
                    }
                }
                Feed::Context(c) => {
                    bail!("input '{name}' reads context '{c}': decoding together takes no context")
                }
            };
            if values.len() != *numel {
                bail!("input '{name}': {} values for {numel}", values.len());
            }
            let (bytes, dt) = typed(&values, *dtype)?;
            feeds.push((name.clone(), bytes, dt));
        }
        let refs: Vec<(&str, &[u8], DType)> = feeds
            .iter()
            .map(|(n, d, t)| (n.as_str(), d.as_slice(), *t))
            .collect();
        let mut outs = exe.run_typed(&refs);
        let as_f32 = |(data, dtype): (Vec<u8>, DType)| {
            super::Tensor::new("out", dtype, vec![], data).to_f32()
        };
        for (k, st) in self.states.iter().enumerate() {
            let out = as_f32(std::mem::replace(
                &mut outs[st.output],
                (Vec::new(), DType::F32),
            ))?;
            if st.kind == StateKind::Replace {
                caches[k] = out;
                continue;
            }
            // Per sequence: its block of the output, `rows` rows.
            let w = st.width.max(1);
            let rows = out.len() / b / w;
            if rows < past + 1 {
                bail!(
                    "cache output {} has {rows} rows, needs {}",
                    st.output,
                    past + 1
                );
            }
            let mut next = Vec::with_capacity(b * (past + 1) * w);
            for s in 0..b {
                let block = &out[s * rows * w..(s + 1) * rows * w];
                if st.kind == StateKind::Rows {
                    next.extend_from_slice(&block[..(past + 1) * w]);
                } else {
                    next.extend_from_slice(&block[..past * w]);
                    next.extend_from_slice(&block[(rows - 1) * w..]);
                }
            }
            caches[k] = next;
        }
        as_f32(std::mem::replace(
            &mut outs[self.logits_out],
            (Vec::new(), DType::F32),
        ))
    }

    fn reset(&mut self) {
        self.cache.iter_mut().for_each(Vec::clear);
        self.tokens.clear();
        self.rope_offset = 0;
        self.last_logits = None;
        self.cache_fresh = false;
    }

    /// A package with no step: the prompt function over the whole of `ids`,
    /// which become the context; the next position's logits.
    fn recompute(&mut self, mut ids: Vec<u32>) -> Result<Vec<f32>> {
        if let Some(logits) = self.prefill(&ids)? {
            return Ok(logits);
        }
        let longest = self.prefill.as_ref().map_or(0, |p| {
            p.buckets
                .iter()
                .filter(|b| b.batch == 1)
                .map(|b| b.seq)
                .max()
                .unwrap_or(0)
        });
        let n = ids.len();
        ids.pop();
        self.tokens = ids;
        bail!(
            "the context is full: {n} tokens (this package's prompt function takes up to {longest})"
        )
    }

    /// The caches as the last step left them: read back from the
    /// executable when they live there, else the host's.
    fn caches(&self) -> Result<Vec<Vec<f32>>> {
        let exe = match &self.compiled {
            Some((_, exe)) if self.resident && !self.cache_fresh => exe,
            _ => return Ok(self.cache.clone()),
        };
        let past = self.tokens.len();
        self.states
            .iter()
            .map(|st| {
                let rows = exe
                    .read_gpu_handle(&st.input)
                    .with_context(|| format!("reading cache '{}'", st.input))?;
                Ok(match st.kind {
                    StateKind::Replace => rows,
                    _ => rows[..(past * st.width).min(rows.len())].to_vec(),
                })
            })
            .collect()
    }

    /// Continue from `caches` holding `tokens`: the next step binds them.
    fn resume(&mut self, caches: Vec<Vec<f32>>, tokens: &[u32]) {
        self.cache = caches;
        self.tokens = tokens.to_vec();
        self.rope_offset = 0;
        self.last_logits = None;
        self.cache_fresh = true;
    }

    /// The whole prompt in one run of the shortest prefill that holds it,
    /// its rows the caches: the next position's logits. `None` with no
    /// prefill, or none that long.
    fn prefill(&mut self, ids: &[u32]) -> Result<Option<Vec<f32>>> {
        let n = ids.len();
        let Some(p) = &mut self.prefill else {
            return Ok(None);
        };
        let Some(b) = p.buckets.iter().position(|b| b.batch == 1 && b.seq >= n) else {
            return Ok(None);
        };
        // Its symbols bound by the context's lengths (an encoder's states).
        let binding = context_binding(&p.buckets[b], &self.context)?;
        let key = (b, binding);
        if p.compiled.as_ref().map(|(k, _)| k) != Some(&key) {
            p.compiled = None;
            let bucket = &p.buckets[b];
            let weights = &mut self.weights;
            let device = p.device.unwrap_or(self.device);
            let graph = if key.1.is_empty() {
                bucket.graph.clone()
            } else {
                rlx_ir::dynamic::specialize(&bucket.graph, &rlx_ir::DimBinding::from_pairs(&key.1))
            };
            let c = crate::with_param_sharing_scope(&self.scope, || {
                let mut c = weights.compile(device, graph);
                weights.bind(&mut c, &bucket.params).map(|()| c)
            })?;
            p.compiled = Some((key, c));
        }
        let bucket = &p.buckets[b];
        // Padding after the prompt: causal attention keeps it out of every
        // real position, and its rows are not kept.
        let mut padded: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
        padded.resize(bucket.seq, 0.0);
        let mut feeds = vec![(bucket.tokens.0.clone(), typed(&padded, bucket.tokens.1)?)];
        if let Some((name, dtype)) = &bucket.last {
            feeds.push((name.clone(), typed(&[(n - 1) as f32], *dtype)?));
        }
        for (input, value, dtype) in &bucket.context {
            let v = self
                .context
                .get(value)
                .with_context(|| format!("prefill input '{input}' reads '{value}', not given"))?;
            feeds.push((input.clone(), typed(v, *dtype)?));
        }
        let refs: Vec<(&str, &[u8], DType)> = feeds
            .iter()
            .map(|(name, (d, t))| (name.as_str(), d.as_slice(), *t))
            .collect();
        let exe = &mut p.compiled.as_mut().expect("compiled above").1;
        let mut outs = exe.run_typed(&refs);
        let mut take = |i: usize| {
            let (data, dtype) = std::mem::replace(
                outs.get_mut(i)
                    .context("the prefill returned too few outputs")?,
                (Vec::new(), DType::F32),
            );
            super::Tensor::new("out", dtype, vec![], data).to_f32()
        };
        for (k, st) in self.states.iter().enumerate() {
            let rows = take(p.state_outputs[k])?;
            if st.kind == StateKind::Replace {
                // The state after the prompt, whole.
                self.cache[k] = rows;
                continue;
            }
            let want = n * st.width;
            if rows.len() < want {
                bail!(
                    "prefill output {} has {} values; cache {k} needs {want}",
                    p.state_outputs[k],
                    rows.len()
                );
            }
            self.cache[k] = rows[..want].to_vec();
        }
        let mut logits = take(p.logits_out)?;
        let len = logits.len();
        logits.drain(..len.saturating_sub(self.vocab));
        if p.release {
            p.compiled = None;
        }
        self.tokens = ids.to_vec();
        self.rope_offset = 0;
        self.last_logits = Some(logits.clone());
        self.cache_fresh = true;
        Ok(Some(logits))
    }

    /// Feed `token` at position `past`; the next position's logits.
    fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        if self.buckets.is_empty() {
            let mut ids = std::mem::take(&mut self.tokens);
            ids.push(token);
            return self.recompute(ids);
        }
        let past = self.tokens.len();
        if self.buckets.iter().all(|b| b.upper < past) {
            self.grow_to(past)?;
        }
        let b = self
            .buckets
            .iter()
            .position(|b| b.upper >= past)
            .with_context(|| {
                format!(
                    "the context is full: {past} tokens (this package decodes up to {})",
                    self.grow
                        .as_ref()
                        .and_then(|g| g.max_context)
                        .unwrap_or_else(|| self.buckets.last().map_or(0, |b| b.upper))
                )
            })?;
        // A longer function than the context needs still runs it: keep it
        // rather than compile a shorter one when a new generation starts.
        let b = match &self.compiled {
            Some((i, _)) if self.buckets[*i].upper >= past => *i,
            _ => b,
        };
        if self.compiled.as_ref().map(|(i, _)| *i) != Some(b) {
            if let Some((_, old)) = self.compiled.take()
                && self.resident
                && !self.cache_fresh
            {
                // The caches leave the old executable for the new one.
                for (k, st) in self.states.iter().enumerate() {
                    let rows = old
                        .read_gpu_handle(&st.input)
                        .with_context(|| format!("reading cache '{}'", st.input))?;
                    self.cache[k] = rows[..(past * st.width).min(rows.len())].to_vec();
                }
            }
            let bucket = &self.buckets[b];
            let weights = &mut self.weights;
            let mut c = crate::with_param_sharing_scope(&self.scope, || {
                let mut c = weights.compile(self.device, bucket.graph.clone());
                weights.bind(&mut c, &bucket.params).map(|()| c)
            })?;
            self.resident = self.replace_state && self.bind_resident(&mut c, b);
            self.compiled = Some((b, c));
        } else if self.resident && self.cache_fresh {
            let (i, mut c) = self.compiled.take().expect("compiled");
            let bound = self.bind_resident(&mut c, b);
            self.compiled = Some((i, c));
            if !bound {
                bail!("the backend took the caches once but not again");
            }
        }
        self.cache_fresh = false;
        if self.resident {
            return self.step_resident(b, token, past);
        }
        let upper = self.buckets[b].upper;
        let mut feeds: Vec<(String, Vec<u8>, DType)> = Vec::new();
        for (name, dtype, numel, feed) in &self.buckets[b].inputs {
            let values: Vec<f32> = match feed {
                Feed::Token => vec![token as f32],
                Feed::Position => vec![past as f32],
                Feed::Starts => vec![0.0; *numel],
                Feed::RopeOffset => vec![self.rope_offset as f32; *numel],
                Feed::DecodeMask => crate::attn_mask::bucket_decode_mask(past, upper),
                Feed::TableRow(t) => {
                    let (_, table, width) = &self.tables[*t];
                    let row = (past as i64 + self.rope_offset).max(0) as usize;
                    table
                        .get(row * width..(row + 1) * width)
                        .with_context(|| format!("table for '{name}' has no row {row}"))?
                        .to_vec()
                }
                Feed::State(k) => {
                    let mut v = self.cache[*k].clone();
                    v.resize(*numel, 0.0);
                    v
                }
                Feed::Context(c) => self
                    .context
                    .get(c)
                    .with_context(|| {
                        format!("input '{name}' reads context '{c}', which no stage made")
                    })?
                    .clone(),
            };
            if values.len() != *numel {
                bail!("input '{name}': {} values for {numel}", values.len());
            }
            let (bytes, dt) = typed(&values, *dtype)?;
            feeds.push((name.clone(), bytes, dt));
        }
        let refs: Vec<(&str, &[u8], DType)> = feeds
            .iter()
            .map(|(n, d, t)| (n.as_str(), d.as_slice(), *t))
            .collect();
        let exe = &mut self.compiled.as_mut().expect("compiled above").1;
        let mut outs = exe.run_typed(&refs);
        let as_f32 = |(data, dtype): (Vec<u8>, DType)| {
            super::Tensor::new("out", dtype, vec![], data).to_f32()
        };
        // The cache's rows, then the new one last (a bucketed output pads
        // between them).
        for (k, st) in self.states.iter().enumerate() {
            let out = as_f32(std::mem::replace(
                &mut outs[st.output],
                (Vec::new(), DType::F32),
            ))?;
            if st.kind == StateKind::Replace {
                self.cache[k] = out;
                continue;
            }
            let w = st.width.max(1);
            let rows = out.len() / w;
            if rows < past + 1 {
                bail!(
                    "cache output {} has {rows} rows, needs {}",
                    st.output,
                    past + 1
                );
            }
            if st.kind == StateKind::Rows {
                // The step wrote the row in place: its cache is the next one.
                self.cache[k] = out[..(past + 1) * w].to_vec();
                continue;
            }
            let mut next = out[..past * w].to_vec();
            next.extend_from_slice(&out[(rows - 1) * w..rows * w]);
            self.cache[k] = next;
        }
        let mut logits = as_f32(std::mem::replace(
            &mut outs[self.logits_out],
            (Vec::new(), DType::F32),
        ))?;
        let n = logits.len();
        logits.drain(..n.saturating_sub(self.vocab));
        self.tokens.push(token);
        self.last_logits = Some(logits.clone());
        Ok(logits)
    }
}

impl GraphLm {
    /// Hand the caches to `c` to keep, each step writing its one new row
    /// there. False, binding nothing, where the caches differ in row width or
    /// the backend keeps no handles: the host then passes them every step.
    fn bind_resident(&self, c: &mut CompiledGraph, b: usize) -> bool {
        let width = self.states.first().map_or(0, |s| s.width);
        // One row a step, the same width each: recurrent state, replaced
        // whole, goes through the host.
        if self.states.is_empty()
            || self
                .states
                .iter()
                .any(|s| s.width != width || width == 0 || s.kind != StateKind::Rows)
        {
            return false;
        }
        for (k, st) in self.states.iter().enumerate() {
            let Some(numel) = self.buckets[b]
                .inputs
                .iter()
                .find(|i| matches!(i.3, Feed::State(j) if j == k))
                .map(|i| i.2)
            else {
                return false;
            };
            let mut rows = self.cache[k].clone();
            rows.resize(numel, 0.0);
            if !c.bind_gpu_handle(&st.input, &rows) || !c.register_kv_row_feed(&st.input, st.output)
            {
                return false;
            }
        }
        true
    }

    /// [`Self::step`] with the caches in the executable: feed the token, its
    /// position and table rows, read only the logits, keep the new rows.
    fn step_resident(&mut self, b: usize, token: u32, past: usize) -> Result<Vec<f32>> {
        let mut feeds: Vec<(&str, Vec<f32>)> = Vec::new();
        for (name, _, _, feed) in &self.buckets[b].inputs {
            let values = match feed {
                Feed::Token => vec![token as f32],
                Feed::Position => vec![past as f32],
                Feed::Starts => vec![0.0],
                Feed::RopeOffset => vec![self.rope_offset as f32],
                Feed::TableRow(t) => {
                    let (_, table, width) = &self.tables[*t];
                    let row = (past as i64 + self.rope_offset).max(0) as usize;
                    table
                        .get(row * width..(row + 1) * width)
                        .with_context(|| format!("table for '{name}' has no row {row}"))?
                        .to_vec()
                }
                Feed::DecodeMask => bail!("a stateful step takes no decode mask ('{name}')"),
                Feed::State(_) => continue,
                Feed::Context(c) => self
                    .context
                    .get(c)
                    .with_context(|| {
                        format!("input '{name}' reads context '{c}', which no stage made")
                    })?
                    .clone(),
            };
            feeds.push((name, values));
        }
        let refs: Vec<(&str, &[f32])> = feeds.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        let exe = &mut self.compiled.as_mut().expect("compiled by step").1;
        let mut outs = exe.run_read_outputs(&refs, Some(&[self.logits_out]));
        // The step wrote row `past` of each cache output: keep it.
        let width = self.states[0].width;
        if !exe.feed_kv_row(past, past, width) {
            bail!("the backend took the caches but cannot update them");
        }
        let mut logits = outs.pop().context("the step returned no logits")?;
        let n = logits.len();
        logits.drain(..n.saturating_sub(self.vocab));
        self.tokens.push(token);
        self.last_logits = Some(logits.clone());
        Ok(logits)
    }
}

impl crate::LmRunner for GraphLm {
    fn family(&self) -> &'static str {
        self.family
    }

    fn prefill_with_media(
        &mut self,
        ids: &[u32],
        media: &[super::Tensor],
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        // One executable's weights at a time: the step's go before the
        // media prompt's are made.
        if self.media.as_ref().is_some_and(|m| m.stages.release) {
            self.compiled = None;
            self.batched = None;
        }
        let Some(m) = self.media.as_mut() else {
            bail!(
                "{} takes no images or other media (its entry has no `media`)",
                self.family
            );
        };
        // Each medium's embedding rows, by its kind (its tensor's name, or
        // the one kind).
        let mut embedded: Vec<(u32, Vec<f32>, usize)> = Vec::new();
        // Each run's grid with `mrope`: its width in rows, rows per frame
        // group, and the time positions between groups.
        let mut grids: Vec<(usize, usize, usize)> = Vec::new();
        // Each medium's further row outputs (`kinds.<kind>.rows`), by name.
        let mut extras: Vec<Vec<(String, Vec<f32>)>> = Vec::new();
        for t in media {
            let k = m
                .kinds
                .iter()
                .find(|k| k.name == t.name)
                .or_else(|| (m.kinds.len() == 1).then(|| &m.kinds[0]))
                .with_context(|| format!("no media kind '{}'", t.name))?;
            // A video's frames, or the one image.
            let (frames, shape) = match (&k.frames, &t.shape[..]) {
                (Some(_), [f, rest @ ..]) => (*f, rest.to_vec()),
                (Some(_), _) => bail!("media '{}' is a video; it has no frames axis", t.name),
                (None, shape) => (1, shape.to_vec()),
            };
            let patch_frames = k.frames.as_ref().map_or(1, |f| f.patch);
            // To the processor's size for it, first.
            let (t, shape) = match k.resize {
                Some((multiple, min_pixels, max_pixels, over_frames)) => {
                    let [h, w, 3] = shape[..] else {
                        bail!(
                            "media '{}' is {:?}, not [height, width, 3]",
                            t.name,
                            t.shape
                        );
                    };
                    if t.dtype != DType::U8 {
                        bail!(
                            "media '{}' is {:?}; a resized image is u8 RGB",
                            t.name,
                            t.dtype
                        );
                    }
                    let (counted, temporal) = if over_frames {
                        (frames, patch_frames)
                    } else {
                        (1, 1)
                    };
                    let (nh, nw) = super::image_resize::smart_resize_frames(
                        counted, temporal, h, w, multiple, min_pixels, max_pixels,
                    )
                    .with_context(|| format!("media '{}' ({h}×{w}) is too narrow", t.name))?;
                    if (nh, nw) == (h, w) {
                        (t.clone(), shape)
                    } else {
                        let frame = h * w * 3;
                        let mut data = Vec::with_capacity(frames * nh * nw * 3);
                        for f in t.data.chunks_exact(frame) {
                            data.extend(super::image_resize::resize_rgb8_bicubic(f, h, w, nh, nw));
                        }
                        let mut dims = vec![nh, nw, 3];
                        if k.frames.is_some() {
                            dims.insert(0, frames);
                        }
                        (
                            super::Tensor::new(t.name.clone(), DType::U8, dims, data),
                            vec![nh, nw, 3],
                        )
                    }
                }
                None => (t.clone(), shape),
            };
            // The inputs the stages take: the image, or each group of frames.
            let pieces: Vec<super::Tensor> = match &k.frames {
                None => vec![t],
                Some(f) => {
                    let frame = t.data.len() / frames.max(1);
                    let groups = frames.div_ceil(f.patch);
                    let mut dims = vec![f.patch];
                    dims.extend(&shape);
                    (0..groups)
                        .map(|g| {
                            let mut data = Vec::with_capacity(f.patch * frame);
                            for i in 0..f.patch {
                                let at = (g * f.patch + i).min(frames - 1) * frame;
                                data.extend_from_slice(&t.data[at..at + frame]);
                            }
                            super::Tensor::new(t.name.clone(), t.dtype, dims.clone(), data)
                        })
                        .collect()
                }
            };
            let groups = pieces.len();
            let joined = k.frames.as_ref().is_some_and(|f| !f.runs);
            let (mut all_rows, mut all_more, mut all_n) = (Vec::new(), Vec::new(), 0);
            let release = m.stages.release;
            for (g, piece) in pieces.into_iter().enumerate() {
                let mut values = HashMap::new();
                values.insert(k.input.clone(), piece);
                let (range, token, output) = (k.stages.clone(), k.token, k.output.clone());
                // One compile for every group of a video: released after its
                // last.
                m.stages.release = release && g + 1 == groups;
                let ran = m.stages.run_range(range, &mut values);
                m.stages.release = release;
                ran?;
                let out = values
                    .remove(&output)
                    .with_context(|| format!("no stage makes '{output}'"))?;
                // `[rows, hidden]`, or with leading 1s (`[1, rows, hidden]`).
                let rows: usize = out.shape[..out.shape.len().saturating_sub(1)]
                    .iter()
                    .product::<usize>()
                    .max(1);
                let mut width = 0;
                if let Some(patch) = k.patch {
                    let [h, w, ..] = shape[..] else {
                        bail!("media '{}' is {:?}, not [height, width, …]", k.name, shape);
                    };
                    let grid = (h / patch, w / patch);
                    if grid.0 * grid.1 != rows {
                        bail!(
                            "media '{}' ({h}×{w}, {patch}-pixel rows) is a {}×{} grid, but its \
                             embedding has {rows} rows",
                            k.name,
                            grid.0,
                            grid.1
                        );
                    }
                    width = grid.1;
                }
                let mut more = Vec::new();
                for name in &k.rows {
                    let t = values
                        .remove(name)
                        .with_context(|| format!("no stage of '{}' makes '{name}'", k.name))?;
                    more.push((name.clone(), t.to_f32()?));
                }
                let t_step = k.frames.as_ref().map_or(0, |f| f.t_step);
                if joined {
                    // One run: the groups' rows (and further rows) end to end.
                    all_rows.extend(out.to_f32()?);
                    if all_more.is_empty() {
                        all_more = more;
                    } else {
                        for ((_, acc), (_, v)) in all_more.iter_mut().zip(more) {
                            acc.extend(v);
                        }
                    }
                    all_n += rows;
                    if k.patch.is_some() && all_n == rows * groups {
                        grids.push((width, rows, t_step));
                    }
                } else {
                    if k.patch.is_some() {
                        grids.push((width, rows, 0));
                    }
                    extras.push(more);
                    embedded.push((token, out.to_f32()?, rows));
                }
            }
            if joined {
                extras.push(all_more);
                embedded.push((k.token, all_rows, all_n));
            }
        }
        // Each run of a placeholder becomes its medium's rows.
        let mut expanded = Vec::new();
        let mut spans = Vec::new();
        let mut next = embedded.iter();
        let mut i = 0;
        while i < ids.len() {
            let id = ids[i];
            if m.kinds.iter().any(|k| k.token == id) {
                let mut j = i;
                while j < ids.len() && ids[j] == id {
                    j += 1;
                }
                let (token, rows, n) = next.next().with_context(|| {
                    format!("the prompt holds more placeholders ({id}) than media")
                })?;
                if *token != id {
                    bail!("placeholder {id} where the next medium's is {token}");
                }
                spans.push((expanded.len(), rows.clone(), *n));
                expanded.extend(std::iter::repeat_n(id, *n));
                i = j;
            } else {
                expanded.push(id);
                i += 1;
            }
        }
        if next.next().is_some() {
            bail!("more media than the prompt has placeholders for");
        }
        // The prompt's embedding, the media's rows in their places.
        let mut values = HashMap::new();
        let id_tensor = super::Tensor::new(
            "ids",
            DType::I32,
            vec![expanded.len()],
            expanded
                .iter()
                .flat_map(|&v| (v as i32).to_le_bytes())
                .collect(),
        );
        values.insert("ids".into(), id_tensor);
        m.stages.run_range(m.embed..m.embed + 1, &mut values)?;
        let embeds = values
            .remove("embeds")
            .context("media.embed makes no 'embeds'")?;
        let mut rows = embeds.to_f32()?;
        let hidden = rows.len() / expanded.len().max(1);
        for (at, media_rows, n) in &spans {
            if media_rows.len() != n * hidden {
                bail!(
                    "a medium's embedding is {} values; {n} rows of {hidden}",
                    media_rows.len()
                );
            }
            rows[at * hidden..(at + n) * hidden].copy_from_slice(media_rows);
        }
        // Padded to a listed length; the prefill reads the rows up to `last`.
        let len = expanded.len();
        let padded = m.sizes.iter().copied().find(|&s| s >= len).unwrap_or(len);
        // The further rows: zeros, but each medium's at its positions.
        let mut names: Vec<&String> = m.kinds.iter().flat_map(|k| &k.rows).collect();
        names.sort();
        names.dedup();
        for name in names {
            let width = extras
                .iter()
                .zip(&spans)
                .find_map(|(more, (_, _, n))| {
                    more.iter()
                        .find(|(m, _)| m == name)
                        .map(|(_, v)| v.len() / n.max(&1))
                })
                .unwrap_or(hidden);
            let mut full = vec![0f32; padded * width];
            for (more, (at, _, n)) in extras.iter().zip(&spans) {
                if let Some((_, v)) = more.iter().find(|(m, _)| m == name) {
                    if v.len() != n * width {
                        bail!("'{name}' is {} values; {n} rows of {width}", v.len());
                    }
                    full[at * width..(at + n) * width].copy_from_slice(v);
                }
            }
            values.insert(
                name.clone(),
                super::Tensor::f32(name.clone(), vec![1, padded, width], &full),
            );
        }
        rows.resize(padded * hidden, 0.0);
        values.insert(
            "prompt_embeds".into(),
            super::Tensor::f32("prompt_embeds", vec![1, padded, hidden], &rows),
        );
        values.insert(
            "last".into(),
            super::Tensor::f32("last", vec![1], &[(len - 1) as f32]),
        );
        // Each token's 3-D position, and its RoPE rows from them.
        let mut next_position = len;
        if let Some(mrope) = &m.mrope {
            let mut positions = Vec::with_capacity(padded);
            let mut grid = grids.iter();
            let mut span = spans.iter().peekable();
            let mut st = 0;
            let mut i = 0;
            while i < len {
                if span.peek().is_some_and(|(at, ..)| *at == i) {
                    let (_, _, n) = span.next().expect("peeked");
                    let (w, per, t_step) = *grid.next().context("a medium without its grid")?;
                    let mut top = st;
                    for r in 0..*n {
                        let (g, r) = (r / per.max(1), r % per.max(1));
                        let p = [st + g * t_step, st + r / w, st + r % w];
                        top = top.max(p[0]).max(p[1]).max(p[2]);
                        positions.push(p);
                    }
                    st = top + 1;
                    i += n;
                } else {
                    positions.push([st; 3]);
                    st += 1;
                    i += 1;
                }
            }
            positions.resize(padded, [0; 3]);
            let (cos, sin) = mrope.rows(&positions)?;
            let h = mrope.half;
            values.insert(
                "rope_cos".into(),
                super::Tensor::f32("rope_cos", vec![padded, h], &cos),
            );
            values.insert(
                "rope_sin".into(),
                super::Tensor::f32("rope_sin", vec![padded, h], &sin),
            );
            next_position = st;
        }
        m.stages.run_range(m.prefill..m.prefill + 1, &mut values)?;
        let outputs = m.stages.list[m.prefill].outputs.clone();
        let mut logits = values
            .remove(&outputs[0])
            .context("the media prefill made no logits")?
            .to_f32()?;
        let total = logits.len();
        logits.drain(..total.saturating_sub(self.vocab));
        self.reset();
        for (k, st) in self.states.iter().enumerate() {
            let name = outputs
                .get(k + 1)
                .with_context(|| format!("the media prefill gives no cache {k}"))?;
            let cache = values
                .remove(name)
                .with_context(|| format!("the media prefill made no '{name}'"))?
                .to_f32()?;
            self.cache[k] = if st.kind == StateKind::Replace {
                cache
            } else {
                cache[..len * st.width].to_vec()
            };
        }
        self.tokens = expanded.clone();
        self.rope_offset = next_position as i64 - len as i64;
        self.last_logits = Some(logits.clone());
        self.cache_fresh = true;
        Ok((expanded, logits))
    }

    fn serve_open(&mut self, slots: usize) -> Result<()> {
        if !self.grow.as_ref().is_some_and(|g| g.batch.is_some()) {
            bail!(
                "{}: its step has no batch symbol to serve requests with",
                self.family
            );
        }
        // The longest context the package runs.
        let reach = self
            .grow
            .as_ref()
            .and_then(|g| g.max_context)
            .unwrap_or_else(|| {
                self.buckets
                    .iter()
                    .map(|b| b.upper)
                    .filter(|&u| u != usize::MAX)
                    .max()
                    .unwrap_or(0)
            });
        self.ensure_batched(slots.max(1), reach)?;
        let caches = vec![Vec::new(); self.states.len()];
        self.serving = Some(Serving {
            slots: (0..slots.max(1)).map(|_| None).collect(),
            past: 0,
            caches,
        });
        Ok(())
    }

    fn serve_add(&mut self, prompt: &[u32]) -> Result<usize> {
        let s = self.serving.as_mut().context("serve_open first")?;
        if prompt.is_empty() {
            bail!("an empty prompt");
        }
        let slot = s
            .slots
            .iter()
            .position(Option::is_none)
            .context("every slot is busy")?;
        s.slots[slot] = Some(ServingSlot {
            start: s.past,
            pending: prompt.iter().copied().collect(),
            last: 0,
        });
        Ok(slot)
    }

    fn serve_remove(&mut self, slot: usize) {
        if let Some(s) = self.serving.as_mut()
            && let Some(x) = s.slots.get_mut(slot)
        {
            *x = None;
        }
    }

    fn serve_step(&mut self) -> Result<Vec<(usize, u32)>> {
        let upper = self
            .batched
            .as_ref()
            .map(|(_, _, b, _)| b.upper)
            .context("serve_open first")?;
        let s = self.serving.as_mut().context("serve_open first")?;
        if s.slots.iter().all(Option::is_none) {
            return Ok(Vec::new());
        }
        if s.past > upper {
            self.serve_rebase()?;
        }
        let s = self.serving.as_mut().expect("open");
        let past = s.past;
        let mut tokens = Vec::with_capacity(s.slots.len());
        let mut starts = Vec::with_capacity(s.slots.len());
        let mut speaks = Vec::new();
        for (i, slot) in s.slots.iter_mut().enumerate() {
            match slot {
                Some(x) => {
                    let t = match x.pending.pop_front() {
                        Some(t) => t,
                        None => x.last,
                    };
                    if x.pending.is_empty() {
                        speaks.push(i);
                    }
                    tokens.push(t);
                    starts.push(x.start as f32);
                }
                // An empty slot sees only its own row.
                None => {
                    tokens.push(0);
                    starts.push(past as f32);
                }
            }
        }
        let mut caches = std::mem::take(&mut s.caches);
        let logits = self.step_together(&tokens, &starts, past, &mut caches);
        let s = self.serving.as_mut().expect("open");
        s.caches = caches;
        let logits = logits?;
        s.past += 1;
        let per = logits.len() / s.slots.len();
        let vocab = self.vocab.min(per);
        let mut out = Vec::with_capacity(speaks.len());
        for i in speaks {
            let row = &logits[(i + 1) * per - vocab..(i + 1) * per];
            let next = crate::lm::argmax_u32(row);
            if let Some(x) = s.slots[i].as_mut() {
                x.last = next;
            }
            out.push((i, next));
        }
        Ok(out)
    }

    /// Several prompts in one step graph sized for them all (the entry's
    /// `batch` symbol); without one, one after another.
    fn generate_batch(
        &mut self,
        prompts: &[Vec<u32>],
        n_new: usize,
        stop: &[u32],
    ) -> Result<Vec<Vec<u32>>> {
        let batched = self.batched_step && self.grow.as_ref().is_some_and(|g| g.batch.is_some());
        if prompts.len() < 2 || !batched || n_new == 0 || prompts.iter().any(Vec::is_empty) {
            return prompts
                .iter()
                .map(|p| {
                    let mut out =
                        crate::LmRunner::generate(self, p, n_new, &mut |t| !stop.contains(&t))?;
                    if out.last().is_some_and(|t| stop.contains(t)) {
                        out.pop();
                    }
                    Ok(out)
                })
                .collect();
        }
        self.decode_together(prompts, n_new, stop)
    }

    /// The hypotheses as the sequences of a batched step (the entry's
    /// `batch` symbol); without one, each extended through the prompt
    /// function and step on its own context.
    fn generate_beam(
        &mut self,
        prompt: &[u32],
        beams: usize,
        n_new: usize,
        stop: &[u32],
        length_penalty: f32,
    ) -> Result<Vec<u32>> {
        let batched = self.batched_step && self.grow.as_ref().is_some_and(|g| g.batch.is_some());
        if !batched || beams < 2 || n_new == 0 || prompt.is_empty() {
            let first = crate::LmRunner::predict_logits(self, prompt)?;
            let mut contexts: Vec<Vec<u32>> = vec![prompt.to_vec()];
            return crate::lm::beam_search(
                &first,
                beams,
                n_new,
                stop,
                length_penalty,
                |parents, tokens| {
                    contexts = parents
                        .iter()
                        .zip(tokens)
                        .map(|(&p, &t)| {
                            let mut c = contexts[p].clone();
                            c.push(t);
                            c
                        })
                        .collect();
                    contexts
                        .iter()
                        .map(|c| crate::LmRunner::predict_logits(self, c))
                        .collect()
                },
            );
        }
        self.decode_beams(prompt, beams, n_new, stop, length_penalty)
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    /// Next-token logits after `ids`. A continuation of what the cache holds
    /// feeds only the new tokens; anything else starts over.
    fn predict_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            bail!("predict_logits needs at least one token");
        }
        if self.buckets.is_empty() {
            return self.recompute(ids.to_vec());
        }
        let cached = self.tokens.len();
        let continues = ids.len() >= cached && ids[..cached] == self.tokens[..];
        if !continues || (ids.len() == cached && self.last_logits.is_none()) {
            self.reset();
        }
        if ids.len() == self.tokens.len()
            && let Some(l) = &self.last_logits
        {
            return Ok(l.clone());
        }
        if self.tokens.is_empty()
            && ids.len() > 1
            && let Some(logits) = self.prefill(ids)?
        {
            return Ok(logits);
        }
        let mut logits = Vec::new();
        for &t in &ids[self.tokens.len()..] {
            logits = self.step(t)?;
        }
        Ok(logits)
    }

    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let mut produced = Vec::new();
        if n_new == 0 {
            return Ok(produced);
        }
        if self.speculative && self.check.is_some() && !prompt_ids.is_empty() {
            return self.decode_speculative(prompt_ids, n_new, on_token);
        }
        let mut logits = self.predict_logits(prompt_ids)?;
        loop {
            let next = crate::lm::argmax_u32(&logits);
            produced.push(next);
            if !on_token(next) || produced.len() == n_new {
                return Ok(produced);
            }
            logits = self.step(next)?;
        }
    }

    fn prefill_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        self.reset();
        self.predict_logits(prompt_ids)
    }

    fn decode_logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.step(token)
    }

    /// The caches and the tokens in them, each cache one of `kv.layers_k`
    /// (in the entry's state order; `layers_v` empty). `None` after a
    /// prompt with media, whose positions are not its rows.
    fn export_session(&self) -> Option<crate::lm::SessionSnapshot> {
        if self.rope_offset != 0 || self.tokens.is_empty() {
            return None;
        }
        let caches = self.caches().ok()?;
        Some(crate::lm::SessionSnapshot {
            kv: crate::kv_cache::LayerKvCache {
                past_len: self.tokens.len(),
                layers_k: caches,
                layers_v: Vec::new(),
                layers_kv_base: Vec::new(),
            },
            tokens: self.tokens.clone(),
        })
    }

    /// A snapshot [`Self::export_session`] took (of this package: one cache
    /// per state, each its size): decoding continues after its tokens.
    fn restore_session(&mut self, snap: &crate::lm::SessionSnapshot) -> bool {
        let kv = &snap.kv;
        let fits = kv.layers_v.is_empty()
            && kv.layers_k.len() == self.states.len()
            && kv.past_len == snap.tokens.len()
            && !snap.tokens.is_empty()
            && self.states.iter().zip(&kv.layers_k).all(|(st, c)| {
                st.kind == StateKind::Replace || c.len() == snap.tokens.len() * st.width
            });
        if fits {
            self.resume(kv.layers_k.clone(), &snap.tokens);
        }
        fits
    }

    /// `prompt` after a snapshot of its first `reuse_len` tokens: only the
    /// rest runs, a token a step.
    fn prefill_logits_reusing(
        &mut self,
        prompt: &[u32],
        snap: &crate::lm::SessionSnapshot,
        reuse_len: usize,
    ) -> Result<Vec<f32>> {
        let reusable = reuse_len < prompt.len()
            && reuse_len == snap.tokens.len()
            && prompt[..reuse_len] == snap.tokens[..]
            && self.restore_session(snap);
        if !reusable {
            return self.prefill_logits(prompt);
        }
        let mut logits = Vec::new();
        for &t in &prompt[reuse_len..] {
            logits = self.step(t)?;
        }
        Ok(logits)
    }
}

// ── stages: functions over named values ──────────────────────

/// Functions run in order over named values: each stage reads earlier
/// values (or a value's dims, `"image#shape"`) by name and names its
/// outputs for later ones. A stage symbolic in its input dims (an image of
/// any size) is specialized to the dims it is given and compiled once per
/// shape.
struct Stages {
    list: Vec<Stage>,
    /// Free each stage's executable after it runs (`prefill: "release"`, or
    /// a package too large to keep several): one copy of the weights at a
    /// time, at a compile per run.
    release: bool,
    /// The stages' weights: 0 is the entry's package (filled by the entry
    /// that keeps them), then one per package a stage names.
    weights: Vec<PackageWeights>,
}

struct Stage {
    function: String,
    /// As packaged: symbolic dims stay until a run binds them.
    graph: Graph,
    device: Device,
    /// Index into [`Stages::weights`].
    weights: usize,
    inputs: Vec<StageInput>,
    outputs: Vec<String>,
    /// Compiled per binding of its symbols (one, empty, when static).
    built: HashMap<Vec<(u32, usize)>, Built>,
}

struct StageInput {
    name: String,
    /// The value it reads, `#shape` for that value's dims.
    source: String,
    dtype: DType,
    dims: Vec<Dim>,
}

struct Built {
    compiled: CompiledGraph,
    /// Each output's dims.
    shapes: Vec<Vec<usize>>,
}

fn u32s(v: Option<&Value>) -> Vec<u32> {
    v.and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|x| x as u32))
                .collect()
        })
        .unwrap_or_default()
}

/// `t`'s values as f32, from any integer or float dtype.
fn values_f32(t: &Tensor) -> Result<Vec<f32>> {
    use rlx_ir::bytes::decode_le_vec;
    Ok(match t.dtype {
        DType::F32 => decode_le_vec::<f32>(&t.data),
        DType::U8 | DType::Bool => t.data.iter().map(|&v| f32::from(v)).collect(),
        DType::I8 => t.data.iter().map(|&v| f32::from(v as i8)).collect(),
        DType::I32 => decode_le_vec::<i32>(&t.data)
            .into_iter()
            .map(|v| v as f32)
            .collect(),
        DType::U32 => decode_le_vec::<u32>(&t.data)
            .into_iter()
            .map(|v| v as f32)
            .collect(),
        DType::I64 => decode_le_vec::<i64>(&t.data)
            .into_iter()
            .map(|v| v as f32)
            .collect(),
        d if d.is_float() => crate::backend::widen_bytes_to_f32(&t.data, d),
        other => bail!("tensor '{}' is {other:?}; stages read numbers", t.name),
    })
}

impl Stages {
    /// The stages `list` names, taken out of `pkg` (or the packages they
    /// name). Static stages compile now; their params bind from `pkg`'s
    /// weights, which stay in `pkg` for whatever else the entry builds.
    fn open(request: &ModelRequest, pkg: &mut Opened, list: &[Value]) -> Result<Self> {
        let mut stages = Stages {
            list: Vec::new(),
            release: false,
            weights: vec![PackageWeights::empty()],
        };
        for s in list {
            let function = s
                .get("function")
                .and_then(Value::as_str)
                .context("a stage names its `function`")?
                .to_string();
            let (graph, weights) = match s.get("package").and_then(Value::as_str) {
                Some(rel) => {
                    // A front end exported once and shared.
                    let mut other = Opened::open(&super::rlxp::beside(&pkg.path, rel))?;
                    other.target = pkg.target.clone();
                    let graph = other.take_graph(&function)?;
                    stages.weights.push(other.weights);
                    (graph, stages.weights.len() - 1)
                }
                None => (pkg.take_graph(&function)?, 0),
            };
            let feeds = s
                .get("inputs")
                .and_then(Value::as_object)
                .with_context(|| format!("stage '{function}' maps its inputs"))?;
            let mut inputs = Vec::new();
            for n in graph.nodes() {
                let Op::Input { name } = &n.op else { continue };
                let source = feeds
                    .get(name.as_str())
                    .and_then(Value::as_str)
                    .with_context(|| {
                        format!("input '{name}' of stage '{function}' has no source")
                    })?;
                inputs.push(StageInput {
                    name: name.clone(),
                    source: source.to_string(),
                    dtype: n.shape.dtype(),
                    dims: n.shape.dims().to_vec(),
                });
            }
            let outputs: Vec<String> = s
                .get("outputs")
                .and_then(Value::as_array)
                .with_context(|| format!("stage '{function}' names its outputs"))?
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect();
            if outputs.len() != graph.outputs.len() {
                bail!(
                    "stage '{function}' names {} outputs; its graph has {}",
                    outputs.len(),
                    graph.outputs.len()
                );
            }
            // The open option `stage_devices: {<function>: <device>}`, else
            // the stage's own `device`, else the model's.
            let chosen = request
                .option("stage_devices")
                .and_then(|m| m.get(&function))
                .or_else(|| s.get("device"))
                .and_then(Value::as_str);
            let device = match chosen {
                Some(d) => crate::parse_device(d)
                    .map_err(|e| anyhow::anyhow!("device of stage '{function}': {e}"))?,
                None => request.device,
            };
            let mut stage = Stage {
                function,
                graph,
                device,
                weights,
                inputs,
                outputs,
                built: HashMap::new(),
            };
            if !stage.graph.has_dynamic_dims() {
                let w = if weights == 0 {
                    &mut pkg.weights
                } else {
                    &mut stages.weights[weights]
                };
                stage.build(Vec::new(), w)?;
            }
            stages.list.push(stage);
        }
        Ok(stages)
    }

    /// How many values the first stage reading `source` takes, when its
    /// input is static.
    fn static_numel(&self, source: &str) -> Option<usize> {
        let input = self
            .list
            .iter()
            .flat_map(|s| &s.inputs)
            .find(|i| i.source == source)?;
        input
            .dims
            .iter()
            .map(|d| match d {
                Dim::Static(n) => Some(*n),
                Dim::Dynamic(_) => None,
            })
            .product()
    }

    /// Run every stage, adding its outputs to `values`.
    fn run(&mut self, values: &mut HashMap<String, Tensor>) -> Result<()> {
        let n = self.list.len();
        self.run_range(0..n, values)
    }

    /// Run stages `range`, adding (or replacing) their outputs in `values`.
    fn run_range(
        &mut self,
        range: std::ops::Range<usize>,
        values: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        let release = self.release;
        for stage in &mut self.list[range] {
            let mut feeds = Vec::new();
            let mut binding: Vec<(u32, usize)> = Vec::new();
            for input in &stage.inputs {
                let (value, shape_of) = match input.source.strip_suffix("#shape") {
                    Some(v) => (v, true),
                    None => (input.source.as_str(), false),
                };
                let t = values.get(value).with_context(|| {
                    format!(
                        "stage '{}' reads '{value}', which nothing made",
                        stage.function
                    )
                })?;
                let (dims, v) = if shape_of {
                    (
                        vec![t.shape.len()],
                        t.shape.iter().map(|&d| d as f32).collect(),
                    )
                } else {
                    (t.shape.clone(), values_f32(t)?)
                };
                // Its dims bind the input's symbols; static ones must match
                // (a flat value of the right count fits a static input).
                let fits_flat = input.dims.iter().all(|d| matches!(d, Dim::Static(_)))
                    && v.len()
                        == input
                            .dims
                            .iter()
                            .map(|d| d.unwrap_static())
                            .product::<usize>();
                if !fits_flat {
                    if dims.len() != input.dims.len() {
                        bail!(
                            "stage '{}' input '{}' has {} dims; '{}' has {dims:?}",
                            stage.function,
                            input.name,
                            input.dims.len(),
                            input.source
                        );
                    }
                    for (d, &n) in input.dims.iter().zip(&dims) {
                        match d {
                            Dim::Static(s) if *s != n => bail!(
                                "stage '{}' input '{}' is {:?}; '{}' is {dims:?}",
                                stage.function,
                                input.name,
                                input.dims,
                                input.source
                            ),
                            Dim::Static(_) => {}
                            Dim::Dynamic(sym) => match binding.iter().find(|(s, _)| s == sym) {
                                Some((_, m)) if *m != n => bail!(
                                    "stage '{}': one size bound to {m} and {n}",
                                    stage.function
                                ),
                                Some(_) => {}
                                None => binding.push((*sym, n)),
                            },
                        }
                    }
                }
                feeds.push((input.name.clone(), typed(&v, input.dtype)?));
            }
            binding.sort();
            if !stage.built.contains_key(&binding) {
                let weights = &mut self.weights[stage.weights];
                stage.build(binding.clone(), weights)?;
            }
            let built = stage.built.get_mut(&binding).expect("just built");
            let refs: Vec<(&str, &[u8], DType)> = feeds
                .iter()
                .map(|(n, (d, t))| (n.as_str(), d.as_slice(), *t))
                .collect();
            let outs = built.compiled.run_typed(&refs);
            for ((name, (data, dtype)), shape) in stage.outputs.iter().zip(outs).zip(&built.shapes)
            {
                values.insert(name.clone(), Tensor::new(name, dtype, shape.clone(), data));
            }
            if release {
                stage.built.clear();
            }
        }
        Ok(())
    }
}

impl Stage {
    /// Compile the stage at `binding` and bind its params.
    fn build(&mut self, binding: Vec<(u32, usize)>, weights: &mut PackageWeights) -> Result<()> {
        let graph = if binding.is_empty() {
            self.graph.clone()
        } else {
            let mut g =
                rlx_ir::dynamic::specialize(&self.graph, &rlx_ir::DimBinding::from_pairs(&binding));
            if g.has_dynamic_dims() {
                bail!(
                    "stage '{}' at {binding:?} still has unbound dims: its inputs' dims must bind \
                     every symbol",
                    self.function
                );
            }
            g.name = self.function.clone();
            g
        };
        let shapes = graph
            .outputs
            .iter()
            .map(|&o| static_dims(&graph, o))
            .collect::<Result<_>>()?;
        let params = params_of(&graph);
        let mut compiled = weights.compile(self.device, graph);
        weights.bind(&mut compiled, &params)?;
        self.built.insert(binding, Built { compiled, shapes });
        Ok(())
    }
}

// ── pipeline: stages, tensors in and out ─────────────────────

/// A pipeline's loop (`loop: {stages, steps, schedule, until}`): its stages
/// run `steps` times, each seeing `step` and `steps` (`f32[1]`) and its
/// value of each `schedule` list (a diffusion's timestep, a sampler's
/// sigma), their outputs replacing what came before (the state). It ends
/// early once the `until` value is true (its first element above 0.5).
struct Looping {
    body: std::ops::Range<usize>,
    steps: usize,
    schedule: Vec<(String, Vec<f32>)>,
    until: Option<String>,
}

/// `metadata` with what the model takes and gives (`inputs`, `outputs`),
/// as a pipeline's contracts read.
fn with_contracts(mut metadata: Value, inputs: Value, outputs: Value) -> Value {
    metadata["inputs"] = inputs;
    metadata["outputs"] = outputs;
    metadata
}

/// Contracts as `info().metadata.inputs` / `outputs` reports them: each
/// with its `name`, `kind` (`"tensor"` unless it says), `dtype` and `shape`
/// filled in from the spec, and whatever else it declares.
fn contract_json(contracts: &[Value], specs: &[TensorSpec]) -> Vec<Value> {
    contracts
        .iter()
        .zip(specs)
        .map(|(c, s)| {
            let mut c = c.clone();
            let o = c.as_object_mut().expect("a contract is an object");
            o.entry("kind").or_insert_with(|| json!("tensor"));
            o.entry("dtype")
                .or_insert_with(|| json!(super::dtype_name(s.dtype)));
            o.entry("shape").or_insert_with(|| json!(s.shape));
            c
        })
        .collect()
}

/// A model that is its stages: named tensors in, named tensors out (an
/// image encoder with its preprocessing, a feature extractor, …).
struct Pipeline {
    stages: Stages,
    /// The stages run once first.
    init: std::ops::Range<usize>,
    looping: Option<Looping>,
    /// The stages run once after the loop.
    finish: std::ops::Range<usize>,
    inputs: Vec<String>,
    /// Each input's declared spec, and whether its contract states a shape
    /// (a tensor of another rank or static size is refused).
    contracts: Vec<(bool, TensorSpec)>,
    outputs: Vec<String>,
    info: ModelInfo,
}

/// The `pipeline` entry of `pkg` as a model.
pub(super) fn pipeline(
    request: &ModelRequest,
    mut pkg: Opened,
    entry: &Value,
) -> Result<Box<dyn Model>> {
    let metadata = pkg.metadata();
    let list = entry
        .get("stages")
        .and_then(Value::as_array)
        .context("entries.pipeline.stages lists the functions it runs")?;
    // The stages before the loop, the loop's, the ones after: one list,
    // one set of weights.
    let looping = entry.get("loop").filter(|l| !l.is_null());
    let body: Vec<Value> = match looping {
        Some(l) => l
            .get("stages")
            .and_then(Value::as_array)
            .context("entries.pipeline.loop.stages lists the functions each step runs")?
            .clone(),
        None => Vec::new(),
    };
    let finish: Vec<Value> = entry
        .get("finish")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let all: Vec<Value> = list.iter().chain(&body).chain(&finish).cloned().collect();
    let mut stages = Stages::open(request, &mut pkg, &all)?;
    stages.weights[0] = std::mem::replace(&mut pkg.weights, PackageWeights::empty());
    let (n_init, n_body) = (list.len(), body.len());
    let looping = match looping {
        Some(l) => Some(Looping {
            body: n_init..n_init + n_body,
            steps: l.get("steps").and_then(Value::as_u64).unwrap_or(1) as usize,
            schedule: l
                .get("schedule")
                .and_then(Value::as_object)
                .map(|o| {
                    o.iter()
                        .map(|(k, v)| {
                            let vals: Vec<f32> = v
                                .as_array()
                                .with_context(|| format!("loop.schedule.{k} is a list of numbers"))?
                                .iter()
                                .map(|x| x.as_f64().map(|x| x as f32))
                                .collect::<Option<_>>()
                                .with_context(|| {
                                    format!("loop.schedule.{k} is a list of numbers")
                                })?;
                            Ok((k.clone(), vals))
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?
                .unwrap_or_default(),
            until: l.get("until").and_then(Value::as_str).map(String::from),
        }),
        None => None,
    };
    let finish = n_init + n_body..n_init + n_body + finish.len();
    // Each input / output: a name, or a contract `{name, kind, dtype,
    // shape, …}` (what a caller feeds: `kind: "image"`, `color: "rgb"`, …).
    let contracts = |key: &str| -> Result<Vec<Value>> {
        entry
            .get(key)
            .and_then(Value::as_array)
            .with_context(|| format!("entries.pipeline.{key} names its values"))?
            .iter()
            .map(|v| match v {
                Value::String(n) => Ok(json!({"name": n})),
                Value::Object(o) if o.get("name").and_then(Value::as_str).is_some() => {
                    Ok(v.clone())
                }
                other => bail!("entries.pipeline.{key}: a name or {{name, …}}, found {other}"),
            })
            .collect()
    };
    let (in_contracts, out_contracts) = (contracts("inputs")?, contracts("outputs")?);
    let name_of = |c: &Value| c["name"].as_str().unwrap_or_default().to_string();
    let inputs: Vec<String> = in_contracts.iter().map(name_of).collect();
    let outputs: Vec<String> = out_contracts.iter().map(name_of).collect();
    // What it takes: the first stage input reading each value.
    let spec = |value: &str| {
        let input = stages
            .list
            .iter()
            .flat_map(|s| &s.inputs)
            .find(|i| i.source == value);
        TensorSpec {
            name: value.to_string(),
            dtype: input.map_or(DType::F32, |i| i.dtype),
            shape: input.map_or_else(Vec::new, |i| {
                i.dims
                    .iter()
                    .map(|d| match d {
                        Dim::Static(n) => Some(*n),
                        Dim::Dynamic(_) => None,
                    })
                    .collect()
            }),
        }
    };
    // A contract's dtype and shape over what the graph says.
    let declared = |c: &Value, mut s: TensorSpec| -> Result<TensorSpec> {
        if let Some(d) = c.get("dtype").and_then(Value::as_str) {
            s.dtype = super::parse_dtype(d)?;
        }
        if let Some(dims) = c.get("shape").and_then(Value::as_array) {
            s.shape = dims
                .iter()
                .map(|d| d.as_u64().map(|n| n as usize))
                .collect();
        }
        Ok(s)
    };
    let in_specs = in_contracts
        .iter()
        .map(|c| declared(c, spec(&name_of(c))))
        .collect::<Result<Vec<_>>>()?;
    let out_specs = out_contracts
        .iter()
        .map(|c| {
            let s = TensorSpec {
                name: name_of(c),
                dtype: DType::F32,
                shape: Vec::new(),
            };
            declared(c, s)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut metadata = metadata;
    metadata["inputs"] = json!(contract_json(&in_contracts, &in_specs));
    metadata["outputs"] = json!(contract_json(&out_contracts, &out_specs));
    let info = ModelInfo {
        provider: "rlxp".into(),
        family: pkg.meta_str("family", "graph"),
        task: entry
            .get("task")
            .and_then(Value::as_str)
            .map_or_else(|| pkg.meta_str("task", "graph"), String::from),
        device: request.device_label(),
        inputs: in_specs.clone(),
        outputs: out_specs,
        metadata,
    };
    Ok(Box::new(Pipeline {
        stages,
        init: 0..n_init,
        looping,
        finish,
        inputs,
        contracts: in_contracts
            .into_iter()
            .zip(in_specs)
            .map(|(c, s)| (c.get("shape").is_some(), s))
            .collect(),
        outputs,
        info,
    }))
}

impl Model for Pipeline {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn run(&mut self, inputs: &[Tensor], options: &Value) -> Result<Vec<Tensor>> {
        let mut values = HashMap::new();
        for name in &self.inputs {
            // By name; a single input may come under any name.
            let t = inputs
                .iter()
                .find(|t| &t.name == name)
                .or_else(|| (self.inputs.len() == 1 && inputs.len() == 1).then(|| &inputs[0]))
                .with_context(|| format!("no input '{name}'"))?;
            let (shaped, spec) =
                &self.contracts[self.inputs.iter().position(|n| n == name).unwrap()];
            if *shaped {
                let fits = t.shape.len() == spec.shape.len()
                    && t.shape
                        .iter()
                        .zip(&spec.shape)
                        .all(|(&n, d)| d.is_none_or(|d| d == n));
                if !fits {
                    bail!(
                        "input '{name}' is {:?}; this model takes {:?} (None: any size)",
                        t.shape,
                        spec.shape
                    );
                }
            }
            values.insert(name.clone(), t.clone());
        }
        // `seed` (option, 0 by default) for stages that draw noise.
        let seed = options.get("seed").and_then(Value::as_u64).unwrap_or(0);
        values.insert("seed".into(), Tensor::f32("seed", vec![1], &[seed as f32]));
        let scalar = |name: &str, v: f32| Tensor::f32(name, vec![1], &[v]);
        self.stages.run_range(self.init.clone(), &mut values)?;
        if let Some(l) = &self.looping {
            // `steps` (option) over the entry's.
            let steps = options
                .get("steps")
                .and_then(Value::as_u64)
                .map_or(l.steps, |s| s as usize);
            for (name, vals) in &l.schedule {
                if vals.len() < steps {
                    bail!(
                        "loop.schedule.{name} has {} values for {steps} steps",
                        vals.len()
                    );
                }
            }
            for step in 0..steps {
                values.insert("step".into(), scalar("step", step as f32));
                values.insert("steps".into(), scalar("steps", steps as f32));
                for (name, vals) in &l.schedule {
                    values.insert(name.clone(), scalar(name, vals[step]));
                }
                self.stages.run_range(l.body.clone(), &mut values)?;
                if let Some(u) = &l.until
                    && values
                        .get(u)
                        .with_context(|| format!("loop.until reads '{u}', which no stage made"))?
                        .to_f32()?
                        .first()
                        .is_some_and(|v| *v > 0.5)
                {
                    break;
                }
            }
        }
        self.stages.run_range(self.finish.clone(), &mut values)?;
        self.outputs
            .iter()
            .map(|n| {
                values
                    .remove(n)
                    .with_context(|| format!("no stage makes '{n}'"))
            })
            .collect()
    }
}

// ── synthesize: text to audio ────────────────────────────────

/// A speech synthesizer from a package (`synthesize` entry): the text,
/// tokenized (into a `template`'s `{text}` when it has one), as the value
/// `ids`; a `voice` vector from the package's `voices` when it has them;
/// then either `stages` straight to the audio, or a `decoder` (a
/// `generate` entry) that writes codec tokens first, their ids less
/// `token_offset` (those within `codebook`) the value `codes` the stages
/// decode. `output` names the audio.
struct GraphTts {
    family: String,
    sample_rate: u32,
    codec: Box<dyn TextCodec>,
    template: Option<String>,
    stages: Stages,
    voices: Vec<(String, Vec<f32>)>,
    decoder: Option<(GraphLm, Vec<u32>)>,
    token_offset: u32,
    codebook: Option<u32>,
    max_tokens: usize,
    output: String,
}

/// The `synthesize` entry of `pkg` as a model.
pub(super) fn text_to_speech(
    request: &ModelRequest,
    mut pkg: Opened,
    entry: &Value,
) -> Result<Box<dyn Model>> {
    let metadata = pkg.metadata();
    // The stages read the package from their own handle: the decoder takes
    // this one's weights.
    let mut own = pkg.reopen()?;
    let list = entry
        .get("stages")
        .and_then(Value::as_array)
        .context("entries.synthesize.stages turn tokens or codes into audio")?;
    let mut stages = Stages::open(request, &mut own, list)?;
    let mut voices = Vec::new();
    if let Some(v) = entry.get("voices").and_then(Value::as_object) {
        for (name, tensor) in v {
            let t = tensor
                .as_str()
                .with_context(|| format!("voices.{name} names a tensor"))?;
            voices.push((name.clone(), own.weights.take_table(t)?));
        }
    }
    stages.weights[0] = std::mem::replace(&mut own.weights, PackageWeights::empty());
    let decoder = match entry.get("decoder").filter(|d| !d.is_null()) {
        Some(d) => Some((graph_lm(request, &mut pkg, d)?, u32s(d.get("stop_ids")))),
        None => None,
    };
    let tokenizer = entry
        .get("tokenizer")
        .and_then(Value::as_str)
        .context("entries.synthesize.tokenizer names the tokenizer.json")?;
    let codec = super::load_tokenizer(&pkg.file(tokenizer)?, &Value::Null)?;
    let sample_rate = entry
        .get("sample_rate")
        .and_then(Value::as_u64)
        .context("entries.synthesize.sample_rate")? as u32;
    let family = pkg.meta_str("family", "tts");
    let voice_names: Vec<&String> = voices.iter().map(|(n, _)| n).collect();
    let info = ModelInfo {
        provider: "rlxp".into(),
        family: family.clone(),
        task: "text-to-speech".into(),
        device: request.device_label(),
        metadata: {
            let mut m = with_contracts(
                metadata,
                json!([{"name": "text", "kind": "text"}]),
                json!([{"name": "audio", "kind": "audio", "dtype": "f32", "shape": [null],
                        "sample_rate": sample_rate, "channels": 1}]),
            );
            m["voices"] = json!(voice_names);
            m
        },
        ..Default::default()
    };
    let tts = GraphTts {
        family,
        sample_rate,
        codec,
        template: entry
            .get("template")
            .and_then(Value::as_str)
            .map(String::from),
        stages,
        voices,
        decoder,
        token_offset: entry
            .get("token_offset")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
        codebook: entry
            .get("codebook")
            .and_then(Value::as_u64)
            .map(|c| c as u32),
        max_tokens: entry
            .get("max_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(1024) as usize,
        output: entry
            .get("output")
            .and_then(Value::as_str)
            .unwrap_or("audio")
            .to_string(),
    };
    Ok(super::tasks::text_to_speech_model(tts, info))
}

impl super::tasks::TextToSpeech for GraphTts {
    fn family(&self) -> &str {
        &self.family
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn synthesize(&mut self, text: &str, options: &Value) -> Result<Vec<f32>> {
        let prompt = match &self.template {
            Some(t) => t.replace("{text}", text),
            None => text.to_string(),
        };
        let ids = self.codec.encode(&prompt)?;
        let as_tensor = |name: &str, v: &[u32]| {
            super::Tensor::new(
                name,
                DType::I32,
                vec![v.len()],
                v.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect(),
            )
        };
        let mut values = HashMap::new();
        values.insert("ids".to_string(), as_tensor("ids", &ids));
        if !self.voices.is_empty() {
            let want = options.get("voice").and_then(Value::as_str);
            let (_, v) = match want {
                Some(w) => self
                    .voices
                    .iter()
                    .find(|(n, _)| n == w)
                    .with_context(|| format!("no voice '{w}'"))?,
                None => &self.voices[0],
            };
            values.insert(
                "voice".into(),
                super::Tensor::f32("voice", vec![v.len()], v),
            );
        }
        if let Some((lm, stop)) = &mut self.decoder {
            // The codec tokens, greedy or sampled as the options say.
            let max = options
                .get("max_tokens")
                .and_then(Value::as_u64)
                .map_or(self.max_tokens, |m| m as usize);
            let opts = super::sample_opts(options);
            let chain = opts.clone().into_chain();
            let seed = options.get("seed").and_then(Value::as_u64).unwrap_or(0);
            let mut rng = rlx_ir::Philox4x32::new(seed);
            let mut state = crate::samplers::SamplerState::default();
            lm.reset();
            let mut context = ids.clone();
            let mut logits = crate::LmRunner::predict_logits(lm, &context)?;
            let mut codes = Vec::new();
            for _ in 0..max {
                let next = if opts.is_greedy() {
                    crate::lm::argmax_u32(&logits)
                } else {
                    chain.sample(&mut logits, &context, &mut state, &mut rng)
                };
                if stop.contains(&next) {
                    break;
                }
                context.push(next);
                if next >= self.token_offset
                    && self.codebook.is_none_or(|c| next < self.token_offset + c)
                {
                    codes.push(next - self.token_offset);
                }
                logits = lm.step(next)?;
            }
            values.insert("codes".into(), as_tensor("codes", &codes));
        }
        self.stages.run(&mut values)?;
        values
            .remove(&self.output)
            .with_context(|| format!("no stage makes '{}'", self.output))?
            .to_f32()
    }
}

// ── transcribe: stages, then a decoder ───────────────────────

/// A speech recognizer from a package: stages (an audio front end, an
/// encoder, its cross-attention K / V) feed a decoder that runs as a
/// `generate` entry's would.
struct GraphStt {
    family: &'static str,
    sample_rate: u32,
    stages: Stages,
    lm: GraphLm,
    codec: Box<dyn TextCodec>,
    prompt: Vec<u32>,
    stop: Vec<u32>,
    /// Ids never emitted, and ids not emitted first (Whisper's
    /// `suppress_tokens` / `begin_suppress_tokens`).
    suppress: Vec<u32>,
    begin_suppress: Vec<u32>,
    max_tokens: usize,
    /// Audio padded with silence to a multiple of this (`audio_multiple`:
    /// whole frames of the front end); the stages also get the real length
    /// as `audio_samples` (`f32[1]`).
    audio_multiple: usize,
    /// Tokens per second of audio, and extra (`token_budget: {per_second,
    /// extra}`), for a model whose decoder stops by a budget, not a size.
    token_budget: Option<(f64, usize)>,
}

/// The `transcribe` entry of `pkg` as a model.
pub(super) fn speech_to_text(
    request: &ModelRequest,
    mut pkg: Opened,
    entry: &Value,
) -> Result<Box<dyn Model>> {
    let metadata = pkg.metadata();
    let list = entry
        .get("stages")
        .and_then(Value::as_array)
        .context("entries.transcribe.stages lists the functions before the decoder")?;
    // The stages read the package from their own handle, so a symbolic one
    // (compiled at its first shape) keeps its weights; the decoder takes
    // this one's.
    let mut own = pkg.reopen()?;
    own.target = pkg.target.clone();
    let mut stages = Stages::open(request, &mut own, list)?;
    stages.weights[0] = std::mem::replace(&mut own.weights, PackageWeights::empty());
    let decoder = entry
        .get("decoder")
        .context("entries.transcribe.decoder is a generate entry")?;
    let lm = graph_lm(request, &mut pkg, decoder)?;
    let tokenizer = entry
        .get("tokenizer")
        .and_then(Value::as_str)
        .context("entries.transcribe.tokenizer names the tokenizer.json")?;
    let codec = super::load_tokenizer(&pkg.file(tokenizer)?, &Value::Null)?;
    let family = lm.family;
    let runner = GraphStt {
        family,
        sample_rate: entry
            .get("sample_rate")
            .and_then(Value::as_u64)
            .unwrap_or(16_000) as u32,
        stages,
        lm,
        codec,
        prompt: u32s(entry.get("prompt_ids")),
        stop: u32s(entry.get("stop_ids")),
        suppress: u32s(entry.get("suppress_ids")),
        begin_suppress: u32s(entry.get("begin_suppress_ids")),
        max_tokens: entry
            .get("max_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(224) as usize,
        audio_multiple: entry
            .get("audio_multiple")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1) as usize,
        token_budget: entry.get("token_budget").and_then(|b| {
            Some((
                b.get("per_second")?.as_f64()?,
                b.get("extra").and_then(Value::as_u64).unwrap_or(0) as usize,
            ))
        }),
    };
    let info = ModelInfo {
        provider: "rlxp".into(),
        family: family.into(),
        task: "speech-recognition".into(),
        device: request.device_label(),
        metadata: with_contracts(
            metadata,
            json!([{
                "name": "audio",
                "kind": "audio",
                "dtype": "f32",
                "shape": [null],
                "sample_rate": runner.sample_rate,
                "channels": 1,
            }]),
            json!([{"name": "text", "kind": "text"}]),
        ),
        ..Default::default()
    };
    Ok(super::tasks::speech_to_text_model(runner, info))
}

impl super::tasks::SpeechToText for GraphStt {
    fn family(&self) -> &str {
        self.family
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32, options: &Value) -> Result<String> {
        if sample_rate != self.sample_rate {
            bail!(
                "audio at {sample_rate} Hz; this model takes {} Hz",
                self.sample_rate
            );
        }
        // Audio is cut or padded with silence to what the front end takes
        // (Whisper: 30 s).
        let samples = audio.len();
        let mut audio = audio.to_vec();
        if let Some(n) = self.stages.static_numel("audio") {
            audio.resize(n, 0.0);
        }
        audio.resize(audio.len().next_multiple_of(self.audio_multiple), 0.0);
        let mut values = HashMap::new();
        values.insert(
            "audio".to_string(),
            Tensor::f32("audio", vec![audio.len()], &audio),
        );
        values.insert(
            "audio_samples".to_string(),
            Tensor::f32("audio_samples", vec![1], &[samples as f32]),
        );
        self.stages.run(&mut values)?;
        // The decoder: the prompt ids, then greedy until a stop id.
        self.lm.reset();
        self.lm.context = values
            .into_iter()
            .map(|(k, t)| Ok((k, t.to_f32()?)))
            .collect::<Result<_>>()?;
        let budget = self.token_budget.map(|(rate, extra)| {
            (samples as f64 / f64::from(self.sample_rate) * rate).ceil() as usize + extra
        });
        let max = options
            .get("max_tokens")
            .and_then(Value::as_u64)
            .map_or(budget.unwrap_or(self.max_tokens), |m| m as usize);
        let mut logits = crate::LmRunner::predict_logits(&mut self.lm, &self.prompt)?;
        let mut out = Vec::new();
        while out.len() < max {
            let first = out.is_empty();
            for &id in self
                .suppress
                .iter()
                .chain(self.begin_suppress.iter().filter(|_| first))
            {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
            let next = crate::lm::argmax_u32(&logits);
            if self.stop.contains(&next) {
                break;
            }
            out.push(next);
            logits = self.lm.step(next)?;
        }
        Ok(self.codec.decode(&out)?.trim().to_string())
    }
}
