// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generic language-model runner trait and shared builder.
//!
//! Until now every `rlx-<family>` model crate carried its own
//! `*RunnerBuilder` (Qwen3RunnerBuilder, Llama32RunnerBuilder, …)
//! with the same fields, the same `*ConfigSource { Embedded |
//! JsonFile | Explicit(T) }` enum, and the same auto-packed-GGUF
//! heuristic. This module hoists those shapes upstream so that:
//!
//!   1. `LmRunner` can live in `rlx-runtime` (today's home in
//!      `rlx-cli` forces every model crate to take a dependency on
//!      the CLI helper crate).
//!   2. Per-family runners can `Deref` to / wrap [`LmRunnerBuilder`]
//!      instead of redefining the same fields.
//!   3. Downstream tools (`skill`, web apps) can talk to runners
//!      through one trait without compiling in every model crate.
//!
//! The trait surface mirrors the existing `rlx_cli::LmRunner`. The
//! CLI re-export is kept for backwards compat.

use std::path::{Path, PathBuf};

use crate::Device;

/// A model-agnostic decode-session snapshot: the KV cache plus the token
/// history that produced it. Used by the prompt cache to reuse a prefix.
#[derive(Debug, Clone)]
pub struct SessionSnapshot {
    pub kv: crate::kv_cache::LayerKvCache,
    pub tokens: Vec<u32>,
}

/// Minimal per-family runner interface used by `auto_dispatch` and
/// the `rlx-text` / `skill` integration.
///
/// Implementations must be `Send` so the boxed trait can move across
/// threads (e.g. when a server runs inference on a worker pool).
/// `Sync` is intentionally not required — runners hold mutable
/// per-call compile / cache state.
pub trait LmRunner: Send {
    /// Short family identifier (`"qwen3"`, `"llama32"`, `"gemma"`).
    fn family(&self) -> &'static str;

    /// LM head vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Run prefill on `prompt_ids` and return last-token logits.
    fn predict_logits(&mut self, prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>>;

    /// Generate up to `n_new` tokens after `prompt_ids` using greedy
    /// (argmax) sampling. The default impl re-prefills on the full
    /// context each step — per-family runners should override with
    /// their cached decode fast path.
    ///
    /// `on_token` returns `true` to continue, `false` to stop.
    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> anyhow::Result<Vec<u32>> {
        let mut context: Vec<u32> = prompt_ids.to_vec();
        let mut produced: Vec<u32> = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let logits = self.predict_logits(&context)?;
            let next = argmax_u32(&logits);
            produced.push(next);
            let cont = on_token(next);
            context.push(next);
            if !cont {
                break;
            }
        }
        Ok(produced)
    }

    /// Greedy generation after each of `prompts`: up to `n_new` tokens
    /// each, a sequence ending at its first id in `stop` (left out). The
    /// default runs them one after another; a runner with a batched step
    /// decodes them together.
    fn generate_batch(
        &mut self,
        prompts: &[Vec<u32>],
        n_new: usize,
        stop: &[u32],
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        prompts
            .iter()
            .map(|p| {
                let mut out = self.generate(p, n_new, &mut |t| !stop.contains(&t))?;
                if out.last().is_some_and(|t| stop.contains(t)) {
                    out.pop();
                }
                Ok(out)
            })
            .collect()
    }

    /// Beam search after `prompt`: the best of `beams` hypotheses kept
    /// each step ([`beam_search`]), up to `n_new` tokens, a hypothesis ending
    /// at an id in `stop` (left out). The default extends each hypothesis
    /// through [`Self::predict_logits`] on its whole context (correct, a
    /// prompt's work a token); a runner with a batched step keeps the beams
    /// in it.
    fn generate_beam(
        &mut self,
        prompt: &[u32],
        beams: usize,
        n_new: usize,
        stop: &[u32],
        length_penalty: f32,
    ) -> anyhow::Result<Vec<u32>> {
        let first = self.predict_logits(prompt)?;
        let mut contexts: Vec<Vec<u32>> = vec![prompt.to_vec()];
        beam_search(
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
                contexts.iter().map(|c| self.predict_logits(c)).collect()
            },
        )
    }

    /// Continuous batching: room for `slots` sequences decoded together,
    /// which requests join ([`Self::serve_add`]) and leave
    /// ([`Self::serve_remove`]) between steps. Default: unsupported.
    fn serve_open(&mut self, _slots: usize) -> anyhow::Result<()> {
        Err(anyhow::anyhow!(
            "{} serves no batch of requests",
            self.family()
        ))
    }

    /// A request in a free slot; its slot. Its prompt goes in a token a
    /// step, from the batch's current position.
    fn serve_add(&mut self, _prompt: &[u32]) -> anyhow::Result<usize> {
        Err(anyhow::anyhow!(
            "{} serves no batch of requests",
            self.family()
        ))
    }

    /// One step of every request in the batch: `(slot, token)` for each
    /// whose prompt is in (its greedy next token, which it is fed next).
    fn serve_step(&mut self) -> anyhow::Result<Vec<(usize, u32)>> {
        Err(anyhow::anyhow!(
            "{} serves no batch of requests",
            self.family()
        ))
    }

    /// The request in `slot` is done; the slot takes the next.
    fn serve_remove(&mut self, _slot: usize) {}

    /// Prefill `ids` with `media` (images, …) in place of their placeholder
    /// tokens: each run of a media kind's placeholder id becomes as many
    /// positions as the media's embedding has rows. The prompt as decoding
    /// continues it (the placeholders expanded), and the next position's
    /// logits. The default takes no media.
    fn prefill_with_media(
        &mut self,
        _ids: &[u32],
        _media: &[crate::model::Tensor],
    ) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
        Err(anyhow::anyhow!(
            "{} takes no images or other media",
            self.family()
        ))
    }

    /// Prefill `prompt_ids`, seed the decode KV cache, and return the
    /// last-position logits `[vocab]`. Together with `decode_logits` this
    /// gives a **host-driven** decode loop: the caller owns sampling, logit
    /// bias, log-probs, and stop detection (used by the HTTP server). The
    /// default reports unsupported so existing runners keep compiling.
    fn prefill_logits(&mut self, _prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        Err(anyhow::anyhow!(
            "{}: host-driven logits decode (prefill_logits) unsupported — wrap \
             this runner in `HostDriven` for a correct, slower version, or \
             forward the family's own cached pair",
            self.family()
        ))
    }

    /// Feed one token, advance the KV cache, and return the next-position
    /// logits `[vocab]`. See [`LmRunner::prefill_logits`]. Default: unsupported.
    fn decode_logits(&mut self, _token: u32) -> anyhow::Result<Vec<f32>> {
        Err(anyhow::anyhow!(
            "{}: host-driven logits decode (decode_logits) unsupported — wrap \
             this runner in `HostDriven` for a correct, slower version, or \
             forward the family's own cached pair",
            self.family()
        ))
    }

    /// Register a callback fired at multimodal-generation **phase boundaries** —
    /// `"vision"` before the image is encoded and `"prefill"` before the LM
    /// prefill — so callers can surface real progress during the pre-token
    /// lead-in (vision encode + prefill are otherwise one opaque blocking call).
    /// Default: no-op (text-only runners ignore it). Pass `None` to clear.
    fn set_multimodal_phase_callback(
        &mut self,
        _cb: Option<std::sync::Arc<dyn Fn(&str) + Send + Sync>>,
    ) {
    }

    /// Prefill `prompt` reusing a session snapshot that already covers its
    /// first `reuse_len` tokens — only the suffix is processed. Returns the
    /// last-position logits. The default ignores the snapshot and does a full
    /// prefill, so callers always get correct logits even without reuse.
    fn prefill_logits_reusing(
        &mut self,
        prompt: &[u32],
        _snap: &SessionSnapshot,
        _reuse_len: usize,
    ) -> anyhow::Result<Vec<f32>> {
        self.prefill_logits(prompt)
    }

    /// Snapshot this runner's decode session (KV cache + token history) for
    /// prompt-cache reuse. `None` ⇒ the family doesn't support it.
    fn export_session(&self) -> Option<SessionSnapshot> {
        None
    }

    /// Restore a previously exported session so generation resumes from a
    /// cached prefix. Returns `true` if applied.
    fn restore_session(&mut self, _snap: &SessionSnapshot) -> bool {
        false
    }

    /// Long-context KV context-store occupancy `(blocks, tokens, disk_bytes)`
    /// when a store is enabled (e.g. rlx-qwen3 HNSW memory), else `None`. Lets a
    /// host inspect/track how the store grows without knowing the concrete runner.
    fn kv_store_stats(&self) -> Option<(usize, usize, usize)> {
        None
    }

    /// Whether this runner supports multimodal (image+text) generation.
    fn supports_multimodal(&self) -> bool {
        false
    }

    /// Multimodal generation: prefill with text where image markers are
    /// spliced with vision embeddings derived from `rgb`.
    fn generate_multimodal(
        &mut self,
        _prompt: &str,
        _rgb: &[u8],
        _img_w: usize,
        _img_h: usize,
        _tokenizer: Option<&Path>,
        _n_new: usize,
        _on_token: &mut dyn FnMut(u32) -> bool,
    ) -> anyhow::Result<Vec<u32>> {
        Err(anyhow::anyhow!(
            "this LmRunner does not support multimodal generation"
        ))
    }
}

/// Beam search from `first` (the logits after the prompt). Each step every
/// live hypothesis proposes its `2 × beams` likeliest tokens; the best
/// `beams` that do not stop live on, and a stop token among the best
/// `beams` finishes its hypothesis, scored by its log-probability over
/// `length ^ length_penalty`. `step(parents, tokens)` extends the
/// hypotheses: the `i`-th new one is hypothesis `parents[i]` of the last
/// step plus `tokens[i]`, and it returns each one's next logits. Ends with
/// `beams` finished hypotheses or `n_new` tokens; returns the best finished
/// or live one's tokens. Ties go to the earlier hypothesis, then the lower
/// id, so every implementation of `step` picks the same.
///
/// The Hugging Face rule of [`crate::search::BeamConfig::new`]; use
/// [`crate::search::beam_search`] directly for n-best, masks, or other rules.
pub fn beam_search(
    first: &[f32],
    beams: usize,
    n_new: usize,
    stop: &[u32],
    length_penalty: f32,
    mut step: impl FnMut(&[usize], &[u32]) -> anyhow::Result<Vec<Vec<f32>>>,
) -> anyhow::Result<Vec<u32>> {
    let cfg = rlx_search::BeamConfig::new(beams, n_new)
        .stop(stop)
        .length_penalty(length_penalty as f64);
    let best = rlx_search::beam_search(cfg, &[], first, &mut (), |s| step(s.parents, s.tokens))?;
    Ok(best
        .into_iter()
        .next()
        .map(|h| h.tokens)
        .unwrap_or_default())
}

/// A boxed runner is a runner.
///
/// Needed because a server holds `Box<dyn LmRunner>` — the engine is chosen
/// from a string at runtime — while anything generic over `R: LmRunner`,
/// [`HostDriven`] included, needs a concrete type. Without this, the
/// adapters in this module are unreachable from exactly the code that needs
/// them most.
impl LmRunner for Box<dyn LmRunner> {
    fn family(&self) -> &'static str {
        (**self).family()
    }

    fn vocab_size(&self) -> usize {
        (**self).vocab_size()
    }

    fn predict_logits(&mut self, prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        (**self).predict_logits(prompt_ids)
    }

    fn generate_batch(
        &mut self,
        prompts: &[Vec<u32>],
        n_new: usize,
        stop: &[u32],
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        (**self).generate_batch(prompts, n_new, stop)
    }

    fn generate_beam(
        &mut self,
        prompt: &[u32],
        beams: usize,
        n_new: usize,
        stop: &[u32],
        length_penalty: f32,
    ) -> anyhow::Result<Vec<u32>> {
        (**self).generate_beam(prompt, beams, n_new, stop, length_penalty)
    }

    fn prefill_with_media(
        &mut self,
        ids: &[u32],
        media: &[crate::model::Tensor],
    ) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
        (**self).prefill_with_media(ids, media)
    }

    fn serve_open(&mut self, slots: usize) -> anyhow::Result<()> {
        (**self).serve_open(slots)
    }

    fn serve_add(&mut self, prompt: &[u32]) -> anyhow::Result<usize> {
        (**self).serve_add(prompt)
    }

    fn serve_step(&mut self) -> anyhow::Result<Vec<(usize, u32)>> {
        (**self).serve_step()
    }

    fn serve_remove(&mut self, slot: usize) {
        (**self).serve_remove(slot)
    }

    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> anyhow::Result<Vec<u32>> {
        (**self).generate(prompt_ids, n_new, on_token)
    }

    fn prefill_logits(&mut self, prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        (**self).prefill_logits(prompt_ids)
    }

    fn decode_logits(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        (**self).decode_logits(token)
    }

    fn set_multimodal_phase_callback(
        &mut self,
        cb: Option<std::sync::Arc<dyn Fn(&str) + Send + Sync>>,
    ) {
        (**self).set_multimodal_phase_callback(cb);
    }

    fn prefill_logits_reusing(
        &mut self,
        prompt: &[u32],
        snap: &SessionSnapshot,
        reuse_len: usize,
    ) -> anyhow::Result<Vec<f32>> {
        (**self).prefill_logits_reusing(prompt, snap, reuse_len)
    }

    fn export_session(&self) -> Option<SessionSnapshot> {
        (**self).export_session()
    }

    fn restore_session(&mut self, snap: &SessionSnapshot) -> bool {
        (**self).restore_session(snap)
    }

    fn kv_store_stats(&self) -> Option<(usize, usize, usize)> {
        (**self).kv_store_stats()
    }

    fn supports_multimodal(&self) -> bool {
        (**self).supports_multimodal()
    }

    fn generate_multimodal(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&Path>,
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> anyhow::Result<Vec<u32>> {
        (**self).generate_multimodal(prompt, rgb, img_w, img_h, tokenizer, n_new, on_token)
    }
}

/// A host-driven decode pair for a runner that does not have one.
///
/// `prefill_logits` and `decode_logits` are what a server needs: the caller
/// owns sampling, so temperature, a logit bias, log-probs and stop strings
/// all live above the model. Both default to an error on this trait, which
/// keeps families compiling — and turns a missing override into a runner
/// that cannot generate at all for any caller that samples for itself. That
/// is not hypothetical: `Llama32Runner` had both methods and simply did not
/// forward them through the trait, so every request failed with
/// "unsupported" while the fast path sat there unused.
///
/// This wrapper supplies the pair for any runner by re-prefilling the whole
/// context each step, on top of `predict_logits`, which every runner must
/// implement. It is O(n²) over a generation and no substitute for a
/// family's cached path — it is the difference between slow and broken.
///
/// ```ignore
/// let runner = HostDriven::new(my_runner);   // now `prefill_logits` works
/// ```
pub struct HostDriven<R: LmRunner> {
    inner: R,
    /// The sequence so far. Held here because a trait default cannot own
    /// state, which is the reason the defaults could not be made to work
    /// without a wrapper.
    context: Vec<u32>,
    warned: bool,
}

impl<R: LmRunner> HostDriven<R> {
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            context: Vec::new(),
            warned: false,
        }
    }

    /// The runner underneath, for the family's own API.
    pub fn inner(&self) -> &R {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Whether the slow path is in use: the family had no cached
    /// host-driven pair and this is re-prefilling the whole context each
    /// token.
    ///
    /// Worth asking, because the cost is quadratic and invisible — a server
    /// that finds this true is serving correct answers at a price it did not
    /// choose, and the fix is in the family's runner rather than here. This
    /// crate does not log, so the answer is returned rather than printed.
    pub fn fell_back(&self) -> bool {
        self.warned
    }

    /// Said once rather than per token, and only where the host has asked
    /// for it: a line per token would bury the output of the thing it is
    /// warning about.
    fn note_fallback(&mut self) {
        if !self.warned {
            self.warned = true;
            if rlx_ir::env::var("RLX_WARN_SLOW_DECODE").is_some() {
                eprintln!(
                    "[lm] {}: no cached host-driven decode; re-prefilling the \
                     whole context each token (O(n^2) over a generation)",
                    self.inner.family()
                );
            }
        }
    }
}

impl<R: LmRunner> LmRunner for HostDriven<R> {
    fn family(&self) -> &'static str {
        self.inner.family()
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn predict_logits(&mut self, prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        self.inner.predict_logits(prompt_ids)
    }

    fn generate_batch(
        &mut self,
        prompts: &[Vec<u32>],
        n_new: usize,
        stop: &[u32],
    ) -> anyhow::Result<Vec<Vec<u32>>> {
        self.inner.generate_batch(prompts, n_new, stop)
    }

    fn generate_beam(
        &mut self,
        prompt: &[u32],
        beams: usize,
        n_new: usize,
        stop: &[u32],
        length_penalty: f32,
    ) -> anyhow::Result<Vec<u32>> {
        self.inner
            .generate_beam(prompt, beams, n_new, stop, length_penalty)
    }

    fn prefill_with_media(
        &mut self,
        ids: &[u32],
        media: &[crate::model::Tensor],
    ) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
        self.inner.prefill_with_media(ids, media)
    }

    fn serve_open(&mut self, slots: usize) -> anyhow::Result<()> {
        self.inner.serve_open(slots)
    }

    fn serve_add(&mut self, prompt: &[u32]) -> anyhow::Result<usize> {
        self.inner.serve_add(prompt)
    }

    fn serve_step(&mut self) -> anyhow::Result<Vec<(usize, u32)>> {
        self.inner.serve_step()
    }

    fn serve_remove(&mut self, slot: usize) {
        self.inner.serve_remove(slot)
    }

    fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> anyhow::Result<Vec<u32>> {
        self.inner.generate(prompt_ids, n_new, on_token)
    }

    fn prefill_logits(&mut self, prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        // The family's own pair if it has one; this wrapper is for when it
        // does not, and wrapping something that does should not make it
        // slower.
        match self.inner.prefill_logits(prompt_ids) {
            Ok(logits) => {
                self.context.clear();
                self.context.extend_from_slice(prompt_ids);
                Ok(logits)
            }
            Err(_) => {
                self.note_fallback();
                self.context.clear();
                self.context.extend_from_slice(prompt_ids);
                self.inner.predict_logits(prompt_ids)
            }
        }
    }

    fn decode_logits(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
        if self.context.is_empty() {
            anyhow::bail!(
                "{}: decode_logits before prefill_logits — there is no \
                 sequence to continue",
                self.inner.family()
            );
        }
        if let Ok(logits) = self.inner.decode_logits(token) {
            self.context.push(token);
            return Ok(logits);
        }
        self.note_fallback();
        self.context.push(token);
        let context = std::mem::take(&mut self.context);
        let out = self.inner.predict_logits(&context);
        self.context = context;
        out
    }

    fn set_multimodal_phase_callback(
        &mut self,
        cb: Option<std::sync::Arc<dyn Fn(&str) + Send + Sync>>,
    ) {
        self.inner.set_multimodal_phase_callback(cb);
    }

    fn prefill_logits_reusing(
        &mut self,
        prompt: &[u32],
        snap: &SessionSnapshot,
        reuse_len: usize,
    ) -> anyhow::Result<Vec<f32>> {
        self.inner.prefill_logits_reusing(prompt, snap, reuse_len)
    }

    fn export_session(&self) -> Option<SessionSnapshot> {
        self.inner.export_session()
    }

    fn restore_session(&mut self, snap: &SessionSnapshot) -> bool {
        self.inner.restore_session(snap)
    }

    fn kv_store_stats(&self) -> Option<(usize, usize, usize)> {
        self.inner.kv_store_stats()
    }

    fn supports_multimodal(&self) -> bool {
        self.inner.supports_multimodal()
    }

    fn generate_multimodal(
        &mut self,
        prompt: &str,
        rgb: &[u8],
        img_w: usize,
        img_h: usize,
        tokenizer: Option<&Path>,
        n_new: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> anyhow::Result<Vec<u32>> {
        self.inner
            .generate_multimodal(prompt, rgb, img_w, img_h, tokenizer, n_new, on_token)
    }
}

pub(crate) fn argmax_u32(logits: &[f32]) -> u32 {
    rlx_search::argmax_u32(logits)
}

// ─────────────────────────────────────────────────────────────────
// Weight format + config source
// ─────────────────────────────────────────────────────────────────

/// Weight file format. Detected from the file extension by default;
/// the CLI accepts `--format` to override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    Safetensors,
    Gguf,
    /// MLX / mlx-community layouts (dir, `.safetensors`, `.npz`, `.npy`).
    Mlx,
    /// HuggingFace DDUF (`.dduf` ZIP of nested safetensors).
    Dduf,
}

impl WeightFormat {
    /// Infer format from a path extension (or mlx-community directory).
    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        if path.is_dir() {
            if path.join("config.json").is_file()
                && (path.join("model.safetensors").is_file()
                    || path.join("model.safetensors.index.json").is_file()
                    || std::fs::read_dir(path)?.any(|e| {
                        e.ok()
                            .and_then(|e| e.path().extension().map(|x| x == "safetensors"))
                            .unwrap_or(false)
                    }))
            {
                return Ok(Self::Mlx);
            }
            return Err(anyhow::anyhow!(
                "directory {:?} is not an mlx-community weight dir (need config.json + safetensors)",
                path
            ));
        }
        match path.extension().and_then(|s| s.to_str()) {
            Some("safetensors") => Ok(Self::Safetensors),
            Some("gguf") => Ok(Self::Gguf),
            Some("npz") | Some("npy") => Ok(Self::Mlx),
            Some("dduf") => Ok(Self::Dduf),
            other => Err(anyhow::anyhow!(
                "cannot autodetect weight format from extension {:?} on {:?}",
                other,
                path
            )),
        }
    }

    /// Parse CLI `--format` values (`safetensors` | `gguf` | `mlx` | `dduf`).
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s {
            "safetensors" => Ok(Self::Safetensors),
            "gguf" => Ok(Self::Gguf),
            "mlx" | "npz" => Ok(Self::Mlx),
            "dduf" => Ok(Self::Dduf),
            other => Err(anyhow::anyhow!(
                "expected safetensors|gguf|mlx|dduf, got {other}"
            )),
        }
    }
}

/// Where to read a model config from.
///
/// Replaces the per-family `Qwen3ConfigSource`, `Llama32ConfigSource`,
/// `GemmaConfigSource`, `Qwen35ConfigSource` enums.
#[derive(Debug, Clone, Default)]
pub enum ConfigSource<T> {
    /// Read from GGUF metadata.
    #[default]
    Embedded,
    /// Read from a HuggingFace `config.json` at this path.
    JsonFile(PathBuf),
    /// Use the supplied config object directly.
    Explicit(T),
}

// ─────────────────────────────────────────────────────────────────
// Sampling
// ─────────────────────────────────────────────────────────────────

/// Mirostat variant selection. See `crate::samplers::{MirostatV1, MirostatV2}`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum MirostatMode {
    #[default]
    Off,
    V1,
    V2,
}

/// Sampling parameters. Greedy when `temperature == 0` and no advanced
/// sampler is enabled. All "advanced" knobs default to off / no-op so
/// legacy callers see classic top-k/top-p/temperature behaviour.
///
/// `into_chain()` turns these flat fields into a `SamplerChain` that
/// downstream backends can execute. Ordering follows llama.cpp's
/// canonical chain (penalties → temperature → top-k → typical → top-p
/// → top-n-σ → xtc → mirostat).
#[derive(Debug, Clone)]
pub struct SampleOpts {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub repetition_penalty: f32,

    // ── advanced samplers ────────────────────────────────────────
    /// Dynamic temperature [min, max] gated by softmax entropy.
    /// `None` ⇒ flat temperature only.
    pub dynamic_temp: Option<(f32, f32)>,
    /// Exponent used by [`crate::samplers::DynamicTemperature`].
    pub dynamic_temp_exponent: f32,
    /// Locally-typical sampling (Meister et al. 2022). 1.0 ⇒ off.
    pub typical_p: f32,
    /// Top-n-σ cutoff (Hewitt et al. 2024). 0 ⇒ off.
    pub top_n_sigma: f32,
    /// Min-p cutoff (Nguyen et al. 2024): keep tokens with prob ≥ `min_p` ·
    /// p_max. 0 ⇒ off.
    pub min_p: f32,
    /// XTC: probability of dropping high-confidence top tokens.
    pub xtc_threshold: f32,
    pub xtc_prob: f32,
    /// DRY repetition penalty knobs.
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_max_ngram: usize,
    pub dry_sequence_breakers: Vec<u32>,
    /// Mirostat mode + parameters.
    pub mirostat: MirostatMode,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub mirostat_m: usize,
    /// Frequency / presence penalties (OpenAI-style).
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repetition_window: usize,
    /// Minimum tokens kept by top-p / typical (avoid one-token nucleus).
    pub min_keep: usize,
}

impl Default for SampleOpts {
    fn default() -> Self {
        Self::greedy()
    }
}

impl SampleOpts {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            dynamic_temp: None,
            dynamic_temp_exponent: 1.0,
            typical_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            xtc_threshold: 0.0,
            xtc_prob: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_max_ngram: 32,
            dry_sequence_breakers: Vec::new(),
            mirostat: MirostatMode::Off,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_m: 100,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_window: 64,
            min_keep: 1,
        }
    }

    pub fn nucleus(temperature: f32, top_p: f32) -> Self {
        Self {
            temperature,
            top_p,
            ..Self::greedy()
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 && self.mirostat == MirostatMode::Off
    }

    /// True when only classic top-k/top-p/temperature are configured;
    /// backends can take a cheap fast path in this case (e.g. the
    /// existing `sample_row` CPU kernel) instead of building a chain.
    pub fn is_classic(&self) -> bool {
        self.dynamic_temp.is_none()
            && self.typical_p >= 1.0
            && self.top_n_sigma <= 0.0
            && self.min_p <= 0.0
            && self.xtc_prob <= 0.0
            && self.dry_multiplier <= 0.0
            && self.mirostat == MirostatMode::Off
            && self.frequency_penalty == 0.0
            && self.presence_penalty == 0.0
            && (self.repetition_penalty - 1.0).abs() < f32::EPSILON
    }

    /// Build the `SamplerChain` corresponding to these options. The
    /// returned chain is ready to drive `SamplerChain::sample` against
    /// a logits row + history. Greedy decoding produces a chain with
    /// one `Temperature{t:1e-6}` step (which collapses to argmax after
    /// softmax) — callers that want true greedy can short-circuit via
    /// `is_greedy()` before building the chain.
    pub fn into_chain(&self) -> crate::samplers::SamplerChain {
        use crate::samplers::*;
        let mut b = SamplerChain::builder();

        // 1. Penalties operate on raw logits, before any temperature
        //    scaling — matches llama.cpp's order.
        if (self.repetition_penalty - 1.0).abs() > f32::EPSILON
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
        {
            b = b.push(RepetitionPenalty {
                penalty: self.repetition_penalty,
                frequency: self.frequency_penalty,
                presence: self.presence_penalty,
                last_n: self.repetition_window,
            });
        }
        if self.dry_multiplier > 0.0 {
            b = b.push(Dry {
                multiplier: self.dry_multiplier,
                base: self.dry_base,
                allowed_length: self.dry_allowed_length,
                max_ngram: self.dry_max_ngram,
                sequence_breakers: self.dry_sequence_breakers.clone(),
            });
        }

        // 2. Temperature (dynamic or static). Mirostat replaces both.
        if self.mirostat == MirostatMode::Off {
            if let Some((mn, mx)) = self.dynamic_temp {
                b = b.push(DynamicTemperature {
                    min: mn,
                    max: mx,
                    exponent: self.dynamic_temp_exponent,
                });
            } else if self.temperature > 0.0 && (self.temperature - 1.0).abs() > f32::EPSILON {
                b = b.push(Temperature {
                    t: self.temperature,
                });
            } else if self.temperature <= 0.0 {
                b = b.push(Temperature { t: 1e-6 });
            }
        }

        // 3. Filters: top-k → typical → top-p → top-n-sigma → xtc.
        if let Some(k) = self.top_k {
            if k > 0 {
                b = b.push(TopK { k: k as usize });
            }
        }
        if self.typical_p < 1.0 && self.typical_p > 0.0 {
            b = b.push(TypicalP {
                p: self.typical_p,
                min_keep: self.min_keep,
            });
        }
        if self.top_p < 1.0 && self.top_p > 0.0 {
            b = b.push(TopP {
                p: self.top_p,
                min_keep: self.min_keep,
            });
        }
        if self.min_p > 0.0 {
            b = b.push(MinP {
                p: self.min_p,
                min_keep: self.min_keep,
            });
        }
        if self.top_n_sigma > 0.0 {
            b = b.push(TopNSigma {
                n: self.top_n_sigma,
            });
        }
        if self.xtc_prob > 0.0 && self.xtc_threshold > 0.0 {
            b = b.push(Xtc {
                threshold: self.xtc_threshold,
                prob: self.xtc_prob,
                min_keep: self.min_keep,
            });
        }

        // 4. Mirostat (replaces softmax+sample at the end of the chain).
        match self.mirostat {
            MirostatMode::Off => {}
            MirostatMode::V1 => {
                b = b.push(MirostatV1 {
                    tau: self.mirostat_tau,
                    eta: self.mirostat_eta,
                    m: self.mirostat_m,
                });
            }
            MirostatMode::V2 => {
                b = b.push(MirostatV2 {
                    tau: self.mirostat_tau,
                    eta: self.mirostat_eta,
                });
            }
        }
        b.build()
    }
}

// ─────────────────────────────────────────────────────────────────
// Shared builder
// ─────────────────────────────────────────────────────────────────

/// Auto-packed threshold: prefer K-quant packed loading for GGUF
/// files >= this size. Cuts host memory ~6× on Q4_K_M models.
pub const PACKED_GGUF_AUTO_THRESHOLD_BYTES: u64 = 256 * 1024 * 1024;

/// Builder fields common to every per-family runner.
///
/// Per-family runner builders should wrap this and forward the
/// methods (or use `#[rlx_runner]` from `rlx-macros`).
#[derive(Debug, Clone)]
pub struct LmRunnerBuilder<Cfg> {
    pub weights: Option<PathBuf>,
    pub config: ConfigSource<Cfg>,
    pub device: Device,
    pub max_seq: usize,
    pub max_memory_gb: Option<f32>,
    pub stream: bool,
    pub sample: SampleOpts,
    pub format: Option<WeightFormat>,
    /// `None` = auto-detect (packed when GGUF ≥ 256 MB).
    pub packed_weights: Option<bool>,
    /// Substring for picking one GGUF in a directory (default `Q4_K_M`).
    pub prefer_gguf: Option<String>,
}

impl<Cfg> Default for LmRunnerBuilder<Cfg> {
    fn default() -> Self {
        Self {
            weights: None,
            config: ConfigSource::Embedded,
            device: Device::Cpu,
            max_seq: 128,
            max_memory_gb: None,
            stream: true,
            sample: SampleOpts::greedy(),
            format: None,
            packed_weights: None,
            prefer_gguf: None,
        }
    }
}

impl<Cfg> LmRunnerBuilder<Cfg> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn weights<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.weights = Some(p.into());
        self
    }

    pub fn config(mut self, src: ConfigSource<Cfg>) -> Self {
        self.config = src;
        self
    }

    pub fn config_value(self, cfg: Cfg) -> Self {
        self.config(ConfigSource::Explicit(cfg))
    }

    pub fn device(mut self, d: Device) -> Self {
        self.device = d;
        self
    }

    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = n;
        self
    }

    pub fn max_memory_gb(mut self, gb: f32) -> Self {
        self.max_memory_gb = Some(gb);
        self
    }

    pub fn stream(mut self, on: bool) -> Self {
        self.stream = on;
        self
    }

    pub fn sample(mut self, s: SampleOpts) -> Self {
        self.sample = s;
        self
    }

    pub fn format(mut self, fmt: WeightFormat) -> Self {
        self.format = Some(fmt);
        self
    }

    pub fn packed_weights(mut self, on: bool) -> Self {
        self.packed_weights = Some(on);
        self
    }

    pub fn prefer_gguf<S: Into<String>>(mut self, q: S) -> Self {
        self.prefer_gguf = Some(q.into());
        self
    }

    /// Resolve the format using the explicit override or the file extension.
    pub fn resolved_format(&self) -> anyhow::Result<WeightFormat> {
        match self.format {
            Some(f) => Ok(f),
            None => {
                let p = self
                    .weights
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("weights path required"))?;
                WeightFormat::from_path(p)
            }
        }
    }

    /// Determine whether packed GGUF loading should be used. Honors an
    /// explicit override; otherwise auto-enables for GGUF files at or
    /// above [`PACKED_GGUF_AUTO_THRESHOLD_BYTES`].
    pub fn resolved_packed(&self, fmt: WeightFormat) -> bool {
        match self.packed_weights {
            Some(b) => b,
            None => {
                if !matches!(fmt, WeightFormat::Gguf) {
                    return false;
                }
                self.weights
                    .as_deref()
                    .and_then(|p| std::fs::metadata(p).ok())
                    .map(|m| m.len() >= PACKED_GGUF_AUTO_THRESHOLD_BYTES)
                    .unwrap_or(false)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Model registry (auto-dispatch by path)
// ─────────────────────────────────────────────────────────────────

/// Family-routing entry: a short name + a probe closure that returns
/// `true` for files this family should handle.
///
/// Registered at process start by `register_model` (or by a
/// `#[rlx_runner]`-generated `inventory` entry). [`auto_runner_name`]
/// walks the registry and returns the first matching family.
pub struct ModelRegistration {
    pub family: &'static str,
    pub description: &'static str,
    /// `(arch_str_lower_case, path) -> bool`. `arch_str_lower_case` is
    /// the GGUF `general.architecture` (`""` for safetensors); `path`
    /// is the concrete weights file. Implementations should return
    /// `true` if the family owns this file.
    pub matches: fn(arch: &str, path: &Path) -> bool,
}

inventory::collect!(ModelRegistration);

/// Re-export of `inventory` so the `register_lm_runner!` proc-macro
/// can call `::rlx_runtime::lm::inventory::submit!` without forcing
/// every caller to add `inventory` to their Cargo.toml.
pub extern crate inventory;

/// Iterate over every registered family.
pub fn registered_models() -> impl Iterator<Item = &'static ModelRegistration> {
    inventory::iter::<ModelRegistration>.into_iter()
}

/// Find the family that claims `(arch, path)`.
pub fn auto_runner_name(arch: &str, path: &Path) -> Option<&'static str> {
    let arch_lc = arch.to_ascii_lowercase();
    registered_models()
        .find(|m| (m.matches)(&arch_lc, path))
        .map(|m| m.family)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Greedy takes the likelier first token into a flat continuation; two
    /// beams find the pair that is likelier as a whole.
    #[test]
    fn beam_search_finds_the_likelier_sequence() {
        let ln = |p: &[f32]| p.iter().map(|v| v.ln()).collect::<Vec<f32>>();
        // Tokens 0 and 1, 2 stops. After 0 nothing is likely; after 1, 0 is.
        let first = ln(&[0.6, 0.4, 1e-6]);
        let after = |t: u32| {
            if t == 0 {
                ln(&[0.34, 0.33, 0.33])
            } else {
                ln(&[0.95, 0.04, 0.01])
            }
        };
        let run = |beams: usize| {
            let mut hyps: Vec<Vec<u32>> = vec![vec![]];
            beam_search(&first, beams, 2, &[2], 1.0, |parents, tokens| {
                hyps = parents
                    .iter()
                    .zip(tokens)
                    .map(|(&p, &t)| {
                        let mut h = hyps[p].clone();
                        h.push(t);
                        h
                    })
                    .collect();
                Ok(hyps.iter().map(|h| after(*h.last().unwrap())).collect())
            })
            .unwrap()
        };
        assert_eq!(run(1), vec![0, 0], "one beam is greedy");
        assert_eq!(run(2), vec![1, 0]);
    }

    #[test]
    fn config_source_default_is_embedded() {
        let s: ConfigSource<()> = ConfigSource::default();
        assert!(matches!(s, ConfigSource::Embedded));
    }

    #[test]
    fn builder_defaults_match_legacy_runners() {
        let b: LmRunnerBuilder<()> = LmRunnerBuilder::new();
        assert_eq!(b.device, Device::Cpu);
        assert_eq!(b.max_seq, 128);
        assert!(b.stream);
        assert!(b.sample.is_greedy());
        assert!(b.packed_weights.is_none());
    }

    #[test]
    fn packed_auto_size_threshold() {
        let mut b: LmRunnerBuilder<()> = LmRunnerBuilder::new();
        b.weights = Some("/nonexistent/file.gguf".into());
        // Missing file → auto returns false (no metadata).
        assert!(!b.resolved_packed(WeightFormat::Gguf));
        // Explicit override wins.
        b.packed_weights = Some(true);
        assert!(b.resolved_packed(WeightFormat::Gguf));
        // Non-GGUF never auto-packs.
        b.packed_weights = None;
        assert!(!b.resolved_packed(WeightFormat::Safetensors));
    }
}

#[cfg(test)]
mod host_driven_tests {
    use super::*;

    /// A runner with no host-driven pair, standing in for a family that has
    /// not wired one up. Its logits are a deterministic function of the
    /// context, so greedy decoding from the same prompt must give the same
    /// tokens however it is driven.
    struct NoPair {
        vocab: usize,
        prefills: usize,
    }

    impl LmRunner for NoPair {
        fn family(&self) -> &'static str {
            "nopair"
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn predict_logits(&mut self, prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
            self.prefills += 1;
            // Peaks at (sum of context + len) % vocab, so the argmax moves
            // with the context rather than being constant.
            let pick = (prompt_ids.iter().map(|&t| t as usize).sum::<usize>() + prompt_ids.len())
                % self.vocab;
            let mut l = vec![0.0f32; self.vocab];
            l[pick] = 1.0;
            Ok(l)
        }
    }

    /// A runner that *does* have the pair, to check the wrapper gets out of
    /// the way rather than adding its slow path on top.
    struct HasPair {
        vocab: usize,
        context: Vec<u32>,
        prefills: usize,
    }

    impl LmRunner for HasPair {
        fn family(&self) -> &'static str {
            "haspair"
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn predict_logits(&mut self, prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
            self.prefills += 1;
            let pick = (prompt_ids.iter().map(|&t| t as usize).sum::<usize>() + prompt_ids.len())
                % self.vocab;
            let mut l = vec![0.0f32; self.vocab];
            l[pick] = 1.0;
            Ok(l)
        }
        fn prefill_logits(&mut self, prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
            self.context = prompt_ids.to_vec();
            self.predict_logits(prompt_ids)
        }
        fn decode_logits(&mut self, token: u32) -> anyhow::Result<Vec<f32>> {
            self.context.push(token);
            let c = std::mem::take(&mut self.context);
            let out = self.predict_logits(&c);
            self.context = c;
            out
        }
    }

    fn host_driven_tokens<R: LmRunner>(r: &mut R, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut logits = LmRunner::prefill_logits(r, prompt).expect("prefill");
        let mut out = Vec::new();
        for _ in 0..n {
            let t = argmax_u32(&logits);
            out.push(t);
            logits = LmRunner::decode_logits(r, t).expect("decode");
        }
        out
    }

    #[test]
    fn a_runner_without_the_pair_cannot_be_driven_by_a_server() {
        // The state of affairs this wrapper exists for, pinned so the error
        // keeps saying what to do about it.
        let mut bare = NoPair {
            vocab: 16,
            prefills: 0,
        };
        let err = LmRunner::prefill_logits(&mut bare, &[1, 2])
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported"), "{err}");
        assert!(
            err.contains("HostDriven"),
            "the error should name the fix: {err}"
        );
    }

    #[test]
    fn wrapping_it_gives_the_same_tokens_as_its_own_generate() {
        // The cross-check that matters: the host-driven loop and the
        // runner's own greedy loop are two ways of doing one thing, and a
        // wrapper that quietly dropped or double-counted a token would
        // still look plausible. Same prompt, same decoding, same tokens.
        let prompt = [3u32, 5, 7];
        let mut own = NoPair {
            vocab: 16,
            prefills: 0,
        };
        let want = own
            .generate(&prompt, 6, &mut |_| true)
            .expect("the default greedy loop");

        let mut wrapped = HostDriven::new(NoPair {
            vocab: 16,
            prefills: 0,
        });
        let got = host_driven_tokens(&mut wrapped, &prompt, 6);
        assert_eq!(got, want);
        assert!(
            wrapped.fell_back(),
            "NoPair has no cached pair, so this is the slow path"
        );
    }

    #[test]
    fn wrapping_a_runner_that_has_the_pair_uses_it() {
        // The wrapper must not impose its fallback on a family that already
        // has a fast path — the cost of being wrapped should be nothing.
        let mut wrapped = HostDriven::new(HasPair {
            vocab: 16,
            context: Vec::new(),
            prefills: 0,
        });
        let got = host_driven_tokens(&mut wrapped, &[3, 5, 7], 6);
        assert_eq!(got.len(), 6);
        assert!(
            !wrapped.fell_back(),
            "the family's own pair should have been used"
        );
    }

    #[test]
    fn decoding_before_prefilling_says_so() {
        // Rather than returning logits for an empty context, which reads as
        // a working model answering nonsense.
        let mut wrapped = HostDriven::new(NoPair {
            vocab: 16,
            prefills: 0,
        });
        let err = LmRunner::decode_logits(&mut wrapped, 4)
            .unwrap_err()
            .to_string();
        assert!(err.contains("before prefill"), "{err}");
    }

    #[test]
    fn the_context_grows_by_exactly_one_token_per_step() {
        // The failure this guards is an off-by-one in the fallback: a
        // context that misses a token still decodes, and the only symptom
        // is a worse answer.
        let mut wrapped = HostDriven::new(NoPair {
            vocab: 1024,
            prefills: 0,
        });
        let prompt = [2u32, 4];
        let mut logits = LmRunner::prefill_logits(&mut wrapped, &prompt).unwrap();
        let mut expect: Vec<u32> = prompt.to_vec();
        for _ in 0..5 {
            let t = argmax_u32(&logits);
            expect.push(t);
            logits = LmRunner::decode_logits(&mut wrapped, t).unwrap();
            // The stand-in's argmax is a function of the whole context, so
            // reproducing it independently proves the wrapper fed the same
            // context.
            let want = (expect.iter().map(|&t| t as usize).sum::<usize>() + expect.len()) % 1024;
            assert_eq!(argmax_u32(&logits) as usize, want, "context: {expect:?}");
        }
    }
}
