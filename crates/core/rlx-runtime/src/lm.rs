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

    /// Prefill `prompt_ids`, seed the decode KV cache, and return the
    /// last-position logits `[vocab]`. Together with [`decode_logits`] this
    /// gives a **host-driven** decode loop: the caller owns sampling, logit
    /// bias, log-probs, and stop detection (used by the HTTP server). The
    /// default reports unsupported so existing runners keep compiling.
    fn prefill_logits(&mut self, _prompt_ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        Err(anyhow::anyhow!(
            "{}: host-driven logits decode (prefill_logits) unsupported",
            self.family()
        ))
    }

    /// Feed one token, advance the KV cache, and return the next-position
    /// logits `[vocab]`. See [`LmRunner::prefill_logits`]. Default: unsupported.
    fn decode_logits(&mut self, _token: u32) -> anyhow::Result<Vec<f32>> {
        Err(anyhow::anyhow!(
            "{}: host-driven logits decode (decode_logits) unsupported",
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

fn argmax_u32(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
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
