// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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
}

impl WeightFormat {
    /// Infer format from a path extension.
    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("safetensors") => Ok(Self::Safetensors),
            Some("gguf") => Ok(Self::Gguf),
            other => Err(anyhow::anyhow!(
                "cannot autodetect weight format from extension {:?} on {:?}",
                other,
                path
            )),
        }
    }

    /// Parse CLI `--format` values (`safetensors` | `gguf`).
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s {
            "safetensors" => Ok(Self::Safetensors),
            "gguf" => Ok(Self::Gguf),
            other => Err(anyhow::anyhow!("expected safetensors|gguf, got {other}")),
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

/// Sampling parameters. Greedy when `temperature == 0`.
#[derive(Debug, Clone, Copy)]
pub struct SampleOpts {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<u32>,
    pub repetition_penalty: f32,
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
        }
    }

    pub fn nucleus(temperature: f32, top_p: f32) -> Self {
        Self {
            temperature,
            top_p,
            top_k: None,
            repetition_penalty: 1.0,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
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
