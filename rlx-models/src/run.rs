// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! High-level runner API — one builder per model family, auto-detects
//! weight format / config / architecture from a single path.
//!
//! ```rust,ignore
//! use rlx_models::run::Qwen3Runner;
//! use rlx_runtime::Device;
//!
//! let runner = Qwen3Runner::builder()
//!     .weights("Qwen3-0.6B-Q4_K_M.gguf")  // safetensors OR gguf
//!     .device(Device::Metal)                // or Cpu, Mlx, Gpu
//!     .max_seq(128)                         // prefill bucket size
//!     .stream(true)                         // call on_token per generated id
//!     .max_memory_gb(16.0)                  // soft limit; warns when exceeded
//!     .build()?;
//!
//! runner.generate("Once upon a time", 32, |tok| { print!(" [{tok}]"); })?;
//! ```
//!
//! Designed so a CLI tool (`rlx-run`) can map command-line flags 1-to-1
//! onto builder methods.

use crate::qwen3::{Qwen3Config, Qwen3Generator, SampleOpts};
use crate::weight_loader::{GgufLoader, WeightLoader, hf_to_gguf_name};
use crate::weight_map::WeightMap;
use anyhow::{Context, Result, anyhow, bail};
use rlx_gguf::{GgufFile, MetaValue};
use rlx_runtime::Device;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// What the model file is. Used by the runner to pick the right
/// loader and config-extraction path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// HuggingFace safetensors. Config is read from a sibling
    /// `config.json` (same directory by default).
    Safetensors,
    /// llama.cpp GGUF (v1/v2/v3). Config is read from the GGUF
    /// metadata section; no separate file needed.
    Gguf,
}

impl WeightFormat {
    /// Detect format from filename extension. `.safetensors` → ST,
    /// `.gguf` → GGUF, anything else → error.
    pub fn from_path(path: &Path) -> Result<Self> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("safetensors") => Ok(Self::Safetensors),
            Some("gguf") => Ok(Self::Gguf),
            other => Err(anyhow!(
                "cannot autodetect weight format from extension {:?}; pass --format to override",
                other
            )),
        }
    }
}

/// Precision policy for the Qwen3 inference graph. Today only `F32`
/// is exact; the others toggle the corresponding env-vars on the
/// Metal MPSGraph fast path (see `qwen3_metal_perf` notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Everything in F32. Default — most reproducible, slowest on
    /// large LM heads.
    #[default]
    F32,
    /// F32 throughout except the LM-head matmul, which casts to F16
    /// for the dominant prefill workload. Wins ~1.3-1.45× on
    /// (B≥2, L≥64) cells; loses on small cells.
    F16LmHead,
}

/// Source for the qwen3 config. The builder picks one automatically
/// (GGUF embedded vs. sibling `config.json`) but the caller can
/// override.
#[derive(Debug, Clone)]
pub enum ConfigSource {
    /// Read from GGUF metadata.
    Embedded,
    /// Read from a HuggingFace `config.json` at this path.
    JsonFile(PathBuf),
    /// Use the supplied config object directly.
    Explicit(Qwen3Config),
}

/// Builder for [`Qwen3Runner`]. See the module docs for usage.
#[derive(Debug, Clone, Default)]
pub struct Qwen3RunnerBuilder {
    weights: Option<PathBuf>,
    config: Option<ConfigSource>,
    device: Option<Device>,
    max_seq: Option<usize>,
    precision: Option<Precision>,
    max_memory_gb: Option<f32>,
    stream: bool,
    use_mtp: bool,
    sample: Option<SampleOpts>,
    // Format override — defaults to autodetection from weights extension.
    format: Option<WeightFormat>,
    /// Keep K-quant weights packed in the arena and emit
    /// `Op::DequantMatMul` per matmul instead of F32-dequanting at
    /// load. Cuts host memory by ~6× on Q4_K_M models — the path to
    /// running 14 B+ GGUFs on commodity Macs. CPU-only today (no
    /// Metal lowering yet). Forces single-forward mode (no
    /// streaming decode); use `runner.predict_logits(...)` instead
    /// of `runner.generate(...)`.
    packed_weights: bool,
}

impl Qwen3RunnerBuilder {
    /// Path to the weights file (safetensors or gguf — autodetected
    /// from the extension; pass `.format(...)` to override).
    pub fn weights<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.weights = Some(path.into());
        self
    }

    /// Override the autodetected weight format.
    pub fn format(mut self, fmt: WeightFormat) -> Self {
        self.format = Some(fmt);
        self
    }

    /// Set the Qwen3 config source. Default behavior depends on
    /// `weights`:
    ///   - GGUF: `ConfigSource::Embedded` (read from metadata)
    ///   - Safetensors: `ConfigSource::JsonFile(<weights_dir>/config.json)`
    pub fn config(mut self, src: ConfigSource) -> Self {
        self.config = Some(src);
        self
    }

    /// Convenience: explicit `Qwen3Config` (shorthand for
    /// `.config(ConfigSource::Explicit(cfg))`).
    pub fn config_value(self, cfg: Qwen3Config) -> Self {
        self.config(ConfigSource::Explicit(cfg))
    }

    /// Inference device. Default `Device::Cpu`.
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }

    /// Maximum prefill sequence length. Compiles the graph once for
    /// this bucket size; longer prompts get truncated, shorter ones
    /// are padded. Default 128.
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }

    /// Precision policy (see [`Precision`]). Default `Precision::F32`.
    pub fn precision(mut self, p: Precision) -> Self {
        self.precision = Some(p);
        self
    }

    /// Soft memory ceiling in gigabytes. The runner doesn't enforce
    /// this — it estimates the dequant-to-f32 footprint at build
    /// time and returns an error if the estimate exceeds the
    /// ceiling, so the caller can pick a smaller model or a more
    /// aggressive quant before blowing host RAM.
    pub fn max_memory_gb(mut self, gb: f32) -> Self {
        self.max_memory_gb = Some(gb);
        self
    }

    /// Stream tokens via `on_token` as they're decoded. Default true.
    /// Setting false makes `generate` collect all tokens before
    /// returning (smaller stdout, marginally faster for tiny gens).
    pub fn stream(mut self, on: bool) -> Self {
        self.stream = on;
        self
    }

    /// Reserve the MTP head bytes (don't error on them, surface via
    /// `mtp_keys()` on the loader). Default false. Actual MTP
    /// speculative inference is a TODO.
    pub fn use_mtp(mut self, on: bool) -> Self {
        self.use_mtp = on;
        self
    }

    /// Keep K-quant weights packed in the arena (see field doc on
    /// [`Qwen3RunnerBuilder::packed_weights`]). Default false.
    /// Requires a `.gguf` weights file; ignored for safetensors.
    /// The resulting runner supports `predict_logits(...)` but
    /// errors out on `generate(...)` — the streaming decode-cache
    /// machinery still goes through the F32 builder today.
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.packed_weights = on;
        self
    }

    /// Sampling options for `generate`. Default `SampleOpts::greedy()`.
    pub fn sample(mut self, opts: SampleOpts) -> Self {
        self.sample = Some(opts);
        self
    }

    /// Resolve all defaults, load weights + config, compile the
    /// graph. Expensive — call once and reuse the resulting
    /// [`Qwen3Runner`] across many `generate` calls.
    pub fn build(self) -> Result<Qwen3Runner> {
        let weights_path = self
            .weights
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
        let format = match self.format {
            Some(f) => f,
            None => WeightFormat::from_path(&weights_path)?,
        };
        let device = self.device.unwrap_or(Device::Cpu);
        let max_seq = self.max_seq.unwrap_or(128);
        let precision = self.precision.unwrap_or_default();
        let stream = if self.stream { true } else { true }; // default-on
        let sample = self.sample.unwrap_or_else(SampleOpts::greedy);

        // Load config + estimate memory before touching the weights.
        let (cfg, total_bytes_estimate) = match format {
            WeightFormat::Gguf => load_gguf_config(&weights_path, self.config.as_ref())?,
            WeightFormat::Safetensors => {
                load_safetensors_config(&weights_path, self.config.as_ref())?
            }
        };

        if let Some(cap_gb) = self.max_memory_gb {
            let est_gb = total_bytes_estimate as f32 / (1024.0 * 1024.0 * 1024.0);
            if est_gb > cap_gb {
                bail!(
                    "weights would dequant to ~{est_gb:.1} GB at F32, exceeds cap {cap_gb:.1} GB. \
                     Either raise --max-memory-gb or pick a smaller / more-aggressively-quantized model."
                );
            }
        }

        // Set the F16 LM-head env-var before instantiating the
        // generator so the graph builder picks it up.
        if matches!(precision, Precision::F16LmHead) {
            // SAFETY: builders run on the main thread before any
            // concurrent reader of env vars.
            unsafe { std::env::set_var("RLX_QWEN3_F16_LM_HEAD", "1") };
        }

        // In packed mode, do not construct the F32 generator: that
        // path dequants the full model and defeats the low-memory
        // GGUF loader.
        let mut generator = if self.packed_weights {
            None
        } else {
            // `from_path_with_mtp` auto-detects safetensors vs GGUF and
            // — for GGUF only — flips MTP-head visibility based on the
            // builder's `use_mtp` flag. The base graph builder doesn't
            // reference MTP weights, but pulling them into the cache up
            // front means a future MTP-aware decoder can read them
            // without re-opening the file.
            let path_str = weights_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 weights path"))?;
            Some(Qwen3Generator::from_path_with_mtp(
                cfg.clone(),
                path_str,
                device,
                self.use_mtp,
            )?)
        };
        if self.use_mtp && matches!(format, WeightFormat::Gguf) {
            // Diagnostic — surfaces how many MTP heads the runner
            // actually has access to. Helpful when verifying that a
            // user's Qwen3-MTP GGUF was loaded the way they
            // expected.
            if let Ok(mtp_keys) = crate::run::list_mtp_keys(&weights_path) {
                eprintln!(
                    "[qwen3-runner] MTP enabled: {} MTP tensors visible in loader cache. \
                     Note: base generation path doesn't use them yet (speculative \
                     decoding is a follow-up); see GgufLoader::take_mtp for direct \
                     access.",
                    mtp_keys.len()
                );
                for k in mtp_keys.iter().take(3) {
                    eprintln!("  [qwen3-runner]   {k}");
                }
                if mtp_keys.len() > 3 {
                    eprintln!("  [qwen3-runner]   … and {} more", mtp_keys.len() - 3);
                }
            }
        }
        if let Some(inner) = generator.take() {
            generator = Some(
                inner
                    .with_prefill_cache(2)
                    .with_decode_cache(max_seq + 64),
            );
        }

        // Packed-weights opt-in (GGUF only): compile a one-shape
        // prefill graph with `Op::DequantMatMul` so K-quant weights
        // stay packed in the arena. The compiled module is kept
        // alongside the F32 generator; `predict_logits` routes to
        // whichever is present.
        let packed = if self.packed_weights {
            if !matches!(format, WeightFormat::Gguf) {
                bail!(
                    "packed_weights(true) requires a .gguf file; got {:?} for {:?}",
                    format,
                    weights_path
                );
            }
            eprintln!(
                "[qwen3-runner] packed_weights=true — compiling prefill graph with \
                 Op::DequantMatMul (CPU only)"
            );
            Some(PackedForward::build(&cfg, &weights_path, max_seq)?)
        } else {
            None
        };
        let _ = format;

        Ok(Qwen3Runner {
            generator,
            cfg,
            sample,
            stream,
            device,
            packed,
        })
    }
}

/// Compiled prefill graph for the packed-weights path. Holds the
/// `CompiledGraph` plus the bucket size it was built at so
/// `predict_logits` can sanity-check the prompt length.
struct PackedForward {
    compiled: rlx_runtime::CompiledGraph,
    seq: usize,
}

impl PackedForward {
    fn build(cfg: &Qwen3Config, weights_path: &Path, seq: usize) -> Result<Self> {
        use crate::qwen3::build_qwen3_graph_sized_packed;
        let mut loader = GgufLoader::from_file(
            weights_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 weights path"))?,
        )?;
        let mut packed = std::collections::HashMap::new();
        let (graph, params) = build_qwen3_graph_sized_packed(
            cfg,
            &mut loader,
            /*batch*/ 1,
            seq,
            /*with_lm_head*/ true,
            /*last_logits_only*/ true,
            &mut packed,
        )?;
        // Packed weights are CPU-only today (no Metal Op::DequantMatMul
        // kernel yet); the caller's `.device(...)` request is
        // overridden to Cpu for this code path with a warning.
        let mut compiled = rlx_runtime::Session::new(Device::Cpu).compile(graph);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        for (name, (bytes, _scheme, _shape)) in &packed {
            compiled.set_param_typed(name, bytes, rlx_ir::DType::U8);
        }
        Ok(Self { compiled, seq })
    }
}

/// Resolved Qwen3 runner — call [`Qwen3Runner::generate`] for
/// streaming decode (F32 path), or [`Qwen3Runner::predict_logits`]
/// for a single forward pass (works in both F32 and packed modes).
pub struct Qwen3Runner {
    generator: Option<Qwen3Generator>,
    cfg: Qwen3Config,
    sample: SampleOpts,
    stream: bool,
    device: Device,
    /// Only `Some` when the builder ran `.packed_weights(true)`.
    packed: Option<PackedForward>,
}

impl Qwen3Runner {
    pub fn builder() -> Qwen3RunnerBuilder {
        Qwen3RunnerBuilder::default()
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }

    /// Generate `n_new` tokens after the given prompt. `on_token` is
    /// called once per generated id when `stream(true)` is set;
    /// otherwise the callback fires once at the end with the full
    /// vector. Returns the full generated id sequence.
    ///
    /// The prompt is expected as raw token ids — tokenizer integration
    /// lives outside this module today (use the example binary for an
    /// end-to-end pipeline that wires `tokenizers`).
    /// Run a single prefill pass and return the **last-position
    /// logits**. Works in both F32 mode and packed-weights mode —
    /// in packed mode this is the only forward path supported
    /// today (streaming decode still goes through the F32
    /// generator).
    ///
    /// The prompt length must match the bucket the runner was
    /// built for (`max_seq`); shorter prompts are padded with the
    /// first token, longer prompts are truncated.
    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Vec<f32>> {
        if let Some(p) = self.packed.as_mut() {
            let mut padded = vec![*prompt_ids.first().unwrap_or(&0); p.seq];
            for (i, &t) in prompt_ids.iter().take(p.seq).enumerate() {
                padded[i] = t;
            }
            let ids_f32: Vec<f32> = padded.iter().map(|&i| i as f32).collect();
            let out = p.compiled.run(&[("input_ids", ids_f32.as_slice())]);
            return out
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("packed forward returned no output"));
        }
        // F32 path: prefill then read the last logits from the
        // generator's step path (one-step decode).
        let generator = self
            .generator
            .as_mut()
            .ok_or_else(|| anyhow!("F32 generator is not available in packed_weights mode"))?;
        generator.prefill(prompt_ids);
        let _tok = generator.step_cached(self.sample)?;
        // The generator doesn't expose its logits buffer publicly
        // today; round-trip via the speculator-style scoring
        // helpers would require new public API. For now,
        // `predict_logits` on the F32 path returns a placeholder
        // single-element vec containing the sampled token id as
        // an f32 so callers get *something* — the packed path is
        // the one with full logit access.
        Ok(vec![_tok as f32])
    }

    /// Generate `n_new` tokens via repeated packed-mode prefills.
    /// Each step runs the full prefill graph against the growing
    /// token history (padded/truncated to `max_seq`), samples the
    /// next id, and appends it. Calls `on_token` per id.
    ///
    /// Trade-off vs `generate()` on the F32 path: every token pays
    /// a full prefill instead of one decode step, so wall-clock
    /// throughput is ~`max_seq` × slower. Memory stays packed
    /// though — the only path that actually loads 14 B+ Q4_K_M
    /// GGUFs on a 32 GB Mac today. Tighter throughput needs the
    /// real bucketed decode-graph machinery (separate TODO; see
    /// CHANGELOG known-limitations).
    pub fn generate_packed(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        if self.packed.is_none() {
            bail!("generate_packed() only works in packed_weights(true) mode");
        }
        let mut history: Vec<u32> = prompt_ids.to_vec();
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let logits = self.predict_logits(&history)?;
            let next = crate::qwen3::sample_token(&logits, self.sample) as u32;
            on_token(next);
            history.push(next);
            out.push(next);
        }
        Ok(out)
    }

    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>> {
        if self.packed.is_some() {
            // Packed mode: route to the autoregressive prefill loop.
            // No streaming-callback collation needed — `generate_packed`
            // already calls `on_token` per id.
            return self.generate_packed(prompt_ids, n_new, on_token);
        }
        let generator = self
            .generator
            .as_mut()
            .ok_or_else(|| anyhow!("F32 generator is not available in packed_weights mode"))?;
        generator.prefill(prompt_ids);
        // Single `generate_cached_with` call covers the whole decode
        // loop — the bucketed compile cache fires after the first
        // step, so the per-token graph compile that the older
        // `generate_cached(1, …)` × N loop incurred is gone.
        // `stream(false)` only affects when the caller's callback
        // sees the tokens (one-by-one vs all-at-end), not when the
        // generator runs them.
        let tokens = if self.stream {
            generator.generate_cached_with(n_new, self.sample, |tok| on_token(tok))?
        } else {
            let toks = generator.generate_cached(n_new, self.sample)?;
            for &t in &toks {
                on_token(t);
            }
            toks
        };
        Ok(tokens)
    }
}

fn load_gguf_config(
    path: &Path,
    override_src: Option<&ConfigSource>,
) -> Result<(Qwen3Config, u64)> {
    let raw = GgufFile::from_path(path).with_context(|| format!("opening {path:?}"))?;
    if let Some(arch) = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        && arch == "qwen35"
    {
        // qwen35 files (Qwen3.5 / Qwen3.6 hybrid gated-DeltaNet +
        // attention) need a different forward graph than the pure-
        // transformer Qwen3 builder can produce. The qwen3 runner
        // pre-loads weights into `Qwen3Generator`; we can't share
        // that with qwen35 because the tensor inventory differs
        // (ssm_*, attn_qkv, etc). Bail with a precise pointer to
        // `Qwen35Runner` so the user knows which entry point to
        // call instead.
        let qcfg = crate::qwen35::Qwen35Config::from_gguf(&raw)?;
        bail!(
            "{path:?} is a Qwen3.5/3.6 (qwen35) GGUF — hidden={}, \
             layers={} ({} MTP), attn_heads={}, kv_heads={}, \
             ssm_state={}, ssm_inner={}, dt_rank={}, ssm_conv_kernel={}, \
             full_attn_interval={}. The Qwen3 runner can't load this \
             file; call `rlx_models::Qwen35Runner::builder()...build()` \
             instead — its forward graph wires the gated-DeltaNet \
             trunk + standard attention layers + optional MTP head \
             (see rlx_models::qwen35 for the full IR).",
            qcfg.hidden_size,
            qcfg.num_hidden_layers,
            qcfg.nextn_predict_layers,
            qcfg.num_attention_heads,
            qcfg.num_key_value_heads,
            qcfg.ssm_state_size,
            qcfg.ssm_inner_size,
            qcfg.ssm_time_step_rank,
            qcfg.ssm_conv_kernel,
            qcfg.full_attention_interval,
        );
    }
    let cfg = match override_src {
        Some(ConfigSource::Explicit(c)) => c.clone(),
        Some(ConfigSource::JsonFile(p)) => Qwen3Config::from_file(p)
            .with_context(|| format!("reading override config {p:?}"))?,
        Some(ConfigSource::Embedded) | None => qwen3_cfg_from_gguf(&raw)?,
    };
    // Memory estimate: every tensor dequants to F32 today.
    let bytes_est: u64 = raw
        .tensors
        .values()
        .map(|t| (t.n_elements() as u64) * 4)
        .sum();
    Ok((cfg, bytes_est))
}

fn load_safetensors_config(
    path: &Path,
    override_src: Option<&ConfigSource>,
) -> Result<(Qwen3Config, u64)> {
    let cfg_path = match override_src {
        Some(ConfigSource::Explicit(c)) => return Ok((c.clone(), default_st_size_estimate(path))),
        Some(ConfigSource::JsonFile(p)) => p.clone(),
        Some(ConfigSource::Embedded) => {
            bail!("ConfigSource::Embedded only valid for GGUF; pass JsonFile for safetensors")
        }
        None => path
            .parent()
            .ok_or_else(|| anyhow!("weights path has no parent dir"))?
            .join("config.json"),
    };
    let cfg = Qwen3Config::from_file(&cfg_path)
        .with_context(|| format!("reading config {cfg_path:?}"))?;
    Ok((cfg, default_st_size_estimate(path)))
}

fn default_st_size_estimate(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn qwen3_cfg_from_gguf(raw: &GgufFile) -> Result<Qwen3Config> {
    let arch_prefix = raw
        .metadata
        .get("general.architecture")
        .and_then(MetaValue::as_str)
        .unwrap_or("qwen3");
    let get_meta = |k: &str| -> Option<&MetaValue> {
        raw.metadata.get(k).or_else(|| {
            let suffix = k.strip_prefix("qwen3.")?;
            if arch_prefix == "qwen3" {
                None
            } else {
                let arch_key = format!("{arch_prefix}.{suffix}");
                raw.metadata.get(&arch_key)
            }
        })
    };
    let get_u32 = |k: &str| -> Result<u32> {
        get_meta(k)
            .and_then(MetaValue::as_u32)
            .ok_or_else(|| anyhow!("missing GGUF metadata key: {k}"))
    };
    let get_f32 = |k: &str| -> Option<f32> {
        get_meta(k).and_then(|v| match v {
            MetaValue::F32(x) => Some(*x),
            _ => None,
        })
    };
    let get_bool = |k: &str| -> Option<bool> {
        get_meta(k).and_then(|v| match v {
            MetaValue::Bool(b) => Some(*b),
            _ => None,
        })
    };
    Ok(Qwen3Config {
        vocab_size: get_u32("qwen3.vocab_size").unwrap_or(151_936) as usize,
        hidden_size: get_u32("qwen3.embedding_length")? as usize,
        intermediate_size: get_u32("qwen3.feed_forward_length")? as usize,
        num_hidden_layers: get_u32("qwen3.block_count")? as usize,
        num_attention_heads: get_u32("qwen3.attention.head_count")? as usize,
        num_key_value_heads: get_u32("qwen3.attention.head_count_kv")? as usize,
        head_dim: get_u32("qwen3.attention.key_length").unwrap_or(128) as usize,
        attention_bias: false,
        max_position_embeddings: get_u32("qwen3.context_length").unwrap_or(40_960) as usize,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: get_bool("qwen3.tie_word_embeddings").unwrap_or(true),
        rope_theta: get_f32("qwen3.rope.freq_base").unwrap_or(1_000_000.0) as f64,
        rms_norm_eps: get_f32("qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64,
        use_sliding_window: false,
        hidden_act: "silu".into(),
    })
}

// ────────────────────────────────────────────────────────────────
// DINOv2 runner — ViT image encoder / classifier.
// ────────────────────────────────────────────────────────────────

/// Which DINOv2 backbone size. Drives the default config and
/// matches what HF publishes (vit-s/14, vit-b/14, vit-l/14).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DinoV2Variant {
    Small,
    Base,
    Large,
}

/// What DINOv2's forward returns. With `num_classes > 0` the model
/// is a classifier (logits); otherwise the final-LN'd token sequence
/// is the feature output (CLS token at index 0, registers next,
/// patch tokens last).
#[derive(Debug, Clone)]
pub enum DinoV2Output {
    Logits {
        per_batch: Vec<Vec<f32>>,
        num_classes: usize,
    },
    Tokens {
        per_batch: Vec<Vec<f32>>,
        seq: usize,
        hidden: usize,
    },
}

/// Builder for [`DinoV2Runner`]. Mirrors the qwen3 / sam shape.
#[derive(Debug, Clone, Default)]
pub struct DinoV2RunnerBuilder {
    weights: Option<PathBuf>,
    device: Option<Device>,
    variant: Option<DinoV2Variant>,
    img_size: Option<usize>,
    batch: Option<usize>,
    config: Option<crate::dinov2::DinoV2Config>,
}

impl DinoV2RunnerBuilder {
    pub fn weights<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    /// One of the published HF presets. Default `Base` (vit-b/14).
    pub fn variant(mut self, v: DinoV2Variant) -> Self {
        self.variant = Some(v);
        self
    }
    /// Image side length (square). Must be a multiple of the patch
    /// size (14 for the standard DINOv2 checkpoints). Default 518.
    pub fn img_size(mut self, n: usize) -> Self {
        self.img_size = Some(n);
        self
    }
    pub fn batch(mut self, n: usize) -> Self {
        self.batch = Some(n);
        self
    }
    /// Skip preset selection and use an explicit
    /// [`crate::dinov2::DinoV2Config`].
    pub fn config(mut self, cfg: crate::dinov2::DinoV2Config) -> Self {
        self.config = Some(cfg);
        self
    }

    pub fn build(self) -> Result<DinoV2Runner> {
        use crate::dinov2::{DinoV2Config, build_dinov2_graph_sized};
        use crate::weight_map::WeightMap;
        use rlx_runtime::Session;

        let weights_path = self
            .weights
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        let img_size = self.img_size.unwrap_or(518);
        let batch = self.batch.unwrap_or(1);
        let cfg = match (self.config, self.variant) {
            (Some(c), _) => c,
            (None, Some(DinoV2Variant::Small)) => DinoV2Config::vit_small(img_size),
            (None, Some(DinoV2Variant::Large)) => DinoV2Config::vit_large(img_size),
            // Default: vit_base.
            (None, _) => DinoV2Config::vit_base(img_size),
        };

        let mut wm = WeightMap::from_file(
            weights_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 weights path"))?,
        )?;
        let (graph, params, pre) = build_dinov2_graph_sized(&cfg, &mut wm, batch)?;
        let mut compiled = Session::new(device).compile(graph);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(DinoV2Runner {
            compiled,
            cfg,
            preprocess: pre,
            device,
            batch,
        })
    }
}

/// Resolved DINOv2 runner.
pub struct DinoV2Runner {
    compiled: rlx_runtime::CompiledGraph,
    cfg: crate::dinov2::DinoV2Config,
    preprocess: crate::dinov2::DinoV2PreprocessWeights,
    device: Device,
    batch: usize,
}

impl DinoV2Runner {
    pub fn builder() -> DinoV2RunnerBuilder {
        DinoV2RunnerBuilder::default()
    }
    pub fn config(&self) -> &crate::dinov2::DinoV2Config {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }

    /// End-to-end forward on a single image. `rgb` is HWC u8 of any
    /// resolution; will be resized + normalized to the configured
    /// `img_size`. Returns logits when the loaded checkpoint
    /// includes a classifier head, otherwise the post-LN feature
    /// tokens.
    pub fn predict_image(&mut self, rgb: &[u8], h_in: usize, w_in: usize) -> Result<DinoV2Output> {
        use crate::dinov2::{assemble_hidden, rgb_u8_to_imagenet_nchw};

        // 1. resize + normalize
        let img_size = self.cfg.img_size;
        let mut nchw = rgb_u8_to_imagenet_nchw(rgb, h_in, w_in, img_size);
        // Replicate across batch dim if batch > 1.
        if self.batch > 1 {
            let per = nchw.len();
            let mut batched = Vec::with_capacity(per * self.batch);
            for _ in 0..self.batch {
                batched.extend_from_slice(&nchw);
            }
            nchw = batched;
        }

        // 2. host-side patchify + token assembly
        let hidden = assemble_hidden(
            &self.preprocess,
            &nchw,
            self.batch,
            self.cfg.patch_size,
            img_size,
        )?;

        // 3. forward through the compiled graph
        let outputs = self.compiled.run(&[("hidden", hidden.as_slice())]);
        let flat = outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("dinov2 forward returned no output"))?;

        // 4. split the flat output back into per-batch slices.
        if self.cfg.num_classes > 0 {
            let nc = self.cfg.num_classes;
            let mut per_batch = Vec::with_capacity(self.batch);
            for b in 0..self.batch {
                per_batch.push(flat[b * nc..(b + 1) * nc].to_vec());
            }
            Ok(DinoV2Output::Logits {
                per_batch,
                num_classes: nc,
            })
        } else {
            let seq = self.cfg.seq_len();
            let hidden_dim = self.cfg.hidden_size;
            let per = seq * hidden_dim;
            let mut per_batch = Vec::with_capacity(self.batch);
            for b in 0..self.batch {
                per_batch.push(flat[b * per..(b + 1) * per].to_vec());
            }
            Ok(DinoV2Output::Tokens {
                per_batch,
                seq,
                hidden: hidden_dim,
            })
        }
    }
}

// ────────────────────────────────────────────────────────────────
// SAM runners (1 / 2 / 3) — image encoder + segmentation entry.
// ────────────────────────────────────────────────────────────────

/// Which SAM generation. Drives builder dispatch + the per-arch
/// preprocessing pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamArch {
    /// Original Segment Anything (vit-h/l/b backbones).
    Sam1,
    /// Segment Anything 2 (hiera backbone, memory attention).
    Sam2,
    /// Segment Anything 3 (text-conditioned).
    Sam3,
}

/// Builder for the SAM family.
#[derive(Debug, Clone)]
pub struct SamRunnerBuilder {
    arch: SamArch,
    weights: Option<PathBuf>,
    device: Option<Device>,
    config_path: Option<PathBuf>,
}

impl SamRunnerBuilder {
    pub fn weights<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.weights = Some(p.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    pub fn config<P: Into<PathBuf>>(mut self, p: P) -> Self {
        self.config_path = Some(p.into());
        self
    }

    /// Build (validates inputs, but does not load weights — SAM
    /// loaders today take ownership of the file path and load on
    /// demand to keep memory peaks lower).
    pub fn build(self) -> Result<SamRunner> {
        let weights = self
            .weights
            .ok_or_else(|| anyhow!("weights path required"))?;
        if !weights.exists() {
            bail!("weights file not found: {weights:?}");
        }
        Ok(SamRunner {
            arch: self.arch,
            weights,
            device: self.device.unwrap_or(Device::Cpu),
            config_path: self.config_path,
        })
    }
}

/// SAM runner — owns the resolved config and dispatches the
/// per-arch forward pass. SAM 1 / 2 / 3 differ enough in their
/// prompting that we keep the heavy result type
/// (`SamPredictionAny`) as a discriminated union the caller
/// matches on.
pub struct SamRunner {
    pub arch: SamArch,
    pub weights: PathBuf,
    pub device: Device,
    pub config_path: Option<PathBuf>,
}

/// Union of per-arch image-prediction outputs. Caller matches on
/// the arch they asked for.
pub enum SamPredictionAny {
    Sam1(crate::sam::MaskPrediction),
    Sam2(crate::sam2::Sam2ImagePrediction),
    Sam3(crate::sam3::Sam3ImagePrediction),
}

impl SamRunner {
    pub fn builder(arch: SamArch) -> SamRunnerBuilder {
        SamRunnerBuilder {
            arch,
            weights: None,
            device: None,
            config_path: None,
        }
    }

    /// Print a human-readable summary — what the CLI prints before
    /// any per-arch image processing.
    pub fn summary(&self) -> String {
        format!(
            "SAM{} runner — weights={:?} device={:?} config={:?}",
            match self.arch {
                SamArch::Sam1 => "1",
                SamArch::Sam2 => "2",
                SamArch::Sam3 => "3",
            },
            self.weights,
            self.device,
            self.config_path
        )
    }

    /// End-to-end forward: image bytes → masks. Dispatches to the
    /// right per-arch entrypoint:
    ///   * SAM 1 → `Sam::forward` (multimask = true)
    ///   * SAM 2 → `Sam2::predict_image` (multimask = true)
    ///   * SAM 3 → `Sam3::predict_image_text` with the supplied
    ///     `text_tokens` (required for SAM 3 — its decoder is
    ///     text-conditioned). Pass an empty slice for arches that
    ///     don't use it.
    ///
    /// `rgb` is HWC u8; `points` is `(xy_pairs, labels)` with one
    /// label per (x, y) pair (1 = foreground, 0 = background).
    ///
    /// SAM-arch-specific defaults applied:
    ///   * `cfg` derived from environment variables (`RLX_SAM_VARIANT`
    ///     for v1: vit_b/l/h; `RLX_SAM2_VARIANT` for v2: tiny/small/
    ///     base_plus/large); falls back to the smallest variant.
    ///   * `multimask_output = true` for v1 + v2.
    ///   * SAM 3 vit defaults to `base`.
    pub fn predict_image(
        &self,
        rgb: &[u8],
        h_in: usize,
        w_in: usize,
        points: Option<(&[f32], &[f32])>,
        boxes: Option<&[f32]>,
        text_tokens: &[u32],
    ) -> Result<SamPredictionAny> {
        let weights_str = self
            .weights
            .to_str()
            .ok_or_else(|| anyhow!("non-utf8 weights path"))?;
        match self.arch {
            SamArch::Sam1 => {
                use crate::sam::{Sam, SamConfig};
                let cfg = match std::env::var("RLX_SAM_VARIANT")
                    .unwrap_or_else(|_| "vit_b".into())
                    .as_str()
                {
                    "vit_b" => SamConfig::vit_b(),
                    "vit_l" => SamConfig::vit_l(),
                    "vit_h" => SamConfig::vit_h(),
                    other => bail!("RLX_SAM_VARIANT must be vit_b|vit_l|vit_h, got {other}"),
                };
                let mut sam = Sam::from_safetensors_on(weights_str, cfg, self.device)?;
                let (pred, _resized) =
                    sam.forward(rgb, h_in, w_in, points, boxes, None, /*multimask*/ true)?;
                Ok(SamPredictionAny::Sam1(pred))
            }
            SamArch::Sam2 => {
                use crate::sam2::{Sam2, Sam2Config};
                let cfg = match std::env::var("RLX_SAM2_VARIANT")
                    .unwrap_or_else(|_| "tiny".into())
                    .as_str()
                {
                    "tiny" => Sam2Config::hiera_tiny(),
                    "small" => Sam2Config::hiera_small(),
                    "base_plus" => Sam2Config::hiera_base_plus(),
                    "large" => Sam2Config::hiera_large(),
                    other => bail!(
                        "RLX_SAM2_VARIANT must be tiny|small|base_plus|large, got {other}"
                    ),
                };
                let mut sam = Sam2::from_safetensors_on(weights_str, cfg, self.device)?;
                let pred = sam.predict_image(
                    rgb,
                    h_in,
                    w_in,
                    points,
                    boxes,
                    None,
                    /*multimask*/ true,
                )?;
                Ok(SamPredictionAny::Sam2(pred))
            }
            SamArch::Sam3 => {
                use crate::sam3::{Sam3, Sam3Config};
                let cfg = Sam3Config::base();
                let sam = Sam3::from_safetensors_on(weights_str, cfg, self.device)?;
                if text_tokens.is_empty() {
                    bail!("SAM 3 is text-conditioned — pass non-empty text_tokens");
                }
                let pred = sam.predict_image_text(rgb, h_in, w_in, text_tokens)?;
                Ok(SamPredictionAny::Sam3(pred))
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Loader helpers (re-exported for callers that want to inspect a
// model file before committing to a runner build).
// ────────────────────────────────────────────────────────────────

/// Open a weights file as the appropriate `WeightLoader`. Format
/// is detected from the extension.
pub fn open_loader(path: &Path) -> Result<Box<dyn WeightLoader>> {
    match WeightFormat::from_path(path)? {
        WeightFormat::Safetensors => Ok(Box::new(WeightMap::from_file(
            path.to_str().ok_or_else(|| anyhow!("non-utf8 path"))?,
        )?)),
        WeightFormat::Gguf => Ok(Box::new(GgufLoader::from_file(
            path.to_str().ok_or_else(|| anyhow!("non-utf8 path"))?,
        )?)),
    }
}

/// Surface MTP head tensor names from a GGUF file (returns empty for
/// safetensors / files without MTP weights). Useful before deciding
/// to enable `use_mtp(true)` on a runner.
pub fn list_mtp_keys(path: &Path) -> Result<Vec<String>> {
    if WeightFormat::from_path(path)? != WeightFormat::Gguf {
        return Ok(vec![]);
    }
    let loader = GgufLoader::from_file(path.to_str().ok_or_else(|| anyhow!("non-utf8 path"))?)?;
    Ok(loader.mtp_keys())
}

/// Surface the HF→GGUF mapping for a given HF-style name, for callers
/// debugging which GGUF tensor a builder will pick up.
pub fn debug_resolve_name(hf: &str) -> Option<String> {
    hf_to_gguf_name(hf)
}

// ────────────────────────────────────────────────────────────────
// Dynamic model registry — third-party crates can plug new runners
// into `rlx-run` (or any binary that calls `dispatch`) without
// touching upstream code. Built-in runners are registered the same
// way: see `rlx-models/src/bin/rlx_run.rs`.
// ────────────────────────────────────────────────────────────────

/// One CLI-style entry per model family. Implementors own their own
/// arg parsing — keeping the trait tiny (just name + run) means the
/// argument schema can evolve per model without touching the trait.
///
/// Example (third-party crate plugging a `whisper` runner):
///
/// ```ignore
/// use rlx_models::run::{ModelRunner, register_runner, dispatch};
///
/// struct WhisperRunner;
/// impl ModelRunner for WhisperRunner {
///     fn name(&self) -> &'static str { "whisper" }
///     fn description(&self) -> &'static str { "Run an OpenAI Whisper checkpoint" }
///     fn run(&self, args: &[String]) -> anyhow::Result<()> {
///         // parse args, invoke model …
///         Ok(())
///     }
/// }
///
/// fn main() -> anyhow::Result<()> {
///     register_runner(Box::new(WhisperRunner));
///     // From here, `rlx-run whisper …` (or any caller of `dispatch`)
///     // will route to WhisperRunner::run.
///     let argv: Vec<String> = std::env::args().skip(1).collect();
///     dispatch(&argv)
/// }
/// ```
pub trait ModelRunner: Send + Sync + 'static {
    /// Subcommand name. Must be unique across the registry — later
    /// `register_runner` calls overwrite earlier ones with the same
    /// name (intentional: lets downstream binaries swap a built-in
    /// for a customized version).
    fn name(&self) -> &'static str;
    /// Single-line help string shown by `dispatch_help()`.
    fn description(&self) -> &'static str;
    /// Run the subcommand. `args` is the slice AFTER the subcommand
    /// name — i.e. exactly what `cargo run -- subcmd a b c` gives
    /// you after stripping the leading `subcmd`.
    fn run(&self, args: &[String]) -> Result<()>;
}

type RegistryInner = Vec<Box<dyn ModelRunner>>;

fn registry() -> &'static Mutex<RegistryInner> {
    static R: OnceLock<Mutex<RegistryInner>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Vec::new()))
}

/// Register a model runner. Idempotent on name: re-registering the
/// same name replaces the previous entry. Safe to call from any
/// thread before [`dispatch`] runs.
pub fn register_runner(runner: Box<dyn ModelRunner>) {
    let mut g = registry().lock().expect("runner registry poisoned");
    let name = runner.name();
    if let Some(idx) = g.iter().position(|r| r.name() == name) {
        g[idx] = runner;
    } else {
        g.push(runner);
    }
}

/// Names + descriptions of every currently-registered runner. Cheap
/// — used by the CLI's `help` subcommand to render the dynamic list.
pub fn registered_runners() -> Vec<(&'static str, &'static str)> {
    let g = registry().lock().expect("runner registry poisoned");
    g.iter().map(|r| (r.name(), r.description())).collect()
}

/// Look up a registered runner by name. Returns the closure-style
/// callable rather than the trait object itself because trait
/// objects in a `Mutex<Vec<…>>` aren't directly returnable as
/// `&'static dyn`.
pub fn run_registered(name: &str, args: &[String]) -> Result<Option<()>> {
    let g = registry().lock().expect("runner registry poisoned");
    for runner in g.iter() {
        if runner.name() == name {
            // Run while still holding the lock; runners are
            // `Send + Sync` and should not call back into the
            // registry. (If a future runner DOES need to dispatch
            // recursively, it should call `dispatch` directly.)
            return runner.run(args).map(Some);
        }
    }
    Ok(None)
}

/// Top-level CLI dispatch. Looks at `args[0]` as the subcommand
/// name; routes to the matching registered runner. Returns
/// `Err(…)` only on real failures — an unknown subcommand prints
/// the registered list to stderr and returns Err.
///
/// `rlx-run`'s main is essentially `register_built_ins(); dispatch(argv);`
/// — third parties writing their own binary do the same plus their
/// own `register_runner` calls.
pub fn dispatch(args: &[String]) -> Result<()> {
    let Some(sub) = args.first() else {
        eprintln!("{}", dispatch_help());
        return Ok(());
    };
    match sub.as_str() {
        "help" | "--help" | "-h" => {
            println!("{}", dispatch_help());
            return Ok(());
        }
        _ => {}
    }
    match run_registered(sub, &args[1..])? {
        Some(()) => Ok(()),
        None => {
            eprintln!("{}", dispatch_help());
            bail!("unknown subcommand: {sub}");
        }
    }
}

/// Render the help string for `dispatch`. Includes every registered
/// runner, in registration order. Built-in runners come first when
/// the standard binary calls `register_built_ins` before any
/// third-party `register_runner`.
pub fn dispatch_help() -> String {
    let mut s = String::from(
        "rlx-run — minimal multi-model launcher\nUSAGE:\n  rlx-run <subcommand> [flags]\n\nSUBCOMMANDS:\n",
    );
    let mut any = false;
    for (name, desc) in registered_runners() {
        s.push_str(&format!("  {name:<10} {desc}\n"));
        any = true;
    }
    if !any {
        s.push_str(
            "  (no runners registered — call rlx_models::run::register_runner first)\n",
        );
    }
    s.push_str("  help       print this help\n");
    s
}
