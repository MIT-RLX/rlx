// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! High-level runner for Qwen3.5 / Qwen3.6 (qwen35 architecture).
//!
//! Mirrors the [`crate::run::Qwen3Runner`] API shape but routes through
//! the gated-DeltaNet trunk + optional MTP head built in
//! [`crate::qwen35::build_qwen35_graph_sized`]. Today the runner is
//! **prefill-only with full F32 weight materialization** — see the
//! builder docs for the memory + decode-cache constraints (small
//! files only: Qwen3.5-0.8B Q4_K_M ≈ 1.5 GB F32 footprint fits, 27B
//! does not until the packed-weights path is extended to qwen35).

use crate::qwen35::config::Qwen35Config;
use crate::qwen35::weights::Qwen35Weights;
use crate::weight_loader::GgufLoader;
use anyhow::{Result, anyhow, bail};
use rlx_ir::DType;
use rlx_runtime::{Device, Session};
use std::path::PathBuf;
use std::time::Instant;

/// Builder for [`Qwen35Runner`]. Mirrors the field set of
/// [`crate::run::Qwen3RunnerBuilder`] minus the precision and packed
/// flags (both still F32-only in this slice).
#[derive(Default, Debug)]
pub struct Qwen35RunnerBuilder {
    weights: Option<PathBuf>,
    device: Option<Device>,
    max_seq: Option<usize>,
    enable_mtp: bool,
    last_logits_only: bool,
    packed_weights: bool,
}

impl Qwen35RunnerBuilder {
    pub fn weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }
    pub fn device(mut self, d: Device) -> Self {
        self.device = Some(d);
        self
    }
    pub fn max_seq(mut self, n: usize) -> Self {
        self.max_seq = Some(n);
        self
    }
    /// Emit the MTP head's `[batch, 1, vocab]` logits as a second
    /// output. The MTP head consumes the trunk's pre-norm hidden
    /// state alongside a fresh embedding of the input ids, so it
    /// fires from the same prefill graph (no separate forward).
    pub fn enable_mtp(mut self, on: bool) -> Self {
        self.enable_mtp = on;
        self
    }
    /// When set, the trunk gathers only the last token's hidden
    /// state before the LM head — saves a `[seq, vocab]` matmul on
    /// long prompts. Default: `true`.
    pub fn last_logits_only(mut self, on: bool) -> Self {
        self.last_logits_only = on;
        self
    }
    /// Keep K-quant matmul weights (Q4_K / Q5_K / Q6_K / Q8_K)
    /// packed in the arena and emit `Op::DequantMatMul` per
    /// projection. Cuts the load-time F32 footprint by ~4×; the
    /// unblocker for ≥14 B Qwen3.5/3.6 GGUFs on commodity Macs.
    /// CPU-only today.
    pub fn packed_weights(mut self, on: bool) -> Self {
        self.packed_weights = on;
        self
    }

    pub fn build(self) -> Result<Qwen35Runner> {
        let weights_path = self
            .weights
            .ok_or_else(|| anyhow!("weights path required (call .weights(...))"))?;
        let device = self.device.unwrap_or(Device::Cpu);
        let max_seq = self.max_seq.unwrap_or(128);

        let t_total = Instant::now();
        let t = Instant::now();
        let mut loader = GgufLoader::from_file(
            weights_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 weights path"))?,
        )?;
        loader.include_mtp(true);
        // Reuse the loader's already-parsed `GgufFile` to read
        // qwen35.* metadata. Re-parsing 800+ tensor headers a second
        // time burned 20 s on the 27 B file before this.
        let cfg = Qwen35Config::from_gguf(loader.file())?;
        eprintln!(
            "[qwen35] loaded GGUF metadata in {:.2?} \
             (layers={}, hidden={}, ssm_state={})",
            t.elapsed(),
            cfg.num_hidden_layers,
            cfg.hidden_size,
            cfg.ssm_state_size,
        );

        if self.enable_mtp && cfg.nextn_predict_layers == 0 {
            bail!(
                "qwen35: enable_mtp(true) but the file has \
                 nextn_predict_layers=0 (no MTP heads to wire)"
            );
        }

        let t = Instant::now();
        let weights = if self.packed_weights {
            Qwen35Weights::from_loader_packed(&mut loader, &cfg)?
        } else {
            Qwen35Weights::from_loader(&mut loader, &cfg)?
        };
        eprintln!(
            "[qwen35] read weights ({}) in {:.2?}",
            if self.packed_weights { "packed" } else { "F32" },
            t.elapsed(),
        );

        let t = Instant::now();
        let (graph, params, packed) = super::build_qwen35_graph_sized(
            &cfg,
            weights,
            /*batch*/ 1,
            max_seq,
            /*with_lm_head*/ true,
            self.last_logits_only,
            self.enable_mtp,
        )?;
        eprintln!(
            "[qwen35] built IR in {:.2?} (params={}, packed={})",
            t.elapsed(),
            params.len(),
            packed.len(),
        );

        let t = Instant::now();
        let mut compiled = Session::new(device).compile(graph);
        eprintln!("[qwen35] compiled graph in {:.2?}", t.elapsed());

        let t = Instant::now();
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        let n_packed = packed.len();
        // Stream packed bytes from the loader's mmap straight into
        // the compiled arena. This is the second half of the
        // `tensor_bytes_borrowed` shortcut: combined with the
        // `take_packed_metadata` change in `from_loader_packed`,
        // we save the ~16 GB Vec<u8> allocation that used to dominate
        // the build time on Qwen3.6-27B Q4_K_M.
        let mut total_packed_bytes: usize = 0;
        for (param_name, (loader_key, _scheme, _shape)) in &packed {
            let bytes = loader
                .tensor_bytes_borrowed(loader_key)
                .ok_or_else(|| anyhow!("packed upload: {loader_key} bytes missing"))?;
            total_packed_bytes += bytes.len();
            compiled.set_param_typed(param_name, bytes, rlx_ir::DType::U8);
        }
        eprintln!(
            "[qwen35] uploaded {} F32 + {} packed params \
             ({:.2} GB packed, zero-copy) in {:.2?} (total build {:.2?})",
            params.len(),
            n_packed,
            total_packed_bytes as f64 / 1e9,
            t.elapsed(),
            t_total.elapsed(),
        );

        Ok(Qwen35Runner {
            compiled,
            cfg,
            device,
            max_seq,
            last_logits_only: self.last_logits_only,
            enable_mtp: self.enable_mtp,
        })
    }
}

/// Compiled Qwen3.5 prefill model. Hold this to run repeated
/// `predict_logits` calls without recompiling.
pub struct Qwen35Runner {
    compiled: rlx_runtime::CompiledGraph,
    cfg: Qwen35Config,
    device: Device,
    max_seq: usize,
    last_logits_only: bool,
    enable_mtp: bool,
}

/// Prefill output. `logits` is the trunk's LM-head logits
/// (`[seq * vocab]` or `[1 * vocab]` depending on `last_logits_only`).
/// `mtp_logits`, when present, is the MTP head's
/// `[1 * vocab]` next-token logits.
#[derive(Debug, Clone)]
pub struct Qwen35PrefillOutput {
    pub logits: Vec<f32>,
    pub mtp_logits: Option<Vec<f32>>,
    pub vocab_size: usize,
}

impl Qwen35Runner {
    pub fn builder() -> Qwen35RunnerBuilder {
        Qwen35RunnerBuilder::default()
    }

    pub fn cfg(&self) -> &Qwen35Config {
        &self.cfg
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Run a single prefill on the prompt ids. Length must be ≤
    /// `max_seq`; shorter prompts are zero-padded at the *end* (the
    /// graph was compiled at fixed shape; padding doesn't change
    /// causal-attn semantics because the model still attends only to
    /// past positions). When `enable_mtp`, the MTP head's logits are
    /// returned as a separate vector.
    pub fn predict_logits(&mut self, prompt_ids: &[u32]) -> Result<Qwen35PrefillOutput> {
        if prompt_ids.len() > self.max_seq {
            bail!(
                "qwen35: prompt length {} exceeds compiled max_seq={}",
                prompt_ids.len(),
                self.max_seq
            );
        }
        let mut padded = vec![0f32; self.max_seq];
        for (i, &t) in prompt_ids.iter().enumerate() {
            padded[i] = t as f32;
        }

        // RoPE cos/sin tables: trivial all-zeros / all-ones for this
        // prefill since we apply RoPE per layer with the same table.
        // The full MRoPE table will replace these in the parity slice.
        let half_d = self.cfg.rope_dim_count / 2;
        let cos = vec![1.0f32; half_d];
        let sin = vec![0.0f32; half_d];

        let n_layer = self.cfg.num_hidden_layers;
        let nextn = self.cfg.nextn_predict_layers;
        let n_main = n_layer - nextn;
        let mtp_il = n_main; // MTP layer index for input naming.

        // Collect per-layer rope inputs. Linear layers don't take
        // RoPE inputs — only full-attn (every `full_attention_interval`)
        // and the MTP head.
        let interval = self.cfg.full_attention_interval.max(1);
        let cos_buf = cos.as_slice();
        let sin_buf = sin.as_slice();

        // Per-layer RoPE feed keys (input_ids is pushed above).
        let mut feed_keys: Vec<String> = Vec::new();
        let _ = DType::F32;
        let mut feeds: Vec<(&str, &[f32])> = Vec::new();
        // input_ids fed as F32 (the embed gather kernel does the
        // implicit cast).
        feeds.push(("input_ids", padded.as_slice()));
        for il in 0..n_main {
            let is_full_attn = ((il + 1) % interval) == 0;
            if !is_full_attn {
                continue;
            }
            feed_keys.push(format!("rope_cos_l{il}"));
            feed_keys.push(format!("rope_sin_l{il}"));
        }
        if self.enable_mtp {
            feed_keys.push(format!("rope_cos_l{mtp_il}"));
            feed_keys.push(format!("rope_sin_l{mtp_il}"));
        }

        for key in feed_keys.iter() {
            let payload: &[f32] = if key.starts_with("rope_cos_l") {
                cos_buf
            } else if key.starts_with("rope_sin_l") {
                sin_buf
            } else {
                bail!("internal: unknown feed key {key}")
            };
            // SAFETY: the slice lives for the entire run() call.
            feeds.push((key.as_str(), payload));
        }

        let outs = self.compiled.run(&feeds);
        if outs.is_empty() {
            bail!("qwen35: forward produced no outputs");
        }
        // Vocab size = trunk logits length / (seq * batch).
        let vocab_size = if self.last_logits_only {
            outs[0].len()
        } else {
            outs[0].len() / self.max_seq
        };
        let mtp_logits = if self.enable_mtp && outs.len() >= 2 {
            Some(outs[1].clone())
        } else {
            None
        };
        Ok(Qwen35PrefillOutput {
            logits: outs[0].clone(),
            mtp_logits,
            vocab_size,
        })
    }

    /// Greedy autoregressive generation via repeated prefills.
    /// Returns the appended `n_new` token ids. Trade-off: each
    /// generated token costs one full prefill (no decode-state
    /// cache yet on the gated-DeltaNet trunk — that's the next
    /// slice). Suitable for short generations on small files;
    /// O(n_new · seq · n_state²) total work scales painfully.
    ///
    /// When `on_token` returns `false` the loop stops early — the
    /// caller can implement EOS detection or streaming-stop logic
    /// without re-running the prefill.
    pub fn generate<F>(
        &mut self,
        prompt_ids: &[u32],
        n_new: usize,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        if prompt_ids.is_empty() {
            bail!("qwen35::generate: prompt must contain at least one id");
        }
        let mut history: Vec<u32> = prompt_ids.to_vec();
        let mut generated: Vec<u32> = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            if history.len() >= self.max_seq {
                // Truncate from the front — simplest sliding-window
                // strategy. The first `keep` tokens stay so the
                // attention sink survives.
                let drop = history.len() - (self.max_seq - 1);
                history.drain(0..drop);
            }
            let out = self.predict_logits(&history)?;
            let next = greedy_argmax(&out.logits);
            history.push(next);
            generated.push(next);
            if !on_token(next) {
                break;
            }
        }
        Ok(generated)
    }
}

fn greedy_argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as u32;
        }
    }
    best_i
}

