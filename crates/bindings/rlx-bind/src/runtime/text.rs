// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The LM text layer: tokenizer (with streaming decode), sampling, chat
//! templates — `rlx-text` behind a binding-neutral surface.
//!
//! Sampling options (`temperature`, `top_k`, `top_p`, `min_p`, `typical_p`,
//! `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `seed`) are
//! one JSON object; no options means greedy. Naming any sampling-only option
//! without a temperature samples at 1.0, so `{top_k: 40}` does what it says.

use std::path::Path;

use rlx_text::{ChatMessage, ChatTemplate, SampleOpts, TokenizerHandle};
use serde_json::Value;

use super::opts::Opts;
use crate::{ApiError, ApiResult};

/// A tokenizer plus the state of one streaming decode.
pub struct Tokenizer {
    pub handle: TokenizerHandle,
    stream_ids: Vec<u32>,
    emitted: usize,
}

fn text_err(what: &str, e: impl std::fmt::Display) -> ApiError {
    ApiError::runtime(format!("{what}: {e:#}"))
}

impl Tokenizer {
    /// From a `tokenizer.json`, a GGUF file, or a model directory.
    pub fn load(path: &str) -> ApiResult<Self> {
        let p = Path::new(path);
        if !p.exists() {
            return Err(ApiError::new(
                crate::ErrorKind::NotFound,
                format!("load_tokenizer('{path}'): no such file or directory"),
            ));
        }
        let handle = rlx_text::load_tokenizer(p)
            .map_err(|e| text_err(&format!("load_tokenizer('{path}')"), e))?;
        Ok(Self {
            handle,
            stream_ids: Vec::new(),
            emitted: 0,
        })
    }

    /// Special tokens default on in the bindings: a prompt without BOS
    /// silently shifts every position.
    pub fn encode(&self, text: &str, add_special: bool) -> ApiResult<Vec<u32>> {
        self.handle
            .encode(text, add_special)
            .map_err(|e| text_err("encode", e))
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> ApiResult<String> {
        self.handle
            .decode(ids, skip_special)
            .map_err(|e| text_err("decode", e))
    }

    /// Feed one generated id; returns only the text that is now final. A
    /// multi-byte character split across tokens comes out once, whole.
    pub fn decode_incremental(&mut self, id: u32) -> ApiResult<String> {
        self.stream_ids.push(id);
        let (delta, emitted) =
            rlx_text::incremental_emit(&self.handle, &self.stream_ids, self.emitted, true)
                .map_err(|e| text_err("decode_incremental", e))?;
        self.emitted = emitted;
        Ok(delta)
    }

    /// Whatever the stream still holds back, flushed.
    pub fn finish_stream(&mut self) -> ApiResult<String> {
        let full = self
            .handle
            .decode(&self.stream_ids, true)
            .map_err(|e| text_err("finish_stream", e))?;
        let tail = full.get(self.emitted..).unwrap_or("").to_string();
        self.emitted = full.len();
        Ok(tail)
    }

    pub fn reset_stream(&mut self) {
        self.stream_ids.clear();
        self.emitted = 0;
    }

    /// Ids fed to the current stream.
    pub fn streamed(&self) -> usize {
        self.stream_ids.len()
    }
}

// ── sampling ────────────────────────────────────────────────

const DEFAULT_SEED: u32 = 0x9E37_79B9;

/// A seed → a non-zero xorshift state (SplitMix64 finalizer).
fn mix_seed(seed: u64) -> u32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x as u32) | 1).max(1)
}

/// Sampling options and the initial RNG state they ask for.
pub fn sample_opts(v: &Value) -> ApiResult<(SampleOpts, u32)> {
    let o = Opts::new("sampling", v)?;
    let mut opts = SampleOpts::greedy();
    let explicit_temperature = match o.f64("temperature")? {
        Some(t) => {
            opts.temperature = t as f32;
            true
        }
        None => false,
    };
    let mut wants_sampling = false;
    if let Some(k) = o.u64("top_k")? {
        // 0 means "no cut" in llama.cpp's convention.
        opts.top_k = if k == 0 { None } else { Some(k as u32) };
        wants_sampling = true;
    }
    for (name, slot) in [
        ("top_p", &mut opts.top_p),
        ("min_p", &mut opts.min_p),
        ("typical_p", &mut opts.typical_p),
        ("repetition_penalty", &mut opts.repetition_penalty),
        ("frequency_penalty", &mut opts.frequency_penalty),
        ("presence_penalty", &mut opts.presence_penalty),
    ] {
        if let Some(x) = o.f64(name)? {
            *slot = x as f32;
            wants_sampling = true;
        }
    }
    if wants_sampling && !explicit_temperature {
        opts.temperature = 1.0;
    }
    let rng = o.u64("seed")?.map(mix_seed).unwrap_or(DEFAULT_SEED);
    o.finish()?;
    Ok((opts, rng))
}

fn nonempty(logits: &[f32], what: &str) -> ApiResult<()> {
    if logits.is_empty() {
        return Err(ApiError::type_(format!("{what}: logits are empty")));
    }
    Ok(())
}

/// Index of the largest logit.
pub fn argmax(logits: &[f32]) -> ApiResult<u32> {
    nonempty(logits, "argmax")?;
    Ok(rlx_text::argmax(logits))
}

/// One token from `logits`, penalized by `history`.
pub fn sample_next(logits: &[f32], history: &[u32], opts: &Value) -> ApiResult<u32> {
    nonempty(logits, "sample_next")?;
    let (opts, mut rng) = sample_opts(opts)?;
    Ok(rlx_text::sample_next(logits, history, &opts, &mut rng))
}

/// A sampler carrying its history and RNG across tokens.
pub struct Sampler {
    pub opts: SampleOpts,
    rng: u32,
    history: Vec<u32>,
}

impl Sampler {
    pub fn new(opts: &Value) -> ApiResult<Self> {
        let (opts, rng) = sample_opts(opts)?;
        Ok(Self {
            opts,
            rng,
            history: Vec::new(),
        })
    }

    pub fn next(&mut self, logits: &[f32]) -> ApiResult<u32> {
        nonempty(logits, "Sampler.next")?;
        let token = rlx_text::sample_next(logits, &self.history, &self.opts, &mut self.rng);
        self.history.push(token);
        Ok(token)
    }

    /// Replace the history with a prompt's ids, so penalties see it.
    pub fn prime(&mut self, ids: &[u32]) -> usize {
        self.history.clear();
        self.history.extend_from_slice(ids);
        self.history.len()
    }

    /// Clear the history; reseed when `seed` is given.
    pub fn reset(&mut self, seed: Option<u64>) {
        self.history.clear();
        if let Some(seed) = seed {
            self.rng = mix_seed(seed);
        }
    }

    pub fn history(&self) -> &[u32] {
        &self.history
    }
}

// ── chat templates ──────────────────────────────────────────

/// `[{role, content}]` → messages.
pub fn messages(v: &Value) -> ApiResult<Vec<ChatMessage>> {
    let Value::Array(items) = v else {
        return Err(ApiError::type_(
            "render_chat: messages must be a list of {role, content}",
        ));
    };
    items
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let field = |k: &str| m.get(k).and_then(Value::as_str);
            let (Some(role), Some(content)) = (field("role"), field("content")) else {
                return Err(ApiError::type_(format!(
                    "render_chat: message {i} needs {{role, content}}"
                )));
            };
            Ok(match role.trim().to_ascii_lowercase().as_str() {
                "system" => ChatMessage::system(content.to_string()),
                "assistant" => ChatMessage::assistant(content.to_string()),
                "user" => ChatMessage::user(content.to_string()),
                other => {
                    return Err(ApiError::type_(format!(
                        "render_chat: message {i} has role '{other}' (system, user, assistant)"
                    )));
                }
            })
        })
        .collect()
}

fn template_from_tokenizer_config(text: &str) -> Option<String> {
    let root: Value = serde_json::from_str(text).ok()?;
    let value = root.get("chat_template")?;
    if let Some(s) = value.as_str() {
        return Some(s.to_string());
    }
    let entries = value.as_array()?;
    let pick = entries
        .iter()
        .find(|e| e.get("name").and_then(Value::as_str) == Some("default"))
        .or_else(|| entries.first())?;
    Some(pick.get("template")?.as_str()?.to_string())
}

/// Render `messages` with a chat template. `source` is the template text, or
/// — when `allow_files` — a `.gguf`, `tokenizer_config.json` or `.jinja` path.
///
/// A host that sandboxes file access passes `allow_files = false`, and the
/// check comes *before* the stat, so a script cannot probe for a file's
/// existence through this call.
pub fn render_chat(
    source: &str,
    messages: &Value,
    add_generation_prompt: bool,
    allow_files: bool,
) -> ApiResult<String> {
    let messages = self::messages(messages)?;
    let path = Path::new(source);
    let template = if allow_files && path.is_file() {
        if source.to_ascii_lowercase().ends_with(".gguf") {
            ChatTemplate::from_gguf(path)
                .map_err(|e| text_err(&format!("render_chat('{source}')"), e))?
        } else {
            let text = std::fs::read_to_string(path)
                .map_err(|e| text_err(&format!("render_chat('{source}')"), e))?;
            let jinja = template_from_tokenizer_config(&text).unwrap_or(text);
            ChatTemplate::from_source(jinja).map_err(|e| {
                text_err(
                    &format!("render_chat('{source}'): no usable chat template"),
                    e,
                )
            })?
        }
    } else {
        ChatTemplate::from_source(source.to_string()).map_err(|e| {
            ApiError::type_(format!(
                "render_chat: '{source}' is neither a readable file nor a valid template: {e:#}"
            ))
        })?
    };
    template
        .render(&messages, add_generation_prompt)
        .map_err(|e| text_err("render_chat render", e))
}
