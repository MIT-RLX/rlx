// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tokenizers, chat templates and sampling — the pieces a language-model
//! script needs either side of the graph.
//!
//! ```js
//! const tok = rlx.loadTokenizer("tokenizer.json");
//! const prompt = rlx.renderChat("model.gguf", [{ role: "user", content: "hi" }]);
//! const ids = tok.encode(prompt);
//! // … run the model, then:
//! const next = rlx.sampleNext(logits, ids, { temperature: 0.7, topP: 0.9 });
//! out += tok.decodeIncremental(next);
//! ```
//!
//! `decodeIncremental` exists because decoding a growing id list from scratch
//! each step is both quadratic and wrong at a partial multi-byte character —
//! the streaming detokenizer holds the boundary state.

use std::path::Path;

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_text::{ChatMessage, ChatTemplate, SampleOpts, TokenizerHandle};

use crate::convert::*;
use crate::handle::{self, CLASS_SAMPLER, CLASS_TOKENIZER};

/// A tokenizer plus the streaming-decode state, so `decodeIncremental` can
/// emit only what became stable.
pub struct TokenizerSlot {
    handle: TokenizerHandle,
    /// Ids already fed to the incremental decoder.
    stream_ids: Vec<u32>,
    /// Byte offset into the full decode already returned to the caller.
    emitted: usize,
}

fn tokenizer_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut TokenizerSlot> {
    handle::borrow_mut(ctx, this, CLASS_TOKENIZER, "rlx.Tokenizer")
}

fn f_load_tokenizer(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "loadTokenizer")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    match rlx_text::load_tokenizer(Path::new(&path)) {
        Ok(handle) => Ok(handle::wrap(
            ctx,
            CLASS_TOKENIZER,
            TokenizerSlot {
                handle,
                stream_ids: Vec::new(),
                emitted: 0,
            },
        )),
        Err(e) => ctx.throw_internal(&format!("loadTokenizer('{path}'): {e:#}")),
    }
}

/// `encode(text, addSpecial?)` → `Uint32Array`-shaped array of ids.
fn m_encode(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let text = ctx.to_rust_string(arg(args, 0))?;
    // Special tokens default ON: a prompt without BOS silently shifts every
    // position, which shows up as bad output rather than an error.
    let add_special = if is_nullish(arg(args, 1)) {
        true
    } else {
        to_bool(arg(args, 1))
    };
    let ids = match tokenizer_of(ctx, this)?.handle.encode(&text, add_special) {
        Ok(ids) => ids,
        Err(e) => return ctx.throw_internal(&format!("encode: {e:#}")),
    };
    let values: Vec<Value> = ids.iter().map(|id| Value::number(*id as f64)).collect();
    Ok(new_array(ctx, values))
}

/// `decode(ids, skipSpecial?)` → string.
fn m_decode(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let ids = to_u32_vec(ctx, arg(args, 0), "decode ids")?;
    let skip_special = if is_nullish(arg(args, 1)) {
        true
    } else {
        to_bool(arg(args, 1))
    };
    let text = match tokenizer_of(ctx, this)?.handle.decode(&ids, skip_special) {
        Ok(text) => text,
        Err(e) => return ctx.throw_internal(&format!("decode: {e:#}")),
    };
    Ok(new_string(ctx, &text))
}

/// `decodeIncremental(id)` → only the text that became *stable* with this id.
///
/// Returns `""` while a multi-byte character or a multi-token grapheme is
/// still partial, which is the right thing to print in a streaming loop. Call
/// `finishStream()` once at the end to flush the held-back tail.
///
/// This delegates to `rlx_text::incremental_emit`, which trims a trailing run
/// of U+FFFD before deciding what is safe to emit. A hand-rolled
/// "emit the suffix if the prefix still matches" version *loses* text: on a
/// ZWJ emoji sequence the decoder rewrites earlier bytes as the grapheme
/// completes, and the mismatch branch drops two characters of `👨‍👩‍👧`.
fn m_decode_incremental(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let id = to_usize(ctx, arg(args, 0), "decodeIncremental id")? as u32;
    let delta = {
        let slot = tokenizer_of(ctx, this)?;
        slot.stream_ids.push(id);
        match rlx_text::incremental_emit(&slot.handle, &slot.stream_ids, slot.emitted, true) {
            Ok((delta, emitted)) => {
                slot.emitted = emitted;
                delta
            }
            Err(e) => return ctx.throw_internal(&format!("decodeIncremental: {e:#}")),
        }
    };
    Ok(new_string(ctx, &delta))
}

/// Flush whatever `decodeIncremental` held back, and end the stream.
///
/// Without this the last partial grapheme of a generation is never printed.
fn m_finish_stream(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let tail = {
        let slot = tokenizer_of(ctx, this)?;
        let full = match slot.handle.decode(&slot.stream_ids, true) {
            Ok(text) => text,
            Err(e) => return ctx.throw_internal(&format!("finishStream: {e:#}")),
        };
        let tail = if slot.emitted < full.len() {
            full[slot.emitted..].to_string()
        } else {
            String::new()
        };
        slot.emitted = full.len();
        tail
    };
    Ok(new_string(ctx, &tail))
}

/// Drop the streaming state so the tokenizer can start a new generation.
fn m_reset_stream(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let slot = tokenizer_of(ctx, this)?;
    slot.stream_ids.clear();
    slot.emitted = 0;
    Ok(Value::Undefined)
}

fn m_to_string(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let text = format!(
        "[rlx.Tokenizer streamed={}]",
        tokenizer_of(ctx, this)?.stream_ids.len()
    );
    Ok(new_string(ctx, &text))
}

// ── sampling ────────────────────────────────────────────────

/// `{temperature, topK, topP, minP, typicalP, repetitionPenalty,
/// frequencyPenalty, presencePenalty, seed}` → `SampleOpts`.
///
/// Starts from `greedy()` and overrides only what the script names, so an empty
/// object is argmax.
///
/// One correction on top of that: `sample_next` short-circuits to argmax
/// whenever `temperature <= 0`, which means every other option — top-p,
/// top-k, both penalties — is silently ignored unless a temperature is also
/// given. `{repetitionPenalty: 1.2}` alone would be a no-op that looks like a
/// working config. So naming any sampling-only option implies
/// `temperature: 1.0` unless the script sets it, and asking for
/// `temperature: 0` explicitly still means greedy.
fn parse_sample_opts(ctx: &mut Context, v: &Value) -> JsResult<SampleOpts> {
    let mut opts = SampleOpts::greedy();
    let temp = field(ctx, v, "temperature")?;
    let explicit_temperature = !is_nullish(&temp);
    if explicit_temperature {
        opts.temperature = to_f32(ctx, &temp)?;
    }

    // Tracks whether anything that only takes effect under sampling was named.
    let mut wants_sampling = false;
    let top_k = field(ctx, v, "topK")?;
    if !is_nullish(&top_k) {
        let k = to_usize(ctx, &top_k, "topK")?;
        // 0 means "no cut" in llama.cpp's convention; keep that readable.
        opts.top_k = if k == 0 { None } else { Some(k as u32) };
        wants_sampling = true;
    }
    let top_p = field(ctx, v, "topP")?;
    if !is_nullish(&top_p) {
        opts.top_p = to_f32(ctx, &top_p)?;
        wants_sampling = true;
    }
    let min_p = field(ctx, v, "minP")?;
    if !is_nullish(&min_p) {
        opts.min_p = to_f32(ctx, &min_p)?;
        wants_sampling = true;
    }
    let typical_p = field(ctx, v, "typicalP")?;
    if !is_nullish(&typical_p) {
        opts.typical_p = to_f32(ctx, &typical_p)?;
        wants_sampling = true;
    }
    let penalty = field(ctx, v, "repetitionPenalty")?;
    if !is_nullish(&penalty) {
        opts.repetition_penalty = to_f32(ctx, &penalty)?;
        wants_sampling = true;
    }
    let frequency = field(ctx, v, "frequencyPenalty")?;
    if !is_nullish(&frequency) {
        opts.frequency_penalty = to_f32(ctx, &frequency)?;
        wants_sampling = true;
    }
    let presence = field(ctx, v, "presencePenalty")?;
    if !is_nullish(&presence) {
        opts.presence_penalty = to_f32(ctx, &presence)?;
        wants_sampling = true;
    }
    if wants_sampling && !explicit_temperature {
        opts.temperature = 1.0;
    }
    Ok(opts)
}

/// `argmax(logits)` — the greedy pick, no options and no RNG.
fn f_argmax(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let logits = to_f32_vec(ctx, arg(args, 0), "argmax logits")?;
    if logits.is_empty() {
        return ctx.throw_type("argmax: logits are empty");
    }
    Ok(Value::number(rlx_text::argmax(&logits) as f64))
}

/// One-shot sample: `sampleNext(logits, history, opts)`.
///
/// **Deterministic by design** — the RNG is seeded per call, so the same
/// `(logits, seed)` always gives the same token. That makes it a test helper,
/// not a generator: a loop calling it repeatedly gets the *same* token every
/// time, and bumping the seed by one does not help either, because a
/// near-identical seed produces a near-identical first draw. Use
/// [`rlx.Sampler`](install) for generation, which carries its RNG forward.
fn f_sample_next(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let logits = to_f32_vec(ctx, arg(args, 0), "sampleNext logits")?;
    if logits.is_empty() {
        return ctx.throw_type("sampleNext: logits are empty");
    }
    let history = to_u32_vec(ctx, arg(args, 1), "sampleNext history")?;
    let options = arg(args, 2).clone();
    let opts = parse_sample_opts(ctx, &options)?;
    let seed_v = field(ctx, &options, "seed")?;
    let mut rng = if is_nullish(&seed_v) {
        DEFAULT_SEED
    } else {
        mix_seed(to_usize(ctx, &seed_v, "seed")? as u64)
    };
    Ok(Value::number(
        rlx_text::sample_next(&logits, &history, &opts, &mut rng) as f64,
    ))
}

/// A sampler that carries its RNG *and* its token history across calls.
///
/// Both pieces of state are why this exists rather than an options bag:
/// re-seeding per call makes every draw identical, and a repetition penalty
/// that cannot see what was already generated does nothing.
pub struct SamplerSlot {
    opts: SampleOpts,
    rng: u32,
    history: Vec<u32>,
}

fn sampler_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut SamplerSlot> {
    handle::borrow_mut(ctx, this, CLASS_SAMPLER, "rlx.Sampler")
}

const DEFAULT_SEED: u32 = 0x9E37_79B9;

/// Spread a small integer seed across all 32 bits.
///
/// A raw `seed: 1` leaves an xorshift stream in a near-zero state whose first
/// several draws barely differ from `seed: 2`'s — which is what made
/// "pass an incrementing seed" useless advice.
fn mix_seed(seed: u64) -> u32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x as u32) | 1).max(1)
}

/// `new rlx.Sampler({temperature, topP, seed, …})`.
fn sampler_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let options = arg(args, 0).clone();
    let opts = parse_sample_opts(ctx, &options)?;
    let seed_v = field(ctx, &options, "seed")?;
    let rng = if is_nullish(&seed_v) {
        DEFAULT_SEED
    } else {
        mix_seed(to_usize(ctx, &seed_v, "seed")? as u64)
    };
    Ok(handle::wrap(
        ctx,
        CLASS_SAMPLER,
        SamplerSlot {
            opts,
            rng,
            history: Vec::new(),
        },
    ))
}

/// `next(logits)` → the sampled token, appended to this sampler's history so
/// the repetition penalty sees it.
fn m_sampler_next(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let logits = to_f32_vec(ctx, arg(args, 0), "Sampler.next logits")?;
    if logits.is_empty() {
        return ctx.throw_type("Sampler.next: logits are empty");
    }
    let slot = sampler_of(ctx, this)?;
    let token = rlx_text::sample_next(&logits, &slot.history, &slot.opts, &mut slot.rng);
    slot.history.push(token);
    Ok(Value::number(token as f64))
}

/// Seed the history with the prompt, so the first generated token already
/// sees it.
fn m_sampler_prime(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let ids = to_u32_vec(ctx, arg(args, 0), "Sampler.prime ids")?;
    let slot = sampler_of(ctx, this)?;
    slot.history.clear();
    slot.history.extend_from_slice(&ids);
    Ok(Value::number(slot.history.len() as f64))
}

/// Clear the history and optionally re-seed — start a new generation.
fn m_sampler_reset(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let seed = if is_nullish(arg(args, 0)) {
        None
    } else {
        Some(to_usize(ctx, arg(args, 0), "Sampler.reset seed")? as u64)
    };
    let slot = sampler_of(ctx, this)?;
    slot.history.clear();
    if let Some(seed) = seed {
        slot.rng = mix_seed(seed);
    }
    Ok(Value::Undefined)
}

fn m_sampler_history(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let ids: Vec<Value> = sampler_of(ctx, this)?
        .history
        .iter()
        .map(|id| Value::number(*id as f64))
        .collect();
    Ok(new_array(ctx, ids))
}

fn m_sampler_to_string(ctx: &mut Context, this: &Value, _a: &[Value], _m: i32) -> JsResult<Value> {
    let text = {
        let slot = sampler_of(ctx, this)?;
        format!(
            "[rlx.Sampler temperature={} topP={} generated={}]",
            slot.opts.temperature,
            slot.opts.top_p,
            slot.history.len()
        )
    };
    Ok(new_string(ctx, &text))
}

// ── chat templates ──────────────────────────────────────────

fn read_messages(ctx: &mut Context, v: &Value) -> JsResult<Vec<ChatMessage>> {
    let n = ctx.length_of(v)? as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let entry = ctx.get_property(v, &quickrs_core::value::PropKey::Index(i as u32))?;
        let role_v = field(ctx, &entry, "role")?;
        let content_v = field(ctx, &entry, "content")?;
        if is_nullish(&role_v) || is_nullish(&content_v) {
            return ctx.throw_type(&format!("renderChat: message {i} needs {{role, content}}"));
        }
        let role = ctx.to_rust_string(&role_v)?;
        let content = ctx.to_rust_string(&content_v)?;
        out.push(match role.trim().to_ascii_lowercase().as_str() {
            "system" => ChatMessage::system(content),
            "assistant" => ChatMessage::assistant(content),
            "user" => ChatMessage::user(content),
            other => {
                return ctx.throw_type(&format!(
                    "renderChat: message {i} has role '{other}' (system, user, assistant)"
                ));
            }
        });
    }
    Ok(out)
}

/// Pull `chat_template` out of a HuggingFace `tokenizer_config.json`.
///
/// Newer checkpoints ship a list of named templates (`default`, `tool_use`, …)
/// rather than one string; the `default` entry wins, else the first.
fn template_from_tokenizer_config(text: &str) -> Option<String> {
    let root: serde_json::Value = serde_json::from_str(text).ok()?;
    let value = root.get("chat_template")?;
    if let Some(s) = value.as_str() {
        return Some(s.to_string());
    }
    let entries = value.as_array()?;
    let pick = entries
        .iter()
        .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))
        .or_else(|| entries.first())?;
    Some(pick.get("template")?.as_str()?.to_string())
}

/// `renderChat(source, messages, addGenerationPrompt?)`.
///
/// `source` may be a `.gguf` (template read from its metadata), a HuggingFace
/// `tokenizer_config.json`, a `.jinja` file, or the template text itself.
/// Resolved in that order, so a path always beats being mistaken for inline
/// Jinja.
fn f_render_chat(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let source = ctx.to_rust_string(arg(args, 0))?;
    let messages = read_messages(ctx, arg(args, 1))?;
    let add_generation_prompt = if is_nullish(arg(args, 2)) {
        true
    } else {
        to_bool(arg(args, 2))
    };

    // A template *string* needs no permission; a path does. Permission is
    // checked *before* the stat, so a sandboxed script cannot probe for the
    // existence of a file by passing its path here.
    let path = Path::new(&source);
    let is_path = handle::fs_allowed(ctx) && path.is_file();
    let template = if is_path {
        let is_gguf = source.to_ascii_lowercase().ends_with(".gguf");
        if is_gguf {
            match ChatTemplate::from_gguf(path) {
                Ok(t) => t,
                Err(e) => return ctx.throw_internal(&format!("renderChat('{source}'): {e:#}")),
            }
        } else {
            let text = match std::fs::read_to_string(path) {
                Ok(t) => t,
                Err(e) => return ctx.throw_internal(&format!("renderChat('{source}'): {e}")),
            };
            // A tokenizer_config.json holds the template under a key; a .jinja
            // file *is* the template.
            let jinja = template_from_tokenizer_config(&text).unwrap_or(text);
            match ChatTemplate::from_source(jinja) {
                Ok(t) => t,
                Err(e) => {
                    return ctx.throw_internal(&format!(
                        "renderChat('{source}'): no usable chat template: {e:#}"
                    ));
                }
            }
        }
    } else {
        match ChatTemplate::from_source(source.clone()) {
            Ok(t) => t,
            Err(e) => {
                return ctx.throw_internal(&format!(
                    "renderChat: '{source}' is neither a readable file nor a valid template: {e:#}"
                ));
            }
        }
    };
    match template.render(&messages, add_generation_prompt) {
        Ok(text) => Ok(new_string(ctx, &text)),
        Err(e) => ctx.throw_internal(&format!("renderChat render: {e:#}")),
    }
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    // No constructor: a tokenizer only ever comes from a file.
    js_class! {
        ctx, namespace;
        name: "Tokenizer",
        class: CLASS_TOKENIZER,
        methods: {
            "encode" => m_encode, 2;
            "decode" => m_decode, 2;
            "decodeIncremental" => m_decode_incremental, 1;
            "finishStream" => m_finish_stream, 0;
            "resetStream" => m_reset_stream, 0;
            "toString" => m_to_string, 0;
        }
    };
    js_class! {
        ctx, namespace;
        name: "Sampler",
        class: CLASS_SAMPLER,
        ctor: sampler_new,
        methods: {
            "next" => m_sampler_next, 1;
            "prime" => m_sampler_prime, 1;
            "reset" => m_sampler_reset, 1;
            "history" => m_sampler_history, 0;
            "toString" => m_sampler_to_string, 0;
        }
    };
    js_functions! {
        ctx, namespace;
        "loadTokenizer" => f_load_tokenizer, 1;
        "argmax" => f_argmax, 1;
        "sampleNext" => f_sample_next, 3;
        "renderChat" => f_render_chat, 3;
    };
}
