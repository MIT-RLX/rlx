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

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_bind::runtime::text::{self, Sampler, Tokenizer};

use crate::api::bind::{lift, to_json};
use crate::convert::*;
use crate::handle::{self, CLASS_SAMPLER, CLASS_TOKENIZER};

fn tokenizer_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut Tokenizer> {
    handle::borrow_mut(ctx, this, CLASS_TOKENIZER, "rlx.Tokenizer")
}

fn ids_value(ctx: &mut Context, ids: &[u32]) -> Value {
    let values: Vec<Value> = ids.iter().map(|id| Value::number(*id as f64)).collect();
    new_array(ctx, values)
}

/// `loadTokenizer(path)` — a `tokenizer.json`, a GGUF file, or a model dir.
fn f_load_tokenizer(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "loadTokenizer")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let tok = lift(ctx, Tokenizer::load(&path))?;
    Ok(handle::wrap(ctx, CLASS_TOKENIZER, tok))
}

/// `encode(text, addSpecial = true)` → ids.
fn m_encode(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let text = ctx.to_rust_string(arg(args, 0))?;
    let add_special = is_nullish(arg(args, 1)) || to_bool(arg(args, 1));
    let r = tokenizer_of(ctx, this)?.encode(&text, add_special);
    let ids = lift(ctx, r)?;
    Ok(ids_value(ctx, &ids))
}

/// `decode(ids, skipSpecial = true)` → text.
fn m_decode(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let ids = to_u32_vec(ctx, arg(args, 0), "decode ids")?;
    let skip_special = is_nullish(arg(args, 1)) || to_bool(arg(args, 1));
    let r = tokenizer_of(ctx, this)?.decode(&ids, skip_special);
    let text = lift(ctx, r)?;
    Ok(new_string(ctx, &text))
}

/// Feed one generated id; returns only the text that is now final.
fn m_decode_incremental(
    ctx: &mut Context,
    this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    let id = to_usize(ctx, arg(args, 0), "decodeIncremental id")? as u32;
    let r = tokenizer_of(ctx, this)?.decode_incremental(id);
    let delta = lift(ctx, r)?;
    Ok(new_string(ctx, &delta))
}

/// Whatever the stream still holds back, flushed.
fn m_finish_stream(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let r = tokenizer_of(ctx, this)?.finish_stream();
    let tail = lift(ctx, r)?;
    Ok(new_string(ctx, &tail))
}

fn m_reset_stream(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    tokenizer_of(ctx, this)?.reset_stream();
    Ok(Value::Undefined)
}

fn m_to_string(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let text = format!(
        "[rlx.Tokenizer streamed={}]",
        tokenizer_of(ctx, this)?.streamed()
    );
    Ok(new_string(ctx, &text))
}

// ── sampling ────────────────────────────────────────────────

fn f_argmax(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let logits = to_f32_vec(ctx, arg(args, 0), "argmax logits")?;
    let id = lift(ctx, text::argmax(&logits))?;
    Ok(Value::number(id as f64))
}

/// `sampleNext(logits, history, {temperature, topK, topP, …, seed})` → id.
/// No options means greedy.
fn f_sample_next(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let logits = to_f32_vec(ctx, arg(args, 0), "sampleNext logits")?;
    let history = if is_nullish(arg(args, 1)) {
        Vec::new()
    } else {
        to_u32_vec(ctx, arg(args, 1), "sampleNext history")?
    };
    let opts = to_json(ctx, arg(args, 2))?;
    let id = lift(ctx, text::sample_next(&logits, &history, &opts))?;
    Ok(Value::number(id as f64))
}

fn sampler_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut Sampler> {
    handle::borrow_mut(ctx, this, CLASS_SAMPLER, "rlx.Sampler")
}

/// `new rlx.Sampler({temperature, topK, …, seed})`.
fn sampler_new(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let opts = to_json(ctx, arg(args, 0))?;
    let sampler = lift(ctx, Sampler::new(&opts))?;
    Ok(handle::wrap(ctx, CLASS_SAMPLER, sampler))
}

fn m_sampler_next(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let logits = to_f32_vec(ctx, arg(args, 0), "Sampler.next logits")?;
    let r = sampler_of(ctx, this)?.next(&logits);
    let id = lift(ctx, r)?;
    Ok(Value::number(id as f64))
}

fn m_sampler_prime(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let ids = to_u32_vec(ctx, arg(args, 0), "Sampler.prime ids")?;
    let n = sampler_of(ctx, this)?.prime(&ids);
    Ok(Value::number(n as f64))
}

fn m_sampler_reset(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let seed = if is_nullish(arg(args, 0)) {
        None
    } else {
        Some(to_u64(ctx, arg(args, 0), "Sampler.reset seed")?)
    };
    sampler_of(ctx, this)?.reset(seed);
    Ok(Value::Undefined)
}

fn m_sampler_history(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let ids = sampler_of(ctx, this)?.history().to_vec();
    Ok(ids_value(ctx, &ids))
}

fn m_sampler_to_string(ctx: &mut Context, this: &Value, _a: &[Value], _m: i32) -> JsResult<Value> {
    let text = {
        let s = sampler_of(ctx, this)?;
        format!(
            "[rlx.Sampler temperature={} topP={} generated={}]",
            s.opts.temperature,
            s.opts.top_p,
            s.history().len()
        )
    };
    Ok(new_string(ctx, &text))
}

// ── chat templates ──────────────────────────────────────────

/// `renderChat(source, [{role, content}], addGenerationPrompt = true)` —
/// `source` is a template, or (with file access) a `.gguf`,
/// `tokenizer_config.json` or `.jinja` path.
fn f_render_chat(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let source = ctx.to_rust_string(arg(args, 0))?;
    let messages = to_json(ctx, arg(args, 1))?;
    let add_generation_prompt = is_nullish(arg(args, 2)) || to_bool(arg(args, 2));
    let allow_files = handle::fs_allowed(ctx);
    let out = lift(
        ctx,
        text::render_chat(&source, &messages, add_generation_prompt, allow_files),
    )?;
    Ok(new_string(ctx, &out))
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
