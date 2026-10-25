// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tokenizers, sampling and chat templates (feature `text`).

use std::ffi::c_char;

use rlx_bind::runtime::text::{self, Sampler, Tokenizer};

use crate::ffi::{RlxU32s, arg, borrow, guard, handle, out, release};

/// A tokenizer plus one streaming decode's state.
pub struct RlxTokenizer(Tokenizer);

/// A sampler carrying its history and RNG across tokens.
pub struct RlxSampler(Sampler);

/// Load a `tokenizer.json`, a GGUF file, or a model directory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_tokenizer_load(
    path: *const c_char,
    out: *mut *mut RlxTokenizer,
) -> i32 {
    guard(|| unsafe {
        let path = arg::str(path, "rlx_tokenizer_load path")?;
        out::put(
            out,
            handle(RlxTokenizer(Tokenizer::load(path)?)),
            "rlx_tokenizer_load out",
        )
    })
}

/// Free a tokenizer (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_tokenizer_free(t: *mut RlxTokenizer) {
    unsafe { release(t) }
}

/// Encode `text`; `add_special` adds BOS / template tokens (usually `true`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_tokenizer_encode(
    t: *mut RlxTokenizer,
    text: *const c_char,
    add_special: bool,
    out: *mut RlxU32s,
) -> i32 {
    guard(|| unsafe {
        let text = arg::str(text, "rlx_tokenizer_encode text")?;
        let ids = borrow(t, "tokenizer")?.0.encode(text, add_special)?;
        out::put(out, ids.into(), "rlx_tokenizer_encode out")
    })
}

/// Decode `n` ids; `skip_special` drops special tokens (usually `true`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_tokenizer_decode(
    t: *mut RlxTokenizer,
    ids: *const u32,
    n: usize,
    skip_special: bool,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let ids = arg::slice(ids, n, "rlx_tokenizer_decode ids")?;
        let text = borrow(t, "tokenizer")?.0.decode(ids, skip_special)?;
        out::string(out, &text, "rlx_tokenizer_decode out")
    })
}

/// Feed one generated id; writes only the text that is now final.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_tokenizer_decode_incremental(
    t: *mut RlxTokenizer,
    id: u32,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let delta = borrow(t, "tokenizer")?.0.decode_incremental(id)?;
        out::string(out, &delta, "rlx_tokenizer_decode_incremental out")
    })
}

/// Flush whatever the stream still holds back.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_tokenizer_finish_stream(
    t: *mut RlxTokenizer,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let tail = borrow(t, "tokenizer")?.0.finish_stream()?;
        out::string(out, &tail, "rlx_tokenizer_finish_stream out")
    })
}

/// Start a new stream.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_tokenizer_reset_stream(t: *mut RlxTokenizer) -> i32 {
    guard(|| unsafe {
        borrow(t, "tokenizer")?.0.reset_stream();
        Ok(())
    })
}

/// Index of the largest of `n` logits.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_argmax(logits: *const f32, n: usize, out: *mut u32) -> i32 {
    guard(|| unsafe {
        let logits = arg::slice(logits, n, "rlx_argmax logits")?;
        out::put(out, text::argmax(logits)?, "rlx_argmax out")
    })
}

/// One token from `n` logits, penalized by `n_history` past ids.
/// `options_json` (`NULL` = greedy): `{"temperature": 0.7, "top_p": 0.9, "seed": 1}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_sample_next(
    logits: *const f32,
    n: usize,
    history: *const u32,
    n_history: usize,
    options_json: *const c_char,
    out: *mut u32,
) -> i32 {
    guard(|| unsafe {
        let logits = arg::slice(logits, n, "rlx_sample_next logits")?;
        let history = arg::slice(history, n_history, "rlx_sample_next history")?;
        let opts = arg::json(options_json, "rlx_sample_next options")?;
        out::put(
            out,
            text::sample_next(logits, history, &opts)?,
            "rlx_sample_next out",
        )
    })
}

/// A sampler; `options_json` as for `rlx_sample_next`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_sampler_new(
    options_json: *const c_char,
    out: *mut *mut RlxSampler,
) -> i32 {
    guard(|| unsafe {
        let opts = arg::json(options_json, "rlx_sampler_new options")?;
        out::put(
            out,
            handle(RlxSampler(Sampler::new(&opts)?)),
            "rlx_sampler_new out",
        )
    })
}

/// Free a sampler (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_sampler_free(s: *mut RlxSampler) {
    unsafe { release(s) }
}

/// The next token from `n` logits (appended to the history).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_sampler_next(
    s: *mut RlxSampler,
    logits: *const f32,
    n: usize,
    out: *mut u32,
) -> i32 {
    guard(|| unsafe {
        let logits = arg::slice(logits, n, "rlx_sampler_next logits")?;
        let id = borrow(s, "sampler")?.0.next(logits)?;
        out::put(out, id, "rlx_sampler_next out")
    })
}

/// Replace the history with a prompt's ids, so penalties see it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_sampler_prime(s: *mut RlxSampler, ids: *const u32, n: usize) -> i32 {
    guard(|| unsafe {
        let ids = arg::slice(ids, n, "rlx_sampler_prime ids")?;
        borrow(s, "sampler")?.0.prime(ids);
        Ok(())
    })
}

/// Clear the history; reseed when `seed` is not `NULL`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_sampler_reset(s: *mut RlxSampler, seed: *const u64) -> i32 {
    guard(|| unsafe {
        borrow(s, "sampler")?.0.reset(arg::opt(seed));
        Ok(())
    })
}

/// The ids sampled (and primed) so far.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_sampler_history(s: *mut RlxSampler, out: *mut RlxU32s) -> i32 {
    guard(|| unsafe {
        let ids = borrow(s, "sampler")?.0.history().to_vec();
        out::put(out, ids.into(), "rlx_sampler_history out")
    })
}

/// Render `messages_json` (`[{"role": "user", "content": "hi"}]`) with a chat
/// template: `source` is the template text, or a `.gguf`,
/// `tokenizer_config.json` or `.jinja` path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_render_chat(
    source: *const c_char,
    messages_json: *const c_char,
    add_generation_prompt: bool,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let source = arg::str(source, "rlx_render_chat source")?;
        let messages = arg::json(messages_json, "rlx_render_chat messages")?;
        let text = text::render_chat(source, &messages, add_generation_prompt, true)?;
        out::string(out, &text, "rlx_render_chat out")
    })
}
