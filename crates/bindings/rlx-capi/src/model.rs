// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Models from any provider — `.rlxp` packages, plugins with downstream
//! family runners — opened by path or id and run by name (feature `models`).

use std::ffi::{CString, c_char};

use rlx_bind::runtime::model::{self, Model};

use crate::ffi::{RlxF32s, RlxU32s, arg, borrow, guard, handle, out, release};
use crate::graph::{RlxGraph, graph_mut};
use crate::runtime::{RlxOutputs, RlxTypedOutputs};

/// A loaded model.
pub struct RlxModel(Model);

/// Every model provider, as JSON `[{name, description, tasks, families}]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_providers(out: *mut *mut c_char) -> i32 {
    guard(|| unsafe { out::json(out, &model::providers(), "rlx_model_providers out") })
}

/// Load a model plugin library; `*out` is its providers' names (JSON array).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_load_model_plugin(path: *const c_char, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let path = arg::str(path, "rlx_load_model_plugin path")?;
        out::strings(out, &model::load_plugin(path)?, "rlx_load_model_plugin out")
    })
}

/// Write `g` (borrowed) and `n` f32 weights as an `.rlxp` model package.
/// `model_json` (`NULL` = `{}`): `{task, family, outputs: [names], metadata}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_save_model(
    g: *mut RlxGraph,
    path: *const c_char,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    model_json: *const c_char,
) -> i32 {
    guard(|| unsafe {
        let path = arg::str(path, "rlx_save_model path")?;
        let weights = arg::inputs(names, data, lens, n, "rlx_save_model weights")?;
        let meta = arg::json(model_json, "rlx_save_model model")?;
        model::save(graph_mut(g)?, path, &weights, &meta)
    })
}

/// Open `source` (a path or id). `options_json` (`NULL` = `{}`):
/// `{device, provider, …the provider's own}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_open(
    source: *const c_char,
    options_json: *const c_char,
    out: *mut *mut RlxModel,
) -> i32 {
    guard(|| unsafe {
        let source = arg::str(source, "rlx_model_open source")?;
        let opts = arg::json(options_json, "rlx_model_open options")?;
        out::put(
            out,
            handle(RlxModel(Model::open(source, &opts)?)),
            "rlx_model_open out",
        )
    })
}

/// Free a model (`NULL` is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_free(m: *mut RlxModel) {
    unsafe { release(m) }
}

/// `{provider, family, task, device, inputs, outputs, metadata}` as JSON.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_info(m: *mut RlxModel, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe { out::json(out, &borrow(m, "model")?.0.info(), "rlx_model_info out") })
}

/// The model's task (`"graph"`, `"text-generation"`, …).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_task(m: *mut RlxModel, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe { out::string(out, &borrow(m, "model")?.0.task(), "rlx_model_task out") })
}

/// Run with `n` f32 inputs. `options_json` (`NULL` = `{}`): `{shapes: {name:
/// dims}, …}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_run(
    m: *mut RlxModel,
    names: *const *const c_char,
    data: *const *const f32,
    lens: *const usize,
    n: usize,
    options_json: *const c_char,
    out: *mut *mut RlxOutputs,
) -> i32 {
    guard(|| unsafe {
        let inputs = arg::inputs(names, data, lens, n, "rlx_model_run inputs")?;
        let opts = arg::json(options_json, "rlx_model_run options")?;
        let outs = borrow(m, "model")?.0.run(&inputs, &opts)?;
        out::put(out, handle(RlxOutputs { outs }), "rlx_model_run out")
    })
}

/// Run with `n` typed inputs (names, byte pointers, byte lengths, dtype names).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_run_typed(
    m: *mut RlxModel,
    names: *const *const c_char,
    data: *const *const u8,
    lens: *const usize,
    dtypes: *const *const c_char,
    n: usize,
    options_json: *const c_char,
    out: *mut *mut RlxTypedOutputs,
) -> i32 {
    guard(|| unsafe {
        let what = "rlx_model_run_typed inputs";
        let names = arg::slice(names, n, what)?;
        let data = arg::slice(data, n, what)?;
        let lens = arg::slice(lens, n, what)?;
        let dtypes = arg::slice(dtypes, n, what)?;
        let mut inputs = Vec::with_capacity(n);
        for i in 0..n {
            inputs.push((
                arg::str(names[i], what)?,
                arg::slice(data[i], lens[i], what)?,
                arg::str(dtypes[i], what)?,
            ));
        }
        let opts = arg::json(options_json, "rlx_model_run_typed options")?;
        let outs = borrow(m, "model")?.0.run_typed(&inputs, &opts)?;
        let labels = outs
            .iter()
            .map(|(_, d)| CString::new(*d).unwrap_or_default())
            .collect();
        out::put(
            out,
            handle(RlxTypedOutputs { outs, labels }),
            "rlx_model_run_typed out",
        )
    })
}

/// Dims of each output of the last run, as JSON (`[[1, 3], …]`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_output_shapes(m: *mut RlxModel, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        let v = rlx_bind::runtime::serde_json::json!(borrow(m, "model")?.0.output_shapes());
        out::json(out, &v, "rlx_model_output_shapes out")
    })
}

/// Names of the outputs of the last run, as a JSON array.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_output_names(m: *mut RlxModel, out: *mut *mut c_char) -> i32 {
    guard(|| unsafe {
        out::strings(
            out,
            &borrow(m, "model")?.0.output_names(),
            "rlx_model_output_names out",
        )
    })
}

/// A language model's last-position logits after `n` ids.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_predict_logits(
    m: *mut RlxModel,
    ids: *const u32,
    n: usize,
    out: *mut RlxF32s,
) -> i32 {
    guard(|| unsafe {
        let ids = arg::slice(ids, n, "rlx_model_predict_logits ids")?;
        let logits = borrow(m, "model")?.0.predict_logits(ids)?;
        out::put(out, logits.into(), "rlx_model_predict_logits out")
    })
}

/// A language model's next `max_tokens` ids after `n` ids. `options_json`
/// (`NULL` = greedy): `{temperature, top_p, top_k, seed, stop_ids}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_generate(
    m: *mut RlxModel,
    ids: *const u32,
    n: usize,
    max_tokens: usize,
    options_json: *const c_char,
    out: *mut RlxU32s,
) -> i32 {
    guard(|| unsafe {
        let ids = arg::slice(ids, n, "rlx_model_generate ids")?;
        let opts = arg::json(options_json, "rlx_model_generate options")?;
        let toks = borrow(m, "model")?.0.generate(ids, max_tokens, &opts)?;
        out::put(out, toks.into(), "rlx_model_generate out")
    })
}

/// Text in, text out, for a model with a tokenizer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_generate_text(
    m: *mut RlxModel,
    prompt: *const c_char,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let prompt = arg::str(prompt, "rlx_model_generate_text prompt")?;
        let opts = arg::json(options_json, "rlx_model_generate_text options")?;
        let text = borrow(m, "model")?.0.generate_text(prompt, &opts)?;
        out::string(out, &text, "rlx_model_generate_text out")
    })
}

/// A reply to `messages_json` (`[{role, content}, …]`, or one JSON string),
/// with the model's chat template applied. `options_json`: as
/// `rlx_model_generate_text`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_chat(
    m: *mut RlxModel,
    messages_json: *const c_char,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let messages = arg::json(messages_json, "rlx_model_chat messages")?;
        let opts = arg::json(options_json, "rlx_model_chat options")?;
        let text = borrow(m, "model")?.0.chat(&messages, &opts)?;
        out::string(out, &text, "rlx_model_chat out")
    })
}

/// The text of `n` mono samples of `audio` at `sample_rate`, for a
/// speech-recognition model.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_transcribe(
    m: *mut RlxModel,
    audio: *const f32,
    n: usize,
    sample_rate: u32,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let audio = arg::slice(audio, n, "rlx_model_transcribe audio")?;
        let opts = arg::json(options_json, "rlx_model_transcribe options")?;
        let text = borrow(m, "model")?
            .0
            .transcribe(audio, sample_rate, &opts)?;
        out::string(out, &text, "rlx_model_transcribe out")
    })
}

/// `text` spoken, for a text-to-speech model: mono samples (free with
/// `rlx_f32s_free`) and their rate.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_synthesize(
    m: *mut RlxModel,
    text: *const c_char,
    options_json: *const c_char,
    audio_out: *mut RlxF32s,
    sample_rate_out: *mut u32,
) -> i32 {
    guard(|| unsafe {
        let text = arg::str(text, "rlx_model_synthesize text")?;
        let opts = arg::json(options_json, "rlx_model_synthesize options")?;
        let (audio, rate) = borrow(m, "model")?.0.synthesize(text, &opts)?;
        out::put(
            sample_rate_out,
            rate,
            "rlx_model_synthesize sample_rate_out",
        )?;
        out::put(audio_out, audio.into(), "rlx_model_synthesize audio_out")
    })
}

/// One vector per text of `n` `texts`, as JSON `[[f32, …], …]`, for an
/// embedding model.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_model_embed(
    m: *mut RlxModel,
    texts: *const *const c_char,
    n: usize,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| unsafe {
        let texts = arg::strs(texts, n, "rlx_model_embed texts")?;
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let opts = arg::json(options_json, "rlx_model_embed options")?;
        let rows = borrow(m, "model")?.0.embed(&refs, &opts)?;
        out::json(out, &rows, "rlx_model_embed out")
    })
}
