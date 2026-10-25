// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx.Model` — a model from any provider (an `.rlxp` package, a plugin's
//! family runner such as an LLM), opened by path or id and run by
//! name.
//!
//! ```js
//! rlx.saveModel(g, "affine.rlxp", { w: [2, 0, 0, 3] }, { outputs: ["y"] });
//! const m = rlx.Model.open("affine.rlxp", { device: "cpu" });
//! const [y] = m.run({ x: [1, 1] });
//! // by task: m.chat([{ role: "user", content: "Hi" }]) · m.transcribe(audio, 16000)
//! //          m.synthesize("Hi") → {audio, sampleRate} · m.embed(["a", "b"]) → [Float32Array, …]
//! ```

use quickrs_core::context::Context;
use quickrs_core::gc::Gc;
use quickrs_core::object::JsObject;
use quickrs_core::value::{JsResult, Value};
use rlx_bind::runtime::model::{self, Model};

use crate::api::bind::{from_json, lift, report, to_json};
use crate::api::graph::graph_ref;
use crate::convert::*;
use crate::handle::{self, CLASS_MODEL};

fn model_of<'a>(ctx: &mut Context, this: &Value) -> JsResult<&'a mut Model> {
    handle::borrow_mut(ctx, this, CLASS_MODEL, "rlx.Model")
}

/// `{name: number[] | Float32Array}` → owned pairs.
fn f32_bag(ctx: &mut Context, bag: &Value, what: &str) -> JsResult<Vec<(String, Vec<f32>)>> {
    let mut out = Vec::new();
    for name in own_keys(ctx, bag)? {
        let v = field(ctx, bag, &name)?;
        out.push((
            name.clone(),
            to_f32_vec(ctx, &v, &format!("{what}('{name}')"))?,
        ));
    }
    Ok(out)
}

fn pairs(owned: &[(String, Vec<f32>)]) -> Vec<(&str, &[f32])> {
    owned
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect()
}

/// `Model.open(source, {device, provider, …})`.
fn model_open(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "Model.open")? {
        return Ok(refused);
    }
    let source = ctx.to_rust_string(arg(args, 0))?;
    let opts = to_json(ctx, arg(args, 1))?;
    let m = lift(ctx, Model::open(&source, &opts))?;
    Ok(handle::wrap(ctx, CLASS_MODEL, m))
}

fn m_info(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let info = model_of(ctx, this)?.info();
    Ok(report(ctx, &info))
}

fn m_task(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let task = model_of(ctx, this)?.task();
    Ok(new_string(ctx, &task))
}

/// `run({x: […]}, {shapes: {x: [1, 2]}})` → one Float32Array per output.
fn m_run(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let owned = f32_bag(ctx, arg(args, 0), "Model.run")?;
    let opts = to_json(ctx, arg(args, 1))?;
    let r = model_of(ctx, this)?.run(&pairs(&owned), &opts);
    let outs = lift(ctx, r)?;
    let values: Vec<Value> = outs.iter().map(|o| new_f32_array(ctx, o)).collect();
    Ok(new_array(ctx, values))
}

/// `runTyped({x: {data, dtype}}, options)` → `[{data: Uint8Array, dtype}, …]`.
fn m_run_typed(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let bag = arg(args, 0).clone();
    let mut owned: Vec<(String, Vec<u8>, String)> = Vec::new();
    for name in own_keys(ctx, &bag)? {
        let entry = field(ctx, &bag, &name)?;
        let data_v = field(ctx, &entry, "data")?;
        let dtype_v = field(ctx, &entry, "dtype")?;
        if is_nullish(&data_v) || is_nullish(&dtype_v) {
            return ctx.throw_type(&format!(
                "Model.runTyped input '{name}': expected {{data: TypedArray, dtype: \"i32\"}}"
            ));
        }
        let bytes = to_bytes(ctx, &data_v, &format!("Model.runTyped('{name}')"))?;
        let label = ctx.to_rust_string(&dtype_v)?;
        owned.push((name, bytes, label));
    }
    let opts = to_json(ctx, arg(args, 1))?;
    let outs = {
        let refs: Vec<(&str, &[u8], &str)> = owned
            .iter()
            .map(|(n, d, t)| (n.as_str(), d.as_slice(), t.as_str()))
            .collect();
        let r = model_of(ctx, this)?.run_typed(&refs, &opts);
        lift(ctx, r)?
    };
    let values: Vec<Value> = outs
        .into_iter()
        .map(|(bytes, dtype)| {
            let data = new_u8_array(ctx, &bytes);
            let label = new_string(ctx, dtype);
            new_object(ctx, vec![("data", data), ("dtype", label)])
        })
        .collect();
    Ok(new_array(ctx, values))
}

fn m_output_shapes(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let shapes = model_of(ctx, this)?.output_shapes();
    let values: Vec<Value> = shapes.iter().map(|s| new_usize_array(ctx, s)).collect();
    Ok(new_array(ctx, values))
}

fn m_output_names(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let names = model_of(ctx, this)?.output_names();
    Ok(new_string_array(ctx, &names))
}

fn ids_value(ctx: &mut Context, ids: &[u32]) -> Value {
    let values: Vec<Value> = ids.iter().map(|id| Value::number(*id as f64)).collect();
    new_array(ctx, values)
}

fn m_predict_logits(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let ids = to_u32_vec(ctx, arg(args, 0), "Model.predictLogits ids")?;
    let r = model_of(ctx, this)?.predict_logits(&ids);
    let logits = lift(ctx, r)?;
    Ok(new_f32_array(ctx, &logits))
}

/// `generate(ids, maxTokens = 16, {temperature, topP, topK, seed, stopIds})`.
fn m_generate(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let ids = to_u32_vec(ctx, arg(args, 0), "Model.generate ids")?;
    let max_tokens = if is_nullish(arg(args, 1)) {
        16
    } else {
        to_usize(ctx, arg(args, 1), "Model.generate maxTokens")?
    };
    let opts = to_json(ctx, arg(args, 2))?;
    let r = model_of(ctx, this)?.generate(&ids, max_tokens, &opts);
    let toks = lift(ctx, r)?;
    Ok(ids_value(ctx, &toks))
}

fn m_generate_text(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let prompt = ctx.to_rust_string(arg(args, 0))?;
    let opts = to_json(ctx, arg(args, 1))?;
    let r = model_of(ctx, this)?.generate_text(&prompt, &opts);
    let text = lift(ctx, r)?;
    Ok(new_string(ctx, &text))
}

/// `chat([{role, content}, …] | "text", {maxTokens, temperature, …})`.
fn m_chat(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let messages = to_json(ctx, arg(args, 0))?;
    let opts = to_json(ctx, arg(args, 1))?;
    let r = model_of(ctx, this)?.chat(&messages, &opts);
    let text = lift(ctx, r)?;
    Ok(new_string(ctx, &text))
}

/// `transcribe(audio, sampleRate = 16000, options)` → the text.
fn m_transcribe(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let audio = to_f32_vec(ctx, arg(args, 0), "Model.transcribe audio")?;
    let rate = if is_nullish(arg(args, 1)) {
        16_000
    } else {
        to_usize(ctx, arg(args, 1), "Model.transcribe sampleRate")? as u32
    };
    let opts = to_json(ctx, arg(args, 2))?;
    let r = model_of(ctx, this)?.transcribe(&audio, rate, &opts);
    let text = lift(ctx, r)?;
    Ok(new_string(ctx, &text))
}

/// `synthesize(text, options)` → `{audio: Float32Array, sampleRate}`.
fn m_synthesize(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let text = ctx.to_rust_string(arg(args, 0))?;
    let opts = to_json(ctx, arg(args, 1))?;
    let r = model_of(ctx, this)?.synthesize(&text, &opts);
    let (audio, rate) = lift(ctx, r)?;
    let audio = new_f32_array(ctx, &audio);
    Ok(new_object(
        ctx,
        vec![("audio", audio), ("sampleRate", Value::number(rate as f64))],
    ))
}

/// `embed(texts | "text", options)` → one Float32Array per text.
fn m_embed(ctx: &mut Context, this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    let texts = if arg(args, 0).is_string() {
        vec![ctx.to_rust_string(arg(args, 0))?]
    } else {
        to_string_vec(ctx, arg(args, 0))?
    };
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let opts = to_json(ctx, arg(args, 1))?;
    let r = model_of(ctx, this)?.embed(&refs, &opts);
    let rows = lift(ctx, r)?;
    let rows: Vec<Vec<f32>> = rlx_bind::runtime::serde_json::from_value(rows).unwrap_or_default();
    let values: Vec<Value> = rows.iter().map(|r| new_f32_array(ctx, r)).collect();
    Ok(new_array(ctx, values))
}

fn m_to_string(ctx: &mut Context, this: &Value, _args: &[Value], _m: i32) -> JsResult<Value> {
    let info = model_of(ctx, this)?.info();
    let s = |k: &str| {
        info.get(k)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    };
    let text = format!(
        "[rlx.Model {} task={} provider={} device={}]",
        s("family"),
        s("task"),
        s("provider"),
        s("device")
    );
    Ok(new_string(ctx, &text))
}

/// `modelProviders()` → `[{name, description, tasks, families}]`.
fn f_model_providers(
    ctx: &mut Context,
    _this: &Value,
    _args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    Ok(from_json(ctx, &model::providers()))
}

/// `loadModelPlugin(path)` → its providers' names.
fn f_load_model_plugin(
    ctx: &mut Context,
    _this: &Value,
    args: &[Value],
    _m: i32,
) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "loadModelPlugin")? {
        return Ok(refused);
    }
    let path = ctx.to_rust_string(arg(args, 0))?;
    let names = lift(ctx, model::load_plugin(&path))?;
    Ok(new_string_array(ctx, &names))
}

/// `saveModel(graph, path, {w: […]}, {task, family, outputs, metadata})`.
fn f_save_model(ctx: &mut Context, _this: &Value, args: &[Value], _m: i32) -> JsResult<Value> {
    if let Some(refused) = handle::require_fs(ctx, "saveModel")? {
        return Ok(refused);
    }
    let graph_v = arg(args, 0).clone();
    let path = ctx.to_rust_string(arg(args, 1))?;
    let owned = f32_bag(ctx, arg(args, 2), "saveModel")?;
    let meta = to_json(ctx, arg(args, 3))?;
    let graph = graph_ref(ctx, &graph_v)?;
    lift(ctx, model::save(graph, &path, &pairs(&owned), &meta))?;
    Ok(Value::Undefined)
}

pub fn install(ctx: &mut Context, namespace: &Gc<JsObject>) {
    // No constructor: a model only ever comes from `Model.open`.
    js_class! {
        ctx, namespace;
        name: "Model",
        class: CLASS_MODEL,
        methods: {
            "info" => m_info, 0;
            "task" => m_task, 0;
            "run" => m_run, 2;
            "runTyped" => m_run_typed, 2;
            "outputShapes" => m_output_shapes, 0;
            "outputNames" => m_output_names, 0;
            "predictLogits" => m_predict_logits, 1;
            "generate" => m_generate, 3;
            "generateText" => m_generate_text, 2;
            "chat" => m_chat, 2;
            "transcribe" => m_transcribe, 3;
            "synthesize" => m_synthesize, 2;
            "embed" => m_embed, 2;
            "toString" => m_to_string, 0;
        }
    };
    js_statics! {
        ctx, namespace, "Model";
        "open" => model_open, 2;
    };
    js_functions! {
        ctx, namespace;
        "modelProviders" => f_model_providers, 0;
        "loadModelPlugin" => f_load_model_plugin, 1;
        "saveModel" => f_save_model, 4;
    };
}
