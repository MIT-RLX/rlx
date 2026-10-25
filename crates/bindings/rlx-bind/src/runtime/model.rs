// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Model`: a model from any provider — an `.rlxp` package, a plugin's family
//! runner (an LLM, a speech or vision model, …) — opened by path or id and run by
//! name (feature `models`).
//!
//! The binding-neutral form of `rlx_runtime::model`: tensors as f32 slices or
//! typed bytes, options and reports as JSON. The language-model calls
//! (`predict_logits`, `generate`) go through the model's `run` with an
//! `input_ids` tensor, so they work the same for a model in a plugin.

use rlx_ir::{DType, Graph, Op};
use rlx_runtime::model::{self as m, Tensor};
use serde_json::{Value, json};

use crate::enums::{dtype, dtype_label};
use crate::panics::catch;
use crate::{ApiError, ApiResult};

fn err(what: &str) -> impl Fn(anyhow::Error) -> ApiError + '_ {
    move |e| ApiError::runtime(format!("{what}: {e:#}"))
}

/// A loaded model, with the shapes and names of its last outputs.
pub struct Model {
    inner: Box<dyn m::Model>,
    last_shapes: Vec<Vec<usize>>,
    last_names: Vec<String>,
}

impl Model {
    /// Open `source`. Options: `{device, provider, …}` (the provider's own).
    pub fn open(source: &str, options: &Value) -> ApiResult<Self> {
        // A package's `tokenizer.json` tokenizes with rlx-text.
        #[cfg(feature = "text")]
        rlx_text::install_model_tokenizer();
        let options = if options.is_null() {
            json!({})
        } else {
            options.clone()
        };
        if let Some(d) = options.get("device").and_then(Value::as_str) {
            super::session::device(d)?;
        }
        let what = format!("Model.open('{source}')");
        let req = m::ModelRequest::new(source, options).map_err(err(&what))?;
        let inner = catch(&what, || m::open(&req))?.map_err(err(&what))?;
        Ok(Self {
            inner,
            last_shapes: Vec::new(),
            last_names: Vec::new(),
        })
    }

    /// `{provider, family, task, device, inputs, outputs, metadata}`.
    pub fn info(&self) -> Value {
        self.inner.info().to_json()
    }

    /// The model's task (`"graph"`, `"text-generation"`, …).
    pub fn task(&self) -> String {
        self.inner.info().task
    }

    fn remember(&mut self, outs: &[Tensor]) {
        self.last_shapes = outs.iter().map(|t| t.shape.clone()).collect();
        self.last_names = outs.iter().map(|t| t.name.clone()).collect();
    }

    /// Shape of input `name`: `options.shapes[name]`, else the model's static
    /// shape, else flat.
    fn shape_of(&self, info: &m::ModelInfo, options: &Value, name: &str, len: usize) -> Vec<usize> {
        if let Some(s) = options
            .get("shapes")
            .and_then(|s| s.get(name))
            .and_then(Value::as_array)
        {
            return s
                .iter()
                .filter_map(|d| d.as_u64().map(|d| d as usize))
                .collect();
        }
        if let Some(spec) = info.inputs.iter().find(|s| s.name == name)
            && spec.shape.iter().all(Option::is_some)
            && !spec.shape.is_empty()
        {
            return spec.shape.iter().map(|d| d.unwrap()).collect();
        }
        vec![len]
    }

    fn run_tensors(&mut self, inputs: Vec<Tensor>, options: &Value) -> ApiResult<Vec<Tensor>> {
        let options = if options.is_null() {
            json!({})
        } else {
            options.clone()
        };
        let outs =
            catch("Model.run", || self.inner.run(&inputs, &options))?.map_err(err("Model.run"))?;
        self.remember(&outs);
        Ok(outs)
    }

    /// f32 tensors in, f32 tensors out (a non-f32 output is an error: use
    /// [`Model::run_typed`]).
    pub fn run(&mut self, inputs: &[(&str, &[f32])], options: &Value) -> ApiResult<Vec<Vec<f32>>> {
        let info = self.inner.info();
        let ts = inputs
            .iter()
            .map(|(name, data)| {
                Tensor::f32(*name, self.shape_of(&info, options, name, data.len()), data)
            })
            .collect();
        self.run_tensors(ts, options)?
            .into_iter()
            .map(|t| {
                t.to_f32().map_err(|_| {
                    ApiError::type_(format!(
                        "Model.run: output '{}' is {}; use run_typed",
                        t.name,
                        dtype_label(t.dtype)
                    ))
                })
            })
            .collect()
    }

    /// Typed tensors in (bytes + dtype), typed out.
    pub fn run_typed(
        &mut self,
        inputs: &[(&str, &[u8], &str)],
        options: &Value,
    ) -> ApiResult<Vec<(Vec<u8>, &'static str)>> {
        let info = self.inner.info();
        let mut ts = Vec::with_capacity(inputs.len());
        for (name, data, dt) in inputs {
            let d = dtype(dt)?;
            let n = data.len() / d.size_bytes().max(1);
            ts.push(Tensor::new(
                *name,
                d,
                self.shape_of(&info, options, name, n),
                data.to_vec(),
            ));
        }
        Ok(self
            .run_tensors(ts, options)?
            .into_iter()
            .map(|t| (t.data, dtype_label(t.dtype)))
            .collect())
    }

    /// Dims of each output of the last run.
    pub fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.last_shapes.clone()
    }

    /// Names of the outputs of the last run.
    pub fn output_names(&self) -> Vec<String> {
        self.last_names.clone()
    }

    fn ids(ids: &[u32]) -> Tensor {
        let data = ids.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect();
        Tensor::new("input_ids", DType::I32, vec![ids.len()], data)
    }

    /// A language model's last-position logits after `ids`.
    pub fn predict_logits(&mut self, ids: &[u32]) -> ApiResult<Vec<f32>> {
        let outs = self.run_tensors(vec![Self::ids(ids)], &json!({}))?;
        let t = outs
            .into_iter()
            .find(|t| t.name == "logits")
            .ok_or_else(|| ApiError::type_("predict_logits: the model returned no `logits`"))?;
        t.to_f32().map_err(|e| ApiError::type_(e.to_string()))
    }

    /// A language model's next `max_tokens` ids after `ids`. Options:
    /// sampling (`temperature`, `top_p`, `top_k`, `seed`, `stop_ids`, …).
    pub fn generate(
        &mut self,
        ids: &[u32],
        max_tokens: usize,
        options: &Value,
    ) -> ApiResult<Vec<u32>> {
        let mut o = if options.is_object() {
            options.clone()
        } else {
            json!({})
        };
        o["max_tokens"] = json!(max_tokens);
        let outs = self.run_tensors(vec![Self::ids(ids)], &o)?;
        let t = outs
            .into_iter()
            .find(|t| t.name == "tokens")
            .ok_or_else(|| ApiError::type_("generate: the model returned no `tokens`"))?;
        m::ids_of(&t).map_err(|e| ApiError::type_(e.to_string()))
    }

    /// Text in, text out, for a model with a tokenizer.
    pub fn generate_text(&mut self, prompt: &str, options: &Value) -> ApiResult<String> {
        let options = if options.is_null() {
            json!({})
        } else {
            options.clone()
        };
        catch("Model.generate_text", || {
            self.inner.generate_text(prompt, &options)
        })?
        .map_err(err("Model.generate_text"))
    }
    /// A reply to `messages` (`[{role, content}, …]`, or one string), with
    /// the model's chat template applied.
    pub fn chat(&mut self, messages: &Value, options: &Value) -> ApiResult<String> {
        let messages = m::ChatMessage::list_from_json(messages)
            .map_err(|e| ApiError::type_(format!("Model.chat: {e}")))?;
        let options = opts(options);
        catch("Model.chat", || self.inner.chat(&messages, &options))?.map_err(err("Model.chat"))
    }

    /// The text of mono `audio` at `sample_rate`.
    pub fn transcribe(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        options: &Value,
    ) -> ApiResult<String> {
        let options = opts(options);
        catch("Model.transcribe", || {
            self.inner.transcribe(audio, sample_rate, &options)
        })?
        .map_err(err("Model.transcribe"))
    }

    /// `text` spoken: mono samples and their rate.
    pub fn synthesize(&mut self, text: &str, options: &Value) -> ApiResult<(Vec<f32>, u32)> {
        let options = opts(options);
        let a = catch("Model.synthesize", || self.inner.synthesize(text, &options))?
            .map_err(err("Model.synthesize"))?;
        Ok((a.samples, a.sample_rate))
    }

    /// One vector per text, as `[[f32, …], …]`.
    pub fn embed(&mut self, texts: &[&str], options: &Value) -> ApiResult<Value> {
        let options = opts(options);
        let rows = catch("Model.embed", || self.inner.embed(texts, &options))?
            .map_err(err("Model.embed"))?;
        Ok(json!(rows))
    }
}

fn opts(options: &Value) -> Value {
    if options.is_null() {
        json!({})
    } else {
        options.clone()
    }
}

/// Every model provider: `[{name, description, tasks, families}]`.
pub fn providers() -> Value {
    m::providers()
}

/// Load a model plugin library; returns its providers' names.
pub fn load_plugin(path: &str) -> ApiResult<Vec<String>> {
    m::plugin::load_plugin(path).map_err(|e| ApiError::runtime(format!("{e:#}")))
}

/// Write `graph` and its f32 `weights` as an `.rlxp` model package `path`.
/// `model`: `{task, family, outputs: [names], metadata}`.
pub fn save(graph: &Graph, path: &str, weights: &[(&str, &[f32])], model: &Value) -> ApiResult<()> {
    let mut ts = Vec::with_capacity(weights.len());
    for (name, data) in weights {
        let shape = graph
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Param { name: p } if p == name))
            .map(|n| {
                n.shape
                    .dims()
                    .iter()
                    .map(|d| match d {
                        rlx_ir::Dim::Static(n) => *n,
                        rlx_ir::Dim::Dynamic(_) => data.len(),
                    })
                    .collect()
            })
            .ok_or_else(|| {
                ApiError::type_(format!("save_model: '{name}' is not a Param of this graph"))
            })?;
        ts.push(Tensor::f32(*name, shape, data));
    }
    let model = if model.is_null() {
        json!({})
    } else {
        model.clone()
    };
    m::rlxp::save_package(std::path::Path::new(path), graph, &ts, &model)
        .map_err(|e| ApiError::runtime(format!("save_model('{path}'): {e:#}")))
}
