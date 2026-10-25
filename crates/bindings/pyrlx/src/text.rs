// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tokenizers, chat templates and sampling (`rlx-text`) — the pieces a
//! language-model loop needs either side of the graph.
//!
//! ```python
//! tok = rlx.load_tokenizer("tokenizer.json")
//! prompt = rlx.render_chat("model.gguf", [{"role": "user", "content": "hi"}])
//! ids = tok.encode(prompt)
//! sampler = rlx.Sampler(temperature=0.7, top_p=0.9, seed=1)
//! sampler.prime(ids)
//! # … run the model, then per step:
//! out += tok.decode_incremental(sampler.next(logits))
//! out += tok.finish_stream()
//! ```
//!
//! `decode_incremental` exists because decoding a growing id list from scratch
//! each step is quadratic and wrong at a partial multi-byte character — the
//! streaming detokenizer holds the boundary state.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rlx_bind::runtime::text::{self, Sampler, Tokenizer};

use crate::bind::{opt_json, to_json, to_py};

/// A tokenizer plus one streaming decode's state. Unsendable: the stream is
/// per-thread state, and the handle is not `Sync`.
#[pyclass(name = "Tokenizer", module = "pyrlx._pyrlx", unsendable)]
pub(crate) struct PyTokenizer {
    inner: Tokenizer,
}

/// `load_tokenizer(path)` — a `tokenizer.json`, a GGUF file, or a model dir.
#[pyfunction]
pub(crate) fn load_tokenizer(path: &str) -> PyResult<PyTokenizer> {
    Ok(PyTokenizer {
        inner: Tokenizer::load(path).map_err(to_py)?,
    })
}

#[pymethods]
impl PyTokenizer {
    /// Special tokens default on: a prompt without BOS silently shifts every
    /// position.
    #[pyo3(signature = (text, add_special = true))]
    fn encode(&self, text: &str, add_special: bool) -> PyResult<Vec<u32>> {
        self.inner.encode(text, add_special).map_err(to_py)
    }

    #[pyo3(signature = (ids, skip_special = true))]
    fn decode(&self, ids: Vec<u32>, skip_special: bool) -> PyResult<String> {
        self.inner.decode(&ids, skip_special).map_err(to_py)
    }

    /// Feed one generated id; returns only the text that is now final.
    fn decode_incremental(&mut self, id: u32) -> PyResult<String> {
        self.inner.decode_incremental(id).map_err(to_py)
    }

    /// Whatever the stream still holds back, flushed.
    fn finish_stream(&mut self) -> PyResult<String> {
        self.inner.finish_stream().map_err(to_py)
    }

    fn reset_stream(&mut self) {
        self.inner.reset_stream();
    }

    fn __repr__(&self) -> String {
        format!("<pyrlx.Tokenizer streamed={}>", self.inner.streamed())
    }
}

fn kwargs_json(kw: Option<&Bound<'_, PyDict>>) -> PyResult<rlx_bind::runtime::Value> {
    match kw {
        Some(d) => to_json(d.as_any()),
        None => Ok(rlx_bind::runtime::Value::Null),
    }
}

/// Index of the largest logit.
#[pyfunction]
pub(crate) fn argmax(logits: Vec<f32>) -> PyResult<u32> {
    text::argmax(&logits).map_err(to_py)
}

/// One token from `logits`, penalized by `history`. Options: `temperature`,
/// `top_k`, `top_p`, `min_p`, `typical_p`, `repetition_penalty`,
/// `frequency_penalty`, `presence_penalty`, `seed`; none means greedy.
#[pyfunction]
#[pyo3(signature = (logits, history = vec![], **opts))]
pub(crate) fn sample_next(
    logits: Vec<f32>,
    history: Vec<u32>,
    opts: Option<&Bound<'_, PyDict>>,
) -> PyResult<u32> {
    let opts = kwargs_json(opts)?;
    text::sample_next(&logits, &history, &opts).map_err(to_py)
}

/// A sampler carrying its history and RNG across tokens.
#[pyclass(name = "Sampler", module = "pyrlx._pyrlx")]
pub(crate) struct PySampler {
    inner: Sampler,
}

#[pymethods]
impl PySampler {
    #[new]
    #[pyo3(signature = (**opts))]
    fn new(opts: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let opts = kwargs_json(opts)?;
        Ok(Self {
            inner: Sampler::new(&opts).map_err(to_py)?,
        })
    }

    fn next(&mut self, logits: Vec<f32>) -> PyResult<u32> {
        self.inner.next(&logits).map_err(to_py)
    }

    /// Replace the history with a prompt's ids, so penalties see it.
    fn prime(&mut self, ids: Vec<u32>) -> usize {
        self.inner.prime(&ids)
    }

    /// Clear the history; reseed when `seed` is given.
    #[pyo3(signature = (seed = None))]
    fn reset(&mut self, seed: Option<u64>) {
        self.inner.reset(seed);
    }

    fn history(&self) -> Vec<u32> {
        self.inner.history().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyrlx.Sampler temperature={} top_p={} generated={}>",
            self.inner.opts.temperature,
            self.inner.opts.top_p,
            self.inner.history().len()
        )
    }
}

/// Render `[{"role", "content"}]` with a chat template. `source` is the
/// template text, or a `.gguf`, `tokenizer_config.json` or `.jinja` path.
#[pyfunction]
#[pyo3(signature = (source, messages, add_generation_prompt = true))]
pub(crate) fn render_chat(
    source: &str,
    messages: &Bound<'_, PyAny>,
    add_generation_prompt: bool,
) -> PyResult<String> {
    let messages = opt_json(Some(messages))?;
    text::render_chat(source, &messages, add_generation_prompt, true).map_err(to_py)
}
