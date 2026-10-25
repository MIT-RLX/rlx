// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Model` — a model from any provider (an `.rlxp` package, a plugin's family
//! runner such as an LLM), opened by path or id and run by name.
//!
//! ```python
//! pyrlx.save_model(g, "affine.rlxp", {"w": w}, {"outputs": ["y"]})
//! m = pyrlx.Model.open("affine.rlxp", {"device": "cpu"})
//! (y,) = m.run({"x": x})
//! # by task: m.chat([{"role": "user", "content": "Hi"}]) · m.transcribe(audio, 16000)
//! #          m.synthesize("Hi") -> (samples, rate) · m.embed(["a", "b"]) -> [2, dim]
//! ```

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rlx_bind::runtime::Value;
use rlx_bind::runtime::model::{self, Model};

use crate::bind::{from_json, opt_json, to_json, to_py};
use crate::graph::{PyGraph, graph_ref};
use crate::session::f32_view;

/// Options from a dict argument merged with keyword arguments.
pub(crate) fn options(
    options: Option<&Bound<'_, PyAny>>,
    kw: Option<&Bound<'_, PyDict>>,
) -> PyResult<Value> {
    let mut o = opt_json(options)?;
    if let Some(kw) = kw {
        if let Value::Object(extra) = to_json(kw.as_any())? {
            if !o.is_object() {
                o = Value::Object(Default::default());
            }
            o.as_object_mut().unwrap().extend(extra);
        }
    }
    Ok(o)
}

/// `{name: float32 array | sequence}` → owned f32 pairs.
fn f32_dict(d: &Bound<'_, PyDict>) -> PyResult<Vec<(String, Vec<f32>)>> {
    let mut out = Vec::with_capacity(d.len());
    for (k, v) in d.iter() {
        let name: String = k.extract()?;
        let data = match f32_view(&name, &v) {
            Ok(s) => s.to_vec(),
            Err(_) => v.extract::<Vec<f32>>()?,
        };
        out.push((name, data));
    }
    Ok(out)
}

fn pairs(owned: &[(String, Vec<f32>)]) -> Vec<(&str, &[f32])> {
    owned
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect()
}

/// A model from any provider.
#[pyclass(name = "Model", module = "pyrlx._pyrlx", unsendable)]
pub(crate) struct PyModel {
    inner: Model,
}

#[pymethods]
impl PyModel {
    /// Open `source` (a path or id); options (dict or keywords): `device`,
    /// `provider`, and the provider's own.
    #[staticmethod]
    #[pyo3(signature = (source, options = None, **kw))]
    fn open(
        source: &str,
        options: Option<&Bound<'_, PyAny>>,
        kw: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let o = self::options(options, kw)?;
        Ok(Self {
            inner: Model::open(source, &o).map_err(to_py)?,
        })
    }

    /// `{provider, family, task, device, inputs, outputs, metadata}`.
    fn info(&self, py: Python<'_>) -> PyObject {
        from_json(py, &self.inner.info())
    }

    /// The task: "graph", "text-generation", ….
    fn task(&self) -> String {
        self.inner.task()
    }

    /// f32 tensors in, one numpy array per output (shaped by the model).
    #[pyo3(signature = (inputs, options = None, **kw))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
        options: Option<&Bound<'py, PyAny>>,
        kw: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let owned = f32_dict(inputs)?;
        let o = self::options(options, kw)?;
        let outs = self.inner.run(&pairs(&owned), &o).map_err(to_py)?;
        let shapes = self.inner.output_shapes();
        let list = PyList::empty_bound(py);
        for (i, out) in outs.into_iter().enumerate() {
            let n = out.len();
            let arr = out.into_pyarray_bound(py);
            match shapes.get(i) {
                Some(s) if !s.is_empty() && s.iter().product::<usize>() == n => {
                    list.append(arr.reshape(s.clone())?)?
                }
                _ => list.append(arr)?,
            }
        }
        Ok(list)
    }

    /// `{name: (bytes, dtype)}` in, `[(bytes, dtype), …]` out.
    #[pyo3(signature = (inputs, options = None, **kw))]
    fn run_typed<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
        options: Option<&Bound<'py, PyAny>>,
        kw: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut owned: Vec<(String, Vec<u8>, String)> = Vec::with_capacity(inputs.len());
        for (k, v) in inputs.iter() {
            let name: String = k.extract()?;
            let (bytes, dtype): (Vec<u8>, String) = v.extract().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "input '{name}': expected tuple (bytes, dtype_str)"
                ))
            })?;
            owned.push((name, bytes, dtype));
        }
        let refs: Vec<(&str, &[u8], &str)> = owned
            .iter()
            .map(|(n, d, t)| (n.as_str(), d.as_slice(), t.as_str()))
            .collect();
        let o = self::options(options, kw)?;
        let outs = self.inner.run_typed(&refs, &o).map_err(to_py)?;
        let list = PyList::empty_bound(py);
        for (bytes, dtype) in outs {
            list.append((PyBytes::new_bound(py, &bytes), dtype))?;
        }
        Ok(list)
    }

    /// Dims of each output of the last run.
    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.inner.output_shapes()
    }

    /// Names of the outputs of the last run.
    fn output_names(&self) -> Vec<String> {
        self.inner.output_names()
    }

    /// A language model's last-position logits after `ids`.
    fn predict_logits(&mut self, ids: Vec<u32>) -> PyResult<Vec<f32>> {
        self.inner.predict_logits(&ids).map_err(to_py)
    }

    /// A language model's next ids after `ids`; sampling options
    /// (`temperature`, `top_p`, `top_k`, `seed`, `stop_ids`) as a dict or keywords.
    #[pyo3(signature = (ids, max_tokens = 16, options = None, **kw))]
    fn generate(
        &mut self,
        ids: Vec<u32>,
        max_tokens: usize,
        options: Option<&Bound<'_, PyAny>>,
        kw: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<u32>> {
        let o = self::options(options, kw)?;
        self.inner.generate(&ids, max_tokens, &o).map_err(to_py)
    }

    /// Text in, text out, for a model with a tokenizer.
    #[pyo3(signature = (prompt, options = None, **kw))]
    fn generate_text(
        &mut self,
        prompt: &str,
        options: Option<&Bound<'_, PyAny>>,
        kw: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let o = self::options(options, kw)?;
        self.inner.generate_text(prompt, &o).map_err(to_py)
    }

    /// A reply to `messages` (`[{"role": …, "content": …}, …]`, or one
    /// string), with the model's chat template applied.
    #[pyo3(signature = (messages, options = None, **kw))]
    fn chat(
        &mut self,
        messages: &Bound<'_, PyAny>,
        options: Option<&Bound<'_, PyAny>>,
        kw: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let m = to_json(messages)?;
        let o = self::options(options, kw)?;
        self.inner.chat(&m, &o).map_err(to_py)
    }

    /// The text of mono `audio` (a float32 array or a list) at `sample_rate`.
    #[pyo3(signature = (audio, sample_rate = 16000, options = None, **kw))]
    fn transcribe(
        &mut self,
        audio: &Bound<'_, PyAny>,
        sample_rate: u32,
        options: Option<&Bound<'_, PyAny>>,
        kw: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let o = self::options(options, kw)?;
        match f32_view("audio", audio) {
            Ok(view) => self.inner.transcribe(view, sample_rate, &o),
            Err(_) => {
                let owned: Vec<f32> = audio.extract()?;
                self.inner.transcribe(&owned, sample_rate, &o)
            }
        }
        .map_err(to_py)
    }

    /// `text` spoken: `(samples, sample_rate)`, samples a float32 array.
    #[pyo3(signature = (text, options = None, **kw))]
    fn synthesize<'py>(
        &mut self,
        py: Python<'py>,
        text: &str,
        options: Option<&Bound<'py, PyAny>>,
        kw: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyAny>, u32)> {
        let o = self::options(options, kw)?;
        let (audio, rate) = self.inner.synthesize(text, &o).map_err(to_py)?;
        Ok((audio.into_pyarray_bound(py).into_any(), rate))
    }

    /// One vector per text: a float32 array `[len(texts), dim]`.
    #[pyo3(signature = (texts, options = None, **kw))]
    fn embed<'py>(
        &mut self,
        py: Python<'py>,
        texts: &Bound<'py, PyAny>,
        options: Option<&Bound<'py, PyAny>>,
        kw: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // One string is one text.
        let texts: Vec<String> = match texts.extract::<String>() {
            Ok(s) => vec![s],
            Err(_) => texts.extract()?,
        };
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let o = self::options(options, kw)?;
        let rows = self.inner.embed(&refs, &o).map_err(to_py)?;
        let rows: Vec<Vec<f32>> = rlx_bind::runtime::serde_json::from_value(rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dim = rows.first().map_or(0, Vec::len);
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        Ok(flat
            .into_pyarray_bound(py)
            .reshape([refs.len(), dim])?
            .into_any())
    }

    fn __repr__(&self) -> String {
        let info = self.inner.info();
        let s = |k: &str| {
            info.get(k)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        };
        format!(
            "<pyrlx.Model {} task={} provider={} device={}>",
            s("family"),
            s("task"),
            s("provider"),
            s("device")
        )
    }
}

/// Every model provider: `[{name, description, tasks, families}]`.
#[pyfunction]
pub(crate) fn model_providers(py: Python<'_>) -> PyObject {
    from_json(py, &model::providers())
}

/// Load a model plugin library; returns its providers' names.
#[pyfunction]
pub(crate) fn load_model_plugin(path: &str) -> PyResult<Vec<String>> {
    model::load_plugin(path).map_err(to_py)
}

/// Write `graph` (not consumed) and its f32 `weights` as an `.rlxp` model
/// package; `model`: `{task, family, outputs, metadata}`.
#[pyfunction]
#[pyo3(signature = (graph, path, weights, model = None))]
pub(crate) fn save_model(
    graph: &Bound<'_, PyGraph>,
    path: &str,
    weights: &Bound<'_, PyDict>,
    model: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let owned = f32_dict(weights)?;
    let meta = opt_json(model)?;
    let g = graph.borrow();
    model::save(graph_ref(&g)?, path, &pairs(&owned), &meta).map_err(to_py)
}
