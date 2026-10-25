// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! An entry of kind `program`: the model's host logic as WebAssembly
//! ([`rlx_program`]), run in a `wasmi` sandbox, its graphs run here.
//!
//! ```text
//! entries: {speak: {
//!   kind: "program",
//!   module: "sidecar:program.wasm",
//!   tasks: ["synthesize"],          // what it answers: run, transcribe,
//!                                   // synthesize, embed, generate, chat, score
//!   task: "text-to-speech",         // what the model is
//!   config: {…},                    // the program's `init`
//! }}
//! ```
//!
//! The module calls back for the package's functions (`rlx.run`: each
//! specialized to its inputs' shapes, compiled once per shape with the
//! package's weights, on the model's device; outputs it `keep`s stay here
//! in named slots, and later runs read them `from` there; inputs it names
//! `weights` are read from the package's weights for that run, e.g. the
//! experts a router picked), its files and
//! tokenizers (`rlx.host`: `file`, `tokenizer.encode`, `tokenizer.decode`,
//! `slot.get`, `slot.set`, `slot.drop`).

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail};
use rlx_ir::{DType, Dim, Graph, Op};
use rlx_program::{Message, Tensor as PTensor, decode, encode};
use serde_json::{Value, json};
use wasmi::{Caller, Engine, Extern, Instance, Linker, Module, Store};

use super::entries::{static_dims, typed};
use super::rlxp::{Opened, params_of};
use super::{Audio, ChatMessage, Model, ModelInfo, ModelRequest, Tensor, TextCodec};
use crate::CompiledGraph;

/// A package function, compiled per binding of its symbols.
struct Function {
    graph: Graph,
    built: HashMap<Vec<(u32, usize)>, (CompiledGraph, Vec<Vec<usize>>)>,
}

/// What the module's imports reach.
pub(super) struct Host {
    pub(super) pkg: Opened,
    device: crate::Device,
    functions: HashMap<String, Function>,
    tokenizers: HashMap<String, Box<dyn TextCodec>>,
    /// Tensors the program keeps here between runs (`keep` / `from`): a
    /// decoder's caches, an encoder's states — never copied into the module.
    slots: HashMap<String, PTensor>,
    /// The package's weight files, opened once, for `weights` inputs.
    reads: crate::dist::WeightCache,
    /// The package's files, for a `wasm32-wasip1` program's `std::fs`.
    pub(super) wasi: super::wasi::Wasi,
    pub(super) log: bool,
    /// What the program was opened with: its package's other entries open
    /// with it (`entry.run`).
    request: super::ModelRequest,
    /// The package's other entries the program has run, by name.
    entries: HashMap<String, Box<dyn super::Model>>,
}

impl Host {
    /// Run the package function `name` on the request's tensors and the
    /// slots its body names (`from: {input: slot}`); outputs it names
    /// (`keep: {output: slot}`) stay in their slots, the rest come back.
    /// The body's `dims: {name: size}` sizes symbols no input's shape
    /// gives (an output as long as the program asks); its `weights:
    /// {input: [tensor, …]}` feeds inputs from the package's weights.
    fn run_request(&mut self, name: &str, req: Message) -> Result<Vec<PTensor>> {
        let mut given = Vec::new();
        if let Some(dims) = req.body.get("dims").and_then(Value::as_object) {
            for (dim, size) in dims {
                let sym = self
                    .pkg
                    .dims
                    .iter()
                    .find(|(n, _)| n == dim)
                    .map(|(_, s)| *s)
                    .with_context(|| format!("the package has no dim '{dim}'"))?;
                let size = size.as_u64().context("a dim's size is a whole number")?;
                given.push((sym, size as usize));
            }
        }
        let mut inputs = req.tensors;
        if let Some(from) = req.body.get("from").and_then(Value::as_object) {
            for (input, slot) in from {
                let slot = slot
                    .as_str()
                    .context("`from` maps an input to a slot name")?;
                let mut t = self
                    .slots
                    .get(slot)
                    .with_context(|| format!("'{name}' reads slot '{slot}', which is empty"))?
                    .clone();
                t.name = input.clone();
                inputs.push(t);
            }
        }
        if let Some(weights) = req.body.get("weights").and_then(Value::as_object) {
            for (input, tensors) in weights {
                let tensors: Vec<&str> = match tensors {
                    Value::String(t) => vec![t.as_str()],
                    Value::Array(a) => a.iter().filter_map(Value::as_str).collect(),
                    _ => bail!("`weights` maps an input to a tensor name or a list of them"),
                };
                inputs.push(self.weights_input(name, input, &tensors)?);
            }
        }
        let outs = self.run(name, &inputs, &given)?;
        let keep = req.body.get("keep").and_then(Value::as_object);
        let mut back = Vec::new();
        for t in outs {
            match keep.and_then(|k| k.get(&t.name)).and_then(Value::as_str) {
                Some(slot) => {
                    self.slots.insert(slot.to_string(), t);
                }
                None => back.push(t),
            }
        }
        Ok(back)
    }

    /// The package function `name`, taken from the package on first use.
    fn function(&mut self, name: &str) -> Result<&mut Function> {
        if !self.functions.contains_key(name) {
            let graph = self
                .pkg
                .take_graph(name)
                .with_context(|| format!("the program runs '{name}', which the package lacks"))?;
            self.functions.insert(
                name.to_string(),
                Function {
                    graph,
                    built: HashMap::new(),
                },
            );
        }
        Ok(self.functions.get_mut(name).expect("inserted above"))
    }

    /// Input `input` of function `name`: the named tensors of the
    /// package's weights, read now and stacked in order, as the input
    /// declares them (static dims; `weights: {input: [tensor, …]}`).
    fn weights_input(&mut self, name: &str, input: &str, tensors: &[&str]) -> Result<PTensor> {
        let f = self.function(name)?;
        let node = f
            .graph
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name } if name == input))
            .with_context(|| format!("'{name}' has no input '{input}'"))?;
        let dtype = node.shape.dtype();
        let dims: Vec<usize> = node
            .shape
            .dims()
            .iter()
            .map(|d| match d {
                Dim::Static(n) => Ok(*n),
                Dim::Dynamic(_) => Err(anyhow!(
                    "'{name}' input '{input}' is read from the weights: its dims are static"
                )),
            })
            .collect::<Result<_>>()?;
        let mut data = Vec::new();
        for t in tensors {
            data.extend(self.pkg.weights.stored(&mut self.reads, t)?);
        }
        let want = dims.iter().product::<usize>() * dtype.size_bytes();
        if data.len() != want {
            bail!(
                "'{name}' input '{input}' is {dims:?} {} ({want} bytes); its tensors are {} bytes",
                super::dtype_name(dtype),
                data.len()
            );
        }
        Ok(PTensor::new(input, super::dtype_name(dtype), dims, data))
    }

    /// Run the package function `name` on `inputs`, the symbols `given`
    /// sized besides those the inputs' shapes size.
    fn run(
        &mut self,
        name: &str,
        inputs: &[PTensor],
        given: &[(u32, usize)],
    ) -> Result<Vec<PTensor>> {
        self.function(name)?;
        let f = self.functions.get_mut(name).expect("taken above");
        // Its symbols, bound by the inputs' shapes.
        let mut binding: Vec<(u32, usize)> = given.to_vec();
        let mut feeds = Vec::new();
        for n in f.graph.nodes() {
            let Op::Input { name: input } = &n.op else {
                continue;
            };
            let t = inputs.iter().find(|t| &t.name == input).with_context(|| {
                format!("'{name}' takes '{input}', which the program did not give")
            })?;
            let dims = n.shape.dims();
            if dims.len() == t.shape.len() {
                for (d, &v) in dims.iter().zip(&t.shape) {
                    match d {
                        Dim::Dynamic(s) => match binding.iter().find(|(x, _)| x == s) {
                            Some((_, m)) if *m != v => {
                                bail!("'{name}': one dim given {m} and {v}")
                            }
                            Some(_) => {}
                            None => binding.push((*s, v)),
                        },
                        Dim::Static(m) if *m != v => bail!(
                            "'{name}' input '{input}' is {dims:?}; the program gave {:?}",
                            t.shape
                        ),
                        Dim::Static(_) => {}
                    }
                }
            }
            let want = n.shape.dtype();
            let (data, dtype) = match super::parse_dtype(&t.dtype)? {
                d if d == want => (t.data.clone(), d),
                _ => typed(&t.to_f32(), want)?,
            };
            feeds.push((input.clone(), data, dtype));
        }
        binding.sort();
        if !f.built.contains_key(&binding) {
            let mut graph = if binding.is_empty() {
                f.graph.clone()
            } else {
                rlx_ir::dynamic::specialize(&f.graph, &rlx_ir::DimBinding::from_pairs(&binding))
            };
            if graph.has_dynamic_dims() {
                bail!("'{name}' at {binding:?} still has unbound dims: its inputs bind every one");
            }
            graph.name = name.to_string();
            let shapes = graph
                .outputs
                .iter()
                .map(|&o| static_dims(&graph, o))
                .collect::<Result<Vec<_>>>()?;
            let params = params_of(&graph);
            let mut c = self.pkg.weights.compile(self.device, graph);
            self.pkg.weights.bind(&mut c, &params)?;
            f.built.insert(binding.clone(), (c, shapes));
        }
        let names: Vec<String> = f
            .graph
            .outputs
            .iter()
            .enumerate()
            .map(|(i, &o)| {
                f.graph
                    .node(o)
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("output_{i}"))
            })
            .collect();
        let (c, shapes) = f.built.get_mut(&binding).expect("built above");
        let refs: Vec<(&str, &[u8], DType)> = feeds
            .iter()
            .map(|(n, d, t)| (n.as_str(), d.as_slice(), *t))
            .collect();
        let outs = c.run_typed(&refs);
        Ok(outs
            .into_iter()
            .zip(shapes.iter())
            .zip(names)
            .map(|(((data, dtype), shape), name)| {
                PTensor::new(name, super::dtype_name(dtype), shape.clone(), data)
            })
            .collect())
    }

    fn tokenizer(&mut self, reference: &str) -> Result<&dyn TextCodec> {
        if !self.tokenizers.contains_key(reference) {
            let bytes = self.pkg.file(reference)?;
            let codec = super::load_tokenizer(&bytes, &Value::Null)?;
            self.tokenizers.insert(reference.to_string(), codec);
        }
        Ok(self.tokenizers[reference].as_ref())
    }

    /// A host service.
    fn service(&mut self, name: &str, req: &Message) -> Result<Message> {
        let s = |k: &str| {
            req.body
                .get(k)
                .and_then(Value::as_str)
                .with_context(|| format!("'{name}' takes `{k}`"))
        };
        Ok(match name {
            // A file of the package (`sidecar:<id>`, or a path beside it).
            "file" => {
                let bytes = self.pkg.file(s("name")?)?;
                let n = bytes.len();
                Message::tensors(vec![PTensor::new("bytes", "u8", vec![n], bytes)])
            }
            // Another entry of the package (`entry`: its name or kind) on
            // the request's tensors with `options` — a language model's
            // `generate` entry, `max_tokens` given, generates ids.
            "entry.run" => {
                let name = s("entry")?.to_string();
                if !self.entries.contains_key(&name) {
                    let (_, kind, entry) = super::rlxp::entries_of(&self.pkg.meta)
                        .into_iter()
                        .find(|(n, k, _)| *n == name || *k == name)
                        .with_context(|| format!("the package has no entry '{name}'"))?;
                    if kind == "program" {
                        bail!("a program runs another entry, not a program ('{name}')");
                    }
                    let model =
                        super::rlxp::open_entry(&self.request, self.pkg.reopen()?, &kind, &entry)?;
                    self.entries.insert(name.clone(), model);
                }
                let options = req.body.get("options").cloned().unwrap_or(Value::Null);
                let inputs: Vec<super::Tensor> = req
                    .tensors
                    .iter()
                    .map(|t| {
                        let dtype = super::parse_dtype(&t.dtype)?;
                        Ok(super::Tensor::new(
                            t.name.clone(),
                            dtype,
                            t.shape.clone(),
                            t.data.clone(),
                        ))
                    })
                    .collect::<Result<_>>()?;
                let model = self.entries.get_mut(&name).expect("opened above");
                let out = model.run(&inputs, &options)?;
                Message::tensors(
                    out.into_iter()
                        .map(|t| PTensor::new(t.name, super::dtype_name(t.dtype), t.shape, t.data))
                        .collect(),
                )
            }
            // One id drawn from the request's `logits` (which the program
            // may have changed: a penalty, a mask) by the sampling options
            // in the body, from the stream `seed + step`.
            "sample" => {
                let t = req
                    .tensors
                    .first()
                    .context("'sample' takes a logits tensor")?;
                if t.dtype != "f32" {
                    bail!("'sample' takes f32 logits, not {}", t.dtype);
                }
                let mut logits = rlx_ir::bytes::decode_le_vec::<f32>(&t.data);
                let n = |k: &str| req.body.get(k).and_then(Value::as_u64).unwrap_or(0);
                let opts = super::sample_opts(&req.body);
                let id = if opts.is_greedy() {
                    crate::lm::argmax_u32(&logits)
                } else {
                    let mut rng = rlx_ir::Philox4x32::new(n("seed").wrapping_add(n("step")));
                    let mut state = crate::samplers::SamplerState::default();
                    opts.into_chain()
                        .sample(&mut logits, &[], &mut state, &mut rng)
                };
                Message::body(json!({"id": id}))
            }
            // `special: false`: the text's ids alone, none added around them.
            "tokenizer.encode" => {
                let text = s("text")?.to_string();
                let plain = req.body.get("special").and_then(Value::as_bool) == Some(false);
                let codec = self.tokenizer(s("tokenizer")?)?;
                let ids = if plain {
                    codec.encode_plain(&text)?
                } else {
                    codec.encode(&text)?
                };
                Message::body(json!({"ids": ids}))
            }
            "tokenizer.decode" => {
                let ids: Vec<u32> = req
                    .body
                    .get("ids")
                    .and_then(Value::as_array)
                    .context("'tokenizer.decode' takes `ids`")?
                    .iter()
                    .filter_map(Value::as_u64)
                    .map(|v| v as u32)
                    .collect();
                let text = self.tokenizer(s("tokenizer")?)?.decode(&ids)?;
                Message::body(json!({"text": text}))
            }
            // A slot's tensor, read back (`slot`), or slots emptied (`drop`).
            "slot.get" => {
                let slot = s("slot")?;
                let t = self
                    .slots
                    .get(slot)
                    .with_context(|| format!("slot '{slot}' is empty"))?;
                Message::tensors(vec![t.clone()])
            }
            "slot.set" => {
                let slot = s("slot")?.to_string();
                let t = req
                    .tensors
                    .first()
                    .context("'slot.set' takes a tensor")?
                    .clone();
                self.slots.insert(slot, t);
                Message::default()
            }
            "slot.drop" => {
                for v in req
                    .body
                    .get("slots")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(k) = v.as_str() {
                        self.slots.remove(k);
                    }
                }
                Message::default()
            }
            other => bail!(
                "no host service '{other}' (file, tokenizer.encode, tokenizer.decode, slot.get, \
                 slot.set, slot.drop)"
            ),
        })
    }
}

fn read(caller: &Caller<'_, Host>, ptr: u32, len: u32) -> Result<Vec<u8>> {
    let memory = caller
        .get_export("memory")
        .and_then(Extern::into_memory)
        .context("the module exports no memory")?;
    let mut buf = vec![0u8; len as usize];
    memory
        .read(caller, ptr as usize, &mut buf)
        .map_err(|e| anyhow!("reading the module's memory: {e}"))?;
    Ok(buf)
}

/// `bytes` written into a buffer the module allocates: `ptr << 32 | len`.
fn give(caller: &mut Caller<'_, Host>, bytes: &[u8]) -> Result<u64> {
    let alloc = caller
        .get_export("rlx_alloc")
        .and_then(Extern::into_func)
        .context("the module exports no rlx_alloc")?
        .typed::<u32, u32>(&*caller)?;
    let ptr = alloc.call(&mut *caller, bytes.len() as u32)?;
    let memory = caller
        .get_export("memory")
        .and_then(Extern::into_memory)
        .context("the module exports no memory")?;
    memory
        .write(&mut *caller, ptr as usize, bytes)
        .map_err(|e| anyhow!("writing the module's memory: {e}"))?;
    Ok((u64::from(ptr) << 32) | bytes.len() as u64)
}

/// An import's answer: its result, or its error, as a message.
fn answer(caller: &mut Caller<'_, Host>, result: Result<Message>) -> u64 {
    let m = result.unwrap_or_else(|e| Message::error(format!("{e:#}")));
    give(caller, &encode(&m)).unwrap_or(0)
}

/// A loaded program and what it answers.
struct ProgramModel {
    store: Store<Host>,
    instance: Instance,
    info: ModelInfo,
    tasks: Vec<String>,
}

impl ProgramModel {
    /// Write `bytes` into the module.
    fn put(&mut self, bytes: &[u8]) -> Result<(u32, u32)> {
        let alloc = self
            .instance
            .get_typed_func::<u32, u32>(&self.store, "rlx_alloc")?;
        let ptr = alloc.call(&mut self.store, bytes.len() as u32)?;
        let memory = self
            .instance
            .get_memory(&self.store, "memory")
            .context("the module exports no memory")?;
        memory
            .write(&mut self.store, ptr as usize, bytes)
            .map_err(|e| anyhow!("{e}"))?;
        Ok((ptr, bytes.len() as u32))
    }

    /// Read and free the response `packed` (`ptr << 32 | len`).
    fn take(&mut self, packed: u64) -> Result<Message> {
        let (ptr, len) = ((packed >> 32) as u32, (packed & 0xffff_ffff) as u32);
        let memory = self
            .instance
            .get_memory(&self.store, "memory")
            .context("the module exports no memory")?;
        let mut buf = vec![0u8; len as usize];
        memory
            .read(&self.store, ptr as usize, &mut buf)
            .map_err(|e| anyhow!("{e}"))?;
        let free = self
            .instance
            .get_typed_func::<(u32, u32), ()>(&self.store, "rlx_free")?;
        free.call(&mut self.store, (ptr, len))?;
        let m = decode(&buf).map_err(|e| anyhow!("the program's response: {e}"))?;
        if let Some(e) = m.as_error() {
            bail!("{e}");
        }
        Ok(m)
    }

    /// Ask the program `task`.
    fn call(&mut self, task: &str, request: &Message) -> Result<Message> {
        if !self.tasks.iter().any(|t| t == task) {
            bail!(
                "this {} program answers {:?}, not '{task}'",
                self.info.family,
                self.tasks
            );
        }
        let t = self.put(task.as_bytes())?;
        let m = self.put(&encode(request))?;
        let f = self
            .instance
            .get_typed_func::<(u32, u32, u32, u32), u64>(&self.store, "rlx_call")?;
        let packed = f
            .call(&mut self.store, (t.0, t.1, m.0, m.1))
            .map_err(|e| anyhow!("the program trapped in '{task}': {e}"))?;
        self.take(packed)
    }
}

fn to_program(t: &Tensor) -> PTensor {
    PTensor::new(
        t.name.clone(),
        super::dtype_name(t.dtype),
        t.shape.clone(),
        t.data.clone(),
    )
}

fn from_program(t: PTensor) -> Result<Tensor> {
    Ok(Tensor::new(
        t.name,
        super::parse_dtype(&t.dtype)?,
        t.shape,
        t.data,
    ))
}

fn text(m: &Message) -> Result<String> {
    m.body
        .get("text")
        .and_then(Value::as_str)
        .map(String::from)
        .context("the program's answer has no `text`")
}

impl Model for ProgramModel {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn run(&mut self, inputs: &[Tensor], options: &Value) -> Result<Vec<Tensor>> {
        let req = Message::new(
            json!({"options": options}),
            inputs.iter().map(to_program).collect(),
        );
        self.call("run", &req)?
            .tensors
            .into_iter()
            .map(from_program)
            .collect()
    }

    fn generate_text(&mut self, prompt: &str, options: &Value) -> Result<String> {
        text(&self.call(
            "generate",
            &Message::body(json!({"prompt": prompt, "options": options})),
        )?)
    }

    fn chat(&mut self, messages: &[ChatMessage], options: &Value) -> Result<String> {
        let messages: Vec<Value> = messages
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect();
        text(&self.call(
            "chat",
            &Message::body(json!({"messages": messages, "options": options})),
        )?)
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32, options: &Value) -> Result<String> {
        let req = Message::new(
            json!({"sample_rate": sample_rate, "options": options}),
            vec![PTensor::f32("audio", vec![audio.len()], audio)],
        );
        text(&self.call("transcribe", &req)?)
    }

    fn synthesize(&mut self, text: &str, options: &Value) -> Result<Audio> {
        let m = self.call(
            "synthesize",
            &Message::body(json!({"text": text, "options": options})),
        )?;
        let samples = m
            .tensor("audio")
            .context("the program's answer has no `audio`")?
            .to_f32();
        let sample_rate = m
            .body
            .get("sample_rate")
            .and_then(Value::as_u64)
            .context("the program's answer has no `sample_rate`")? as u32;
        Ok(Audio {
            samples,
            sample_rate,
        })
    }

    fn embed(&mut self, texts: &[&str], options: &Value) -> Result<Vec<Vec<f32>>> {
        let m = self.call(
            "embed",
            &Message::body(json!({"texts": texts, "options": options})),
        )?;
        let t = m
            .tensor("embeddings")
            .context("the program's answer has no `embeddings`")?;
        let d = *t.shape.last().unwrap_or(&0);
        Ok(t.to_f32().chunks(d.max(1)).map(<[f32]>::to_vec).collect())
    }

    fn score(&mut self, query: &str, passages: &[&str], options: &Value) -> Result<Vec<f32>> {
        let m = self.call(
            "score",
            &Message::body(json!({"query": query, "passages": passages, "options": options})),
        )?;
        Ok(m.body
            .get("scores")
            .and_then(Value::as_array)
            .context("the program's answer has no `scores`")?
            .iter()
            .filter_map(Value::as_f64)
            .map(|v| v as f32)
            .collect())
    }
}

/// The `program` entry `entry` of `pkg` as a model.
pub(super) fn program(
    request: &ModelRequest,
    mut pkg: Opened,
    entry: &Value,
) -> Result<Box<dyn Model>> {
    let module = entry
        .get("module")
        .and_then(Value::as_str)
        .context("a program entry names its `module`")?;
    let wasm = pkg.file(module)?;
    let tasks: Vec<String> = entry
        .get("tasks")
        .and_then(Value::as_array)
        .context("a program entry lists the `tasks` it answers")?
        .iter()
        .filter_map(Value::as_str)
        .map(String::from)
        .collect();
    let mut metadata = pkg.metadata();
    metadata["tasks"] = json!(tasks);
    let family = pkg.meta["family"].as_str().unwrap_or("program").to_string();
    let task = entry
        .get("task")
        .and_then(Value::as_str)
        .or_else(|| pkg.meta["task"].as_str())
        .unwrap_or("program")
        .to_string();
    pkg.target = request.device_label();
    let engine = Engine::default();
    let module = Module::new(&engine, &wasm).map_err(|e| anyhow!("{module}: {e}"))?;
    let sidecars: Vec<String> = pkg
        .pack
        .sidecar_ids()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let host = Host {
        wasi: super::wasi::Wasi::new(sidecars),
        pkg,
        device: request.device,
        functions: HashMap::new(),
        tokenizers: HashMap::new(),
        slots: HashMap::new(),
        reads: crate::dist::WeightCache::new(),
        log: rlx_ir::env::flag("RLX_PROGRAM_LOG"),
        request: request.clone(),
        entries: HashMap::new(),
    };
    let mut store = Store::new(&engine, host);
    let mut linker = Linker::<Host>::new(&engine);
    super::wasi::link(&mut linker).map_err(|e| anyhow!("{e}"))?;
    linker
        .func_wrap(
            "rlx",
            "run",
            |mut caller: Caller<'_, Host>,
             name: u32,
             name_len: u32,
             msg: u32,
             msg_len: u32|
             -> u64 {
                let result = (|| -> Result<Message> {
                    let name = String::from_utf8(read(&caller, name, name_len)?)?;
                    let req = decode(&read(&caller, msg, msg_len)?).map_err(|e| anyhow!(e))?;
                    let outs = caller.data_mut().run_request(&name, req)?;
                    Ok(Message::tensors(outs))
                })();
                answer(&mut caller, result)
            },
        )
        .map_err(|e| anyhow!("{e}"))?;
    linker
        .func_wrap(
            "rlx",
            "host",
            |mut caller: Caller<'_, Host>,
             name: u32,
             name_len: u32,
             msg: u32,
             msg_len: u32|
             -> u64 {
                let result = (|| -> Result<Message> {
                    let name = String::from_utf8(read(&caller, name, name_len)?)?;
                    let req = decode(&read(&caller, msg, msg_len)?).map_err(|e| anyhow!(e))?;
                    caller.data_mut().service(&name, &req)
                })();
                answer(&mut caller, result)
            },
        )
        .map_err(|e| anyhow!("{e}"))?;
    linker
        .func_wrap(
            "rlx",
            "log",
            |caller: Caller<'_, Host>, text: u32, len: u32| {
                if caller.data().log
                    && let Ok(b) = read(&caller, text, len)
                {
                    eprintln!("[rlx program] {}", String::from_utf8_lossy(&b));
                }
            },
        )
        .map_err(|e| anyhow!("{e}"))?;
    let instance = linker
        .instantiate_and_start(&mut store, &module)
        .map_err(|e| anyhow!("instantiating the program: {e}"))?;
    // A WASI reactor's constructors.
    if let Ok(init) = instance.get_typed_func::<(), ()>(&store, "_initialize") {
        init.call(&mut store, ())
            .map_err(|e| anyhow!("the program's _initialize: {e}"))?;
    }
    let mut model = ProgramModel {
        store,
        instance,
        info: ModelInfo {
            provider: "rlxp".into(),
            family,
            task,
            device: request.device_label(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata,
        },
        tasks,
    };
    // `init` with the entry's config.
    let config = encode(&Message::body(
        entry.get("config").cloned().unwrap_or(Value::Null),
    ));
    let (ptr, len) = model.put(&config)?;
    let init = model
        .instance
        .get_typed_func::<(u32, u32), u64>(&model.store, "rlx_init")?;
    let packed = init
        .call(&mut model.store, (ptr, len))
        .map_err(|e| anyhow!("the program trapped in init: {e}"))?;
    model.take(packed).context("the program's init")?;
    Ok(Box::new(model))
}
