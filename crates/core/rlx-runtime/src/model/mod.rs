// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Models: one interface over whatever runs a checkpoint — a family runner
//! (an LLM, a speech or vision model, …), an `.rlxp` package, or a user's own
//! code — so the bindings, and through the C ABI every language, open and run
//! any of them the same way.
//!
//! * [`Model`] — named typed tensors in, named typed tensors out
//!   ([`Model::run`]), plus the token-level LM interface ([`Model::lm`]) and
//!   text generation ([`Model::generate_text`]) for the models that have them.
//! * [`ModelProvider`] — turns a [`ModelRequest`] (a path or id, a device,
//!   options) into a [`Model`]. Providers register statically
//!   ([`register_provider!`](crate::register_provider)) or at run time
//!   ([`register_provider`]), and [`open`] asks the one that claims the
//!   request most specifically ([`ModelProvider::probe`]).
//! * [`plugin`] — a provider library loaded at run time through a versioned C
//!   vtable, so a model collection built separately (downstream) plugs into an
//!   already-built `librlx_capi` and so into every language binding.
//! * [`lm_model`] — any [`LmRunner`] as a [`Model`]; [`tasks`] — the same
//!   for speech recognition, text to speech and embeddings, plus [`Provider`]
//!   to declare a provider in a few lines. Every [`Model`] then answers the
//!   task methods ([`Model::chat`], [`Model::transcribe`],
//!   [`Model::synthesize`], [`Model::embed`]) in Rust and every binding.
//!
//! The built-in `rlxp` provider (feature `model-io`) runs an `.rlxp` package
//! with an embedded graph on any device.

use std::path::Path;
use std::sync::RwLock;

use anyhow::{Result, anyhow, bail};
use rlx_ir::DType;
use serde_json::{Value, json};

use crate::lm::LmRunner;
use crate::{Device, SampleOpts};

#[cfg(feature = "model-io")]
mod entries;
pub mod image_resize;
pub mod json_prefix;
pub mod plugin;
#[cfg(feature = "program")]
mod program;
#[cfg(feature = "model-io")]
pub mod rlxp;
#[cfg(feature = "model-io")]
mod sae;
pub mod tasks;
#[cfg(all(feature = "model-io", feature = "training"))]
pub mod trainer;
#[cfg(feature = "program")]
mod wasi;

pub use tasks::{
    Audio, ChatMessage, Claim, Provider, SpeechToText, TextEmbedding, TextRanking, TextToSpeech,
    speech_to_text_model, text_embedding_model, text_ranking_model, text_to_speech_model,
};

/// A named tensor: its dtype, dims, and little-endian bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn new(name: impl Into<String>, dtype: DType, shape: Vec<usize>, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
            data,
        }
    }

    /// An f32 tensor.
    pub fn f32(name: impl Into<String>, shape: Vec<usize>, values: &[f32]) -> Self {
        let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::new(name, DType::F32, shape, data)
    }

    /// The values of an f32 tensor.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            bail!("tensor '{}' is {:?}, not f32", self.name, self.dtype);
        }
        Ok(rlx_ir::bytes::decode_le_vec::<f32>(&self.data))
    }

    /// Element count from the shape.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A model input or output: name, dtype, dims (`None` = sized at run time).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<Option<usize>>,
}

/// What a model is and what it takes.
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    /// The provider that opened it (`"rlxp"`, `"demo"`, …).
    pub provider: String,
    /// The architecture family (`"qwen3"`, `"whisper"`, `"graph"`, …).
    pub family: String,
    /// What it does: `"graph"`, `"text-generation"`, `"embedding"`,
    /// `"speech-recognition"`, `"image-segmentation"`, … — free-form, but the
    /// providers here use the Hugging Face task names.
    pub task: String,
    /// The device it runs on.
    pub device: String,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    /// Whatever else the provider reports (vocab size, sample rate, …).
    pub metadata: Value,
}

impl ModelInfo {
    /// As JSON (`{provider, family, task, device, inputs, outputs, metadata}`).
    pub fn to_json(&self) -> Value {
        let spec = |s: &TensorSpec| {
            json!({
                "name": s.name,
                "dtype": dtype_name(s.dtype),
                "shape": s.shape,
            })
        };
        json!({
            "provider": self.provider,
            "family": self.family,
            "task": self.task,
            "device": self.device,
            "inputs": self.inputs.iter().map(spec).collect::<Vec<_>>(),
            "outputs": self.outputs.iter().map(spec).collect::<Vec<_>>(),
            "metadata": self.metadata,
        })
    }

    /// Read back [`ModelInfo::to_json`].
    pub fn from_json(v: &Value) -> Result<Self> {
        let s = |k: &str| {
            v.get(k)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        let specs = |k: &str| -> Result<Vec<TensorSpec>> {
            v.get(k)
                .and_then(Value::as_array)
                .map(|a| a.iter().map(spec_from_json).collect())
                .unwrap_or_else(|| Ok(Vec::new()))
        };
        Ok(Self {
            provider: s("provider"),
            family: s("family"),
            task: s("task"),
            device: s("device"),
            inputs: specs("inputs")?,
            outputs: specs("outputs")?,
            metadata: v.get("metadata").cloned().unwrap_or(Value::Null),
        })
    }
}

fn spec_from_json(v: &Value) -> Result<TensorSpec> {
    let name = v.get("name").and_then(Value::as_str).unwrap_or_default();
    let dtype = parse_dtype(v.get("dtype").and_then(Value::as_str).unwrap_or("f32"))?;
    let shape = v
        .get("shape")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|d| d.as_u64().map(|d| d as usize)).collect())
        .unwrap_or_default();
    Ok(TensorSpec {
        name: name.into(),
        dtype,
        shape,
    })
}

/// A loaded model.
///
/// `Send` (a model may move to a worker thread), not `Sync`: runners hold
/// mutable cache state.
pub trait Model: Send {
    /// What it is and what it takes.
    fn info(&self) -> ModelInfo;

    /// Named tensors in, named tensors out. `options` are the model's own
    /// (JSON object; `{}` when the caller has none).
    fn run(&mut self, inputs: &[Tensor], options: &Value) -> Result<Vec<Tensor>>;

    /// The token-level interface, for a language model.
    fn lm(&mut self) -> Option<&mut dyn LmRunner> {
        None
    }

    /// Text in, text out, for a model that carries a tokenizer (chat,
    /// translation, transcription prompts, …). `options`: sampling
    /// (`max_tokens`, `temperature`, `top_p`, …) and the model's own.
    fn generate_text(&mut self, _prompt: &str, _options: &Value) -> Result<String> {
        let info = self.info();
        bail!(
            "{} does not generate text (it is a {} model)",
            describe(&info),
            info.task
        )
    }

    /// The text after each of `prompts` (`options` as
    /// [`Model::generate_text`]). A model that can decode several sequences
    /// at once does (a package step with a `batch` symbol, greedy); the
    /// default generates them one after another.
    fn generate_batch(&mut self, prompts: &[&str], options: &Value) -> Result<Vec<String>> {
        prompts
            .iter()
            .map(|p| self.generate_text(p, options))
            .collect()
    }

    /// The text after `prompt` with `media` (images, …) where the prompt
    /// holds their placeholders (`options.messages` as for
    /// [`Model::generate_text`]), for a model that takes media.
    fn generate_with_media(
        &mut self,
        _prompt: &str,
        _media: &[Tensor],
        _options: &Value,
    ) -> Result<String> {
        bail!("{} takes no images or other media", describe(&self.info()))
    }

    /// A reply to `messages`, with the model's chat template applied.
    /// `options`: as [`Model::generate_text`].
    fn chat(&mut self, messages: &[ChatMessage], options: &Value) -> Result<String> {
        tasks::chat_via_generate(self, messages, options)
    }

    /// The text of mono `audio` at `sample_rate`, for a
    /// `"speech-recognition"` model.
    fn transcribe(&mut self, audio: &[f32], sample_rate: u32, options: &Value) -> Result<String> {
        tasks::transcribe_via_run(self, audio, sample_rate, options)
    }

    /// `text` spoken, for a `"text-to-speech"` model.
    fn synthesize(&mut self, text: &str, options: &Value) -> Result<Audio> {
        tasks::synthesize_via_run(self, text, options)
    }

    /// One vector per text, for an `"embedding"` model.
    fn embed(&mut self, texts: &[&str], options: &Value) -> Result<Vec<Vec<f32>>> {
        tasks::embed_via_run(self, texts, options)
    }

    /// Each of `passages` scored against `query` (higher: more relevant),
    /// for a `"text-ranking"` model.
    fn score(&mut self, query: &str, passages: &[&str], options: &Value) -> Result<Vec<f32>> {
        tasks::score_via_run(self, query, passages, options)
    }
}

fn describe(info: &ModelInfo) -> String {
    format!("this {} model ({})", info.task, info.family)
}

/// What to open: a path, a hub id, or whatever a provider understands, with
/// a device and the provider's options.
#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub source: String,
    pub device: Device,
    /// The full options object (`device` and `provider` included).
    pub options: Value,
}

impl ModelRequest {
    /// From a source and `{device, provider, …}` options.
    pub fn new(source: impl Into<String>, options: Value) -> Result<Self> {
        let options = if options.is_null() {
            json!({})
        } else {
            options
        };
        if !options.is_object() {
            bail!("model options must be a JSON object");
        }
        let device = match options.get("device").and_then(Value::as_str) {
            Some(d) => crate::device_parse::parse_device(d).map_err(|e| anyhow!("{e}"))?,
            None => Device::Cpu,
        };
        Ok(Self {
            source: source.into(),
            device,
            options,
        })
    }

    /// The device's label (`"cpu"`, `"metal"`, …), for [`ModelInfo::device`].
    pub fn device_label(&self) -> String {
        crate::device_label(self.device).to_string()
    }

    /// The source as a path.
    pub fn path(&self) -> &Path {
        Path::new(&self.source)
    }

    /// The provider named in the options, if any.
    pub fn provider(&self) -> Option<&str> {
        self.options.get("provider").and_then(Value::as_str)
    }
}

/// Opens models of some kind.
pub trait ModelProvider: Send + Sync {
    /// Short unique name (`"rlxp"`, `"demo"`).
    fn name(&self) -> &str;

    /// One line for listings.
    fn description(&self) -> &str {
        ""
    }

    /// The tasks its models perform.
    fn tasks(&self) -> Vec<String> {
        Vec::new()
    }

    /// How specifically it claims `request`: 0 = not at all; higher wins
    /// (an extension match might be 10, a parsed config naming the
    /// architecture 100).
    fn probe(&self, request: &ModelRequest) -> u32;

    /// Open the model.
    fn open(&self, request: &ModelRequest) -> Result<Box<dyn Model>>;

    /// Families / ids it knows how to open, for listings (`["qwen3", …]`).
    fn families(&self) -> Vec<String> {
        Vec::new()
    }
}

/// A statically registered provider (see [`register_provider!`](crate::register_provider)).
pub struct ProviderRegistration(pub &'static dyn ModelProvider);

inventory::collect!(ProviderRegistration);

/// Register a provider at program start:
///
/// ```ignore
/// struct MyProvider;
/// impl rlx_runtime::model::ModelProvider for MyProvider { /* … */ }
/// rlx_runtime::register_provider!(MyProvider);
/// ```
#[macro_export]
macro_rules! register_provider {
    ($provider:expr) => {
        $crate::lm::inventory::submit! {
            $crate::model::ProviderRegistration(&$provider)
        }
    };
}

static DYNAMIC: RwLock<Vec<Box<dyn ModelProvider>>> = RwLock::new(Vec::new());

/// Register a provider at run time (a plugin, or one built from config).
/// A provider with the name of one already registered replaces it.
pub fn register_provider(provider: Box<dyn ModelProvider>) {
    let mut all = DYNAMIC.write().unwrap_or_else(|e| e.into_inner());
    all.retain(|p| p.name() != provider.name());
    all.push(provider);
}

fn with_providers<R>(f: impl FnOnce(&[&dyn ModelProvider]) -> R) -> R {
    plugin::load_env_plugins();
    let dynamic = DYNAMIC.read().unwrap_or_else(|e| e.into_inner());
    // Run-time registrations first: a plugin may override a built-in.
    let mut all: Vec<&dyn ModelProvider> = dynamic.iter().map(|p| p.as_ref()).collect();
    // Linked plugins and built-ins are named in code, not collected at
    // start-up: a C linker drops the start-up registrations of a static
    // library's objects that nothing references.
    let named = plugin::linked_providers()
        .into_iter()
        .chain(builtin_providers());
    let collected = inventory::iter::<ProviderRegistration>
        .into_iter()
        .map(|r| r.0);
    for p in named.chain(collected) {
        if !all.iter().any(|q| q.name() == p.name()) {
            all.push(p);
        }
    }
    f(&all)
}

fn builtin_providers() -> Vec<&'static dyn ModelProvider> {
    #[cfg(feature = "model-io")]
    return vec![&rlxp::RlxpProvider];
    #[cfg(not(feature = "model-io"))]
    Vec::new()
}

/// Every provider: `[{name, description, tasks, families}]`.
pub fn providers() -> Value {
    with_providers(|all| {
        Value::Array(
            all.iter()
                .map(|p| {
                    json!({
                        "name": p.name(),
                        "description": p.description(),
                        "tasks": p.tasks(),
                        "families": p.families(),
                    })
                })
                .collect(),
        )
    })
}

/// Open `request` with its named provider, else the one that claims it most
/// specifically.
pub fn open(request: &ModelRequest) -> Result<Box<dyn Model>> {
    with_providers(|all| {
        if let Some(name) = request.provider() {
            let p = all.iter().find(|p| p.name() == name).ok_or_else(|| {
                anyhow!(
                    "no model provider '{name}' (have: {})",
                    all.iter().map(|p| p.name()).collect::<Vec<_>>().join(", ")
                )
            })?;
            return p.open(request);
        }
        let best = all
            .iter()
            .map(|p| (p.probe(request), *p))
            .filter(|(score, _)| *score > 0)
            .max_by_key(|(score, _)| *score);
        match best {
            Some((_, p)) => p.open(request),
            None => bail!(
                "no model provider claims '{}' (providers: {}). {}; or pass \
                 {{\"provider\": \"…\"}}",
                request.source,
                all.iter().map(|p| p.name()).collect::<Vec<_>>().join(", "),
                match plugin::home_plugin_dir() {
                    Some(d) => format!(
                        "Model plugins load from {} and RLX_MODEL_PLUGINS",
                        d.display()
                    ),
                    None => "On this platform, link model plugins into the app \
                             (link_model_plugins!, then rlx_link_model_plugins()) or load \
                             a bundled one with load_plugin"
                        .to_string(),
                }
            ),
        }
    })
}

/// Open `source` with `{device, provider, …}` options.
pub fn open_source(source: &str, options: Value) -> Result<Box<dyn Model>> {
    open(&ModelRequest::new(source, options)?)
}

// ── dtypes ──────────────────────────────────────────────────

/// The canonical dtype spelling (`"f32"`, `"bf16"`, `"i32"`, …).
pub fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F64 => "f64",
        DType::I8 => "i8",
        DType::U8 => "u8",
        DType::I16 => "i16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I64 => "i64",
        DType::Bool => "bool",
        DType::C64 => "c64",
        DType::C128 => "c128",
    }
}

/// Read [`dtype_name`] back.
pub fn parse_dtype(s: &str) -> Result<DType> {
    Ok(match s.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" => DType::F32,
        "f16" | "float16" => DType::F16,
        "bf16" | "bfloat16" => DType::BF16,
        "f64" | "float64" => DType::F64,
        "i8" | "int8" => DType::I8,
        "u8" | "uint8" => DType::U8,
        "i16" | "int16" => DType::I16,
        "i32" | "int32" => DType::I32,
        "u32" | "uint32" => DType::U32,
        "i64" | "int64" => DType::I64,
        "bool" => DType::Bool,
        "c64" => DType::C64,
        "c128" => DType::C128,
        other => bail!("unknown dtype '{other}'"),
    })
}

// ── a language model as a Model ─────────────────────────────

/// Option `key` (snake_case) or its camelCase spelling, as bindings in
/// camelCase languages pass it (`top_p` / `topP`).
pub fn option<'a>(options: &'a Value, key: &str) -> Option<&'a Value> {
    options.get(key).or_else(|| {
        let mut camel = String::with_capacity(key.len());
        let mut up = false;
        for c in key.chars() {
            if c == '_' {
                up = true;
            } else if up {
                camel.extend(c.to_uppercase());
                up = false;
            } else {
                camel.push(c);
            }
        }
        options.get(&camel)
    })
}

/// Sampling options from JSON: `{temperature, top_p, top_k,
/// repetition_penalty, min_p}` (or camelCase) over greedy defaults.
pub fn sample_opts(options: &Value) -> SampleOpts {
    let mut s = SampleOpts::greedy();
    let f = |k: &str| option(options, k).and_then(Value::as_f64).map(|v| v as f32);
    if let Some(t) = f("temperature") {
        s.temperature = t;
    }
    if let Some(p) = f("top_p") {
        s.top_p = p;
    }
    if let Some(k) = option(options, "top_k").and_then(Value::as_u64) {
        s.top_k = Some(k as u32);
    }
    if let Some(r) = f("repetition_penalty") {
        s.repetition_penalty = r;
    }
    if let Some(m) = f("min_p") {
        s.min_p = m;
    }
    s
}

/// Turns ids into text and back, for [`lm_model`]'s `generate_text`.
pub trait TextCodec: Send {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    /// `text`'s ids alone: none of the special tokens [`Self::encode`] adds
    /// around them (a template's `<s> … </s>`).
    fn encode_plain(&self, text: &str) -> Result<Vec<u32>> {
        self.encode(text)
    }
    fn decode(&self, ids: &[u32]) -> Result<String>;
    /// Two texts as one sequence, the way the tokenizer pairs them
    /// (`[CLS] a [SEP] b [SEP]`, `<s> a </s></s> b </s>`), and each token's
    /// segment (0 in `a`, 1 in `b`).
    fn encode_pair(&self, _a: &str, _b: &str) -> Result<(Vec<u32>, Vec<u32>)> {
        bail!("this tokenizer does not encode text pairs")
    }
    /// Ids that end generation.
    fn stop_ids(&self) -> Vec<u32> {
        Vec::new()
    }

    /// `messages` as one prompt, in the model's chat format. The default
    /// refuses: a wrong template silently degrades a model.
    fn apply_chat_template(&self, _messages: &[ChatMessage]) -> Result<String> {
        bail!("this tokenizer has no chat template; use generate_text with a prompt")
    }

    /// [`Self::apply_chat_template`] from JSON messages, which may carry
    /// more than `role` and `content` (an assistant turn's `tool_calls`, a
    /// tool turn's `tool_call_id`), with the template's options: `tools`
    /// (OpenAI-style function specs), `enable_thinking`,
    /// `add_generation_prompt` (true). The default renders `role` and
    /// `content` only, and refuses tools rather than drop them.
    fn apply_chat_template_json(&self, messages: &[Value], options: &Value) -> Result<String> {
        if options
            .get("tools")
            .and_then(Value::as_array)
            .is_some_and(|t| !t.is_empty())
        {
            bail!("this tokenizer's chat template takes no tools");
        }
        let turns: Vec<ChatMessage> = messages
            .iter()
            .map(|m| serde_json::from_value(m.clone()))
            .collect::<std::result::Result<_, _>>()?;
        self.apply_chat_template(&turns)
    }
}

/// Turns a tokenizer file (a `tokenizer.json`'s bytes) into a [`TextCodec`].
/// rlx-runtime has no tokenizer of its own, so a crate that has one installs
/// it with [`set_tokenizer_loader`]; rlx-text's `install_model_tokenizer`
/// does, and every language binding calls that. A package's tokenizer is
/// data, read through this.
///
/// The second argument is the chat setup, `{}` when there is none:
/// `{chat_template: "<Jinja source>", bos_token: "…", eos_token: "…"}`, which
/// the codec's [`TextCodec::apply_chat_template`] renders.
pub type TokenizerLoader = fn(&[u8], &Value) -> Result<Box<dyn TextCodec>>;

static TOKENIZER_LOADER: RwLock<Option<TokenizerLoader>> = RwLock::new(None);

/// Install the tokenizer implementation packages' tokenizers load with.
pub fn set_tokenizer_loader(loader: TokenizerLoader) {
    *TOKENIZER_LOADER.write().unwrap_or_else(|e| e.into_inner()) = Some(loader);
}

/// A [`TextCodec`] from a tokenizer file's bytes and its chat setup, through
/// the installed [`TokenizerLoader`].
pub fn load_tokenizer(bytes: &[u8], chat: &Value) -> Result<Box<dyn TextCodec>> {
    let loader = *TOKENIZER_LOADER.read().unwrap_or_else(|e| e.into_inner());
    match loader {
        Some(load) => load(bytes, chat),
        None => bail!(
            "this model has a tokenizer, and no tokenizer is installed: call \
             rlx_text::install_model_tokenizer() (the language bindings do)"
        ),
    }
}

struct LmModel {
    runner: Box<dyn LmRunner>,
    codec: Option<Box<dyn TextCodec>>,
    info: ModelInfo,
}

/// `runner` as a [`Model`] of task `"text-generation"`.
///
/// `run` takes `input_ids` (i32 / u32 / i64 / f32 ids) and returns `logits`
/// (`[vocab]`, the last position); with `{"max_tokens": n}` it generates and
/// returns `tokens` (i32) instead. `generate_text` needs `codec`.
/// `info.task`, `info.family` and `info.vocab_size` metadata are filled in
/// when empty.
pub fn lm_model(
    runner: Box<dyn LmRunner>,
    codec: Option<Box<dyn TextCodec>>,
    mut info: ModelInfo,
) -> Box<dyn Model> {
    if info.task.is_empty() {
        info.task = "text-generation".into();
    }
    if info.family.is_empty() {
        info.family = runner.family().into();
    }
    let vocab = runner.vocab_size();
    if info.inputs.is_empty() {
        info.inputs = vec![TensorSpec {
            name: "input_ids".into(),
            dtype: DType::I32,
            shape: vec![None],
        }];
        info.outputs = vec![TensorSpec {
            name: "logits".into(),
            dtype: DType::F32,
            shape: vec![Some(vocab)],
        }];
    }
    if info.metadata.is_null() {
        info.metadata = json!({});
    }
    if let Some(m) = info.metadata.as_object_mut() {
        m.entry("vocab_size").or_insert(json!(vocab));
        m.entry("has_tokenizer").or_insert(json!(codec.is_some()));
    }
    Box::new(LmModel {
        runner,
        codec,
        info,
    })
}

/// Token ids from a tensor of i32 / u32 / i64 / f32 values.
pub fn ids_of(t: &Tensor) -> Result<Vec<u32>> {
    Ok(match t.dtype {
        DType::I32 => rlx_ir::bytes::decode_le_vec::<i32>(&t.data)
            .into_iter()
            .map(|v| v as u32)
            .collect(),
        DType::U32 => rlx_ir::bytes::decode_le_vec::<u32>(&t.data),
        DType::I64 => rlx_ir::bytes::decode_le_vec::<i64>(&t.data)
            .into_iter()
            .map(|v| v as u32)
            .collect(),
        DType::F32 => rlx_ir::bytes::decode_le_vec::<f32>(&t.data)
            .into_iter()
            .map(|v| v as u32)
            .collect(),
        d => bail!("token ids must be i32, u32, i64 or f32, not {d:?}"),
    })
}

fn ids_tensor(name: &str, ids: &[u32]) -> Tensor {
    let data = ids.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect();
    Tensor::new(name, DType::I32, vec![ids.len()], data)
}

/// `response_format: "json"` (or `{"type": "json_object"}`): the reply is
/// one JSON value.
fn wants_json(options: &Value) -> bool {
    match option(options, "response_format") {
        Some(Value::String(s)) => s == "json" || s == "json_object",
        Some(Value::Object(o)) => {
            matches!(
                o.get("type").and_then(Value::as_str),
                Some("json" | "json_object")
            )
        }
        _ => false,
    }
}

/// `options.max_tokens` (or the model's `max_new_tokens`), 16 by default.
fn max_tokens(options: &Value) -> usize {
    option(options, "max_tokens")
        .or_else(|| option(options, "max_new_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(16) as usize
}

impl LmModel {
    /// `options` over the model's generation defaults
    /// (`metadata.generation`, from its `generation_config.json`): what the
    /// caller does not say, the model does. `do_sample: false` (the
    /// model's, or the caller's) is greedy whatever the temperature.
    fn with_defaults(&self, options: &Value) -> Value {
        let mut merged = self
            .info
            .metadata
            .get("generation")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        if let Some(o) = options.as_object() {
            for (k, v) in o {
                merged.insert(k.clone(), v.clone());
            }
        }
        let caller_temperature = options.get("temperature").is_some();
        if merged.get("do_sample").and_then(Value::as_bool) == Some(false)
            && !(caller_temperature && options.get("do_sample").is_none())
        {
            merged.insert("temperature".into(), json!(0.0));
        }
        Value::Object(merged)
    }

    /// Generation kept to one JSON value: each step, of the likeliest
    /// tokens, only those whose text continues a JSON prefix may be picked
    /// (the likeliest of them greedy, else sampled among them), and a stop
    /// id only once the value is whole. Ends at that stop, or when the
    /// value is over.
    fn generate_json(
        &mut self,
        prompt: &[u32],
        max_tokens: usize,
        stops: &[u32],
        options: &Value,
    ) -> Result<Vec<u32>> {
        const CANDIDATES: usize = 256;
        let codec = self
            .codec
            .as_ref()
            .ok_or_else(|| anyhow!("JSON output needs the model's tokenizer"))?;
        let opts = sample_opts(options);
        let chain = opts.clone().into_chain();
        let seed = options.get("seed").and_then(Value::as_u64).unwrap_or(0);
        let mut rng = rlx_ir::Philox4x32::new(seed);
        let mut sampler = crate::samplers::SamplerState::default();
        let mut json = json_prefix::JsonPrefix::new();
        let mut context = prompt.to_vec();
        let mut produced = Vec::new();
        while produced.len() < max_tokens && !json.done() {
            let mut logits = self.runner.predict_logits(&context)?;
            let mut order: Vec<u32> = (0..logits.len() as u32).collect();
            let take = CANDIDATES.min(order.len());
            order.select_nth_unstable_by(take - 1, |&a, &b| {
                logits[b as usize].total_cmp(&logits[a as usize])
            });
            order.truncate(take);
            order.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
            let mut allowed: Vec<(u32, json_prefix::JsonPrefix)> = Vec::new();
            for &id in &order {
                if stops.contains(&id) {
                    if json.complete() {
                        allowed.push((id, json.clone()));
                    }
                    continue;
                }
                let piece = codec.decode(&[id])?;
                let mut next = json.clone();
                if !piece.is_empty() && next.feed_str(&piece) {
                    allowed.push((id, next));
                }
            }
            let Some(first) = allowed.first().map(|(id, _)| *id) else {
                bail!("no token among the likeliest {take} continues the JSON value");
            };
            let pick = if opts.is_greedy() {
                first
            } else {
                // Sample among the allowed only.
                let keep: std::collections::HashSet<u32> =
                    allowed.iter().map(|(id, _)| *id).collect();
                for (i, l) in logits.iter_mut().enumerate() {
                    if !keep.contains(&(i as u32)) {
                        *l = f32::NEG_INFINITY;
                    }
                }
                chain.sample(&mut logits, &context, &mut sampler, &mut rng)
            };
            if stops.contains(&pick) {
                break;
            }
            json = allowed
                .into_iter()
                .find(|(id, _)| *id == pick)
                .map(|(_, s)| s)
                .expect("picked among the allowed");
            produced.push(pick);
            context.push(pick);
        }
        Ok(produced)
    }

    /// `options.stop_ids` and the tokenizer's.
    fn stop_ids(&self, options: &Value) -> Vec<u32> {
        let mut stops: Vec<u32> = option(options, "stop_ids")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|v| v as u32))
                    .collect()
            })
            .unwrap_or_default();
        if let Some(c) = &self.codec {
            stops.extend(c.stop_ids());
        }
        stops
    }

    /// Generate up to `max_tokens` after `prompt` with `options` sampling,
    /// stopping at a stop id.
    fn generate(&mut self, prompt: &[u32], options: &Value) -> Result<Vec<u32>> {
        let max_tokens = max_tokens(options);
        let stops = self.stop_ids(options);
        if wants_json(options) {
            return self.generate_json(prompt, max_tokens, &stops, options);
        }
        // `num_beams > 1`: beam search (`length_penalty`, 1 by default).
        let beams = option(options, "num_beams")
            .and_then(Value::as_u64)
            .unwrap_or(1) as usize;
        if beams > 1 {
            let penalty = option(options, "length_penalty")
                .and_then(Value::as_f64)
                .unwrap_or(1.0) as f32;
            return self
                .runner
                .generate_beam(prompt, beams, max_tokens, &stops, penalty);
        }
        let opts = sample_opts(options);
        if opts.is_greedy() {
            let mut produced = Vec::new();
            self.runner.generate(prompt, max_tokens, &mut |t| {
                produced.push(t);
                !stops.contains(&t)
            })?;
            if produced.last().is_some_and(|t| stops.contains(t)) {
                produced.pop();
            }
            return Ok(produced);
        }
        // Sampled: drive the decode loop from logits. `seed_per_step`: each
        // token's draw from its own stream (`seed + step`), its history
        // empty — as a runner that reseeds every step samples.
        let chain = opts.into_chain();
        let seed = options.get("seed").and_then(Value::as_u64).unwrap_or(0);
        let per_step = option(options, "seed_per_step")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let mut rng = rlx_ir::Philox4x32::new(seed);
        let mut state = crate::samplers::SamplerState::default();
        let mut context = prompt.to_vec();
        let mut produced = Vec::new();
        for step in 0..max_tokens {
            let mut logits = self.runner.predict_logits(&context)?;
            let next = if per_step {
                let mut rng = rlx_ir::Philox4x32::new(seed.wrapping_add(step as u64));
                let mut state = crate::samplers::SamplerState::default();
                chain.sample(&mut logits, &[], &mut state, &mut rng)
            } else {
                chain.sample(&mut logits, &context, &mut state, &mut rng)
            };
            if stops.contains(&next) {
                break;
            }
            produced.push(next);
            context.push(next);
        }
        Ok(produced)
    }
}

impl Model for LmModel {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn run(&mut self, inputs: &[Tensor], options: &Value) -> Result<Vec<Tensor>> {
        let ids = inputs
            .iter()
            .find(|t| t.name == "input_ids")
            .or_else(|| inputs.first())
            .ok_or_else(|| anyhow!("{} needs an `input_ids` tensor", describe(&self.info)))?;
        let ids = ids_of(ids)?;
        if option(options, "max_tokens").is_some() {
            let tokens = self.generate(&ids, options)?;
            return Ok(vec![ids_tensor("tokens", &tokens)]);
        }
        // Media beside the ids (an image, a speaker vector): the prompt's
        // logits with each spliced in at its placeholder token; the next
        // run, the context grown by a token, goes on from its cache.
        let media: Vec<Tensor> = inputs
            .iter()
            .filter(|t| t.name != "input_ids")
            .cloned()
            .collect();
        if !media.is_empty() && inputs.iter().any(|t| t.name == "input_ids") {
            let (_, logits) = self.runner.prefill_with_media(&ids, &media)?;
            return Ok(vec![Tensor::f32("logits", vec![logits.len()], &logits)]);
        }
        let logits = self.runner.predict_logits(&ids)?;
        Ok(vec![Tensor::f32("logits", vec![logits.len()], &logits)])
    }

    fn lm(&mut self) -> Option<&mut dyn LmRunner> {
        Some(self.runner.as_mut())
    }

    fn generate_text(&mut self, prompt: &str, options: &Value) -> Result<String> {
        let options = &self.with_defaults(options);
        let Some(codec) = &self.codec else {
            bail!(
                "{} has no tokenizer; use run / generate with ids",
                describe(&self.info)
            );
        };
        let prompt = match option(options, "messages") {
            Some(Value::Array(m)) => codec.apply_chat_template_json(m, options)? + prompt,
            Some(m) => codec.apply_chat_template(&ChatMessage::list_from_json(m)?)? + prompt,
            None => prompt.to_string(),
        };
        let ids = codec.encode(&prompt)?;
        let out = self.generate(&ids, options)?;
        self.codec.as_ref().unwrap().decode(&out)
    }

    fn generate_with_media(
        &mut self,
        prompt: &str,
        media: &[Tensor],
        options: &Value,
    ) -> Result<String> {
        let options = &self.with_defaults(options);
        let codec = self
            .codec
            .as_ref()
            .ok_or_else(|| anyhow!("{} has no tokenizer", describe(&self.info)))?;
        let prompt = match option(options, "messages") {
            Some(Value::Array(m)) => codec.apply_chat_template_json(m, options)? + prompt,
            Some(m) => codec.apply_chat_template(&ChatMessage::list_from_json(m)?)? + prompt,
            None => prompt.to_string(),
        };
        let ids = codec.encode(&prompt)?;
        let (mut context, mut logits) = self.runner.prefill_with_media(&ids, media)?;
        let stops = self.stop_ids(options);
        let opts = sample_opts(options);
        let chain = opts.clone().into_chain();
        let seed = options.get("seed").and_then(Value::as_u64).unwrap_or(0);
        let mut rng = rlx_ir::Philox4x32::new(seed);
        let mut state = crate::samplers::SamplerState::default();
        let mut produced = Vec::new();
        for _ in 0..max_tokens(options) {
            let next = if opts.is_greedy() {
                crate::lm::argmax_u32(&logits)
            } else {
                chain.sample(&mut logits, &context, &mut state, &mut rng)
            };
            if stops.contains(&next) {
                break;
            }
            produced.push(next);
            context.push(next);
            logits = self.runner.predict_logits(&context)?;
        }
        self.codec.as_ref().expect("checked").decode(&produced)
    }

    fn generate_batch(&mut self, prompts: &[&str], options: &Value) -> Result<Vec<String>> {
        let options = &self.with_defaults(options);
        // Greedy, plain prompts: the runner's batch. Else one by one.
        let together = self.codec.is_some()
            && sample_opts(options).is_greedy()
            && option(options, "messages").is_none()
            && option(options, "num_beams")
                .and_then(Value::as_u64)
                .is_none_or(|b| b <= 1);
        if !together {
            return prompts
                .iter()
                .map(|p| self.generate_text(p, options))
                .collect();
        }
        let codec = self.codec.as_ref().expect("checked");
        let ids = prompts
            .iter()
            .map(|p| codec.encode(p))
            .collect::<Result<Vec<_>>>()?;
        let stops = self.stop_ids(options);
        let outs = self
            .runner
            .generate_batch(&ids, max_tokens(options), &stops)?;
        let codec = self.codec.as_ref().expect("checked");
        outs.iter().map(|o| codec.decode(o)).collect()
    }
}

#[cfg(test)]
mod tests;
