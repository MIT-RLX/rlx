// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tasks: the short way to make a model runnable from every language.
//!
//! Implement the one trait for what the model does, and declare a
//! [`Provider`] that says which sources it opens:
//!
//! ```ignore
//! use rlx_runtime::model::{Claim, ModelRequest, Provider, SpeechToText, speech_to_text_model};
//!
//! struct Whisper { /* … */ }
//! impl SpeechToText for Whisper {
//!     fn family(&self) -> &str { "whisper" }
//!     fn transcribe(&mut self, audio: &[f32], sample_rate: u32, _o: &serde_json::Value)
//!         -> anyhow::Result<String> { /* … */ }
//! }
//!
//! static WHISPER: Provider = Provider::new("whisper", "speech-recognition", |req| {
//!     Ok(speech_to_text_model(Whisper::load(req.path(), req.device)?, req.model_info()))
//! })
//! .claims(&[Claim::GgufArch("whisper"), Claim::HfModelType("whisper")]);
//! rlx_runtime::register_provider!(WHISPER);
//! ```
//!
//! Callers then use the task's method on any [`Model`] — `transcribe`,
//! `synthesize`, `embed`, `chat` — in Rust and through every binding, for a
//! model built in or loaded from a plugin alike. The methods travel as named
//! tensors over [`Model::run`] (`audio`, `text`, `embeddings`), so the plugin
//! ABI needs nothing new.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use super::{Model, ModelInfo, ModelProvider, ModelRequest, Tensor, TensorSpec};
use rlx_ir::DType;

/// `"speech-recognition"`.
pub const SPEECH_RECOGNITION: &str = "speech-recognition";
/// `"text-to-speech"`.
pub const TEXT_TO_SPEECH: &str = "text-to-speech";
/// `"embedding"`.
pub const EMBEDDING: &str = "embedding";
/// `"text-generation"`.
pub const TEXT_GENERATION: &str = "text-generation";
/// `"text-ranking"`: a cross-encoder scoring (query, passage) pairs.
pub const TEXT_RANKING: &str = "text-ranking";

/// One turn of a conversation, for [`Model::chat`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    /// `"system"`, `"user"`, `"assistant"`, …
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// From `[{role, content}, …]`, or a bare string as one user turn.
    pub fn list_from_json(v: &Value) -> Result<Vec<Self>> {
        if let Some(s) = v.as_str() {
            return Ok(vec![Self::user(s)]);
        }
        serde_json::from_value(v.clone())
            .map_err(|e| anyhow!("messages must be [{{role, content}}, …]: {e}"))
    }
}

/// Mono audio and its sample rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

// ── the task traits ─────────────────────────────────────────

/// Audio in, text out.
pub trait SpeechToText: Send {
    /// The architecture family (`"whisper"`, …).
    fn family(&self) -> &str;

    /// The rate the model expects; audio at another rate is the model's to
    /// resample or refuse.
    fn sample_rate(&self) -> u32 {
        16_000
    }

    /// Transcribe mono `audio` at `sample_rate`. `options`: the model's own
    /// (`language`, `timestamps`, …).
    fn transcribe(&mut self, audio: &[f32], sample_rate: u32, options: &Value) -> Result<String>;
}

/// Text in, audio out.
pub trait TextToSpeech: Send {
    fn family(&self) -> &str;

    /// The rate of the audio [`TextToSpeech::synthesize`] returns.
    fn sample_rate(&self) -> u32;

    /// Speak `text`. `options`: the model's own (`voice`, `speed`, …).
    fn synthesize(&mut self, text: &str, options: &Value) -> Result<Vec<f32>>;
}

/// Texts in, one vector each out.
pub trait TextEmbedding: Send {
    fn family(&self) -> &str;

    /// The length of each vector.
    fn dim(&self) -> usize;

    fn embed(&mut self, texts: &[&str], options: &Value) -> Result<Vec<Vec<f32>>>;
}

/// A query and passages in, one relevance score per passage out.
pub trait TextRanking: Send {
    fn family(&self) -> &str;

    /// Each of `passages` scored against `query`: higher is more relevant.
    fn score(&mut self, query: &str, passages: &[&str], options: &Value) -> Result<Vec<f32>>;
}

// ── tensors for text and audio ──────────────────────────────

impl Tensor {
    /// A UTF-8 string as a `u8` tensor.
    pub fn text(name: impl Into<String>, s: &str) -> Self {
        Self::new(name, DType::U8, vec![s.len()], s.as_bytes().to_vec())
    }

    /// A `u8` tensor's bytes as a string.
    pub fn to_text(&self) -> Result<String> {
        if self.dtype != DType::U8 {
            bail!("tensor '{}' is {:?}, not text (u8)", self.name, self.dtype);
        }
        String::from_utf8(self.data.clone())
            .map_err(|_| anyhow!("tensor '{}' is not valid UTF-8", self.name))
    }
}

fn text_spec(name: &str) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dtype: DType::U8,
        shape: vec![None],
    }
}

fn audio_spec() -> TensorSpec {
    TensorSpec {
        name: "audio".into(),
        dtype: DType::F32,
        shape: vec![None],
    }
}

fn fill(info: &mut ModelInfo, task: &str, family: &str) {
    if info.task.is_empty() {
        info.task = task.into();
    }
    if info.family.is_empty() {
        info.family = family.into();
    }
    if !info.metadata.is_object() {
        info.metadata = json!({});
    }
}

fn named<'a>(outs: &'a [Tensor], name: &str, what: &str) -> Result<&'a Tensor> {
    outs.iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow!("{what}: the model returned no `{name}` output"))
}

fn rate_option(options: &Value) -> Option<u32> {
    super::option(options, "sample_rate")
        .and_then(Value::as_u64)
        .map(|r| r as u32)
}

// ── adapters: a task trait as a Model ───────────────────────

struct SttModel<R> {
    runner: R,
    info: ModelInfo,
}

/// `runner` as a [`Model`] of task `"speech-recognition"`: `run` takes
/// `audio` (f32, mono; `{"sample_rate": n}`) and returns `text`.
pub fn speech_to_text_model(
    runner: impl SpeechToText + 'static,
    mut info: ModelInfo,
) -> Box<dyn Model> {
    fill(&mut info, SPEECH_RECOGNITION, runner.family());
    if info.inputs.is_empty() {
        info.inputs = vec![audio_spec()];
        info.outputs = vec![text_spec("text")];
    }
    let rate = runner.sample_rate();
    info.metadata
        .as_object_mut()
        .unwrap()
        .entry("sample_rate")
        .or_insert(json!(rate));
    Box::new(SttModel { runner, info })
}

impl<R: SpeechToText + 'static> Model for SttModel<R> {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn run(&mut self, inputs: &[Tensor], options: &Value) -> Result<Vec<Tensor>> {
        let audio = inputs
            .iter()
            .find(|t| t.name == "audio")
            .or_else(|| inputs.first())
            .ok_or_else(|| anyhow!("{} needs an `audio` tensor", super::describe(&self.info)))?
            .to_f32()?;
        let rate = rate_option(options).unwrap_or_else(|| self.runner.sample_rate());
        let text = self.runner.transcribe(&audio, rate, options)?;
        Ok(vec![Tensor::text("text", &text)])
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32, options: &Value) -> Result<String> {
        self.runner.transcribe(audio, sample_rate, options)
    }
}

struct TtsModel<R> {
    runner: R,
    info: ModelInfo,
}

/// `runner` as a [`Model`] of task `"text-to-speech"`: `run` takes `text`
/// and returns `audio` (f32) and `sample_rate` (i32 `[1]`).
pub fn text_to_speech_model(
    runner: impl TextToSpeech + 'static,
    mut info: ModelInfo,
) -> Box<dyn Model> {
    fill(&mut info, TEXT_TO_SPEECH, runner.family());
    if info.inputs.is_empty() {
        info.inputs = vec![text_spec("text")];
        info.outputs = vec![audio_spec()];
    }
    let rate = runner.sample_rate();
    info.metadata
        .as_object_mut()
        .unwrap()
        .entry("sample_rate")
        .or_insert(json!(rate));
    Box::new(TtsModel { runner, info })
}

impl<R: TextToSpeech + 'static> Model for TtsModel<R> {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn run(&mut self, inputs: &[Tensor], options: &Value) -> Result<Vec<Tensor>> {
        let text = inputs
            .iter()
            .find(|t| t.name == "text")
            .or_else(|| inputs.first())
            .ok_or_else(|| anyhow!("{} needs a `text` tensor", super::describe(&self.info)))?
            .to_text()?;
        let audio = self.runner.synthesize(&text, options)?;
        let rate = self.runner.sample_rate() as i32;
        Ok(vec![
            Tensor::f32("audio", vec![audio.len()], &audio),
            Tensor::new(
                "sample_rate",
                DType::I32,
                vec![1],
                rate.to_le_bytes().to_vec(),
            ),
        ])
    }

    fn synthesize(&mut self, text: &str, options: &Value) -> Result<Audio> {
        Ok(Audio {
            samples: self.runner.synthesize(text, options)?,
            sample_rate: self.runner.sample_rate(),
        })
    }
}

struct EmbedModel<R> {
    runner: R,
    info: ModelInfo,
}

/// `runner` as a [`Model`] of task `"embedding"`: `run` takes one `text`
/// tensor per text and returns `embeddings` (f32 `[n, dim]`).
pub fn text_embedding_model(
    runner: impl TextEmbedding + 'static,
    mut info: ModelInfo,
) -> Box<dyn Model> {
    fill(&mut info, EMBEDDING, runner.family());
    let dim = runner.dim();
    if info.inputs.is_empty() {
        info.inputs = vec![text_spec("text")];
        info.outputs = vec![TensorSpec {
            name: "embeddings".into(),
            dtype: DType::F32,
            shape: vec![None, Some(dim)],
        }];
    }
    info.metadata
        .as_object_mut()
        .unwrap()
        .entry("dim")
        .or_insert(json!(dim));
    Box::new(EmbedModel { runner, info })
}

impl<R: TextEmbedding + 'static> Model for EmbedModel<R> {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn run(&mut self, inputs: &[Tensor], options: &Value) -> Result<Vec<Tensor>> {
        let texts = inputs
            .iter()
            .filter(|t| t.name == "text")
            .map(Tensor::to_text)
            .collect::<Result<Vec<_>>>()?;
        if texts.is_empty() {
            bail!(
                "{} needs one or more `text` tensors",
                super::describe(&self.info)
            );
        }
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let rows = self.runner.embed(&refs, options)?;
        let dim = rows.first().map_or(self.runner.dim(), Vec::len);
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        Ok(vec![Tensor::f32(
            "embeddings",
            vec![refs.len(), dim],
            &flat,
        )])
    }

    fn embed(&mut self, texts: &[&str], options: &Value) -> Result<Vec<Vec<f32>>> {
        self.runner.embed(texts, options)
    }
}

struct RankModel<R> {
    runner: R,
    info: ModelInfo,
}

/// `runner` as a [`Model`] of task `"text-ranking"`: `run` takes a `query`
/// text tensor and one `passage` text tensor per passage, and returns
/// `scores` (f32 `[n]`).
pub fn text_ranking_model(
    runner: impl TextRanking + 'static,
    mut info: ModelInfo,
) -> Box<dyn Model> {
    fill(&mut info, TEXT_RANKING, runner.family());
    if info.inputs.is_empty() {
        info.inputs = vec![text_spec("query"), text_spec("passage")];
        info.outputs = vec![TensorSpec {
            name: "scores".into(),
            dtype: DType::F32,
            shape: vec![None],
        }];
    }
    Box::new(RankModel { runner, info })
}

impl<R: TextRanking + 'static> Model for RankModel<R> {
    fn info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn run(&mut self, inputs: &[Tensor], options: &Value) -> Result<Vec<Tensor>> {
        let text = |name: &str| {
            inputs
                .iter()
                .filter(|t| t.name == name)
                .map(Tensor::to_text)
                .collect::<Result<Vec<_>>>()
        };
        let (query, passages) = (text("query")?, text("passage")?);
        let [query] = &query[..] else {
            bail!(
                "{} needs one `query` text tensor and `passage` ones",
                super::describe(&self.info)
            );
        };
        let refs: Vec<&str> = passages.iter().map(String::as_str).collect();
        let scores = self.runner.score(query, &refs, options)?;
        Ok(vec![Tensor::f32("scores", vec![scores.len()], &scores)])
    }

    fn score(&mut self, query: &str, passages: &[&str], options: &Value) -> Result<Vec<f32>> {
        self.runner.score(query, passages, options)
    }
}

// ── the task methods over `run` (any Model, plugins too) ────

/// What a caller does with a model of `task`, for the error that says a
/// method does not fit.
fn methods_for(task: &str) -> &'static str {
    match task {
        TEXT_GENERATION => "chat / generate_text / generate",
        SPEECH_RECOGNITION => "transcribe",
        TEXT_TO_SPEECH => "synthesize",
        EMBEDDING => "embed",
        TEXT_RANKING => "score",
        _ => "run",
    }
}

fn expect_task(info: &ModelInfo, task: &str, method: &str) -> Result<()> {
    if info.task == task {
        return Ok(());
    }
    bail!(
        "{method} needs a {task} model, but '{}' ({} provider) is a {} model — use {} instead",
        info.family,
        info.provider,
        info.task,
        methods_for(&info.task)
    )
}

fn with_option(options: &Value, key: &str, v: Value) -> Value {
    let mut o = if options.is_object() {
        options.clone()
    } else {
        json!({})
    };
    o[key] = v;
    o
}

pub(super) fn transcribe_via_run(
    m: &mut (impl Model + ?Sized),
    audio: &[f32],
    sample_rate: u32,
    options: &Value,
) -> Result<String> {
    expect_task(&m.info(), SPEECH_RECOGNITION, "transcribe")?;
    let o = with_option(options, "sample_rate", json!(sample_rate));
    let outs = m.run(&[Tensor::f32("audio", vec![audio.len()], audio)], &o)?;
    named(&outs, "text", "transcribe")?.to_text()
}

pub(super) fn synthesize_via_run(
    m: &mut (impl Model + ?Sized),
    text: &str,
    options: &Value,
) -> Result<Audio> {
    let info = m.info();
    expect_task(&info, TEXT_TO_SPEECH, "synthesize")?;
    let outs = m.run(&[Tensor::text("text", text)], options)?;
    let samples = named(&outs, "audio", "synthesize")?.to_f32()?;
    let sample_rate = match outs.iter().find(|t| t.name == "sample_rate") {
        Some(t) => super::ids_of(t)?.first().copied(),
        None => info
            .metadata
            .get("sample_rate")
            .and_then(Value::as_u64)
            .map(|r| r as u32),
    }
    .ok_or_else(|| anyhow!("synthesize: the model reports no sample rate"))?;
    Ok(Audio {
        samples,
        sample_rate,
    })
}

pub(super) fn embed_via_run(
    m: &mut (impl Model + ?Sized),
    texts: &[&str],
    options: &Value,
) -> Result<Vec<Vec<f32>>> {
    expect_task(&m.info(), EMBEDDING, "embed")?;
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let inputs: Vec<Tensor> = texts.iter().map(|t| Tensor::text("text", t)).collect();
    let outs = m.run(&inputs, options)?;
    let t = named(&outs, "embeddings", "embed")?;
    let flat = t.to_f32()?;
    let dim = flat.len() / texts.len();
    Ok(flat.chunks(dim.max(1)).map(<[f32]>::to_vec).collect())
}

pub(super) fn score_via_run(
    m: &mut (impl Model + ?Sized),
    query: &str,
    passages: &[&str],
    options: &Value,
) -> Result<Vec<f32>> {
    expect_task(&m.info(), TEXT_RANKING, "score")?;
    if passages.is_empty() {
        return Ok(Vec::new());
    }
    let mut inputs = vec![Tensor::text("query", query)];
    inputs.extend(passages.iter().map(|p| Tensor::text("passage", p)));
    let outs = m.run(&inputs, options)?;
    named(&outs, "scores", "score")?.to_f32()
}

pub(super) fn chat_via_generate(
    m: &mut (impl Model + ?Sized),
    messages: &[ChatMessage],
    options: &Value,
) -> Result<String> {
    let o = with_option(options, "messages", serde_json::to_value(messages)?);
    m.generate_text("", &o)
}

// ── declaring a provider ────────────────────────────────────

/// A way a [`Provider`] recognises a source. The best match across claims
/// is the provider's [`ModelProvider::probe`] score.
#[derive(Debug, Clone, Copy)]
pub enum Claim {
    /// An exact id (`"kokoro"`, `"demo:count"`). Score 100.
    Id(&'static str),
    /// Ids starting with a prefix (`"kokoro:"` claims `kokoro:af_heart`). Score 90.
    Prefix(&'static str),
    /// A GGUF file whose `general.architecture` is this. Score 70.
    GgufArch(&'static str),
    /// A Hugging Face checkpoint (a directory, or a file in one) whose
    /// `config.json` has this `model_type`. Score 70.
    HfModelType(&'static str),
    /// A file or directory with this name. Score 50.
    FileName(&'static str),
    /// A file with this extension (no dot). Score 10.
    Extension(&'static str),
}

impl Claim {
    /// How well `request` matches (0 = not at all).
    pub fn score(&self, request: &ModelRequest) -> u32 {
        let path = request.path();
        let hit = match *self {
            Claim::Id(id) => request.source == id,
            Claim::Prefix(p) => request.source.starts_with(p),
            Claim::GgufArch(a) => gguf_architecture(path).as_deref() == Some(a),
            Claim::HfModelType(t) => hf_model_type(path).as_deref() == Some(t),
            Claim::FileName(n) => path.file_name().and_then(|f| f.to_str()) == Some(n),
            Claim::Extension(e) => path
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| x.eq_ignore_ascii_case(e)),
        };
        if !hit {
            return 0;
        }
        match self {
            Claim::Id(_) => 100,
            Claim::Prefix(_) => 90,
            Claim::GgufArch(_) | Claim::HfModelType(_) => 70,
            Claim::FileName(_) => 50,
            Claim::Extension(_) => 10,
        }
    }
}

type OpenFn = fn(&ModelRequest) -> Result<Box<dyn Model>>;

/// A provider declared as data: a name, a task, what it claims, and how to
/// open. `static` + [`register_provider!`](crate::register_provider):
///
/// ```ignore
/// static KOKORO: Provider = Provider::new("kokoro", "text-to-speech", |req| { /* … */ })
///     .families(&["kokoro"])
///     .claims(&[Claim::Id("kokoro"), Claim::HfModelType("kokoro")]);
/// rlx_runtime::register_provider!(KOKORO);
/// ```
///
/// The model it opens gets its provider, task and device filled in when the
/// open function left them empty.
pub struct Provider {
    name: &'static str,
    task: &'static str,
    description: &'static str,
    families: &'static [&'static str],
    claims: &'static [Claim],
    probe: Option<fn(&ModelRequest) -> u32>,
    open: OpenFn,
}

impl Provider {
    pub const fn new(name: &'static str, task: &'static str, open: OpenFn) -> Self {
        Self {
            name,
            task,
            description: "",
            families: &[],
            claims: &[],
            probe: None,
            open,
        }
    }

    pub const fn description(mut self, d: &'static str) -> Self {
        self.description = d;
        self
    }

    pub const fn families(mut self, f: &'static [&'static str]) -> Self {
        self.families = f;
        self
    }

    pub const fn claims(mut self, c: &'static [Claim]) -> Self {
        self.claims = c;
        self
    }

    /// A custom score, combined (max) with the claims'.
    pub const fn probe(mut self, p: fn(&ModelRequest) -> u32) -> Self {
        self.probe = Some(p);
        self
    }
}

impl ModelProvider for Provider {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        self.description
    }

    fn tasks(&self) -> Vec<String> {
        vec![self.task.into()]
    }

    fn families(&self) -> Vec<String> {
        self.families.iter().map(|f| f.to_string()).collect()
    }

    fn probe(&self, request: &ModelRequest) -> u32 {
        let claimed = self
            .claims
            .iter()
            .map(|c| c.score(request))
            .max()
            .unwrap_or(0);
        claimed.max(self.probe.map_or(0, |p| p(request)))
    }

    fn open(&self, request: &ModelRequest) -> Result<Box<dyn Model>> {
        let inner = (self.open)(request)?;
        Ok(Box::new(Stamped {
            inner,
            provider: self.name,
            task: self.task,
            device: request.device_label(),
        }))
    }
}

/// A model with its provider / task / device filled in.
struct Stamped {
    inner: Box<dyn Model>,
    provider: &'static str,
    task: &'static str,
    device: String,
}

impl Model for Stamped {
    fn info(&self) -> ModelInfo {
        let mut i = self.inner.info();
        if i.provider.is_empty() {
            i.provider = self.provider.into();
        }
        if i.task.is_empty() {
            i.task = self.task.into();
        }
        if i.device.is_empty() {
            i.device = self.device.clone();
        }
        i
    }

    fn run(&mut self, inputs: &[Tensor], options: &Value) -> Result<Vec<Tensor>> {
        self.inner.run(inputs, options)
    }

    fn lm(&mut self) -> Option<&mut dyn crate::lm::LmRunner> {
        self.inner.lm()
    }

    fn generate_text(&mut self, prompt: &str, options: &Value) -> Result<String> {
        self.inner.generate_text(prompt, options)
    }

    fn generate_batch(&mut self, prompts: &[&str], options: &Value) -> Result<Vec<String>> {
        self.inner.generate_batch(prompts, options)
    }

    fn generate_with_media(
        &mut self,
        prompt: &str,
        media: &[Tensor],
        options: &Value,
    ) -> Result<String> {
        self.inner.generate_with_media(prompt, media, options)
    }

    fn chat(&mut self, messages: &[ChatMessage], options: &Value) -> Result<String> {
        self.inner.chat(messages, options)
    }

    fn transcribe(&mut self, audio: &[f32], sample_rate: u32, options: &Value) -> Result<String> {
        expect_task(&self.info(), SPEECH_RECOGNITION, "transcribe")?;
        self.inner.transcribe(audio, sample_rate, options)
    }

    fn synthesize(&mut self, text: &str, options: &Value) -> Result<Audio> {
        expect_task(&self.info(), TEXT_TO_SPEECH, "synthesize")?;
        self.inner.synthesize(text, options)
    }

    fn embed(&mut self, texts: &[&str], options: &Value) -> Result<Vec<Vec<f32>>> {
        expect_task(&self.info(), EMBEDDING, "embed")?;
        self.inner.embed(texts, options)
    }

    fn score(&mut self, query: &str, passages: &[&str], options: &Value) -> Result<Vec<f32>> {
        expect_task(&self.info(), TEXT_RANKING, "score")?;
        self.inner.score(query, passages, options)
    }
}

impl ModelRequest {
    /// A [`ModelInfo`] with this request's device and `{source}` metadata —
    /// the usual start for an adapter's `info`.
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            device: self.device_label(),
            metadata: json!({"source": self.source}),
            ..Default::default()
        }
    }

    /// An option of the provider's own, by snake_case or camelCase key.
    pub fn option(&self, key: &str) -> Option<&Value> {
        super::option(&self.options, key)
    }
}

// ── sniffing a source ───────────────────────────────────────

/// `config.json` beside a file, or in a directory.
fn config_json(path: &Path) -> Option<PathBuf> {
    let dir = if path.is_dir() { path } else { path.parent()? };
    let p = dir.join("config.json");
    p.is_file().then_some(p)
}

/// The `model_type` of a Hugging Face checkpoint's `config.json`.
pub fn hf_model_type(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(config_json(path)?).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    v.get("model_type")?.as_str().map(String::from)
}

/// The `general.architecture` of a GGUF file, read from its header.
pub fn gguf_architecture(path: &Path) -> Option<String> {
    use std::io::{BufReader, Read};
    if !path.is_file() {
        return None;
    }
    let mut r = BufReader::new(std::fs::File::open(path).ok()?);
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b4).ok()?;
    if &b4 != b"GGUF" {
        return None;
    }
    r.read_exact(&mut b4).ok()?; // version
    r.read_exact(&mut b8).ok()?; // tensor count
    r.read_exact(&mut b8).ok()?;
    let kvs = u64::from_le_bytes(b8);
    let read_u64 = |r: &mut BufReader<std::fs::File>| -> Option<u64> {
        let mut b = [0u8; 8];
        r.read_exact(&mut b).ok()?;
        Some(u64::from_le_bytes(b))
    };
    // The architecture is near the front; give up after a few keys rather
    // than walk a large tokenizer array.
    for _ in 0..kvs.min(64) {
        let klen = read_u64(&mut r)? as usize;
        if klen > 1 << 16 {
            return None;
        }
        let mut key = vec![0u8; klen];
        r.read_exact(&mut key).ok()?;
        r.read_exact(&mut b4).ok()?;
        let ty = u32::from_le_bytes(b4);
        if key == b"general.architecture" && ty == 8 {
            let n = read_u64(&mut r)? as usize;
            if n > 1 << 12 {
                return None;
            }
            let mut s = vec![0u8; n];
            r.read_exact(&mut s).ok()?;
            return String::from_utf8(s).ok();
        }
        skip_gguf_value(&mut r, ty)?;
    }
    None
}

/// Skip one GGUF metadata value of type `ty`.
fn skip_gguf_value(r: &mut impl std::io::Read, ty: u32) -> Option<()> {
    let fixed = |ty: u32| match ty {
        0 | 1 | 7 => Some(1u64),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    };
    let mut b8 = [0u8; 8];
    match ty {
        8 => {
            r.read_exact(&mut b8).ok()?;
            std::io::copy(
                &mut std::io::Read::take(&mut *r, u64::from_le_bytes(b8)),
                &mut std::io::sink(),
            )
            .ok()?;
        }
        9 => {
            let mut b4 = [0u8; 4];
            r.read_exact(&mut b4).ok()?;
            let et = u32::from_le_bytes(b4);
            r.read_exact(&mut b8).ok()?;
            let n = u64::from_le_bytes(b8);
            match fixed(et) {
                Some(w) => {
                    std::io::copy(
                        &mut std::io::Read::take(&mut *r, w * n),
                        &mut std::io::sink(),
                    )
                    .ok()?;
                }
                None => {
                    for _ in 0..n {
                        skip_gguf_value(r, et)?;
                    }
                }
            }
        }
        t => {
            std::io::copy(
                &mut std::io::Read::take(&mut *r, fixed(t)?),
                &mut std::io::sink(),
            )
            .ok()?;
        }
    }
    Some(())
}
