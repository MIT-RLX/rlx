// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reference model plugin: one demo model per task, in a shared library
//! that `librlx_capi` loads at run time — so every language binding opens its
//! models with `Model.open("demo:count")` without rlx linking this crate —
//! or linked into a host that cannot load one.
//!
//! It is what a downstream model collection builds, in miniature:
//!
//! 1. the model, behind its task's trait — [`LmRunner`] (+ a [`TextCodec`]),
//!    [`SpeechToText`], [`TextToSpeech`] or [`TextEmbedding`];
//! 2. a [`Provider`] per model: its name, task, the sources it claims, and
//!    how to open one;
//! 3. [`export_model_plugin!`](rlx_runtime::export_model_plugin) listing
//!    them: loaded, they are the `rlx_model_plugin_v1` table; linked into a
//!    host that cannot load libraries (iOS, wasm), they are
//!    [`link_model_plugins!`](rlx_runtime::link_model_plugins)'s.
//!
//! The models: `demo:count` counts on from the last letter
//! (`chat("c")` is `"defg…"`); `demo:voice` speaks letters, one sample each,
//! and `demo:speech` reads them back; `demo:embed` maps a text to
//! `[length, vowels]`.
//!
//! ```text
//! cargo build -p rlx-model-plugin
//! RLX_MODEL_PLUGINS=target/debug/librlx_model_plugin.dylib python -c \
//!   'import pyrlx; print(pyrlx.Model.open("demo:count").generate_text("c", max_tokens=4))'
//! ```

use anyhow::{Result, bail};
use rlx_runtime::LmRunner;
use rlx_runtime::model::{
    ChatMessage, Claim, Provider, SpeechToText, TextCodec, TextEmbedding, TextToSpeech, lm_model,
    speech_to_text_model, text_embedding_model, text_to_speech_model,
};
use serde_json::Value;

const VOCAB: usize = 26;

/// A language model whose next token is `(last + 1) % 26`, sharply.
pub struct CountLm;

impl LmRunner for CountLm {
    fn family(&self) -> &'static str {
        "count"
    }

    fn vocab_size(&self) -> usize {
        VOCAB
    }

    fn predict_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let next = (ids.last().copied().unwrap_or(25) as usize + 1) % VOCAB;
        Ok((0..VOCAB)
            .map(|i| if i == next { 20.0 } else { 0.0 })
            .collect())
    }
}

/// Lower-case letters as ids, `a` = 0; `z` ends a generation.
pub struct Letters;

impl TextCodec for Letters {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        text.bytes()
            .map(|b| match b {
                b'a'..=b'z' => Ok((b - b'a') as u32),
                _ => bail!("the demo tokenizer knows only a-z, not {:?}", b as char),
            })
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(ids
            .iter()
            .map(|&i| (b'a' + (i % 26) as u8) as char)
            .collect())
    }

    fn stop_ids(&self) -> Vec<u32> {
        vec![25]
    }

    /// The conversation's last user turn: enough for a model that counts on.
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .ok_or_else(|| anyhow::anyhow!("demo:count needs a user message"))
    }
}

/// Speech, one sample per letter: `a` = 1/32 … `z` = 26/32, so
/// `demo:speech` reads back what `demo:voice` says.
fn letter_to_sample(b: u8) -> f32 {
    f32::from(b - b'a' + 1) / 32.0
}

fn sample_to_letter(v: f32) -> Option<char> {
    let i = (v * 32.0).round() as i32 - 1;
    (0..26).contains(&i).then(|| (b'a' + i as u8) as char)
}

/// `demo:speech` — a speech recogniser for `demo:voice`'s audio.
pub struct Ear;

impl SpeechToText for Ear {
    fn family(&self) -> &str {
        "letters"
    }

    fn transcribe(&mut self, audio: &[f32], _sample_rate: u32, _: &Value) -> Result<String> {
        Ok(audio.iter().filter_map(|&v| sample_to_letter(v)).collect())
    }
}

/// `demo:voice` — speaks lower-case letters at 16 kHz.
pub struct Voice;

impl TextToSpeech for Voice {
    fn family(&self) -> &str {
        "letters"
    }

    fn sample_rate(&self) -> u32 {
        16_000
    }

    fn synthesize(&mut self, text: &str, _: &Value) -> Result<Vec<f32>> {
        Ok(text
            .bytes()
            .filter(u8::is_ascii_lowercase)
            .map(letter_to_sample)
            .collect())
    }
}

/// `demo:embed` — a text as `[length, vowels]`.
pub struct Shape;

impl TextEmbedding for Shape {
    fn family(&self) -> &str {
        "shape"
    }

    fn dim(&self) -> usize {
        2
    }

    fn embed(&mut self, texts: &[&str], _: &Value) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let vowels = t.chars().filter(|c| "aeiou".contains(*c)).count();
                vec![t.chars().count() as f32, vowels as f32]
            })
            .collect())
    }
}

// Each provider: a name, a task, what it claims, how to open.

static COUNT: Provider = Provider::new("demo", "text-generation", |req| {
    Ok(lm_model(
        Box::new(CountLm),
        Some(Box::new(Letters)),
        req.model_info(),
    ))
})
.description("the reference plugin's counting language model (demo:count)")
.families(&["count"])
.claims(&[Claim::Id("demo:count")]);

static SPEECH: Provider = Provider::new("demo-speech", "speech-recognition", |req| {
    Ok(speech_to_text_model(Ear, req.model_info()))
})
.description("reads back demo:voice's audio (demo:speech)")
.families(&["letters"])
.claims(&[Claim::Id("demo:speech")]);

static VOICE: Provider = Provider::new("demo-voice", "text-to-speech", |req| {
    Ok(text_to_speech_model(Voice, req.model_info()))
})
.description("speaks lower-case letters, one sample each (demo:voice)")
.families(&["letters"])
.claims(&[Claim::Id("demo:voice")]);

static EMBED: Provider = Provider::new("demo-embed", "embedding", |req| {
    Ok(text_embedding_model(Shape, req.model_info()))
})
.description("embeds a text as [length, vowels] (demo:embed)")
.families(&["shape"])
.claims(&[Claim::Id("demo:embed")]);

rlx_runtime::export_model_plugin!("rlx-model-plugin", [COUNT, SPEECH, VOICE, EMBED]);

#[cfg(test)]
mod tests {
    use rlx_runtime::model::{self, ChatMessage, plugin};
    use serde_json::json;

    #[test]
    fn the_exported_table_opens_and_runs_every_demo_model() {
        // Through the exported C table, as librlx_capi would.
        let mut names = unsafe { plugin::register_table(super::rlx_model_plugin_v1()) }.unwrap();
        names.sort();
        assert_eq!(names, ["demo", "demo-embed", "demo-speech", "demo-voice"]);
        let none = json!({});

        let mut lm = model::open_source("demo:count", json!({})).unwrap();
        assert_eq!(lm.info().task, "text-generation");
        assert_eq!(
            lm.generate_text("c", &json!({"max_tokens": 4})).unwrap(),
            "defg"
        );
        // `y` → `z` is the stop id.
        assert_eq!(
            lm.generate_text("x", &json!({"max_tokens": 9})).unwrap(),
            "y"
        );
        let reply = lm.chat(&[ChatMessage::user("c")], &json!({"max_tokens": 4}));
        assert_eq!(reply.unwrap(), "defg");

        let mut voice = model::open_source("demo:voice", json!({})).unwrap();
        let audio = voice.synthesize("hello", &none).unwrap();
        assert_eq!((audio.samples.len(), audio.sample_rate), (5, 16_000));
        let mut ear = model::open_source("demo:speech", json!({})).unwrap();
        assert_eq!(
            ear.transcribe(&audio.samples, audio.sample_rate, &none)
                .unwrap(),
            "hello"
        );

        let mut e = model::open_source("demo:embed", json!({})).unwrap();
        assert_eq!(
            e.embed(&["hello", "sky"], &none).unwrap(),
            [[5.0, 2.0], [3.0, 0.0]]
        );
    }
}
