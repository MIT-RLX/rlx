// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! This crate's tokenizer as rlx-runtime's [`TextCodec`], so a model package
//! that ships a `tokenizer.json` (and a chat template) tokenizes with it.

use anyhow::Result;
use rlx_runtime::model::TextCodec;
use serde_json::Value;

use crate::chat::{ChatMessage, ChatTemplate};
use crate::tokenizer::{RawTokenizer, TokenizerHandle};

/// A tokenizer as a model's codec: special tokens added on encode (`<s> …
/// </s>`, `[CLS] … [SEP]`), skipped on decode, and the model's chat template
/// when it has one.
pub struct ModelTokenizer {
    pub tokenizer: TokenizerHandle,
    pub template: Option<ChatTemplate>,
}

impl TextCodec for ModelTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // A chat model's prompt carries its special tokens in the text.
        self.tokenizer.encode(text, self.template.is_none())
    }

    fn encode_plain(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text, false)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, true)
    }

    fn encode_pair(&self, a: &str, b: &str) -> Result<(Vec<u32>, Vec<u32>)> {
        let enc = self
            .tokenizer
            .raw()
            .encode((a, b), true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
        Ok((enc.get_ids().to_vec(), enc.get_type_ids().to_vec()))
    }

    fn apply_chat_template(&self, messages: &[rlx_runtime::model::ChatMessage]) -> Result<String> {
        let Some(t) = &self.template else {
            anyhow::bail!("this model has no chat template; use generate_text with a prompt");
        };
        let turns: Vec<ChatMessage> = messages
            .iter()
            .map(|m| ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();
        t.render(&turns, true)
    }

    fn apply_chat_template_json(&self, messages: &[Value], options: &Value) -> Result<String> {
        let Some(t) = &self.template else {
            anyhow::bail!("this model has no chat template; use generate_text with a prompt");
        };
        render_chat_json(t, messages, options)
    }
}

/// `messages` (JSON, with whatever keys the template reads) through `t`,
/// with the options a model's `chat` takes: `tools`, `enable_thinking`,
/// `add_generation_prompt` (true). Tools for a template with no `tools`
/// branch are refused rather than dropped.
pub fn render_chat_json(t: &ChatTemplate, messages: &[Value], options: &Value) -> Result<String> {
    let tools: Vec<Value> = options
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if !tools.is_empty() && !t.supports_tools() {
        anyhow::bail!("this model's chat template has no tools; it cannot be told about them");
    }
    let flag = |k: &str, d: bool| options.get(k).and_then(Value::as_bool).unwrap_or(d);
    t.render_json_with_options(
        messages,
        crate::chat::ChatRenderOptions {
            add_generation_prompt: flag("add_generation_prompt", true),
            enable_thinking: flag("enable_thinking", false),
            tools,
        },
    )
}

/// A `tokenizer.json`'s bytes as a [`TextCodec`], with the chat setup
/// `{chat_template, bos_token, eos_token}` (or `null`).
pub fn model_tokenizer(bytes: &[u8], chat: &Value) -> Result<Box<dyn TextCodec>> {
    let raw =
        RawTokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
    let token = |k: &str| chat.get(k).and_then(Value::as_str).map(String::from);
    let template = match chat.get("chat_template").and_then(Value::as_str) {
        Some(src) => Some(
            ChatTemplate::from_source(src)?.with_tokens(token("bos_token"), token("eos_token")),
        ),
        None => None,
    };
    Ok(Box::new(ModelTokenizer {
        tokenizer: TokenizerHandle::from_raw(raw),
        template,
    }))
}

/// Make rlx-runtime load packages' tokenizers with this crate
/// ([`rlx_runtime::model::set_tokenizer_loader`]). Idempotent.
pub fn install_model_tokenizer() {
    rlx_runtime::model::set_tokenizer_loader(model_tokenizer);
}
