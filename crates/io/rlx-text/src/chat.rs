// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Chat-template engine for RLX runners.
//!
//! Replaces `LlamaModel::apply_chat_template` (llama-cpp-4) end-to-end. Two
//! sources: an inline Jinja2 string, or `tokenizer.chat_template` (and
//! `tokenizer.ggml.chat_template`) read directly from a GGUF file's
//! metadata. Rendering uses `minijinja`.
//!
//! BOS/EOS strings are looked up via `tokenizer.ggml.bos_token_id` /
//! `eos_token_id` against the `tokenizer.ggml.tokens` array (the GGUF
//! convention).

use anyhow::{Context, Result, anyhow};
use minijinja::value::Object;
use minijinja::{Environment, Error as JinjaError, ErrorKind, State, Value};
use rlx_gguf::{GgufFile, MetaValue};
use serde_json::Value as JsonValue;
use std::path::Path;
use std::sync::Arc;

/// Convenience for the M3 auto-dispatch family: load the chat template
/// + BOS/EOS strings directly from a GGUF path.
///
/// Alias for [`ChatTemplate::from_gguf`]. Use `rlx_models::run::auto_chat_template(path)`
/// next to `rlx_models::run::auto_runner(path)`.
pub fn auto_chat_template(path: &Path) -> Result<ChatTemplate> {
    ChatTemplate::from_gguf(path)
}

/// One chat turn. `role` is conventionally one of `system`, `user`,
/// `assistant`, `tool` — but templates can accept anything.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Extra Jinja variables for templates that need more than the ChatML
/// baseline (Gemma 4 thinking channel, tool schemas, …).
#[derive(Debug, Clone, Copy)]
pub struct ChatRenderOptions {
    pub add_generation_prompt: bool,
    /// Gemma 4 unified templates gate the `<|think|>` prefix and thought
    /// channel on this flag (HF `enable_thinking`, default true for IT).
    pub enable_thinking: bool,
}

impl Default for ChatRenderOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: false,
        }
    }
}

impl ChatRenderOptions {
    pub fn user_turn(add_generation_prompt: bool) -> Self {
        Self {
            add_generation_prompt,
            ..Self::default()
        }
    }

    pub fn gemma4_thinking(add_generation_prompt: bool) -> Self {
        Self {
            add_generation_prompt,
            enable_thinking: true,
        }
    }
}

/// HF chat templates call `.get(key)` / `.get(key, default)` on message
/// dicts. minijinja maps from `serde_json` do not expose that method —
/// wrap them so Gemma 4 / tool-use templates render.
#[derive(Debug, Clone)]
struct GettableValue(JsonValue);

impl GettableValue {
    fn from_json(v: JsonValue) -> Value {
        match v {
            JsonValue::Object(_) | JsonValue::Array(_) => Value::from_object(Self(v)),
            JsonValue::String(s) => Value::from(s),
            JsonValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::from(i)
                } else if let Some(u) = n.as_u64() {
                    Value::from(u)
                } else if let Some(f) = n.as_f64() {
                    Value::from(f)
                } else {
                    Value::from(n.to_string())
                }
            }
            JsonValue::Bool(b) => Value::from(b),
            JsonValue::Null => Value::from(()),
        }
    }
}

impl Object for GettableValue {
    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        match &self.0 {
            JsonValue::Object(map) => {
                let k = key.as_str()?;
                map.get(k).map(|v| GettableValue::from_json(v.clone()))
            }
            JsonValue::Array(items) => {
                let idx = key.as_usize()?;
                items.get(idx).map(|v| GettableValue::from_json(v.clone()))
            }
            _ => None,
        }
    }

    fn call_method(
        self: &Arc<Self>,
        _state: &State<'_, '_>,
        name: &str,
        args: &[Value],
    ) -> Result<Value, JinjaError> {
        if name != "get" {
            return Err(JinjaError::new(
                ErrorKind::UnknownMethod,
                format!("GettableValue has no method named {name}"),
            ));
        }
        let key = args
            .first()
            .and_then(|v| v.as_str())
            .ok_or_else(|| JinjaError::new(ErrorKind::InvalidOperation, "get() needs a key"))?;
        let default = args.get(1).cloned().unwrap_or(Value::UNDEFINED);
        Ok(match &self.0 {
            JsonValue::Object(map) => map
                .get(key)
                .map(|v| GettableValue::from_json(v.clone()))
                .unwrap_or(default),
            _ => default,
        })
    }

    fn render(self: &Arc<Self>, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            JsonValue::String(s) => write!(f, "{s}"),
            JsonValue::Number(n) => write!(f, "{n}"),
            JsonValue::Bool(b) => write!(f, "{b}"),
            JsonValue::Null => write!(f, "null"),
            JsonValue::Array(_) => write!(f, "[...]"),
            JsonValue::Object(_) => write!(f, "{{...}}"),
        }
    }
}

fn chat_messages_to_values(messages: &[ChatMessage]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            GettableValue::from_json(serde_json::json!({
                "role": m.role,
                "content": m.content,
            }))
        })
        .collect()
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
        }
    }
}

/// Where a [`ChatTemplate`] was loaded from. Useful for diagnostics and
/// for letting a caller round-trip the source string into config.
#[derive(Debug, Clone)]
pub enum ChatTemplateSource {
    Inline,
    GgufMetadata(String),
}

/// Compiled Jinja chat template + BOS/EOS strings.
pub struct ChatTemplate {
    env: Environment<'static>,
    source_text: String,
    source_kind: ChatTemplateSource,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

const TEMPLATE_NAME: &str = "chat";

fn build_env(source: String) -> Result<Environment<'static>> {
    let mut env = Environment::new();
    // HF templates occasionally call `raise_exception(msg)` for invariant
    // checks (e.g. "system must come first"). Wire it to a Jinja error.
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<Value, JinjaError> {
            Err(JinjaError::new(ErrorKind::InvalidOperation, msg))
        },
    );
    // Poolside / Unsloth Jinja uses Python str methods (`.strip()`, `.rstrip()`,
    // …). MiniJinja does not expose those on builtins — bridge the common ones.
    env.set_unknown_method_callback(hf_string_method_callback);
    env.add_template_owned(TEMPLATE_NAME, source)
        .context("compiling chat template")?;
    Ok(env)
}

/// Minimal HF/Python string method bridge for chat templates.
fn hf_string_method_callback(
    _state: &State<'_, '_>,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, JinjaError> {
    let Some(s) = value.as_str() else {
        return Err(JinjaError::new(
            ErrorKind::UnknownMethod,
            format!("object has no method named {method}"),
        ));
    };
    if !args.is_empty() {
        return Err(JinjaError::new(
            ErrorKind::InvalidOperation,
            format!("{method}() takes no arguments in this bridge"),
        ));
    }
    let out = match method {
        "strip" => s.trim(),
        "lstrip" => s.trim_start(),
        "rstrip" => s.trim_end(),
        "lower" => return Ok(Value::from(s.to_lowercase())),
        "upper" => return Ok(Value::from(s.to_uppercase())),
        other => {
            return Err(JinjaError::new(
                ErrorKind::UnknownMethod,
                format!("string has no method named {other}"),
            ));
        }
    };
    Ok(Value::from(out))
}

impl ChatTemplate {
    /// Compile a chat template from a raw Jinja string.
    pub fn from_source(src: impl Into<String>) -> Result<Self> {
        let source_text: String = src.into();
        let env = build_env(source_text.clone())?;
        Ok(Self {
            env,
            source_text,
            source_kind: ChatTemplateSource::Inline,
            bos_token: None,
            eos_token: None,
        })
    }

    /// Override BOS/EOS strings (passed to the template as `bos_token` /
    /// `eos_token` Jinja variables).
    pub fn with_tokens(mut self, bos: Option<String>, eos: Option<String>) -> Self {
        self.bos_token = bos;
        self.eos_token = eos;
        self
    }

    /// Load template + BOS/EOS from a GGUF file. Reads
    /// `tokenizer.chat_template` first, then `tokenizer.ggml.chat_template`.
    pub fn from_gguf(path: &Path) -> Result<Self> {
        let raw = GgufFile::from_path(path).with_context(|| format!("opening GGUF {path:?}"))?;
        Self::from_gguf_file(&raw)
    }

    /// Same as [`from_gguf`](Self::from_gguf), but reuses an already-parsed file.
    pub fn from_gguf_file(raw: &GgufFile) -> Result<Self> {
        let (key, src) = pick_chat_template_meta(raw).ok_or_else(|| {
            anyhow!("no tokenizer.chat_template or tokenizer.ggml.chat_template in GGUF metadata")
        })?;
        let env = build_env(src.clone())?;
        let bos = resolve_special_token(raw, "tokenizer.ggml.bos_token_id");
        let eos = resolve_special_token(raw, "tokenizer.ggml.eos_token_id");
        Ok(Self {
            env,
            source_text: src,
            source_kind: ChatTemplateSource::GgufMetadata(key.to_owned()),
            bos_token: bos,
            eos_token: eos,
        })
    }

    pub fn source_text(&self) -> &str {
        &self.source_text
    }

    pub fn source_kind(&self) -> &ChatTemplateSource {
        &self.source_kind
    }

    pub fn bos_token(&self) -> Option<&str> {
        self.bos_token.as_deref()
    }

    pub fn eos_token(&self) -> Option<&str> {
        self.eos_token.as_deref()
    }

    /// Render the template with the given messages.
    ///
    /// The template sees Jinja variables: `messages` (list of
    /// `{role, content}` maps), `add_generation_prompt` (bool), and
    /// `bos_token` / `eos_token` strings (empty if unknown).
    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        self.render_with_options(
            messages,
            ChatRenderOptions {
                add_generation_prompt,
                ..ChatRenderOptions::default()
            },
        )
    }

    /// Render with extra template knobs (`enable_thinking`, …).
    pub fn render_with_options(
        &self,
        messages: &[ChatMessage],
        opts: ChatRenderOptions,
    ) -> Result<String> {
        let msgs = chat_messages_to_values(messages);
        let ctx = minijinja::context! {
            messages => Value::from(msgs),
            add_generation_prompt => opts.add_generation_prompt,
            enable_thinking => opts.enable_thinking,
            bos_token => self.bos_token.clone().unwrap_or_default(),
            eos_token => self.eos_token.clone().unwrap_or_default(),
        };
        let tmpl = self
            .env
            .get_template(TEMPLATE_NAME)
            .expect("template registered in build_env");
        tmpl.render(ctx).context("rendering chat template")
    }
}

fn pick_chat_template_meta(raw: &GgufFile) -> Option<(&'static str, String)> {
    for key in ["tokenizer.chat_template", "tokenizer.ggml.chat_template"] {
        if let Some(MetaValue::String(s)) = raw.metadata.get(key) {
            return Some((key, s.clone()));
        }
    }
    None
}

fn resolve_special_token(raw: &GgufFile, id_key: &str) -> Option<String> {
    let id = raw.metadata.get(id_key).and_then(MetaValue::as_u32)? as usize;
    let toks = raw.metadata.get("tokenizer.ggml.tokens")?;
    let MetaValue::Array(arr) = toks else {
        return None;
    };
    match arr.get(id)? {
        MetaValue::String(s) => Some(s.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal Qwen / ChatML-style template — same shape as Qwen3's, simplified
    // enough that test failures point at our rendering plumbing not at
    // upstream Jinja quirks. Whitespace-trim markers are intentionally
    // avoided so the literal `\n` inside the template survives.
    const QWEN_TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    // Minimal Llama-3-style template using bos_token + headers.
    const LLAMA3_TEMPLATE: &str = "{% for m in messages %}{% if loop.first %}{{ bos_token }}{% endif %}<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";

    // Minimal Gemma-style template.
    const GEMMA_TEMPLATE: &str = "{% for m in messages %}{% set role = 'user' if m.role == 'system' else m.role %}<start_of_turn>{{ role }}\n{{ m.content }}<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}";

    fn sample_conv() -> Vec<ChatMessage> {
        vec![ChatMessage::system("be concise"), ChatMessage::user("hi")]
    }

    #[test]
    fn qwen_template_renders_with_generation_prompt() {
        let t = ChatTemplate::from_source(QWEN_TEMPLATE).unwrap();
        let out = t.render(&sample_conv(), true).unwrap();
        let expected = "<|im_start|>system\nbe concise<|im_end|>\n\
                        <|im_start|>user\nhi<|im_end|>\n\
                        <|im_start|>assistant\n";
        assert_eq!(out, expected);
    }

    #[test]
    fn qwen_template_omits_generation_prompt_when_disabled() {
        let t = ChatTemplate::from_source(QWEN_TEMPLATE).unwrap();
        let out = t.render(&sample_conv(), false).unwrap();
        assert!(out.ends_with("<|im_end|>\n"));
        assert!(!out.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn llama3_template_uses_bos_token() {
        let t = ChatTemplate::from_source(LLAMA3_TEMPLATE)
            .unwrap()
            .with_tokens(Some("<|begin_of_text|>".into()), Some("<|eot_id|>".into()));
        let out = t.render(&sample_conv(), true).unwrap();
        let expected = "<|begin_of_text|>\
                        <|start_header_id|>system<|end_header_id|>\n\nbe concise<|eot_id|>\
                        <|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>\
                        <|start_header_id|>assistant<|end_header_id|>\n\n";
        assert_eq!(out, expected);
        assert_eq!(t.bos_token(), Some("<|begin_of_text|>"));
        assert_eq!(t.eos_token(), Some("<|eot_id|>"));
    }

    #[test]
    fn gemma_template_rewrites_system_to_user() {
        let t = ChatTemplate::from_source(GEMMA_TEMPLATE).unwrap();
        let out = t.render(&sample_conv(), true).unwrap();
        let expected = "<start_of_turn>user\nbe concise<end_of_turn>\n\
                        <start_of_turn>user\nhi<end_of_turn>\n\
                        <start_of_turn>model\n";
        assert_eq!(out, expected);
    }

    #[test]
    fn dict_get_method_works_like_hf_templates() {
        const TEMPLATE: &str =
            "{% for m in messages %}{{ m.get('role') }}:{{ m.get('content') }};{% endfor %}";
        let t = ChatTemplate::from_source(TEMPLATE).unwrap();
        let out = t
            .render(
                &[ChatMessage::user("hi"), ChatMessage::assistant("yo")],
                false,
            )
            .unwrap();
        assert_eq!(out, "user:hi;assistant:yo;");
    }

    #[test]
    fn enable_thinking_is_visible_to_template() {
        const TEMPLATE: &str = "{% if enable_thinking %}think{% else %}plain{% endif %}";
        let t = ChatTemplate::from_source(TEMPLATE).unwrap();
        let on = t
            .render_with_options(&[], ChatRenderOptions::gemma4_thinking(false))
            .unwrap();
        let off = t.render(&[], false).unwrap();
        assert_eq!(on, "think");
        assert_eq!(off, "plain");
    }

    #[test]
    fn raise_exception_propagates_as_error() {
        let t = ChatTemplate::from_source("{{ raise_exception('nope') }}").unwrap();
        let err = t.render(&[], false).unwrap_err();
        assert!(format!("{err:#}").contains("nope"));
    }

    #[test]
    fn python_string_strip_methods_work() {
        const TEMPLATE: &str =
            "{% set s = '  hi  ' %}{{ s.strip() }}|{{ s.lstrip() }}|{{ s.rstrip() }}";
        let t = ChatTemplate::from_source(TEMPLATE).unwrap();
        let out = t.render(&[], false).unwrap();
        assert_eq!(out, "hi|hi  |  hi");
    }

    /// Builds a minimal GGUF in a temp file with a chat_template + token
    /// table, then verifies BOS/EOS resolve and rendering works.
    #[test]
    fn from_gguf_reads_template_and_special_tokens() {
        // We build a v3 GGUF with three metadata keys:
        //   tokenizer.chat_template      (String)
        //   tokenizer.ggml.tokens        (Array of String)
        //   tokenizer.ggml.bos_token_id  (U32)
        //   tokenizer.ggml.eos_token_id  (U32)
        // and one tiny f32 tensor so the file passes the loader.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&rlx_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&4u64.to_le_bytes()); // kv count

        let write_string_kv = |buf: &mut Vec<u8>, k: &str, v: &str| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&8u32.to_le_bytes());
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            buf.extend_from_slice(v.as_bytes());
        };
        let write_u32_kv = |buf: &mut Vec<u8>, k: &str, v: u32| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        };
        let write_string_array_kv = |buf: &mut Vec<u8>, k: &str, items: &[&str]| {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            // type = Array(9)
            buf.extend_from_slice(&9u32.to_le_bytes());
            // element type = String(8)
            buf.extend_from_slice(&8u32.to_le_bytes());
            // length (u64)
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for s in items {
                buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                buf.extend_from_slice(s.as_bytes());
            }
        };

        write_string_kv(&mut buf, "tokenizer.chat_template", QWEN_TEMPLATE);
        write_string_array_kv(
            &mut buf,
            "tokenizer.ggml.tokens",
            &["<pad>", "<bos>", "<eos>", "hi"],
        );
        write_u32_kv(&mut buf, "tokenizer.ggml.bos_token_id", 1);
        write_u32_kv(&mut buf, "tokenizer.ggml.eos_token_id", 2);

        // tiny f32 tensor
        let name = "w";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&(rlx_gguf::GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        while !buf
            .len()
            .is_multiple_of(rlx_gguf::DEFAULT_ALIGNMENT as usize)
        {
            buf.push(0);
        }
        for _ in 0..4 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let path = std::env::temp_dir().join("rlx_chat_template_from_gguf.gguf");
        std::fs::write(&path, &buf).unwrap();

        let t = ChatTemplate::from_gguf(&path).expect("from_gguf");
        assert_eq!(t.bos_token(), Some("<bos>"));
        assert_eq!(t.eos_token(), Some("<eos>"));
        let out = t.render(&sample_conv(), true).unwrap();
        assert!(out.contains("<|im_start|>assistant\n"));
        match t.source_kind() {
            ChatTemplateSource::GgufMetadata(k) => assert_eq!(k, "tokenizer.chat_template"),
            other => panic!("unexpected source: {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }
}
