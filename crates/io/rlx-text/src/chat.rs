// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
#[derive(Debug, Clone)]
pub struct ChatRenderOptions {
    pub add_generation_prompt: bool,
    /// Gemma 4 unified templates gate the `<|think|>` prefix and thought
    /// channel on this flag (HF `enable_thinking`, default true for IT).
    pub enable_thinking: bool,
    /// OpenAI-style tool specifications, exposed to the template as `tools`.
    ///
    /// Chat templates for tool-capable models (Qwen3, Llama 3.x, Mistral, …)
    /// branch on `tools` to emit a system block describing the callable
    /// functions. Leaving it unset meant that branch never fired, so a caller
    /// that passed tools got a model that had never been told about them — it
    /// answers in prose and the tools look broken rather than unsupported.
    pub tools: Vec<JsonValue>,
}

impl Default for ChatRenderOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: false,
            tools: Vec::new(),
        }
    }
}

impl ChatTemplate {
    /// Whether this template has a `tools` branch.
    ///
    /// Checked against the template source because that is the only honest
    /// signal available: a template with no `tools` reference will silently
    /// ignore anything passed, and a caller needs to be able to say
    /// "this model cannot do tool calling" rather than answer as if it had.
    pub fn supports_tools(&self) -> bool {
        self.source_text.contains("tools")
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
            ..Self::default()
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
            // Arrays become real minijinja sequences, not wrapped objects. A
            // wrapper only answers `get_value`, which is indexing — `{% for %}`
            // over it yields nothing, so a nested array (an assistant turn's
            // `tool_calls`, a multi-part `content`) rendered as empty with no
            // error. Objects stay wrapped, because that is what gives templates
            // the `.get(key)` method they call.
            JsonValue::Array(items) => Value::from(
                items
                    .into_iter()
                    .map(Self::from_json)
                    .collect::<Vec<Value>>(),
            ),
            JsonValue::Object(_) => Value::from_object(Self(v)),
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
            // Arrays never reach here: `from_json` turns them into sequences.
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
    // Tool-use templates serialize each tool's schema with `| tojson`. MiniJinja
    // ships that filter only with its `json` feature, which this crate does not
    // enable, so a tools render fails outright with "unknown filter" — not a
    // degraded prompt, no render at all. Provide it over `serde_json`, which is
    // already a dependency, and honour the `indent` argument HF templates pass.
    env.add_filter(
        "tojson",
        |v: Value, kwargs: minijinja::value::Kwargs| -> Result<String, JinjaError> {
            // `tojson(indent=2)` passes a keyword, which arrives as `Kwargs`
            // rather than positionally — taking it as `Option<usize>` makes
            // minijinja try to convert the whole kwargs map and the render dies.
            let indent: Option<usize> = kwargs.get("indent").ok();
            kwargs.assert_all_used()?;
            // A wrapped dict has to be unwrapped first: `GettableValue` exists to
            // give templates a `.get()` method, and serializing the wrapper
            // yields `{}` — which renders a tools block listing no tools, the
            // exact silent-but-plausible failure this is all guarding against.
            let json: JsonValue = match v.downcast_object_ref::<GettableValue>() {
                Some(g) => g.0.clone(),
                None => serde_json::to_value(&v).map_err(|e| {
                    JinjaError::new(ErrorKind::InvalidOperation, format!("tojson: {e}"))
                })?,
            };
            let out = match indent {
                Some(n) if n > 0 => {
                    let pad = vec![b' '; n];
                    let fmt = serde_json::ser::PrettyFormatter::with_indent(&pad);
                    let mut buf = Vec::new();
                    let mut ser = serde_json::Serializer::with_formatter(&mut buf, fmt);
                    serde::Serialize::serialize(&json, &mut ser).map_err(|e| {
                        JinjaError::new(ErrorKind::InvalidOperation, format!("tojson: {e}"))
                    })?;
                    String::from_utf8(buf).unwrap_or_default()
                }
                _ => json.to_string(),
            };
            Ok(out)
        },
    );
    env.add_template_owned(TEMPLATE_NAME, source)
        .context("compiling chat template")?;
    Ok(env)
}

/// Python string methods that HF chat templates call, bridged onto MiniJinja.
///
/// The set is driven by what real templates use, not by completeness. The
/// previous version rejected *every* call that had an argument, which meant
/// `startswith(prefix)`, `endswith`, `split(sep)` and `replace(a, b)` all
/// failed — Qwen3's template could not render at all, and the error named only
/// the first method it happened to hit.
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

    let arg_str =
        |i: usize| -> Option<String> { args.get(i).and_then(|v| v.as_str()).map(str::to_owned) };
    let wrong_args = |want: &str| {
        JinjaError::new(
            ErrorKind::InvalidOperation,
            format!("{method}() expects {want}"),
        )
    };

    match method {
        // ── no arguments ──
        "strip" | "lstrip" | "rstrip" if args.is_empty() => Ok(Value::from(match method {
            "strip" => s.trim(),
            "lstrip" => s.trim_start(),
            _ => s.trim_end(),
        })),
        "lower" => Ok(Value::from(s.to_lowercase())),
        "upper" => Ok(Value::from(s.to_uppercase())),
        "title" => Ok(Value::from(title_case(s))),
        "capitalize" => {
            let mut chars = s.chars();
            Ok(Value::from(match chars.next() {
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                }
                None => String::new(),
            }))
        }
        "splitlines" => Ok(Value::from(s.lines().map(Value::from).collect::<Vec<_>>())),

        // ── stripping a character set, as Python allows ──
        "strip" | "lstrip" | "rstrip" => {
            let chars: Vec<char> = arg_str(0)
                .ok_or_else(|| wrong_args("a string of characters"))?
                .chars()
                .collect();
            let trimmed = match method {
                "strip" => s.trim_matches(|c| chars.contains(&c)),
                "lstrip" => s.trim_start_matches(|c| chars.contains(&c)),
                _ => s.trim_end_matches(|c| chars.contains(&c)),
            };
            Ok(Value::from(trimmed))
        }

        // ── prefix / suffix tests ──
        //
        // Python accepts a tuple of candidates as well as a single string, and
        // templates do use that form, so both are handled.
        "startswith" | "endswith" => {
            let first = args.first().ok_or_else(|| wrong_args("a prefix"))?;
            let candidates: Vec<String> = match first.as_str() {
                Some(one) => vec![one.to_owned()],
                None => first
                    .try_iter()
                    .map_err(|_| wrong_args("a string or a sequence of strings"))?
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect(),
            };
            let hit = candidates.iter().any(|c| {
                if method == "startswith" {
                    s.starts_with(c.as_str())
                } else {
                    s.ends_with(c.as_str())
                }
            });
            Ok(Value::from(hit))
        }

        // ── splitting ──
        //
        // Bare `split()` splits on any whitespace run and drops empties, which
        // is Python's behaviour and *not* the same as splitting on " ".
        "split" | "rsplit" => {
            let sep = arg_str(0);
            let limit = args
                .get(1)
                .and_then(|v| v.as_i64())
                .filter(|n| *n >= 0)
                .map(|n| n as usize);
            let parts: Vec<Value> = match (&sep, limit) {
                (None, _) => s.split_whitespace().map(Value::from).collect(),
                (Some(sep), None) => s.split(sep.as_str()).map(Value::from).collect(),
                (Some(sep), Some(n)) if method == "split" => {
                    s.splitn(n + 1, sep.as_str()).map(Value::from).collect()
                }
                (Some(sep), Some(n)) => {
                    let mut v: Vec<Value> =
                        s.rsplitn(n + 1, sep.as_str()).map(Value::from).collect();
                    v.reverse();
                    v
                }
            };
            Ok(Value::from(parts))
        }

        "replace" => {
            let from = arg_str(0).ok_or_else(|| wrong_args("(old, new)"))?;
            let to = arg_str(1).ok_or_else(|| wrong_args("(old, new)"))?;
            let count = args.get(2).and_then(|v| v.as_i64()).filter(|n| *n >= 0);
            Ok(Value::from(match count {
                Some(n) => s.replacen(from.as_str(), &to, n as usize),
                None => s.replace(from.as_str(), &to),
            }))
        }

        "join" => {
            let items = args.first().ok_or_else(|| wrong_args("a sequence"))?;
            let parts: Vec<String> = items
                .try_iter()
                .map_err(|_| wrong_args("a sequence"))?
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| v.to_string())
                })
                .collect();
            Ok(Value::from(parts.join(s)))
        }

        "count" => {
            let needle = arg_str(0).ok_or_else(|| wrong_args("a substring"))?;
            Ok(Value::from(s.matches(needle.as_str()).count()))
        }
        // Python's `find` answers -1 rather than raising, and returns a *byte*
        // index here; templates use it as a containment test.
        "find" => {
            let needle = arg_str(0).ok_or_else(|| wrong_args("a substring"))?;
            Ok(Value::from(
                s.find(needle.as_str()).map(|i| i as i64).unwrap_or(-1),
            ))
        }

        other => Err(JinjaError::new(
            ErrorKind::UnknownMethod,
            format!("string has no method named {other} in this bridge"),
        )),
    }
}

/// Python's `str.title()`: capitalize each run of letters.
fn title_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut start_of_word = true;
    for c in s.chars() {
        if c.is_alphabetic() {
            if start_of_word {
                out.extend(c.to_uppercase());
            } else {
                out.extend(c.to_lowercase());
            }
            start_of_word = false;
        } else {
            out.push(c);
            start_of_word = true;
        }
    }
    out
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
        let msgs: Vec<JsonValue> = messages
            .iter()
            .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
            .collect();
        self.render_json_with_options(&msgs, opts)
    }

    /// Render from raw JSON messages, for fields [`ChatMessage`] does not carry.
    ///
    /// Tool-use templates read more than `role` and `content`: an assistant turn
    /// is rendered with its `tool_calls`, and a `tool` turn with the result it
    /// carries. Without those a multi-turn tool conversation loses the model's own
    /// call from its history — it is told a result arrived for a call it has no
    /// record of making, and typically calls the same tool again.
    ///
    /// Taking JSON keeps that open-ended: a caller passes whatever keys its
    /// template reads (`tool_call_id`, `name`, …) without every consumer of
    /// `ChatMessage` having to grow a field.
    pub fn render_json_with_options(
        &self,
        messages: &[JsonValue],
        opts: ChatRenderOptions,
    ) -> Result<String> {
        let msgs: Vec<Value> = messages
            .iter()
            .cloned()
            .map(GettableValue::from_json)
            .collect();
        // `tools` is passed as `none` when empty rather than as an empty list:
        // templates test `{% if tools %}`, and some then iterate assuming at
        // least one entry, so an empty list can render an empty tool block.
        let tools = if opts.tools.is_empty() {
            Value::from(())
        } else {
            // Through `GettableValue`, same as messages: tool-use templates call
            // `.get("function")` / `.get("parameters", {})` on these dicts, and a
            // plain serde map has no `get` method — the render fails outright
            // rather than degrading, so the wrapper is required, not cosmetic.
            Value::from(
                opts.tools
                    .iter()
                    .cloned()
                    .map(GettableValue::from_json)
                    .collect::<Vec<_>>(),
            )
        };
        let ctx = minijinja::context! {
            messages => Value::from(msgs),
            add_generation_prompt => opts.add_generation_prompt,
            enable_thinking => opts.enable_thinking,
            tools => tools,
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

    /// Render a template against one user turn.
    fn render(source: &str) -> String {
        ChatTemplate::from_source(source)
            .expect("compiles")
            .render(&[ChatMessage::user("hi")], false)
            .expect("renders")
    }

    #[test]
    fn string_methods_take_arguments() {
        // The bridge used to reject *every* call with an argument, so
        // `startswith`, `endswith`, `split` and `replace` all failed and Qwen3's
        // template could not render at all. The error named only whichever one
        // the template reached first, which made it look like a single missing
        // method.
        let cases = [
            (r#"{{ "hello world".startswith("hello") }}"#, "true"),
            (r#"{{ "hello world".startswith("nope") }}"#, "false"),
            (r#"{{ "a.txt".endswith(".txt") }}"#, "true"),
            // Python accepts a tuple of candidates, and templates use that form.
            (r#"{{ "a.md".endswith([".txt", ".md"]) }}"#, "true"),
            (r#"{{ "a.rs".endswith([".txt", ".md"]) }}"#, "false"),
            (r#"{{ "a,b,c".split(",") | join("|") }}"#, "a|b|c"),
            (r#"{{ "a,b,c".split(",", 1) | join("|") }}"#, "a|b,c"),
            (r#"{{ "a b  c".split() | length }}"#, "3"),
            (r#"{{ "aXbXc".replace("X", "-") }}"#, "a-b-c"),
            (r#"{{ "aXbXc".replace("X", "-", 1) }}"#, "a-bXc"),
            (r#"{{ "-".join(["a", "b"]) }}"#, "a-b"),
            (r#"{{ "xxhixx".strip("x") }}"#, "hi"),
            (r#"{{ "abcabc".count("bc") }}"#, "2"),
            (r#"{{ "abc".find("c") }}"#, "2"),
            (r#"{{ "abc".find("z") }}"#, "-1"),
        ];
        for (source, want) in cases {
            assert_eq!(render(source), want, "rendering {source}");
        }
    }

    #[test]
    fn no_argument_string_methods_still_work() {
        for (source, want) in [
            (r#"{{ "  hi  ".strip() }}"#, "hi"),
            (r#"{{ "  hi".lstrip() + "!" }}"#, "hi!"),
            (r#"{{ "hi  ".rstrip() + "!" }}"#, "hi!"),
            (r#"{{ "Hi".lower() }}"#, "hi"),
            (r#"{{ "hi".upper() }}"#, "HI"),
            (r#"{{ "hello world".title() }}"#, "Hello World"),
            (r#"{{ "hELLO".capitalize() }}"#, "Hello"),
        ] {
            assert_eq!(render(source), want, "rendering {source}");
        }
    }

    #[test]
    fn split_with_no_separator_collapses_whitespace() {
        // Python's bare `split()` splits on runs of whitespace and drops the
        // empties; splitting on a literal " " does not, and templates rely on
        // the difference.
        assert_eq!(render(r#"{{ "  a   b ".split() | length }}"#), "2");
        // `['', '', 'a', '', '', 'b', '']` — the empties are the point.
        assert_eq!(render(r#"{{ "  a   b ".split(" ") | length }}"#), "7");
    }

    #[test]
    fn an_unknown_string_method_still_reports_its_name() {
        let err = ChatTemplate::from_source(r#"{{ "x".partition("y") }}"#)
            .expect("compiles")
            .render(&[ChatMessage::user("hi")], false)
            .expect_err("partition is not bridged");
        assert!(
            format!("{err:#}").contains("partition"),
            "the error should name the method: {err:#}"
        );
    }

    #[test]
    fn a_qwen_shaped_template_renders() {
        // The shape that failed: a `startswith` guard over the message content,
        // which is how Qwen3 decides whether a system turn is already present.
        let source = "\
{%- for m in messages %}\
{%- if m['role'] == 'system' and not m['content'].startswith('<|') %}\
<|im_start|>system\n{{ m['content'] }}<|im_end|>\n\
{%- else %}\
<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n\
{%- endif %}\
{%- endfor %}";
        let out = ChatTemplate::from_source(source)
            .expect("compiles")
            .render(
                &[ChatMessage::system("Be brief."), ChatMessage::user("hi")],
                false,
            )
            .expect("renders");
        assert!(out.contains("<|im_start|>system\nBe brief."), "{out}");
        assert!(out.contains("<|im_start|>user\nhi"), "{out}");
    }
}
