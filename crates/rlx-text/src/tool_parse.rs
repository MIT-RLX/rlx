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

//! Parsing model-emitted tool / function calls into structured form.
//!
//! Models emit tool calls in a few conventions; this module recognizes the
//! common ones and returns `{name, arguments}` (mirroring mlx-lm's
//! `tool_parsers/`). All host-side string work — no model/runtime coupling.
//!
//! Formats:
//!   - **Hermes / Qwen** — `<tool_call>{"name": …, "arguments": {…}}</tool_call>`
//!     (possibly several blocks).
//!   - **JSON** — the whole text is an object (or array of objects) with
//!     `name` + `arguments`/`parameters`.
//!   - **Pythonic** — `[fn(a="x", b=2)]` (optionally wrapped in tags).

use serde_json::{Map, Value};

/// A parsed tool call: function name + a JSON object of arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Known tool-call output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// `<tool_call>{json}</tool_call>` blocks (Qwen3, Hermes, …).
    Hermes,
    /// Bare JSON object or array of `{name, arguments|parameters}`.
    Json,
    /// `[fn(arg="v", n=2)]` (Llama/`pythonic`).
    Pythonic,
}

/// Parse tool calls from `text` in a known `fmt`. Returns an empty vec when
/// nothing parses (a plain assistant message, not a tool call).
pub fn parse(text: &str, fmt: ToolFormat) -> Vec<ToolCall> {
    match fmt {
        ToolFormat::Hermes => parse_hermes(text),
        ToolFormat::Json => parse_json(text.trim()),
        ToolFormat::Pythonic => parse_pythonic(text),
    }
}

/// Try each known format, in order of specificity, and return the first that
/// yields calls. Useful when the chat template's format isn't known up front.
pub fn detect_and_parse(text: &str) -> Vec<ToolCall> {
    for fmt in [ToolFormat::Hermes, ToolFormat::Pythonic, ToolFormat::Json] {
        let calls = parse(text, fmt);
        if !calls.is_empty() {
            return calls;
        }
    }
    Vec::new()
}

/// Normalize a JSON object that should contain `name` + arguments into a
/// [`ToolCall`]. Accepts `arguments` or `parameters`; arguments may be an
/// object or a JSON-encoded string.
fn tool_call_from_obj(obj: &Map<String, Value>) -> Option<ToolCall> {
    let name = obj.get("name")?.as_str()?.to_string();
    let args = obj
        .get("arguments")
        .or_else(|| obj.get("parameters"))
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    // Some models double-encode arguments as a JSON string.
    let args = match args {
        Value::String(s) => serde_json::from_str(&s).unwrap_or(Value::String(s)),
        other => other,
    };
    Some(ToolCall {
        name,
        arguments: args,
    })
}

fn parse_json(text: &str) -> Vec<ToolCall> {
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(o)) => tool_call_from_obj(&o).into_iter().collect(),
        Ok(Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_object().and_then(tool_call_from_obj))
            .collect(),
        _ => Vec::new(),
    }
}

fn parse_hermes(text: &str) -> Vec<ToolCall> {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        let after = &rest[start + OPEN.len()..];
        let Some(end) = after.find(CLOSE) else { break };
        let inner = after[..end].trim();
        out.extend(parse_json(inner));
        rest = &after[end + CLOSE.len()..];
    }
    out
}

fn parse_pythonic(text: &str) -> Vec<ToolCall> {
    // Locate the innermost `[ ... ]` payload (ignore any tag wrappers).
    let Some(open) = text.find('[') else {
        return Vec::new();
    };
    let Some(close_rel) = text[open..].rfind(']') else {
        return Vec::new();
    };
    let body = &text[open + 1..open + close_rel];
    let mut out = Vec::new();
    for call in split_top_level(body, ',') {
        let call = call.trim();
        let Some(paren) = call.find('(') else {
            continue;
        };
        if !call.ends_with(')') {
            continue;
        }
        let name = call[..paren].trim();
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let args_str = &call[paren + 1..call.len() - 1];
        let mut args = Map::new();
        for pair in split_top_level(args_str, ',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            if let Some(eq) = pair.find('=') {
                let key = pair[..eq].trim().to_string();
                let val = pair[eq + 1..].trim();
                args.insert(key, literal_to_value(val));
            }
        }
        out.push(ToolCall {
            name: name.to_string(),
            arguments: Value::Object(args),
        });
    }
    out
}

/// Convert a Python-ish literal to a JSON value: try JSON first (numbers,
/// `true`/`false`, quoted strings), then strip simple quotes, else keep raw.
fn literal_to_value(s: &str) -> Value {
    if let Ok(v) = serde_json::from_str::<Value>(s) {
        return v;
    }
    // Python single-quoted string → strip quotes.
    if s.len() >= 2 && s.starts_with('\'') && s.ends_with('\'') {
        return Value::String(s[1..s.len() - 1].to_string());
    }
    match s {
        "True" => Value::Bool(true),
        "False" => Value::Bool(false),
        "None" => Value::Null,
        other => Value::String(other.to_string()),
    }
}

/// Split `s` on `sep`, but only at the top nesting level (respecting `()`,
/// `[]`, `{}`, and quoted strings). Avoids splitting inside argument values.
fn split_top_level(s: &str, sep: char) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut quote: Option<char> = None;
    let mut cur = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some(q) => {
                cur.push(c);
                if c == '\\' {
                    if let Some(&n) = chars.peek() {
                        cur.push(n);
                        chars.next();
                    }
                } else if c == q {
                    quote = None;
                }
            }
            None => match c {
                '"' | '\'' => {
                    quote = Some(c);
                    cur.push(c);
                }
                '(' | '[' | '{' => {
                    depth += 1;
                    cur.push(c);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    cur.push(c);
                }
                _ if c == sep && depth == 0 => {
                    parts.push(std::mem::take(&mut cur));
                }
                _ => cur.push(c),
            },
        }
    }
    if !cur.trim().is_empty() || !parts.is_empty() {
        parts.push(cur);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hermes_single_call() {
        let text = r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#;
        let calls = parse(text, ToolFormat::Hermes);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, json!({"city":"Paris"}));
    }

    #[test]
    fn hermes_multiple_calls_with_surrounding_text() {
        let text = "sure!\n<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\n\
                    <tool_call>{\"name\":\"b\",\"parameters\":{\"x\":1}}</tool_call>";
        let calls = parse(text, ToolFormat::Hermes);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].arguments, json!({"x":1}));
    }

    #[test]
    fn json_double_encoded_arguments() {
        let text = r#"{"name":"f","arguments":"{\"k\":2}"}"#;
        let calls = parse(text, ToolFormat::Json);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, json!({"k":2}));
    }

    #[test]
    fn pythonic_call() {
        let text = r#"[get_weather(city="Paris", days=2, metric=true)]"#;
        let calls = parse(text, ToolFormat::Pythonic);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments,
            json!({"city":"Paris","days":2,"metric":true})
        );
    }

    #[test]
    fn pythonic_with_tag_wrapper_and_commas_in_strings() {
        let text = r#"<|tool_call_start|>[note(text="a, b, c")]<|tool_call_end|>"#;
        let calls = parse(text, ToolFormat::Pythonic);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, json!({"text":"a, b, c"}));
    }

    #[test]
    fn detect_picks_the_right_format() {
        assert_eq!(detect_and_parse("just text"), vec![]);
        assert_eq!(
            detect_and_parse(r#"<tool_call>{"name":"x","arguments":{}}</tool_call>"#)[0].name,
            "x"
        );
        assert_eq!(detect_and_parse(r#"[y(a=1)]"#)[0].name, "y");
    }
}
