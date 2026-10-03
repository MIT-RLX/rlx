// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rendering tool schemas into a chat prompt.
//!
//! A tools render has three ways to fail quietly, and all three were real:
//! `tools` missing from the template context (the branch never fires), `tojson`
//! unregistered (the whole render errors), and the dict wrapper serializing as
//! `{}` (a tools block listing no tools). The last is the dangerous one — the
//! model is told it may call functions and shown none, so it answers in prose
//! and the caller reads that as the model declining.

use rlx_text::{ChatMessage, ChatRenderOptions, ChatTemplate};

/// The shape every tool-use template shares: a block gated on `tools`, with
/// each schema serialized by `tojson`. Qwen3, Llama 3.x and Mistral differ in
/// wording, not in these two mechanics.
/// Extended with the assistant/tool branches real tool-use templates have, so a
/// round trip can be checked and not just the tool list.
const TEMPLATE: &str = r#"
{%- if tools %}<|im_start|>system
# Tools
<tools>
{%- for t in tools %}
{{ t | tojson }}
{%- endfor %}
</tools><|im_end|>
{%- endif %}
{%- for m in messages %}
{%- if m.role == "tool" %}<|im_start|>user
<tool_response>
{{ m.content }}
</tool_response><|im_end|>
{%- else %}<|im_start|>{{ m.role }}
{{ m.content }}
{%- if m.tool_calls %}
{%- for c in m.tool_calls %}
{%- set fn = c.function %}
<tool_call>
{"name": "{{ fn.name }}", "arguments": {{ fn.arguments | tojson }}}
</tool_call>
{%- endfor %}
{%- endif %}<|im_end|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{%- endif %}
"#;

fn tool(name: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }
        }
    })
}

fn render(tools: Vec<serde_json::Value>) -> String {
    let t = ChatTemplate::from_source(TEMPLATE).expect("compile");
    t.render_with_options(
        &[ChatMessage {
            role: "user".into(),
            content: "weather in Paris?".into(),
        }],
        ChatRenderOptions {
            add_generation_prompt: true,
            tools,
            ..Default::default()
        },
    )
    .expect("render")
}

#[test]
fn a_tool_schema_reaches_the_prompt() {
    let out = render(vec![tool("get_weather")]);
    assert!(
        out.contains("# Tools"),
        "the tools branch should fire:\n{out}"
    );
    assert!(
        out.contains("get_weather"),
        "the tool's name must survive serialization:\n{out}"
    );
    // The guard against the wrapper serializing as `{}`: the nested schema has
    // to come through, not just the name.
    assert!(
        out.contains("\"city\""),
        "the schema must come through:\n{out}"
    );
    assert!(out.contains("required"), "nested arrays too:\n{out}");
}

#[test]
fn every_tool_is_listed() {
    let out = render(vec![tool("a"), tool("b")]);
    assert!(
        out.contains("\"name\":\"a\"") || out.contains("\"name\": \"a\""),
        "{out}"
    );
    assert!(
        out.contains("\"name\":\"b\"") || out.contains("\"name\": \"b\""),
        "{out}"
    );
}

/// No tools must leave the prompt exactly as it was, not emit an empty block.
/// Templates test `{% if tools %}`, so an empty list has to read as falsey.
#[test]
fn no_tools_renders_no_tools_block() {
    let out = render(Vec::new());
    assert!(
        !out.contains("# Tools"),
        "empty tools should not open a block:\n{out}"
    );
}

#[test]
fn supports_tools_reflects_the_template() {
    assert!(
        ChatTemplate::from_source(TEMPLATE)
            .unwrap()
            .supports_tools()
    );
    let plain = "{% for m in messages %}{{ m.content }}{% endfor %}";
    assert!(
        !ChatTemplate::from_source(plain).unwrap().supports_tools(),
        "a template with no tools branch must say so, so a caller can refuse"
    );
}

/// `tojson(indent=n)` is what some templates use for readability; it must not
/// error out just because the argument is present.
#[test]
fn tojson_accepts_an_indent_argument() {
    let t = ChatTemplate::from_source("{{ tools[0] | tojson(indent=2) }}").expect("compile");
    let out = t
        .render_with_options(
            &[],
            ChatRenderOptions {
                tools: vec![tool("get_weather")],
                ..Default::default()
            },
        )
        .expect("render with indent");
    assert!(out.contains("get_weather"), "{out}");
    assert!(
        out.contains('\n'),
        "indented json should be multi-line:\n{out}"
    );
}

/// A tool loop's second leg: the assistant's own call and the tool's result both
/// have to reach the prompt.
///
/// Dropping the call is the subtle half — the result still renders, so the prompt
/// looks fine and simply omits what the result is answering. Templates read
/// `message.tool_calls`, so the field has to arrive under that name, with
/// `arguments` still an object for `tojson` to encode.
#[test]
fn an_assistant_tool_call_and_its_result_both_render() {
    let t = ChatTemplate::from_source(TEMPLATE).expect("compile");
    let messages = vec![
        serde_json::json!({ "role": "user", "content": "weather in Paris?" }),
        serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": { "name": "get_weather", "arguments": { "city": "Paris" } }
            }]
        }),
        serde_json::json!({ "role": "tool", "content": "{\"temp_c\": 14}" }),
    ];
    let out = t
        .render_json_with_options(
            &messages,
            ChatRenderOptions {
                add_generation_prompt: true,
                tools: vec![tool("get_weather")],
                ..Default::default()
            },
        )
        .expect("render");

    assert!(out.contains("<tool_call>"), "the call must render:\n{out}");
    assert!(out.contains("get_weather"), "with its name:\n{out}");
    assert!(
        out.contains("\"city\":\"Paris\"") || out.contains("\"city\": \"Paris\""),
        "and its arguments as json, not a stringified blob:\n{out}"
    );
    assert!(
        out.contains("<tool_response>"),
        "the result must render:\n{out}"
    );
    assert!(out.contains("temp_c"), "with its content:\n{out}");
    // Ordering is what makes it a conversation: the call comes before the result.
    assert!(
        out.find("<tool_call>") < out.find("<tool_response>"),
        "the call must precede the result:\n{out}"
    );
}

/// Templates branch on whether `arguments` is a string, so both shapes have to
/// come through as themselves — an object must not arrive pre-stringified (the
/// template would `tojson` it again into a quoted blob) and a string must not be
/// re-quoted.
#[test]
fn arguments_keep_their_shape() {
    // The branch Qwen3's template uses, verbatim in spirit.
    let t = ChatTemplate::from_source(
        "{%- for m in messages %}{%- for c in m.tool_calls %}         {%- if c.function.arguments is string %}STR:{{ c.function.arguments }}         {%- else %}OBJ:{{ c.function.arguments | tojson }}{%- endif %}         {%- endfor %}{%- endfor %}",
    )
    .expect("compile");

    let render = |args: serde_json::Value| {
        t.render_json_with_options(
            &[serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{ "function": { "name": "f", "arguments": args } }]
            })],
            ChatRenderOptions::default(),
        )
        .expect("render")
    };

    let obj = render(serde_json::json!({ "a": 1 }));
    assert!(
        obj.starts_with("OBJ:"),
        "an object must take the tojson branch: {obj:?}"
    );
    assert!(obj.contains("\"a\":1"), "and encode once: {obj:?}");
    assert!(
        !obj.contains("\\\""),
        "no escaped quotes from double encoding: {obj:?}"
    );

    let s = render(serde_json::json!("{\"a\": 1}"));
    assert!(
        s.starts_with("STR:"),
        "a string must take the string branch: {s:?}"
    );
    assert_eq!(s, "STR:{\"a\": 1}", "and pass through unchanged");
}
