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

//! Multi-protocol request router (plan #32).
//!
//! Borrowed from MAX's `serve/router/{openai_routes, kserve_routes,
//! sagemaker_routes, openresponses_routes}.py`. The shape: one
//! inference engine, multiple wire protocols layered as thin
//! per-protocol adapters. Adding a new protocol = implementing
//! [`WireProtocol`] for its raw request type, not editing the hot
//! path.
//!
//! Today's adapter is OpenAI-shaped (chat completions +
//! embeddings) since [`crate::mock_requests`] already defines
//! those structs. KServe / SageMaker / OpenResponses slot in by
//! impl'ing `WireProtocol` for their respective request types.
//!
//! All conversion is pure-data: no I/O, no async. The actual HTTP
//! parsing happens upstream (in the future serving crate); this
//! module owns the translation between wire types and the
//! internal [`RoutedRequest`].

use crate::mock_requests::{ChatCompletionRequest, EmbeddingRequest, Input};

/// What kind of inference the request is asking for. Drives
/// which downstream pipeline serves it (text-gen vs embedding
/// pool, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    /// Chat completion (autoregressive token generation).
    ChatCompletion,
    /// Embedding (one forward pass, return pooled output).
    Embedding,
    /// Plain text completion (legacy `/v1/completions` shape).
    TextCompletion,
}

/// Internal canonical request shape. Every wire protocol parses
/// into this; downstream schedulers / engines consume only this
/// type.
#[derive(Debug, Clone)]
pub struct RoutedRequest {
    pub id: u64,
    pub kind: RequestKind,
    /// Pre-tokenized input (one entry per text in a batched embed
    /// request). For chat completion: the system + user history
    /// flattened into a single token list.
    pub inputs: Vec<Vec<u32>>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub stream: bool,
    /// Optional LoRA adapter name; passed through to
    /// [`crate::lora_scheduler::LoraRequest::adapter`].
    pub adapter: Option<String>,
    /// Model name from the wire request — kept for telemetry /
    /// validation; the actual model selection happens upstream.
    pub model: String,
}

#[derive(Debug, Clone)]
pub enum RouteError {
    UnknownProtocol { name: String },
    InvalidRequest { reason: String },
    UnsupportedFeature { feature: &'static str },
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownProtocol { name } => write!(f, "unknown protocol: {name}"),
            Self::InvalidRequest { reason } => write!(f, "invalid request: {reason}"),
            Self::UnsupportedFeature { feature } => write!(f, "unsupported feature: {feature}"),
        }
    }
}

impl std::error::Error for RouteError {}

/// Adapter trait — implement once per wire protocol.
///
/// Implementations are tiny by design: they parse / validate the
/// wire shape and produce a `RoutedRequest`. They do NOT touch
/// the engine.
pub trait WireProtocol {
    type Request;
    fn name(&self) -> &'static str;
    fn parse(&self, req: Self::Request) -> Result<RoutedRequest, RouteError>;
}

/// OpenAI-style adapter — handles ChatCompletionRequest and
/// EmbeddingRequest from [`crate::mock_requests`].
pub struct OpenAIProtocol;

impl WireProtocol for OpenAIProtocol {
    type Request = OpenAIRequest;
    fn name(&self) -> &'static str {
        "openai"
    }
    fn parse(&self, req: OpenAIRequest) -> Result<RoutedRequest, RouteError> {
        match req {
            OpenAIRequest::Chat(c) => parse_chat(c),
            OpenAIRequest::Embedding(e) => parse_embed(e),
        }
    }
}

#[derive(Debug, Clone)]
pub enum OpenAIRequest {
    Chat(ChatCompletionRequest),
    Embedding(EmbeddingRequest),
}

fn parse_chat(req: ChatCompletionRequest) -> Result<RoutedRequest, RouteError> {
    if req.messages.is_empty() {
        return Err(RouteError::InvalidRequest {
            reason: "messages cannot be empty".into(),
        });
    }
    // Tokenization is the consumer's job — here we synthesize a
    // single placeholder token-list per message. Real serving
    // hooks call into a tokenizer before this point and feeds
    // the resulting vec<u32> in directly.
    let flat: Vec<u32> = req
        .messages
        .iter()
        .flat_map(|m| pseudo_tokenize(&m.role, &m.content))
        .collect();
    Ok(RoutedRequest {
        id: hash_request_id(&req.model, &flat),
        kind: RequestKind::ChatCompletion,
        inputs: vec![flat],
        max_tokens: req.max_tokens.unwrap_or(256),
        temperature: req.temperature.unwrap_or(1.0),
        stream: req.stream.unwrap_or(false),
        adapter: None, // OpenAI shape doesn't carry adapter; future ext.
        model: req.model,
    })
}

fn parse_embed(req: EmbeddingRequest) -> Result<RoutedRequest, RouteError> {
    let inputs: Vec<Vec<u32>> = match req.input {
        Input::Single(s) => vec![pseudo_tokenize("input", &s)],
        Input::Batch(v) => v.iter().map(|s| pseudo_tokenize("input", s)).collect(),
    };
    if inputs.is_empty() {
        return Err(RouteError::InvalidRequest {
            reason: "embedding input cannot be empty".into(),
        });
    }
    Ok(RoutedRequest {
        id: hash_request_id(
            &req.model,
            inputs.first().map(|v| v.as_slice()).unwrap_or(&[]),
        ),
        kind: RequestKind::Embedding,
        inputs,
        max_tokens: 0, // not meaningful for embeddings
        temperature: 0.0,
        stream: false,
        adapter: None,
        model: req.model,
    })
}

/// Placeholder tokenizer: maps each char to its u32 code point,
/// prefixed by a role-header pseudo-token (1=system, 2=user, 3=...).
/// Real consumers replace this with a real tokenizer; the routing
/// layer doesn't depend on which tokenizer.
fn pseudo_tokenize(role: &str, text: &str) -> Vec<u32> {
    let role_token = match role {
        "system" => 1u32,
        "user" => 2,
        "assistant" => 3,
        _ => 4,
    };
    let mut tokens = Vec::with_capacity(text.len() + 1);
    tokens.push(role_token);
    tokens.extend(text.chars().map(|c| c as u32));
    tokens
}

/// Stable-ish u64 from `(model, first_input_tokens)`. The router
/// itself doesn't need uniqueness — the consumer can override
/// `RoutedRequest::id` with a real UUID after parsing.
fn hash_request_id(model: &str, tokens: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    model.hash(&mut h);
    tokens.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_requests::*;

    #[test]
    fn openai_chat_routes_to_chat_completion() {
        let req = ChatCompletionRequest {
            model: "gpt-4o-mini".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Hi".into(),
            }],
            max_tokens: Some(64),
            temperature: Some(0.7),
            stream: Some(false),
        };
        let routed = OpenAIProtocol.parse(OpenAIRequest::Chat(req)).unwrap();
        assert_eq!(routed.kind, RequestKind::ChatCompletion);
        assert_eq!(routed.inputs.len(), 1);
        assert_eq!(routed.max_tokens, 64);
        assert!((routed.temperature - 0.7).abs() < 1e-6);
        assert_eq!(routed.model, "gpt-4o-mini");
    }

    #[test]
    fn openai_embedding_single_string() {
        let req = EmbeddingRequest {
            model: "text-embedding-3-small".into(),
            input: Input::Single("Hello".into()),
            encoding_format: None,
        };
        let routed = OpenAIProtocol.parse(OpenAIRequest::Embedding(req)).unwrap();
        assert_eq!(routed.kind, RequestKind::Embedding);
        assert_eq!(routed.inputs.len(), 1);
        // role(=4 since "input" isn't a known role) + 5 chars
        assert_eq!(routed.inputs[0].len(), 6);
    }

    #[test]
    fn openai_embedding_batch_input() {
        let req = EmbeddingRequest {
            model: "text-embedding-3-small".into(),
            input: Input::Batch(vec!["a".into(), "bb".into(), "ccc".into()]),
            encoding_format: None,
        };
        let routed = OpenAIProtocol.parse(OpenAIRequest::Embedding(req)).unwrap();
        assert_eq!(routed.inputs.len(), 3);
        assert_eq!(routed.inputs[1].len(), 3); // role + 2 chars
    }

    #[test]
    fn empty_chat_messages_rejected() {
        let req = ChatCompletionRequest {
            model: "x".into(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
            stream: None,
        };
        let err = OpenAIProtocol.parse(OpenAIRequest::Chat(req)).unwrap_err();
        assert!(matches!(err, RouteError::InvalidRequest { .. }));
    }

    #[test]
    fn defaults_applied_when_optional_fields_missing() {
        let req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "x".into(),
            }],
            max_tokens: None,
            temperature: None,
            stream: None,
        };
        let routed = OpenAIProtocol.parse(OpenAIRequest::Chat(req)).unwrap();
        assert_eq!(routed.max_tokens, 256);
        assert_eq!(routed.temperature, 1.0);
        assert!(!routed.stream);
    }

    #[test]
    fn protocol_name_introspectable() {
        assert_eq!(OpenAIProtocol.name(), "openai");
    }
}
