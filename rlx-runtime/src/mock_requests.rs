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

//! Mock request payloads for tests (plan #64).
//!
//! Borrowed from MAX's `serve/mocks/mock_api_requests.py`. One
//! source of truth for "what does an OpenAI / OpenAI-style request
//! actually look like" so tests don't redefine payloads each time.
//!
//! These are *test fixtures*, not a serving impl — the structs are
//! intentionally small and serde-friendly. When RLX grows a serving
//! crate it should consume these as a starting point.

use serde::{Deserialize, Serialize};

/// A single message in a chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String, // "system" | "user" | "assistant"
    pub content: String,
}

/// OpenAI-shaped chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

/// OpenAI-shaped embedding request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingRequest {
    pub model: String,
    /// Either a single string or a list — accept both via the
    /// untagged `Input` enum below.
    pub input: Input,
    pub encoding_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Input {
    Single(String),
    Batch(Vec<String>),
}

// ── Canned fixtures ───────────────────────────────────────────────

pub fn chat_simple() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-4o-mini".to_string(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "What is the capital of France?".into(),
        }],
        max_tokens: Some(64),
        temperature: Some(0.0),
        stream: Some(false),
    }
}

pub fn chat_system_user() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "gpt-4o-mini".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".into(),
                content: "You are a terse oracle.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Color of the sky?".into(),
            },
        ],
        max_tokens: Some(8),
        temperature: Some(0.7),
        stream: Some(false),
    }
}

pub fn chat_streaming() -> ChatCompletionRequest {
    let mut r = chat_simple();
    r.stream = Some(true);
    r
}

pub fn embed_single() -> EmbeddingRequest {
    EmbeddingRequest {
        model: "text-embedding-3-small".to_string(),
        input: Input::Single("Hello, World!".to_string()),
        encoding_format: Some("float".to_string()),
    }
}

pub fn embed_batch() -> EmbeddingRequest {
    EmbeddingRequest {
        model: "text-embedding-3-small".to_string(),
        input: Input::Batch(vec![
            "Hello, World!".into(),
            "fastembed-rs is licensed under Apache-2.0".into(),
            "Some other short text here".into(),
        ]),
        encoding_format: Some("float".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_round_trip_through_json() {
        // Serialize + deserialize each canned fixture; the round
        // trip catches any field rename / type drift early.
        let cases: Vec<serde_json::Value> = vec![
            serde_json::to_value(chat_simple()).unwrap(),
            serde_json::to_value(chat_system_user()).unwrap(),
            serde_json::to_value(chat_streaming()).unwrap(),
            serde_json::to_value(embed_single()).unwrap(),
            serde_json::to_value(embed_batch()).unwrap(),
        ];
        for v in cases {
            let s = serde_json::to_string(&v).unwrap();
            let _: serde_json::Value = serde_json::from_str(&s).unwrap();
        }
    }

    #[test]
    fn embed_input_accepts_both_string_and_array() {
        let single =
            serde_json::from_str::<EmbeddingRequest>(r#"{"model":"x","input":"hi"}"#).unwrap();
        let batch =
            serde_json::from_str::<EmbeddingRequest>(r#"{"model":"x","input":["a","b"]}"#).unwrap();
        assert!(matches!(single.input, Input::Single(_)));
        assert!(matches!(batch.input, Input::Batch(_)));
    }
}
