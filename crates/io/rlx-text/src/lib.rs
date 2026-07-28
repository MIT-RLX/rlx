// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RLX text: tokenizer + chat template + sampling helpers.
//!
//! Promoted from `rlx-models/crates/rlx-cli` so that downstream LM apps
//! (server, web playground, training tools) have one published crate to
//! depend on without taking the CLI helper layer.

pub mod chat;
pub mod detokenize;
pub mod sampling;
pub mod tokenizer;
pub mod tool_parse;

pub use chat::{ChatMessage, ChatTemplate, ChatTemplateSource, auto_chat_template};
pub use detokenize::{StreamingDetokenizer, incremental_emit};
pub use rlx_runtime::SampleOpts;
pub use sampling::{argmax, sample_next};
pub use tokenizer::{TokenizerHandle, decode_ids, load_tokenizer};
pub use tool_parse::{ToolCall, ToolFormat};
