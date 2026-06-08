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

//! RLX text: tokenizer + chat template + sampling helpers.
//!
//! Promoted from `rlx-models/crates/rlx-cli` so that downstream LM apps
//! (server, web playground, training tools) have one published crate to
//! depend on without taking the CLI helper layer.

pub mod chat;
pub mod sampling;
pub mod tokenizer;

pub use chat::{ChatMessage, ChatTemplate, ChatTemplateSource, auto_chat_template};
pub use rlx_runtime::SampleOpts;
pub use sampling::{argmax, sample_next};
pub use tokenizer::{TokenizerHandle, decode_ids, load_tokenizer};
