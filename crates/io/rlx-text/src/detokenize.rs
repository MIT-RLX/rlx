// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Incremental detokenization for streaming generation.
//!
//! `TokenizerHandle::decode` is batch-only. Decoding tokens one at a time
//! and concatenating breaks on two fronts:
//!   - **Byte-level BPE** (GPT-2/Qwen) splits a single multi-byte UTF-8
//!     char across two tokens; decoding the first alone yields the U+FFFD
//!     replacement char, which must not be emitted to the client.
//!   - **SentencePiece** decides leading spaces from context, so per-token
//!     decode mis-spaces word boundaries.
//!
//! [`StreamingDetokenizer`] re-decodes the full id sequence each step and
//! emits only the newly *stable* suffix, holding back a trailing run of
//! replacement chars until the next token completes them. This is O(T²)
//! over a generation (one decode per token); fine for chat-length outputs.
//! A per-tokenizer byte-buffer fast path (O(T)) is a future optimization.

use crate::tokenizer::TokenizerHandle;
use anyhow::Result;

/// Streaming detokenizer over a borrowed [`TokenizerHandle`].
///
/// ```ignore
/// let mut detok = StreamingDetokenizer::new(&tok, true);
/// for id in decoded_token_stream {
///     let delta = detok.push(id)?;   // emit to the client as it stabilizes
///     if !delta.is_empty() { send(delta); }
/// }
/// send(detok.finish()?);             // flush any held-back tail
/// ```
pub struct StreamingDetokenizer<'t> {
    tok: &'t TokenizerHandle,
    skip_special: bool,
    ids: Vec<u32>,
    /// Byte offset into the full decode that has already been emitted.
    emitted: usize,
}

impl<'t> StreamingDetokenizer<'t> {
    pub fn new(tok: &'t TokenizerHandle, skip_special: bool) -> Self {
        Self {
            tok,
            skip_special,
            ids: Vec::new(),
            emitted: 0,
        }
    }

    /// Append a token and return any newly-stable text (may be empty when
    /// the token only extends an incomplete multi-byte char).
    pub fn push(&mut self, id: u32) -> Result<String> {
        self.ids.push(id);
        let full = self.tok.decode(&self.ids, self.skip_special)?;
        let (delta, emitted) = advance(&full, self.emitted);
        self.emitted = emitted;
        Ok(delta.to_string())
    }

    /// Flush everything decoded so far, including a held-back tail. Call
    /// once at the end of generation (after the last `push`).
    pub fn finish(&mut self) -> Result<String> {
        let full = self.tok.decode(&self.ids, self.skip_special)?;
        let out = if self.emitted < full.len() {
            full[self.emitted..].to_string()
        } else {
            String::new()
        };
        self.emitted = full.len();
        Ok(out)
    }

    /// The full text decoded so far (including any not-yet-emitted tail).
    pub fn text(&self) -> Result<String> {
        self.tok.decode(&self.ids, self.skip_special)
    }

    /// Tokens accumulated so far.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
}

/// Core incremental-emit step, decoupled from the tokenizer for testing:
/// given the full decode `full` and the byte offset `emitted` already
/// returned, yield the next stable slice and the updated offset. A trailing
/// run of U+FFFD (incomplete multi-byte bytes) is held back.
/// Stateless incremental detokenization: decode `ids` and return the next
/// *stable* text beyond byte offset `emitted`, plus the new offset. Lets a
/// caller (e.g. a continuous-batching scheduler) keep per-sequence `(ids,
/// emitted)` state without holding a borrow of the tokenizer.
pub fn incremental_emit(
    tok: &TokenizerHandle,
    ids: &[u32],
    emitted: usize,
    skip_special: bool,
) -> Result<(String, usize)> {
    let full = tok.decode(ids, skip_special)?;
    let (delta, new_emitted) = advance(&full, emitted);
    Ok((delta.to_string(), new_emitted))
}

fn advance(full: &str, emitted: usize) -> (&str, usize) {
    let stable = full.trim_end_matches('\u{FFFD}').len();
    // Invariant: append-only generation keeps `full[..emitted]` fixed, so
    // `emitted` stays a char boundary. Guard defensively anyway.
    if stable <= emitted || emitted > full.len() || !full.is_char_boundary(emitted) {
        return ("", emitted.min(full.len()));
    }
    (&full[emitted..stable], stable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emits_growing_suffix() {
        // First step emits "Hello".
        let (d, e) = advance("Hello", 0);
        assert_eq!(d, "Hello");
        assert_eq!(e, 5);
        // Next decode extends it; emit only " world".
        let (d, e) = advance("Hello world", e);
        assert_eq!(d, " world");
        assert_eq!(e, 11);
    }

    #[test]
    fn holds_back_incomplete_multibyte() {
        // A byte-level-BPE first half of a 2-byte char decodes to U+FFFD;
        // it must be held back, not emitted.
        let (d, e) = advance("hi\u{FFFD}", 2);
        assert_eq!(d, "");
        assert_eq!(e, 2);
        // Next token completes the char (é); now it's stable.
        let (d, e) = advance("hié", 2);
        assert_eq!(d, "é");
        assert_eq!(e, "hié".len());
    }

    #[test]
    fn no_change_emits_nothing() {
        let (d, e) = advance("abc", 3);
        assert_eq!(d, "");
        assert_eq!(e, 3);
    }

    #[test]
    fn trailing_replacement_run_held_back() {
        // Multiple incomplete bytes pending.
        let (d, e) = advance("ok\u{FFFD}\u{FFFD}", 2);
        assert_eq!(d, "");
        assert_eq!(e, 2);
    }
}
