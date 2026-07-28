// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thin wrapper around the `tokenizers` crate.
//!
//! Many `rlx-<family>` runners import `tokenizers::Tokenizer` directly
//! and re-implement the same "load from path → encode → decode" dance.
//! This module collapses that to one entry point.

use anyhow::{Context, Result};
use std::path::Path;

pub use tokenizers::Tokenizer as RawTokenizer;

/// Owned tokenizer handle. Use [`load_tokenizer`] to construct.
pub struct TokenizerHandle {
    inner: RawTokenizer,
}

impl TokenizerHandle {
    pub fn from_raw(raw: RawTokenizer) -> Self {
        Self { inner: raw }
    }

    pub fn raw(&self) -> &RawTokenizer {
        &self.inner
    }

    pub fn raw_mut(&mut self) -> &mut RawTokenizer {
        &mut self.inner
    }

    /// Encode `text` and return token ids.
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decode ids back to a string.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))
    }
}

/// Load a HF-format `tokenizer.json` from disk.
pub fn load_tokenizer(path: &Path) -> Result<TokenizerHandle> {
    let raw = RawTokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer at {path:?}: {e}"))
        .with_context(|| format!("tokenizer.json at {path:?}"))?;
    Ok(TokenizerHandle::from_raw(raw))
}

/// Standalone decode helper (no handle needed).
pub fn decode_ids(t: &TokenizerHandle, ids: &[u32], skip_special: bool) -> Result<String> {
    t.decode(ids, skip_special)
}
