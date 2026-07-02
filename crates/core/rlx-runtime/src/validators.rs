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

//! Composable request validators (plan #84).
//!
//! Borrowed from MAX's `pipelines/core/context_validators.py`. Each
//! validation rule is a small composable function returning
//! `Result<(), ValidationError>`. Pipelines pick which to apply for
//! their request type.
//!
//! Why composable instead of a single `validate_request()`?
//!   - Different request types (text embed, image embed, generation)
//!     share some rules (max length) but not others (image bounds).
//!   - Adding a rule = one function, not editing a monolith.
//!   - Easy to test rules in isolation.
//!
//! Used by future serving paths (#31, #32 in PLAN.md). Today the
//! benchmark runners can use it for input sanity checks.

use std::fmt;

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub rule: &'static str,
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.rule, self.message)
    }
}

impl std::error::Error for ValidationError {}

pub type ValidationResult = Result<(), ValidationError>;

/// A single check on a context value.
pub trait Validator<C>: Send + Sync {
    fn check(&self, ctx: &C) -> ValidationResult;
    fn name(&self) -> &'static str;
}

/// Run a chain of validators; return the first error or `Ok(())`.
pub fn run_chain<C>(ctx: &C, chain: &[&dyn Validator<C>]) -> ValidationResult {
    for v in chain {
        v.check(ctx)?;
    }
    Ok(())
}

// ── Text-context validators ─────────────────────────────────────

/// Context fields a text request typically carries.
#[derive(Debug, Clone)]
pub struct TextContext {
    pub seq_len: usize,
    pub batch_size: usize,
    pub vocab_id_max: usize,
    pub max_token_id_seen: usize,
}

pub struct MaxSeqLen(pub usize);
impl Validator<TextContext> for MaxSeqLen {
    fn check(&self, ctx: &TextContext) -> ValidationResult {
        if ctx.seq_len > self.0 {
            Err(ValidationError {
                rule: self.name(),
                message: format!("seq_len {} exceeds max {}", ctx.seq_len, self.0),
            })
        } else {
            Ok(())
        }
    }
    fn name(&self) -> &'static str {
        "max_seq_len"
    }
}

pub struct MaxBatchSize(pub usize);
impl Validator<TextContext> for MaxBatchSize {
    fn check(&self, ctx: &TextContext) -> ValidationResult {
        if ctx.batch_size > self.0 {
            Err(ValidationError {
                rule: self.name(),
                message: format!("batch_size {} exceeds max {}", ctx.batch_size, self.0),
            })
        } else {
            Ok(())
        }
    }
    fn name(&self) -> &'static str {
        "max_batch_size"
    }
}

pub struct TokenIdsInVocab;
impl Validator<TextContext> for TokenIdsInVocab {
    fn check(&self, ctx: &TextContext) -> ValidationResult {
        if ctx.max_token_id_seen >= ctx.vocab_id_max {
            Err(ValidationError {
                rule: self.name(),
                message: format!(
                    "saw token_id {} but vocab is {}",
                    ctx.max_token_id_seen, ctx.vocab_id_max
                ),
            })
        } else {
            Ok(())
        }
    }
    fn name(&self) -> &'static str {
        "token_ids_in_vocab"
    }
}

// ── Image-context validators ────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ImageContext {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

pub struct ImageMaxBounds {
    pub max_w: u32,
    pub max_h: u32,
}
impl Validator<ImageContext> for ImageMaxBounds {
    fn check(&self, ctx: &ImageContext) -> ValidationResult {
        if ctx.width > self.max_w || ctx.height > self.max_h {
            Err(ValidationError {
                rule: self.name(),
                message: format!(
                    "{}×{} exceeds max {}×{}",
                    ctx.width, ctx.height, self.max_w, self.max_h
                ),
            })
        } else {
            Ok(())
        }
    }
    fn name(&self) -> &'static str {
        "image_max_bounds"
    }
}

pub struct ChannelsAllowed(pub &'static [u32]);
impl Validator<ImageContext> for ChannelsAllowed {
    fn check(&self, ctx: &ImageContext) -> ValidationResult {
        if !self.0.contains(&ctx.channels) {
            Err(ValidationError {
                rule: self.name(),
                message: format!("channels={} not in allowed set {:?}", ctx.channels, self.0),
            })
        } else {
            Ok(())
        }
    }
    fn name(&self) -> &'static str {
        "channels_allowed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_chain_short_circuits() {
        let ctx = TextContext {
            seq_len: 600,
            batch_size: 1,
            vocab_id_max: 30000,
            max_token_id_seen: 100,
        };
        let max_seq = MaxSeqLen(512);
        let max_batch = MaxBatchSize(64);
        let tok = TokenIdsInVocab;
        let chain: Vec<&dyn Validator<TextContext>> = vec![&max_seq, &max_batch, &tok];
        let err = run_chain(&ctx, &chain).unwrap_err();
        assert_eq!(err.rule, "max_seq_len");
    }

    #[test]
    fn image_chain_passes() {
        let ctx = ImageContext {
            width: 224,
            height: 224,
            channels: 3,
        };
        let bounds = ImageMaxBounds {
            max_w: 1024,
            max_h: 1024,
        };
        let chans = ChannelsAllowed(&[1, 3, 4]);
        let chain: Vec<&dyn Validator<ImageContext>> = vec![&bounds, &chans];
        assert!(run_chain(&ctx, &chain).is_ok());
    }

    #[test]
    fn image_chain_catches_bad_channels() {
        let ctx = ImageContext {
            width: 224,
            height: 224,
            channels: 2,
        };
        let chans = ChannelsAllowed(&[1, 3, 4]);
        let chain: Vec<&dyn Validator<ImageContext>> = vec![&chans];
        let err = run_chain(&ctx, &chain).unwrap_err();
        assert_eq!(err.rule, "channels_allowed");
    }
}
