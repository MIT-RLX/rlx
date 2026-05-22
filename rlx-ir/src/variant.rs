// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Model execution variants — one object drives cache keys and [`DimBinding`].
//!
//! Mirrors the “shader components” idea from extensible shading systems: the same
//! granularity selects **what to specialize** and **which symbolic dims to bind**.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::dynamic::sym;
use crate::shape::DimBinding;

/// Coarse execution phase (prefill vs decode vs encoder).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelPhase {
    Prefill,
    Decode,
    Encoder,
    Inference,
}

/// Concrete shape bucket for compile-once / specialize-at-runtime workflows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelVariant {
    pub batch: usize,
    pub seq: usize,
    pub past_seq: Option<usize>,
    pub phase: ModelPhase,
    /// Extra dynamic symbols beyond batch/seq/past (e.g. custom ragged axes).
    pub extra: Vec<(u32, usize)>,
}

impl ModelVariant {
    pub fn prefill(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            seq,
            past_seq: None,
            phase: ModelPhase::Prefill,
            extra: Vec::new(),
        }
    }

    /// Single-step decode: `seq` is the new token count (often 1); `past_seq` is KV length.
    pub fn decode(batch: usize, past_seq: usize, new_tokens: usize) -> Self {
        Self {
            batch,
            seq: new_tokens,
            past_seq: Some(past_seq),
            phase: ModelPhase::Decode,
            extra: Vec::new(),
        }
    }

    pub fn encoder(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            seq,
            past_seq: None,
            phase: ModelPhase::Encoder,
            extra: Vec::new(),
        }
    }

    pub fn with_extra(mut self, symbol: u32, size: usize) -> Self {
        self.extra.push((symbol, size));
        self
    }

    /// Stable cache key: phase + bound leading dims + extra symbols.
    pub fn cache_key(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.phase.hash(&mut h);
        self.batch.hash(&mut h);
        self.seq.hash(&mut h);
        self.past_seq.hash(&mut h);
        for (sym, size) in &self.extra {
            sym.hash(&mut h);
            size.hash(&mut h);
        }
        h.finish()
    }

    /// Symbol bindings used by [`crate::dynamic::bind_graph`] / compile specialization.
    pub fn dim_binding(&self) -> DimBinding {
        let mut b = match (self.phase, self.past_seq) {
            (ModelPhase::Decode, Some(past)) => DimBinding::batch_past_seq(self.batch, past),
            _ => DimBinding::batch_seq(self.batch, self.seq),
        };
        if self.phase == ModelPhase::Decode {
            b.set(sym::SEQ, self.seq);
        }
        for (sym, size) in &self.extra {
            b.set(*sym, *size);
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_binding_sets_batch_seq() {
        let v = ModelVariant::prefill(2, 128);
        let b = v.dim_binding();
        assert_eq!(b.get(sym::BATCH), Some(2));
        assert_eq!(b.get(sym::SEQ), Some(128));
    }

    #[test]
    fn decode_binding_sets_past_and_new_seq() {
        let v = ModelVariant::decode(1, 64, 1);
        let b = v.dim_binding();
        assert_eq!(b.get(sym::BATCH), Some(1));
        assert_eq!(b.get(sym::PAST_SEQ), Some(64));
        assert_eq!(b.get(sym::SEQ), Some(1));
    }

    #[test]
    fn cache_key_differs_by_phase() {
        let a = ModelVariant::prefill(1, 8).cache_key();
        let b = ModelVariant::decode(1, 7, 1).cache_key();
        assert_ne!(a, b);
    }
}
