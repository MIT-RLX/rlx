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

//! Named weights registry (plan #24).
//!
//! Borrowed from MAX's `weights_registry/weights_registry.mojo`.
//! Promotes the passive `weights.rs` loader contract to an active
//! per-process registry: named handles + reference counts + LoRA
//! adapter accounting.
//!
//! Why a registry beyond the existing `WeightLoader`?
//!   - **LoRA hot-swap.** Multiple adapters share the same base
//!     weights. The registry owns the bytes; adapters are
//!     handle-pointers with their own metadata.
//!   - **Tied embeddings.** GPT-2 / LLaMA / Gemma tie input
//!     embedding to output projection. With a registry both
//!     positions resolve to the same `WeightEntry`.
//!   - **Weight streaming.** Future "load layer N when running
//!     layer N" patterns need to refcount which weights are
//!     in-memory.
//!   - **Memory accounting.** [`#35`] memory estimation queries
//!     `total_bytes()` to gate model loads against unified-memory
//!     budget.

use rlx_ir::Shape;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Stable handle into a [`WeightRegistry`]. Cheap to copy; passing
/// it around is the recommended way to refer to a weight without
/// keeping the registry borrowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightHandle(u64);

impl WeightHandle {
    pub fn id(self) -> u64 {
        self.0
    }
}

/// What role a weight plays. Drives downstream scheduling
/// (LoRA-aware request grouping) and accounting.
#[derive(Debug, Clone)]
pub enum WeightKind {
    /// A base model weight — independent storage.
    Base,
    /// A LoRA adapter's slice (down-proj A or up-proj B).
    /// Multiple adapters can attach to the same base; the
    /// scheduler groups requests by `adapter` name.
    LoraAdapter { adapter: String },
    /// A view that resolves to another weight's storage. Used for
    /// tied embeddings — `embed_tokens.weight` and
    /// `lm_head.weight` are the same buffer, two names.
    TiedAlias { target: WeightHandle },
}

/// One entry in the registry.
#[derive(Debug)]
pub struct WeightEntry {
    pub name: String,
    pub shape: Shape,
    pub kind: WeightKind,
    /// `Arc` so multiple consumers (different graphs / different
    /// LoRA combinations / hot-reload) can hold the same bytes
    /// without copying.
    pub bytes: Arc<[u8]>,
    /// Ref count. Goes up on `pin`, down on `release`. Hitting
    /// zero on `release` keeps the entry in the registry; explicit
    /// `unregister` is required to drop. This separation matters
    /// for "weight streaming" use cases that re-pin frequently.
    pub refs: AtomicUsize,
}

/// The registry itself. One per process is the typical setup; a
/// `Session` can borrow a registry to resolve weight names during
/// graph compile.
pub struct WeightRegistry {
    by_name: HashMap<String, WeightHandle>,
    by_handle: HashMap<u64, Arc<WeightEntry>>,
    next_id: AtomicU64,
}

impl WeightRegistry {
    pub fn new() -> Self {
        Self {
            by_name: HashMap::new(),
            by_handle: HashMap::new(),
            next_id: AtomicU64::new(0),
        }
    }

    fn alloc_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Register a fresh weight under `name`. If `name` already
    /// exists, returns the existing handle (idempotent — useful for
    /// weight loaders that may see the same tensor twice).
    pub fn register(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        bytes: Arc<[u8]>,
        kind: WeightKind,
    ) -> WeightHandle {
        let name = name.into();
        if let Some(&h) = self.by_name.get(&name) {
            return h;
        }
        let id = self.alloc_id();
        let h = WeightHandle(id);
        let entry = Arc::new(WeightEntry {
            name: name.clone(),
            shape,
            kind,
            bytes,
            refs: AtomicUsize::new(0),
        });
        self.by_name.insert(name, h);
        self.by_handle.insert(id, entry);
        h
    }

    /// Resolve a name → handle.
    pub fn lookup(&self, name: &str) -> Option<WeightHandle> {
        self.by_name.get(name).copied()
    }

    /// Read the entry for `handle`. Resolves a TiedAlias one step.
    pub fn get(&self, handle: WeightHandle) -> Option<&Arc<WeightEntry>> {
        let entry = self.by_handle.get(&handle.0)?;
        if let WeightKind::TiedAlias { target } = entry.kind {
            return self.by_handle.get(&target.0);
        }
        Some(entry)
    }

    /// Increment the entry's refcount. Returns the new count.
    pub fn pin(&self, handle: WeightHandle) -> Option<usize> {
        let entry = self.by_handle.get(&handle.0)?;
        Some(entry.refs.fetch_add(1, Ordering::Relaxed) + 1)
    }

    /// Decrement the entry's refcount. Returns the new count.
    /// Hitting zero does NOT drop the entry — call `unregister` to
    /// drop. Returns `None` if the handle is unknown.
    pub fn release(&self, handle: WeightHandle) -> Option<usize> {
        let entry = self.by_handle.get(&handle.0)?;
        let prev = entry.refs.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(prev >= 1, "release on a zero-refcount entry");
        Some(prev - 1)
    }

    /// Drop an entry. Caller must have already `release`d to zero
    /// (debug-asserted). Returns the entry's name on success.
    pub fn unregister(&mut self, handle: WeightHandle) -> Option<String> {
        let entry = self.by_handle.remove(&handle.0)?;
        debug_assert_eq!(
            entry.refs.load(Ordering::Relaxed),
            0,
            "unregister on a still-referenced entry: refs={}",
            entry.refs.load(Ordering::Relaxed)
        );
        self.by_name.remove(&entry.name);
        Some(entry.name.clone())
    }

    /// Total registered weight bytes — sums each base entry's
    /// `bytes.len()` plus each adapter; tied-alias entries don't
    /// double-count (they share storage).
    pub fn total_bytes(&self) -> usize {
        self.by_handle
            .values()
            .filter(|e| !matches!(e.kind, WeightKind::TiedAlias { .. }))
            .map(|e| e.bytes.len())
            .sum()
    }

    /// All handles whose kind is `LoraAdapter { adapter: <name> }`.
    /// Used by LoRA-aware scheduling (#33) to group requests.
    pub fn lora_adapter_handles(&self, adapter: &str) -> Vec<WeightHandle> {
        let mut v: Vec<WeightHandle> = self
            .by_handle
            .iter()
            .filter_map(|(&id, e)| match &e.kind {
                WeightKind::LoraAdapter { adapter: a } if a == adapter => Some(WeightHandle(id)),
                _ => None,
            })
            .collect();
        v.sort_by_key(|h| h.0);
        v
    }

    /// All distinct LoRA adapter names currently registered.
    pub fn lora_adapter_names(&self) -> Vec<String> {
        let mut s: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for e in self.by_handle.values() {
            if let WeightKind::LoraAdapter { adapter } = &e.kind {
                s.insert(adapter.clone());
            }
        }
        s.into_iter().collect()
    }

    pub fn len(&self) -> usize {
        self.by_handle.len()
    }
    pub fn is_empty(&self) -> bool {
        self.by_handle.is_empty()
    }
}

impl Default for WeightRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;

    fn shape() -> Shape {
        Shape::new(&[8, 8], DType::F32)
    }
    fn bytes(n: usize) -> Arc<[u8]> {
        vec![0u8; n].into()
    }

    #[test]
    fn register_and_lookup() {
        let mut r = WeightRegistry::new();
        let h = r.register("w", shape(), bytes(256), WeightKind::Base);
        assert_eq!(r.lookup("w"), Some(h));
        let entry = r.get(h).unwrap();
        assert_eq!(entry.name, "w");
        assert_eq!(entry.bytes.len(), 256);
    }

    #[test]
    fn register_is_idempotent() {
        let mut r = WeightRegistry::new();
        let h1 = r.register("w", shape(), bytes(128), WeightKind::Base);
        let h2 = r.register("w", shape(), bytes(999), WeightKind::Base);
        // Same handle; the second register call doesn't overwrite.
        assert_eq!(h1, h2);
        assert_eq!(r.get(h1).unwrap().bytes.len(), 128);
    }

    #[test]
    fn pin_release_balance() {
        let mut r = WeightRegistry::new();
        let h = r.register("w", shape(), bytes(64), WeightKind::Base);
        assert_eq!(r.pin(h), Some(1));
        assert_eq!(r.pin(h), Some(2));
        assert_eq!(r.release(h), Some(1));
        assert_eq!(r.release(h), Some(0));
        // Unregister now ok.
        assert_eq!(r.unregister(h), Some("w".to_string()));
        assert!(r.lookup("w").is_none());
    }

    #[test]
    fn tied_alias_resolves_to_target() {
        let mut r = WeightRegistry::new();
        let target = r.register("embed", shape(), bytes(128), WeightKind::Base);
        let alias = r.register(
            "lm_head",
            shape(),
            bytes(0), // alias has no bytes of its own
            WeightKind::TiedAlias { target },
        );
        let resolved = r.get(alias).unwrap();
        assert_eq!(resolved.name, "embed");
        assert_eq!(resolved.bytes.len(), 128);
    }

    #[test]
    fn total_bytes_skips_aliases() {
        let mut r = WeightRegistry::new();
        let _t = r.register("embed", shape(), bytes(100), WeightKind::Base);
        let _a = r.register(
            "lm_head",
            shape(),
            bytes(0),
            WeightKind::TiedAlias {
                target: r.lookup("embed").unwrap(),
            },
        );
        let _b = r.register("ffn", shape(), bytes(200), WeightKind::Base);
        assert_eq!(r.total_bytes(), 300, "alias must not double-count");
    }

    #[test]
    fn lora_grouping() {
        let mut r = WeightRegistry::new();
        let _b = r.register("ffn", shape(), bytes(100), WeightKind::Base);
        r.register(
            "ffn.lora.a",
            shape(),
            bytes(8),
            WeightKind::LoraAdapter {
                adapter: "code".into(),
            },
        );
        r.register(
            "ffn.lora.b",
            shape(),
            bytes(8),
            WeightKind::LoraAdapter {
                adapter: "code".into(),
            },
        );
        r.register(
            "attn.lora.a",
            shape(),
            bytes(8),
            WeightKind::LoraAdapter {
                adapter: "math".into(),
            },
        );

        let mut adapters = r.lora_adapter_names();
        adapters.sort();
        assert_eq!(adapters, vec!["code".to_string(), "math".to_string()]);

        let code_handles = r.lora_adapter_handles("code");
        assert_eq!(code_handles.len(), 2);
    }
}
