// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Curated Public `RLX_*` catalog — thin facade over [`crate::env_registry`].
//!
//! Prefer [`crate::env_registry`] for new code. Exhaustive inventory:
//! `docs/rlx-env-vars.md` (`just gen-rlx-env-vars`).

pub use crate::env_registry::{
    EnvVarDoc, catalog_for_group, format_catalog, public_catalog_docs, public_entries,
};

/// Public catalog entries (compatibility with `ENV_CATALOG`).
pub fn catalog_slice() -> &'static [EnvVarDoc] {
    public_catalog_docs()
}
