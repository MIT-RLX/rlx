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
