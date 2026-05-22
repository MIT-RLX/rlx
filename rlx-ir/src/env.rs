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

//! Unified `RLX_*` configuration — readable from **code overrides** or process env.
//!
//! Code overrides (via [`set`], [`RlxEnv::apply`], or [`RuntimeOverrides::install`])
//! take precedence over `std::env` for the same key.
//!
//! ```rust
//! use rlx_ir::env::{self, RlxEnv};
//!
//! // Single knob
//! env::set("RLX_VERBOSE", "1");
//! assert!(env::flag("RLX_VERBOSE"));
//!
//! // Bulk
//! RlxEnv::new()
//!     .set("RLX_DISABLE_MPSGRAPH", "1")
//!     .set("RLX_MPSGRAPH_MIN_FLOPS", "100000")
//!     .apply();
//! ```

use std::collections::HashMap;
use std::ffi::OsString;
use std::str::FromStr;
use std::sync::{OnceLock, RwLock};

static OVERRIDES: OnceLock<RwLock<HashMap<String, String>>> = OnceLock::new();

fn map() -> &'static RwLock<HashMap<String, String>> {
    OVERRIDES.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Normalize to `RLX_*` form.
pub fn normalize_key(key: &str) -> String {
    if key.starts_with("RLX_") {
        key.to_string()
    } else {
        format!("RLX_{key}")
    }
}

/// Set a code-side override. Pass `"0"` / `"false"` to force a flag off even
/// when the process environment has it enabled.
pub fn set(key: impl AsRef<str>, value: impl Into<String>) {
    let key = normalize_key(key.as_ref());
    if let Ok(mut g) = map().write() {
        g.insert(key, value.into());
    }
}

/// Remove a code override for `key`; subsequent reads fall back to process env.
pub fn unset(key: impl AsRef<str>) {
    let key = normalize_key(key.as_ref());
    if let Ok(mut g) = map().write() {
        g.remove(&key);
    }
}

/// Drop every code override.
pub fn clear_overrides() {
    if let Ok(mut g) = map().write() {
        g.clear();
    }
}

/// Read configuration: code override first, then `std::env::var`.
pub fn var(key: &str) -> Option<String> {
    let key = normalize_key(key);
    if let Ok(g) = map().read() {
        if let Some(v) = g.get(&key) {
            return Some(v.clone());
        }
    }
    std::env::var(&key).ok()
}

/// Like [`var`] but returns an `OsString` (mirrors `std::env::var_os`).
pub fn var_os(key: &str) -> Option<OsString> {
    var(key).map(Into::into)
}

/// True when the variable is set to a truthy value (`1`, `true`, `yes`, `on`, …).
/// False when unset or set to `0` / `false` / `off` / `no` / empty.
pub fn flag(key: &str) -> bool {
    match var(key) {
        Some(v) => truthy(&v),
        None => false,
    }
}

/// True when neither a code override nor process env provides the key.
pub fn is_unset(key: &str) -> bool {
    var(key).is_none()
}

/// Parse an integer/bool/string knob, falling back to `default`.
pub fn parse_or<T: FromStr>(key: &str, default: T) -> T {
    var(key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn truthy(v: &str) -> bool {
    let s = v.trim();
    if s.is_empty() {
        return false;
    }
    match s.to_ascii_lowercase().as_str() {
        "0" | "false" | "off" | "no" => false,
        "1" | "true" | "yes" | "on" => true,
        _ if s.chars().all(|c| c.is_ascii_digit()) => s != "0",
        _ => true, // any other non-empty value counts as enabled
    }
}

/// Bulk builder for code-side `RLX_*` overrides.
#[derive(Debug, Clone, Default)]
pub struct RlxEnv {
    pairs: Vec<(String, String)>,
}

impl RlxEnv {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(mut self, key: impl AsRef<str>, value: impl Into<String>) -> Self {
        self.pairs
            .push((normalize_key(key.as_ref()), value.into()));
        self
    }

    pub fn flag(mut self, key: impl AsRef<str>, on: bool) -> Self {
        self.pairs
            .push((normalize_key(key.as_ref()), if on { "1" } else { "0" }.into()));
        self
    }

    /// Apply all pairs to the global override map.
    pub fn apply(self) {
        for (k, v) in self.pairs {
            set(&k, v);
        }
    }
}

/// RAII guard: installs overrides on construction, restores previous values on drop.
pub struct RuntimeOverrides {
    saved: Vec<(String, Option<String>)>,
}

impl RuntimeOverrides {
    /// Install `pairs` for the lifetime of the returned guard.
    pub fn install(pairs: impl IntoIterator<Item = (impl AsRef<str>, impl Into<String>)>) -> Self {
        let mut saved = Vec::new();
        for (key, value) in pairs {
            let key = normalize_key(key.as_ref());
            let prev = map()
                .read()
                .ok()
                .and_then(|g| g.get(&key).cloned());
            saved.push((key.clone(), prev));
            set(&key, value);
        }
        Self { saved }
    }
}

impl Drop for RuntimeOverrides {
    fn drop(&mut self) {
        for (key, prev) in self.saved.drain(..) {
            match prev {
                Some(v) => set(&key, v),
                None => unset(&key),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_override_wins_over_process_env() {
        clear_overrides();
        let _g = RuntimeOverrides::install([("VERBOSE", "2")]);
        assert_eq!(var("RLX_VERBOSE"), Some("2".into()));
        assert!(flag("RLX_VERBOSE"));
    }

    #[test]
    fn flag_parses_falsy_override() {
        clear_overrides();
        set("RLX_DISABLE_MPSGRAPH", "0");
        assert!(!flag("RLX_DISABLE_MPSGRAPH"));
    }

    #[test]
    fn rlx_env_bulk_apply() {
        clear_overrides();
        RlxEnv::new()
            .set("MPSGRAPH_MIN_FLOPS", "42")
            .flag("USE_ICB", true)
            .apply();
        assert_eq!(parse_or("RLX_MPSGRAPH_MIN_FLOPS", 0u64), 42);
        assert!(flag("RLX_USE_ICB"));
        clear_overrides();
    }
}
