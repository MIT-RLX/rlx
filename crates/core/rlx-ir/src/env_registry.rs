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

//! Single source of truth for `RLX_*` environment variables.
//!
//! Every process-env / [`crate::env`] read should eventually resolve through
//! this registry (name, type, default, stability, aliases, layer). Prefer
//! typed `*Config::from_env()` loaders and [`crate::env`] for I/O — this
//! module owns **metadata** and alias resolution.
//!
//! - Public discoverability: [`format_catalog`] / `just env-catalog`
//! - Full inventory docs: `docs/rlx-env-vars.md` (`just gen-rlx-env-vars`)
//! - Deprecated aliases warn when `RLX_ENV_DEPRECATIONS=1` or `RLX_VERBOSE=1`
//! - Prefer `CompileOptions` for compile semantics; env is CLI / bisect

use crate::env;
use std::sync::OnceLock;

/// Value kind for documentation and typed loaders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnvKind {
    Bool,
    /// Unset → `default`; explicit `0`/`false` disables.
    BoolOr {
        default: bool,
    },
    U64,
    String,
    Enum(&'static [&'static str]),
    Path,
}

/// How stable / user-facing a variable is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvStability {
    /// Shown by `just env-catalog`; safe for scripts and docs.
    Public,
    /// Escape hatch / parity bisect; documented in the full inventory.
    Bisect,
    /// Bench / tooling / example-only.
    Internal,
    /// Still accepted; prefer `replace_with`.
    Deprecated {
        replace_with: &'static str,
    },
}

/// Which subsystem owns the option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvLayer {
    Compile,
    Device,
    Runtime,
    Backend(&'static str),
    Tooling,
}

/// One registered `RLX_*` variable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvVarEntry {
    pub name: &'static str,
    pub group: &'static str,
    pub summary: &'static str,
    pub kind: EnvKind,
    pub stability: EnvStability,
    pub aliases: &'static [&'static str],
    pub layer: EnvLayer,
}

/// Backward-compatible view used by older callers / docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvVarDoc {
    pub name: &'static str,
    pub group: &'static str,
    pub summary: &'static str,
}

impl From<&EnvVarEntry> for EnvVarDoc {
    fn from(e: &EnvVarEntry) -> Self {
        Self {
            name: e.name,
            group: e.group,
            summary: e.summary,
        }
    }
}

/// Full registry (Public + Bisect + Internal + Deprecated).
pub const REGISTRY: &[EnvVarEntry] = &include!("env_registry_data.inc.rs");

/// Public-only entries (curated catalog).
pub fn public_entries() -> impl Iterator<Item = &'static EnvVarEntry> {
    REGISTRY
        .iter()
        .filter(|e| matches!(e.stability, EnvStability::Public))
}

/// Look up by canonical name or alias.
pub fn lookup(name: &str) -> Option<&'static EnvVarEntry> {
    REGISTRY
        .iter()
        .find(|e| e.name == name || e.aliases.iter().any(|a| *a == name))
        .or_else(|| {
            // Deprecated rows are looked up by their own `name`.
            REGISTRY.iter().find(|e| {
                matches!(e.stability, EnvStability::Deprecated { .. }) && e.name == name
            })
        })
}

/// True when `name` is a registered canonical name or alias.
pub fn is_registered(name: &str) -> bool {
    lookup(name).is_some()
}

/// Canonical name for `name` (identity if already canonical / unknown).
pub fn canonical_name(name: &str) -> &str {
    match lookup(name) {
        Some(e) => e.name,
        None => name,
    }
}

fn maybe_warn_deprecated(used: &str, entry: &EnvVarEntry) {
    if used == entry.name {
        return;
    }
    if !matches!(
        entry.stability,
        EnvStability::Deprecated { .. }
    ) && !entry.aliases.iter().any(|a| *a == used)
    {
        return;
    }
    // Alias used, or deprecated entry itself.
    if !env::flag("RLX_ENV_DEPRECATIONS") && !env::flag("RLX_VERBOSE") {
        return;
    }
    static WARNED: OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let set = WARNED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    if let Ok(mut g) = set.lock() {
        if !g.insert(used.to_string()) {
            return;
        }
    }
    let replace = match entry.stability {
        EnvStability::Deprecated { replace_with } => replace_with,
        _ => entry.name,
    };
    eprintln!("rlx: `{used}` is deprecated; use `{replace}`");
}

/// Read a string via [`env::var`], resolving aliases and deprecation notices.
pub fn var(name: &str) -> Option<String> {
    if let Some(e) = lookup(name) {
        // Prefer canonical, then aliases (first set wins: canonical first).
        if let Some(v) = env::var(e.name) {
            if name != e.name {
                maybe_warn_deprecated(name, e);
            }
            return Some(v);
        }
        for a in e.aliases {
            if let Some(v) = env::var(a) {
                maybe_warn_deprecated(a, e);
                return Some(v);
            }
        }
        // Deprecated entry looked up by old name: try replace_with already covered
        // via lookup landing on Deprecated row — also try its replace_with.
        if let EnvStability::Deprecated { replace_with } = e.stability {
            maybe_warn_deprecated(name, e);
            return env::var(replace_with);
        }
        return None;
    }
    env::var(name)
}

/// Flag read with alias resolution.
pub fn flag(name: &str) -> bool {
    match var(name) {
        Some(v) => truthy(&v),
        None => false,
    }
}

/// Flag with default when unset (alias-aware).
pub fn flag_or(name: &str, default: bool) -> bool {
    match var(name) {
        Some(v) => truthy(&v),
        None => {
            if let Some(e) = lookup(name)
                && let EnvKind::BoolOr { default: d } = e.kind
            {
                return d;
            }
            default
        }
    }
}

/// Parse with default (alias-aware).
pub fn parse_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    var(name).and_then(|s| s.parse().ok()).unwrap_or(default)
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
        _ => true,
    }
}

/// Entries whose `group` matches, or all Public when `group` is empty.
/// When `public_only`, Non-Public are excluded.
pub fn entries_for_group(group: &str, public_only: bool) -> Vec<&'static EnvVarEntry> {
    REGISTRY
        .iter()
        .filter(|e| {
            if public_only && !matches!(e.stability, EnvStability::Public) {
                return false;
            }
            group.is_empty() || e.group == group
        })
        .collect()
}

/// Pretty-print the Public catalog (optionally filtered by group).
pub fn format_catalog(group: Option<&str>) -> String {
    let g = group.unwrap_or("");
    let entries = entries_for_group(g, true);
    let mut out = String::from("# RLX environment catalog (Public)\n\n");
    let mut cur = "";
    for e in entries {
        if e.group != cur {
            cur = e.group;
            out.push_str(&format!("## {cur}\n\n"));
        }
        out.push_str(&format!("- `{}` — {}\n", e.name, e.summary));
    }
    out.push_str(
        "\nFull registry (Bisect / Internal / Deprecated): `docs/rlx-env-vars.md` \
         (`just gen-rlx-env-vars`). Prefer CompileOptions when a setting changes \
         compile semantics.\n",
    );
    out
}

/// Markdown dump of the full registry (for `gen-rlx-env-vars`).
pub fn format_registry_markdown() -> String {
    let mut out = String::from("# RLX environment variables (`RLX_*`)\n\n");
    out.push_str(
        "Generated from the [`env_registry`](../crates/core/rlx-ir/src/env_registry.rs) \
         source of truth. Prefer `CompileOptions` when a setting changes compile \
         semantics. Curated Public list: `just env-catalog`.\n\n",
    );
    out.push_str("## Legend\n\n");
    out.push_str("| Mark | Meaning |\n|------|---------|\n");
    out.push_str("| Public | Stable / documented (`just env-catalog`) |\n");
    out.push_str("| Bisect | Escape hatch / parity |\n");
    out.push_str("| Internal | Bench / tooling |\n");
    out.push_str("| Deprecated | Use `replace_with` |\n\n");
    out.push_str(&format!("**Registered names:** {}\n\n", REGISTRY.len()));

    let mut by_group: std::collections::BTreeMap<&str, Vec<&EnvVarEntry>> =
        std::collections::BTreeMap::new();
    for e in REGISTRY {
        by_group.entry(e.group).or_default().push(e);
    }
    out.push_str("## Groups\n\n");
    for (g, es) in &by_group {
        out.push_str(&format!("- [{g}](#{g}) — {}\n", es.len()));
    }
    out.push('\n');
    for (g, es) in &by_group {
        out.push_str(&format!("## {g}\n\n"));
        out.push_str("| Name | Stability | Kind | Layer | Summary |\n");
        out.push_str("|------|-----------|------|-------|----------|\n");
        for e in es {
            let stab = match e.stability {
                EnvStability::Public => "Public",
                EnvStability::Bisect => "Bisect",
                EnvStability::Internal => "Internal",
                EnvStability::Deprecated { replace_with } => {
                    out.push_str(&format!(
                        "| `{}` | Deprecated → `{replace_with}` | {:?} | {:?} | {} |\n",
                        e.name, e.kind, e.layer, e.summary
                    ));
                    continue;
                }
            };
            let kind = match e.kind {
                EnvKind::Bool => "Bool".into(),
                EnvKind::BoolOr { default } => format!("BoolOr({default})"),
                EnvKind::U64 => "U64".into(),
                EnvKind::String => "String".into(),
                EnvKind::Enum(v) => format!("Enum{v:?}"),
                EnvKind::Path => "Path".into(),
            };
            let layer = match e.layer {
                EnvLayer::Compile => "Compile".into(),
                EnvLayer::Device => "Device".into(),
                EnvLayer::Runtime => "Runtime".into(),
                EnvLayer::Backend(b) => format!("Backend({b})"),
                EnvLayer::Tooling => "Tooling".into(),
            };
            let aliases = if e.aliases.is_empty() {
                String::new()
            } else {
                format!(" aliases: {}", e.aliases.join(", "))
            };
            out.push_str(&format!(
                "| `{}` | {stab} | {kind} | {layer} | {}{aliases} |\n",
                e.name, e.summary
            ));
        }
        out.push('\n');
    }
    out.push_str("## Maintenance\n\n");
    out.push_str("```sh\njust gen-rlx-env-vars\n```\n\n");
    out.push_str(
        "Add new names to `env_registry_data.inc.rs` (or regenerate via \
         `scripts/gen-rlx-env-vars.py --seed-reads`). Unregistered \
         `env::flag(\"RLX_…\")` call sites fail `just check-rlx-env-vars`.\n",
    );
    out
}

/// Public catalog as docs (for `catalog_for_group` compat).
pub fn catalog_for_group(group: &str) -> Vec<EnvVarDoc> {
    public_entries()
        .filter(|e| group.is_empty() || e.group == group)
        .map(EnvVarDoc::from)
        .collect()
}

/// Public [`EnvVarDoc`] slice built once (for `ENV_CATALOG` re-export).
pub fn public_catalog_docs() -> &'static [EnvVarDoc] {
    static DOCS: OnceLock<Vec<EnvVarDoc>> = OnceLock::new();
    DOCS.get_or_init(|| public_entries().map(EnvVarDoc::from).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_nonempty_and_prefixed() {
        assert!(!REGISTRY.is_empty());
        for e in REGISTRY {
            assert!(e.name.starts_with("RLX_"), "{}", e.name);
            assert!(!e.group.is_empty());
            assert!(!e.summary.is_empty());
        }
    }

    #[test]
    fn public_catalog_has_device() {
        let s = format_catalog(Some("device"));
        assert!(s.contains("RLX_DEVICE"));
        assert!(!s.contains("RLX_DISABLE_MPSGRAPH"));
    }

    #[test]
    fn alias_resolves_metal_dequant() {
        assert_eq!(
            canonical_name("RLX_DISABLE_METAL_DEQUANT_GPU"),
            "RLX_METAL_DEQUANT_GPU_DISABLE"
        );
        assert!(is_registered("RLX_DISABLE_METAL_DEQUANT_GPU"));
        assert!(is_registered("RLX_METAL_DEQUANT_GPU_DISABLE"));
    }

    #[test]
    fn lookup_cache_param() {
        let e = lookup("RLX_CACHE_PARAM_INVARIANT").expect("registered");
        assert!(matches!(e.stability, EnvStability::Public));
        assert!(matches!(e.layer, EnvLayer::Compile));
    }
}
