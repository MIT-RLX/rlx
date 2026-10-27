// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
    Deprecated { replace_with: &'static str },
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
        .find(|e| e.name == name || e.aliases.contains(&name))
        .or_else(|| {
            // Deprecated rows are looked up by their own `name`.
            REGISTRY
                .iter()
                .find(|e| matches!(e.stability, EnvStability::Deprecated { .. }) && e.name == name)
        })
}

/// True when `name` is a registered canonical name or alias.
pub fn is_registered(name: &str) -> bool {
    lookup(name).is_some()
}

/// The registered name (or alias) nearest to `name` by edit distance, when it
/// is close enough to be a plausible typo: at most a quarter of the name's
/// length, and at least one edit. Feeds the unknown-variable warning and
/// `cargo rlx doctor`.
pub fn suggest(name: &str) -> Option<&'static str> {
    let budget = (name.len() / 4).max(1);
    REGISTRY
        .iter()
        .flat_map(|e| std::iter::once(e.name).chain(e.aliases.iter().copied()))
        .map(|cand| (levenshtein(name, cand), cand))
        .filter(|(d, _)| *d <= budget)
        .min_by_key(|(d, c)| (*d, c.len()))
        .map(|(_, c)| c)
}

fn levenshtein(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != *cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// `RLX_*` variables set in `vars` that the registry does not know — read by
/// nothing, so most likely typos — each with [`suggest`]'s nearest name.
pub fn unknown_vars(vars: impl IntoIterator<Item = String>) -> Vec<(String, Option<&'static str>)> {
    let mut out: Vec<(String, Option<&'static str>)> = vars
        .into_iter()
        .filter(|k| k.starts_with("RLX_") && !is_registered(k))
        .map(|k| {
            let s = suggest(&k);
            (k, s)
        })
        .collect();
    out.sort();
    out
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
    if !matches!(entry.stability, EnvStability::Deprecated { .. }) && !entry.aliases.contains(&used)
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

/// The value of `name` from `raw` (an override, else the process
/// environment), resolving aliases and deprecation notices: a registered
/// name reads its canonical spelling first, then its aliases, then — for a
/// deprecated row — its replacement. [`env`]'s reads resolve through this,
/// once a key, and keep the result.
pub(crate) fn resolve(name: &str, raw: impl Fn(&str) -> Option<String>) -> Option<String> {
    let Some(e) = lookup(name) else {
        return raw(name);
    };
    if let Some(v) = raw(e.name) {
        if name != e.name {
            maybe_warn_deprecated(name, e);
        }
        return Some(v);
    }
    for a in e.aliases {
        if let Some(v) = raw(a) {
            maybe_warn_deprecated(a, e);
            return Some(v);
        }
    }
    if let EnvStability::Deprecated { replace_with } = e.stability {
        maybe_warn_deprecated(name, e);
        return raw(replace_with).or_else(|| layered(replace_with, &raw).map(|(v, _)| v));
    }
    layered(e.name, &raw).map(|(v, _)| v)
}

/// List variables that switch on a whole family at once. `RLX_DISABLE=cse,sccp`
/// sets `RLX_DISABLE_CSE` and `RLX_DISABLE_SCCP`; `RLX_TRACE=metal,gguf` sets
/// `RLX_METAL_TRACE` and `RLX_GGUF_TRACE`; `RLX_DEBUG` likewise for `*_DEBUG`.
/// Tokens are case-insensitive. The long names keep working unchanged.
pub const FAMILIES: &[(&str, &str, &str)] = &[
    ("RLX_DISABLE", "RLX_DISABLE_", ""), // env-lint: ignore
    ("RLX_TRACE", "RLX_", "_TRACE"),
    ("RLX_DEBUG", "RLX_", "_DEBUG"),
];

/// Built-in `RLX_PROFILE` bundles: `(name, what it is for, settings)`. Each
/// setting sits below anything set explicitly. Only switches whose effect the
/// registry documents belong here; a profile that guesses is worse than none.
pub const PROFILES: &[(&str, &str, &[(&str, &str)])] = &[
    (
        "debug",
        "runtime logging, the dispatch report, static NaN lint, deprecation notices",
        &[
            ("RLX_VERBOSE", "1"),
            ("RLX_DISPATCH_REPORT", "1"),
            ("RLX_LINT_NUMERICS", "1"),
            ("RLX_ENV_DEPRECATIONS", "1"),
        ],
    ),
    (
        "nans",
        "localize the first NaN/Inf at runtime (slow), plus the static lint",
        &[("RLX_DEBUG_NANS", "1"), ("RLX_LINT_NUMERICS", "1")],
    ),
    (
        "parity",
        "tightest numerics on CUDA, ROCm and Metal, for cross-backend comparison",
        &[
            ("RLX_CUDA_PARITY", "1"),
            ("RLX_CUDA_MATMUL_PRECISE", "1"),
            ("RLX_ROCM_PARITY", "1"),
            ("RLX_METAL_PRECISE", "1"),
            ("RLX_METAL_SGEMM_PRECISE", "1"),
        ],
    ),
    (
        "deterministic",
        "reproducible collective reductions (set on every rank)",
        &[("RLX_DETERMINISTIC_REDUCE", "1")],
    ),
    (
        "ci",
        "a missing GPU fails tests instead of skipping; memory-plan self-check",
        &[("RLX_REQUIRE_DEVICE", "1"), ("RLX_MEM_VERIFY", "1")],
    ),
];

/// Where a value that was not set explicitly came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layer {
    /// A list variable (`RLX_DISABLE`, `RLX_TRACE`, `RLX_DEBUG`).
    Family(&'static str),
    /// The top level of the `RLX_CONFIG` file.
    Config(String),
    /// A profile named in `RLX_PROFILE` (built-in or from the config file).
    Profile(String),
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Layer::Family(list) => write!(f, "{list}"),
            Layer::Config(path) => write!(f, "RLX_CONFIG {path}"),
            Layer::Profile(p) => write!(f, "RLX_PROFILE {p}"),
        }
    }
}

/// The value of registered `name` from the layers below explicit settings —
/// families, then the config file, then profiles (the last one listed wins) —
/// and which layer supplied it.
pub fn layered(name: &str, raw: &impl Fn(&str) -> Option<String>) -> Option<(String, Layer)> {
    for &(list, prefix, suffix) in FAMILIES {
        let Some(token) = name
            .strip_prefix(prefix)
            .and_then(|r| r.strip_suffix(suffix))
            .filter(|t| !t.is_empty() && name != list)
        else {
            continue;
        };
        if raw(list).is_some_and(|l| l.split(',').any(|t| t.trim().eq_ignore_ascii_case(token))) {
            return Some(("1".into(), Layer::Family(list)));
        }
    }
    let config = raw("RLX_CONFIG").map(|p| (load_config(&p), p));
    if let Some((cfg, path)) = &config
        && let Some(v) = cfg.top.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    {
        return Some((v.clone(), Layer::Config(path.clone())));
    }
    let profiles = raw("RLX_PROFILE")?;
    for p in profiles
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .rev()
    {
        let from_file = config
            .as_ref()
            .and_then(|(cfg, _)| cfg.profiles.iter().find(|(n, _)| n == p))
            .and_then(|(_, kv)| kv.iter().find(|(k, _)| k == name))
            .map(|(_, v)| v.clone());
        let builtin = || {
            PROFILES
                .iter()
                .find(|(n, _, _)| *n == p)
                .and_then(|(_, _, kv)| kv.iter().find(|(k, _)| *k == name))
                .map(|(_, v)| v.to_string())
        };
        if let Some(v) = from_file.or_else(builtin) {
            return Some((v, Layer::Profile(p.to_string())));
        }
    }
    None
}

/// Every registered variable the layers set, with its value and source — what
/// `cargo rlx doctor` shows. Explicit settings are not included.
pub fn layered_values(raw: &impl Fn(&str) -> Option<String>) -> Vec<(&'static str, String, Layer)> {
    REGISTRY
        .iter()
        .filter(|e| raw(e.name).is_none())
        .filter_map(|e| layered(e.name, raw).map(|(v, l)| (e.name, v, l)))
        .collect()
}

/// A parsed `RLX_CONFIG` file. The format is a TOML subset:
///
/// ```toml
/// # applied whenever RLX_CONFIG points here
/// RLX_DISPATCH_REPORT = 1
/// LINT_NUMERICS = true          # the RLX_ prefix is optional
///
/// [profile.bench]               # selected with RLX_PROFILE=bench
/// RLX_CUDA_TMA = true
/// RLX_METAL_SDPA_TUNE_CACHE = "/tmp/sdpa.cache"
/// ```
///
/// `true`/`false` become `1`/`0`; quotes are stripped. Unknown keys and
/// malformed lines are reported once on stderr and skipped.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConfigFile {
    pub top: Vec<(String, String)>,
    pub profiles: Vec<(String, Vec<(String, String)>)>,
    /// One message per skipped line.
    pub problems: Vec<String>,
}

pub fn parse_config(text: &str) -> ConfigFile {
    let mut cfg = ConfigFile::default();
    let mut section: Option<usize> = None;
    for (i, line) in text.lines().enumerate() {
        let line = strip_comment(line).trim();
        if line.is_empty() {
            continue;
        }
        if let Some(head) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            match head.trim().strip_prefix("profile.") {
                Some(name) if !name.is_empty() => {
                    cfg.profiles.push((name.to_string(), Vec::new()));
                    section = Some(cfg.profiles.len() - 1);
                }
                _ => cfg.problems.push(format!(
                    "line {}: unknown section [{head}] (only [profile.NAME])",
                    i + 1
                )),
            }
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            cfg.problems
                .push(format!("line {}: expected KEY = value", i + 1));
            continue;
        };
        let key = env::normalize_key(k.trim());
        if !is_registered(&key) {
            let hint = suggest(&key)
                .map(|s| format!("; did you mean {s}?"))
                .unwrap_or_default();
            cfg.problems.push(format!(
                "line {}: {key} is not a variable rlx reads{hint}",
                i + 1
            ));
            continue;
        }
        let v = v.trim();
        let v = v
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or(v);
        let v = match v {
            "true" => "1",
            "false" => "0",
            v => v,
        };
        let target = match section {
            Some(s) => &mut cfg.profiles[s].1,
            None => &mut cfg.top,
        };
        target.push((canonical_name(&key).to_string(), v.to_string()));
    }
    cfg
}

/// `#` starts a comment outside double quotes.
fn strip_comment(line: &str) -> &str {
    let mut quoted = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '#' if !quoted => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Parse `path` once per process; problems (and an unreadable file) are
/// reported on stderr the first time.
fn load_config(path: &str) -> std::sync::Arc<ConfigFile> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    static FILES: OnceLock<Mutex<HashMap<String, Arc<ConfigFile>>>> = OnceLock::new();
    let files = FILES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = files.lock().unwrap_or_else(|e| e.into_inner());
    g.entry(path.to_string())
        .or_insert_with(|| {
            let cfg = match std::fs::read_to_string(path) {
                Ok(text) => parse_config(&text),
                Err(e) => ConfigFile {
                    problems: vec![format!("cannot read: {e}")],
                    ..ConfigFile::default()
                },
            };
            for p in &cfg.problems {
                eprintln!("rlx: RLX_CONFIG {path}: {p}");
            }
            Arc::new(cfg)
        })
        .clone()
}

/// [`env::var`]: it resolves aliases and deprecations itself.
pub fn var(name: &str) -> Option<String> {
    env::var(name)
}

/// [`env::flag`].
pub fn flag(name: &str) -> bool {
    env::flag(name)
}

/// [`env::flag`], else the registry's declared default (`BoolOr`), else
/// `default`, when the key is unset.
pub fn flag_or(name: &str, default: bool) -> bool {
    if !env::is_unset(name) {
        return env::flag(name);
    }
    match lookup(name).map(|e| e.kind) {
        Some(EnvKind::BoolOr { default: d }) => d,
        _ => default,
    }
}

/// [`env::parse_or`].
pub fn parse_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    env::parse_or(name, default)
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
    use std::collections::HashMap;

    fn reader(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| m.get(k).cloned()
    }

    #[test]
    fn family_lists_switch_on_their_members() {
        let raw = reader(&[("RLX_DISABLE", "cse, sccp"), ("RLX_TRACE", "gguf")]);
        assert_eq!(
            super::resolve("RLX_DISABLE_CSE", &raw).as_deref(),
            Some("1")
        );
        assert_eq!(
            super::resolve("RLX_DISABLE_SCCP", &raw).as_deref(),
            Some("1")
        );
        assert_eq!(super::resolve("RLX_GGUF_TRACE", &raw).as_deref(), Some("1"));
        assert_eq!(super::resolve("RLX_METAL_TRACE", &raw), None);
        assert_eq!(super::resolve("RLX_DISABLE_MPS", &raw), None);
        // An explicit value beats the family.
        let raw = reader(&[("RLX_DISABLE", "cse"), ("RLX_DISABLE_CSE", "0")]);
        assert_eq!(
            super::resolve("RLX_DISABLE_CSE", &raw).as_deref(),
            Some("0")
        );
    }

    #[test]
    fn profiles_layer_below_explicit_values_and_later_ones_win() {
        let raw = reader(&[("RLX_PROFILE", "debug")]);
        assert_eq!(
            super::resolve("RLX_DISPATCH_REPORT", &raw).as_deref(),
            Some("1")
        );
        let raw = reader(&[("RLX_PROFILE", "debug"), ("RLX_DISPATCH_REPORT", "0")]);
        assert_eq!(
            super::resolve("RLX_DISPATCH_REPORT", &raw).as_deref(),
            Some("0")
        );
        let raw = reader(&[("RLX_PROFILE", "debug,nans")]);
        assert_eq!(super::resolve("RLX_DEBUG_NANS", &raw).as_deref(), Some("1"));
        assert_eq!(super::resolve("RLX_VERBOSE", &raw).as_deref(), Some("1"));
        let layered = super::layered_values(&raw);
        assert!(layered.iter().any(
            |(n, _, l)| *n == "RLX_LINT_NUMERICS" && *l == super::Layer::Profile("nans".into())
        ));
    }

    /// Every profile setting must name a registered variable, or the profile
    /// silently sets nothing.
    #[test]
    fn every_profile_setting_is_registered() {
        for (p, _, kv) in super::PROFILES {
            for (k, _) in *kv {
                assert!(
                    super::is_registered(k),
                    "profile {p}: {k} is not registered"
                );
            }
        }
        for (list, _, _) in super::FAMILIES {
            assert!(super::is_registered(list), "{list} is not registered");
        }
    }

    #[test]
    fn config_file_parses_top_level_profiles_and_reports_problems() {
        let cfg = super::parse_config(
            "# comment\n\
             RLX_DISPATCH_REPORT = true\n\
             LINT_NUMERICS = 1   # prefix optional\n\
             RLX_DISPACH_REPORT = 1\n\
             [profile.bench]\n\
             RLX_METAL_SDPA_TUNE_CACHE = \"/tmp/a#b\"\n\
             [weird]\n\
             nonsense\n",
        );
        assert_eq!(
            cfg.top,
            vec![
                ("RLX_DISPATCH_REPORT".to_string(), "1".to_string()),
                ("RLX_LINT_NUMERICS".to_string(), "1".to_string()),
            ]
        );
        assert_eq!(cfg.profiles.len(), 1);
        assert_eq!(cfg.profiles[0].0, "bench");
        assert_eq!(cfg.profiles[0].1[0].1, "/tmp/a#b");
        assert_eq!(cfg.problems.len(), 3, "{:?}", cfg.problems);
        assert!(cfg.problems[0].contains("did you mean RLX_DISPATCH_REPORT"));
    }

    #[test]
    fn typos_get_the_nearest_registered_name() {
        const TYPO: &str = "RLX_DISPATCH_REPOT"; // env-lint: ignore
        const SHORT_TYPO: &str = "RLX_VERBSE"; // env-lint: ignore
        const FAR: &str = "RLX_ZZZZZZZZZZZZZZZZZZZZ"; // env-lint: ignore
        assert_eq!(super::suggest(TYPO), Some("RLX_DISPATCH_REPORT"));
        assert_eq!(super::suggest(FAR), None);
        assert_eq!(super::levenshtein("kitten", "sitting"), 3);
        let unknown = super::unknown_vars(["RLX_VERBOSE", SHORT_TYPO, "PATH"].map(String::from));
        assert_eq!(unknown, vec![(SHORT_TYPO.to_string(), Some("RLX_VERBOSE"))]);
    }

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
