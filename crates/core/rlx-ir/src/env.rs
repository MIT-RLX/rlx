// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

/// Code overrides. `None` is an explicit removal ([`remove`]): the key reads
/// as unset even when the process environment has it.
static OVERRIDES: OnceLock<RwLock<HashMap<String, Option<String>>>> = OnceLock::new();

fn map() -> &'static RwLock<HashMap<String, Option<String>>> {
    OVERRIDES.get_or_init(|| RwLock::new(HashMap::new()))
}

thread_local! {
    /// Thread-scoped overrides ([`scoped`]), newest last. Checked before the
    /// global overrides and the resolved cache, and seen only by this thread.
    static SCOPED: std::cell::RefCell<Vec<(String, Option<String>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The thread-scoped value of `key`, if this thread has one: `Some(None)` is
/// a scoped removal.
fn scoped_value(key: &str) -> Option<Option<String>> {
    SCOPED.with(|s| {
        let s = s.borrow();
        if s.is_empty() {
            return None;
        }
        s.iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    })
}

/// Override `key` on **this thread only** until the guard drops.
///
/// For per-model settings read at graph-build time (`RLX_DROPOUT`) when two
/// models build on two threads at once: a process-wide [`set`] or
/// `std::env::set_var` would hand each the other's value. Reads on other
/// threads — including rayon workers this thread hands work to — do not see
/// it, so scope only keys read by the thread that builds.
#[must_use = "the override ends when the guard drops"]
pub fn scoped(key: impl AsRef<str>, value: impl Into<String>) -> ScopedVar {
    push_scoped(normalize_key(key.as_ref()), Some(value.into()))
}

/// [`scoped`], making `key` read as unset on this thread.
#[must_use = "the override ends when the guard drops"]
pub fn scoped_remove(key: impl AsRef<str>) -> ScopedVar {
    push_scoped(normalize_key(key.as_ref()), None)
}

fn push_scoped(key: String, value: Option<String>) -> ScopedVar {
    let depth = SCOPED.with(|s| {
        let mut s = s.borrow_mut();
        s.push((key, value));
        s.len() - 1
    });
    ScopedVar {
        depth,
        _not_send: std::marker::PhantomData,
    }
}

/// Guard from [`scoped`] / [`scoped_remove`]; restores on drop. Not `Send`:
/// it belongs to the thread whose reads it changes.
pub struct ScopedVar {
    depth: usize,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Drop for ScopedVar {
    fn drop(&mut self) {
        SCOPED.with(|s| s.borrow_mut().truncate(self.depth));
    }
}

/// Every key read so far, resolved (override, else process environment).
/// A read is then one map lookup under a read lock, no allocation and no
/// `getenv`: backends read their `RLX_*` switches per dispatch, ~80-100 ns
/// each uncached. Any override change empties it; a process-environment
/// change made with `std::env::set_var` after a read needs [`reload`].
static RESOLVED: OnceLock<RwLock<HashMap<String, Option<String>>>> = OnceLock::new();

fn resolved() -> &'static RwLock<HashMap<String, Option<String>>> {
    RESOLVED.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Forget every resolved value, so the next read sees the process
/// environment as it is now. For code that changes `RLX_*` variables with
/// `std::env::set_var` / `remove_var` after they were read; [`set`],
/// [`remove`], [`unset`] and [`clear_overrides`] do it themselves.
pub fn reload() {
    if let Ok(mut g) = resolved().write() {
        g.clear();
    }
}

/// `f` of the value of `key` (an `RLX_*` name, or one without the prefix):
/// from the resolved cache, else resolved now and kept.
fn with_value<R>(key: &str, f: impl FnOnce(Option<&str>) -> R) -> R {
    let key: std::borrow::Cow<'_, str> = if key.starts_with("RLX_") {
        std::borrow::Cow::Borrowed(key)
    } else {
        std::borrow::Cow::Owned(format!("RLX_{key}"))
    };
    if let Some(v) = scoped_value(key.as_ref()) {
        return f(v.as_deref());
    }
    if let Ok(g) = resolved().read()
        && let Some(v) = g.get(key.as_ref())
    {
        return f(v.as_deref());
    }
    warn_unknown_once();
    // An override, else the process environment, by the registry's
    // spelling of the key (canonical, aliases, deprecated replacement).
    let raw = |k: &str| match map().read().ok().and_then(|g| g.get(k).cloned()) {
        Some(v) => v,
        None => std::env::var(k).ok(),
    };
    let value = crate::env_registry::resolve(key.as_ref(), raw);
    let out = f(value.as_deref());
    if let Ok(mut g) = resolved().write() {
        g.insert(key.into_owned(), value);
    }
    out
}

/// Registered variables set by `RLX_PROFILE`, `RLX_CONFIG` or a family list
/// (`RLX_DISABLE`/`RLX_TRACE`/`RLX_DEBUG`) rather than explicitly, with their
/// value and source. Code overrides count as explicit.
pub fn implied() -> Vec<(&'static str, String, crate::env_registry::Layer)> {
    let raw = |k: &str| match map().read().ok().and_then(|g| g.get(k).cloned()) {
        Some(v) => v,
        None => std::env::var(k).ok(),
    };
    crate::env_registry::layered_values(&raw)
}

/// On the first `RLX_*` read, name every `RLX_*` variable in the process
/// environment that the registry does not know. Nothing reads such a variable,
/// so a typo (`RLX_DISPATCH_REPOT=1`) otherwise does nothing, silently.
/// `RLX_ENV_WARN_UNKNOWN=0` turns it off; `cargo rlx doctor` lists the same.
fn warn_unknown_once() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DONE: AtomicBool = AtomicBool::new(false);
    // Set before reading the opt-out: that read comes back through here.
    if DONE.swap(true, Ordering::Relaxed) || !flag_or("RLX_ENV_WARN_UNKNOWN", true) {
        return;
    }
    // a browser has no process environment (`std::env::vars` panics there)
    if cfg!(all(target_family = "wasm", target_os = "unknown")) {
        return;
    }
    for (name, near) in crate::env_registry::unknown_vars(std::env::vars().map(|(k, _)| k)) {
        match near {
            Some(near) => {
                eprintln!("rlx: {name} is not a variable rlx reads; did you mean {near}?")
            }
            None => eprintln!(
                "rlx: {name} is not a variable rlx reads (RLX_ENV_WARN_UNKNOWN=0 silences this)"
            ),
        }
    }
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
        g.insert(key, Some(value.into()));
    }
    reload();
}

/// Make `key` read as unset — even when the process environment sets it.
/// The in-process counterpart of `std::env::remove_var`; [`unset`] instead
/// drops the code override and lets the process value show through again.
pub fn remove(key: impl AsRef<str>) {
    let key = normalize_key(key.as_ref());
    if let Ok(mut g) = map().write() {
        g.insert(key, None);
    }
    reload();
}

/// Remove a code override for `key`; subsequent reads fall back to process env.
pub fn unset(key: impl AsRef<str>) {
    let key = normalize_key(key.as_ref());
    if let Ok(mut g) = map().write() {
        g.remove(&key);
    }
    reload();
}

/// Drop every code override.
pub fn clear_overrides() {
    if let Ok(mut g) = map().write() {
        g.clear();
    }
    reload();
}

/// Read configuration: code override first, then `std::env::var`.
pub fn var(key: &str) -> Option<String> {
    with_value(key, |v| v.map(String::from))
}

/// Read ANY variable, `std::env::var`-shaped: `RLX_*` keys through [`var`]
/// (code overrides first), every other key straight from the process
/// environment. For helpers handed a key at run time, which may be `RLX_…`
/// or a foreign name such as `HF_TOKEN` — [`var`] would prefix the latter to
/// `RLX_HF_TOKEN`.
pub fn read(key: &str) -> Result<String, std::env::VarError> {
    if key.starts_with("RLX_") {
        var_result(key)
    } else {
        std::env::var(key)
    }
}

/// [`read`] returning an `OsString`, `std::env::var_os`-shaped.
pub fn read_os(key: &str) -> Option<OsString> {
    if key.starts_with("RLX_") {
        var_os(key)
    } else {
        std::env::var_os(key)
    }
}

/// [`var`] shaped like `std::env::var` — `Err(NotPresent)` when unset — so a
/// `crate::env::var_result("RLX_…")` call site can switch over without changing its
/// handling. Every `RLX_*` read goes through this module, so code overrides
/// (and wasm32, which has no process environment) reach it.
pub fn var_result(key: &str) -> Result<String, std::env::VarError> {
    var(key).ok_or(std::env::VarError::NotPresent)
}

/// Like [`var`] but returns an `OsString` (mirrors `std::env::var_os`).
///
/// **Not** a drop-in for `std::env::var_os` on non-UTF-8 values: this routes
/// through [`var`], which uses `std::env::var(..).ok()` and therefore drops a
/// value that is not valid UTF-8, where `std::env::var_os` would return it.
/// That is deliberate — an override is stored as a `String`, so there is no
/// representation for non-UTF-8 here — but it means a path-valued `RLX_*`
/// variable containing invalid UTF-8 (possible on Linux, not on macOS) is
/// treated as unset rather than used.
pub fn var_os(key: &str) -> Option<OsString> {
    var(key).map(Into::into)
}

/// True when the variable is set to a truthy value (`1`, `true`, `yes`, `on`, …).
/// False when unset or set to `0` / `false` / `off` / `no` / empty.
pub fn flag(key: &str) -> bool {
    with_value(key, |v| v.is_some_and(truthy))
}

/// Like [`flag`], but returns `default` when the key is unset. Use for
/// failsafe defaults that must stay on unless explicitly disabled (`=0`).
pub fn flag_or(key: &str, default: bool) -> bool {
    with_value(key, |v| v.map_or(default, truthy))
}

/// True when neither a code override nor process env provides the key.
pub fn is_unset(key: &str) -> bool {
    with_value(key, |v| v.is_none())
}

/// Should a device-gated test skip, given whether the backend is built and
/// whether it could be instantiated? Panics instead under `RLX_REQUIRE_DEVICE`.
///
/// A skipped test still prints `ok`, which is indistinguishable from a real
/// pass in the summary line. On a rig whose card has fallen off the PCIe bus,
/// a whole suite goes green having executed nothing — this repo has watched
/// that happen. `RLX_REQUIRE_DEVICE=1` (set by `rig.sh` and the `Justfile` GPU
/// recipes) turns the skip into a failure.
///
/// The `compiled` argument is what keeps that honest in both directions: a
/// backend the build does not contain is *absent*, not *broken*, and failing
/// for it would make every host fail for every backend it wasn't built with.
/// Only a backend that IS built and still could not be instantiated is what the
/// flag is about.
///
/// This lives in `rlx-ir` rather than in `rlx-runtime`'s test helpers because
/// the backend crates need it too and cannot depend on `rlx-runtime` — it
/// depends on them. Taking the two facts as plain booleans keeps it free of any
/// backend or `Device` type:
///
/// ```no_run
/// # fn is_available() -> bool { true }
/// if rlx_ir::env::skip_unless_device("metal", cfg!(feature = "metal"), is_available()) {
///     return;
/// }
/// ```
#[must_use = "the caller must actually return when this is true, or the test runs without a device"]
pub fn skip_unless_device(label: &str, compiled: bool, available: bool) -> bool {
    if available {
        return false;
    }
    assert!(
        !flag("RLX_REQUIRE_DEVICE") || !compiled,
        "RLX_REQUIRE_DEVICE=1, the `{label}` backend is compiled in, and it still \
         could not be instantiated — this run would have reported `ok` without \
         executing anything on it"
    );
    if compiled {
        eprintln!("{label} unavailable on this host — skipping");
    }
    true
}

/// Parse an integer/bool/string knob, falling back to `default`.
pub fn parse_or<T: FromStr>(key: &str, default: T) -> T {
    with_value(key, |v| v.and_then(|s| s.parse().ok())).unwrap_or(default)
}

fn truthy(v: &str) -> bool {
    let s = v.trim();
    if s.is_empty() {
        return false;
    }
    let is = |w: &str| s.eq_ignore_ascii_case(w);
    if is("0") || is("false") || is("off") || is("no") {
        return false;
    }
    if is("1") || is("true") || is("yes") || is("on") {
        return true;
    }
    if s.chars().all(|c| c.is_ascii_digit()) {
        return s != "0";
    }
    true // any other non-empty value counts as enabled
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
        self.pairs.push((normalize_key(key.as_ref()), value.into()));
        self
    }

    pub fn flag(mut self, key: impl AsRef<str>, on: bool) -> Self {
        self.pairs.push((
            normalize_key(key.as_ref()),
            if on { "1" } else { "0" }.into(),
        ));
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
    /// Previous override per key: `None` = there was none.
    saved: Vec<(String, Option<Option<String>>)>,
}

impl RuntimeOverrides {
    /// Install `pairs` for the lifetime of the returned guard.
    pub fn install(pairs: impl IntoIterator<Item = (impl AsRef<str>, impl Into<String>)>) -> Self {
        let mut saved = Vec::new();
        for (key, value) in pairs {
            let key = normalize_key(key.as_ref());
            let prev = map().read().ok().and_then(|g| g.get(&key).cloned());
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
                Some(v) => {
                    if let Ok(mut g) = map().write() {
                        g.insert(key, v);
                    }
                    reload();
                }
                None => unset(&key),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Env tests share the process-global override map.
    static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn with_clean_overrides(f: impl FnOnce()) {
        let _guard = ENV_TEST_LOCK.lock().expect("env test lock poisoned");
        clear_overrides();
        f();
        clear_overrides();
    }

    /// The layers reach real reads, follow code overrides, and refresh when
    /// the list or profile changes (the resolved cache is cleared by `set`).
    #[test]
    fn family_and_profile_values_reach_flag_reads() {
        with_clean_overrides(|| {
            assert!(!flag("RLX_DISABLE_CSE"));
            set("RLX_DISABLE", "cse");
            assert!(flag("RLX_DISABLE_CSE"));
            set("RLX_DISABLE", "sccp");
            assert!(!flag("RLX_DISABLE_CSE"));
            set("RLX_PROFILE", "deterministic");
            assert!(flag("RLX_DETERMINISTIC_REDUCE"));
            assert!(
                implied()
                    .iter()
                    .any(|(n, _, _)| *n == "RLX_DETERMINISTIC_REDUCE")
            );
        });
    }

    #[test]
    fn remove_hides_and_unset_restores() {
        with_clean_overrides(|| {
            const REMOVE_KEY: &str = "RLX_TEST_REMOVE_KEY"; // env-lint: ignore
            set(REMOVE_KEY, "1");
            assert_eq!(var_result(REMOVE_KEY).as_deref(), Ok("1"));
            remove(REMOVE_KEY);
            assert!(var(REMOVE_KEY).is_none());
            assert!(var_result(REMOVE_KEY).is_err());
            unset(REMOVE_KEY);
            assert!(var(REMOVE_KEY).is_none());
        });
    }

    #[test]
    fn code_override_wins_over_process_env() {
        with_clean_overrides(|| {
            let _g = RuntimeOverrides::install([("VERBOSE", "2")]);
            assert_eq!(var("RLX_VERBOSE"), Some("2".into()));
            assert!(flag("RLX_VERBOSE"));
        });
    }

    #[test]
    fn flag_parses_falsy_override() {
        with_clean_overrides(|| {
            set("RLX_DISABLE_MPSGRAPH", "0");
            assert!(!flag("RLX_DISABLE_MPSGRAPH"));
        });
    }

    #[test]
    fn scoped_is_per_thread_and_nests() {
        with_clean_overrides(|| {
            const SCOPED_KEY: &str = "RLX_TEST_SCOPED_KEY"; // env-lint: ignore
            set(SCOPED_KEY, "global");
            {
                let _a = scoped("TEST_SCOPED_KEY", "0.25");
                assert_eq!(var(SCOPED_KEY).as_deref(), Some("0.25"));
                // Another thread keeps the global value.
                let other = std::thread::spawn(|| var(SCOPED_KEY)).join().unwrap();
                assert_eq!(other.as_deref(), Some("global"));
                {
                    let _b = scoped_remove(SCOPED_KEY);
                    assert_eq!(var(SCOPED_KEY), None);
                }
                assert_eq!(var(SCOPED_KEY).as_deref(), Some("0.25"));
            }
            assert_eq!(var(SCOPED_KEY).as_deref(), Some("global"));
        });
    }

    #[test]
    fn rlx_env_bulk_apply() {
        with_clean_overrides(|| {
            RlxEnv::new()
                .set("MPSGRAPH_MIN_FLOPS", "42")
                .flag("USE_ICB", true)
                .apply();
            assert_eq!(parse_or("RLX_MPSGRAPH_MIN_FLOPS", 0u64), 42);
            assert!(flag("RLX_USE_ICB"));
        });
    }
}
