// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Known failures, kept as data next to the tests they belong to.
//!
//! A backend that computes one case wrong used to have three options: a skip
//! (silent — the case never runs again, and nobody notices when it is fixed),
//! a `#[ignore]` (the same, per whole test), or a red suite (which hides every
//! *new* failure behind the old one). A known-failures file is the fourth: the
//! case still runs on every pass, its failure is reported but expected, and the
//! moment it starts passing the suite fails until the entry is removed.
//!
//! One file per test binary, `tests/known_failures/<test>.txt`:
//!
//! ```text
//! # case          device[@os]   issue                    note
//! conv_reverb     wgpu@linux    owner/repo#41            PartitionedConv wrong at element 0 on Vulkan
//! c64_mul_div     wgpu@linux    #41
//! scatter_rank1   *             #57
//! ```
//!
//! `device` is the label the test passes (`{device:?}` lowercased, or a
//! harness-specific label); `*` matches any device. An `@os` suffix
//! restricts the entry to one `std::env::consts::OS` — for failures that
//! only one platform's driver stack shows. The issue column is required: an
//! expected failure with nowhere to read why it is expected is a skip with
//! extra steps. Malformed lines panic at load.
//!
//! A second directory, `RLX_KNOWN_FAILURES_DIR`, adds rig-local entries (a
//! driver version one rig has) without editing the tree.
//!
//! | env | effect |
//! |---|---|
//! | `RLX_RUN_KNOWN_FAILURES=1` | listed failures fail like any other (is it still failing for the stated reason?) |
//! | `RLX_KNOWN_FAILURES_ISSUE=#41` | run only the cases listed under that issue |
//! | `RLX_KNOWN_FAILURES_REPORT=path` | append one JSON line per verdict, for `scripts/known-failures.py update` |
//!
//! Two shapes of test use it: a harness looping over a case matrix calls
//! [`KnownFailures::record`] with each case's `Result`; a single `#[test]`
//! wraps its body in [`KnownFailures::check`].

use std::io::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

/// One line of a known-failures file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub case: String,
    /// Lowercased device label, or `*`.
    pub device: String,
    /// `std::env::consts::OS` the entry is limited to.
    pub os: Option<String>,
    pub issue: String,
    pub note: String,
    /// `path:line`, for messages.
    pub source: String,
}

impl Entry {
    fn matches(&self, case: &str, device: &str) -> bool {
        self.case == case
            && (self.device == "*" || self.device == device.to_ascii_lowercase())
            && self
                .os
                .as_deref()
                .is_none_or(|os| os == std::env::consts::OS)
    }
}

/// How one case came out against its entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    /// Failed, and listed.
    KnownFail,
    /// Passed while listed: the entry is stale.
    KnownPass,
}

impl Verdict {
    fn as_str(self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Fail => "fail",
            Verdict::KnownFail => "known-fail",
            Verdict::KnownPass => "known-pass",
        }
    }
}

/// The known failures of one test binary.
#[derive(Debug, Clone)]
pub struct KnownFailures {
    test: String,
    /// The in-tree list, written into report lines so the updater edits it.
    file: PathBuf,
    entries: Vec<Entry>,
}

impl KnownFailures {
    /// Load `<manifest_dir>/tests/known_failures/<test>.txt` plus the same file
    /// under `RLX_KNOWN_FAILURES_DIR`. Either may be absent.
    ///
    /// ```no_run
    /// let known = rlx_ir::known_failures::KnownFailures::load(
    ///     env!("CARGO_MANIFEST_DIR"),
    ///     "fd_backward_gate",
    /// );
    /// ```
    pub fn load(manifest_dir: &str, test: &str) -> Self {
        let name = format!("{test}.txt");
        let file = Path::new(manifest_dir)
            .join("tests")
            .join("known_failures")
            .join(&name);
        let mut paths = vec![file.clone()];
        if let Some(dir) = crate::env::var("RLX_KNOWN_FAILURES_DIR") {
            paths.push(PathBuf::from(dir).join(&name));
        }
        let mut entries = Vec::new();
        for path in paths {
            if let Ok(text) = std::fs::read_to_string(&path) {
                entries.extend(parse(&text, &path.display().to_string()));
            }
        }
        Self {
            test: test.to_string(),
            file,
            entries,
        }
    }

    /// From text, for tests of this module and of callers.
    pub fn from_text(test: &str, text: &str) -> Self {
        Self {
            test: test.to_string(),
            file: PathBuf::new(),
            entries: parse(text, test),
        }
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The entry covering `case` on `device` here, if any.
    pub fn lookup(&self, case: &str, device: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.matches(case, device))
    }

    /// Whether to run `case` at all. False only under
    /// `RLX_KNOWN_FAILURES_ISSUE` for cases not listed under that issue.
    pub fn selected(&self, case: &str, device: &str) -> bool {
        match crate::env::var("RLX_KNOWN_FAILURES_ISSUE") {
            Some(issue) => self
                .lookup(case, device)
                .is_some_and(|e| issue_eq(&e.issue, &issue)),
            None => true,
        }
    }

    /// Classify one outcome, print expected verdicts, append to the report,
    /// and return the message the test should fail with — `None` when the
    /// outcome is acceptable.
    pub fn record(&self, case: &str, device: &str, result: &Result<(), String>) -> Option<String> {
        let entry = self.lookup(case, device);
        let strict = crate::env::flag("RLX_RUN_KNOWN_FAILURES");
        let verdict = match (result, entry) {
            (Ok(()), None) => Verdict::Pass,
            (Ok(()), Some(_)) => Verdict::KnownPass,
            (Err(_), Some(_)) if !strict => Verdict::KnownFail,
            (Err(_), _) => Verdict::Fail,
        };
        let detail = result.as_ref().err().map(String::as_str).unwrap_or("");
        self.report(case, device, verdict, entry, detail);
        let where_ = |e: &Entry| format!("{} ({})", e.issue, e.source);
        match verdict {
            Verdict::Pass => None,
            Verdict::KnownFail => {
                let e = entry.expect("known-fail has an entry");
                eprintln!(
                    "[known failure] {}::{case} on {device}: {} — {}",
                    self.test,
                    where_(e),
                    first_line(detail)
                );
                None
            }
            Verdict::KnownPass => Some(format!(
                "{}::{case} on {device} PASSED but is listed as a known failure at {} — \
                 remove the entry",
                self.test,
                where_(entry.expect("known-pass has an entry"))
            )),
            Verdict::Fail => Some(match entry {
                Some(e) => format!(
                    "{}::{case} on {device}: {detail} (listed at {}, run with RLX_RUN_KNOWN_FAILURES)",
                    self.test,
                    where_(e)
                ),
                None => format!("{}::{case} on {device}: {detail}", self.test),
            }),
        }
    }

    /// Run `body` as `case` on `device`: a panic is the failure. Panics with
    /// the [`record`](Self::record) message when the outcome is not acceptable.
    pub fn check(&self, case: &str, device: &str, body: impl FnOnce()) {
        if !self.selected(case, device) {
            eprintln!(
                "[known failure] {}::{case} on {device}: not under RLX_KNOWN_FAILURES_ISSUE, skipped",
                self.test
            );
            return;
        }
        let listed = self.lookup(case, device).is_some();
        let result = if listed {
            // Keep the expected panic's message and backtrace out of the log.
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let r = catch_unwind(AssertUnwindSafe(body));
            std::panic::set_hook(prev);
            r
        } else {
            catch_unwind(AssertUnwindSafe(body))
        };
        let result = result.map_err(|payload| panic_message(payload.as_ref()));
        if let Some(msg) = self.record(case, device, &result) {
            panic!("{msg}");
        }
    }

    fn report(
        &self,
        case: &str,
        device: &str,
        verdict: Verdict,
        entry: Option<&Entry>,
        detail: &str,
    ) {
        let Some(path) = crate::env::var("RLX_KNOWN_FAILURES_REPORT") else {
            return;
        };
        let line = format!(
            "{{\"test\":{},\"file\":{},\"case\":{},\"device\":{},\"os\":{},\"verdict\":{},\"issue\":{},\"source\":{},\"detail\":{}}}\n",
            json_str(&self.test),
            json_str(&self.file.display().to_string()),
            json_str(case),
            json_str(&device.to_ascii_lowercase()),
            json_str(std::env::consts::OS),
            json_str(verdict.as_str()),
            entry.map_or("null".to_string(), |e| json_str(&e.issue)),
            entry.map_or("null".to_string(), |e| json_str(&e.source)),
            json_str(first_line(detail)),
        );
        // One write per line with O_APPEND: lines from parallel tests interleave
        // whole, never torn.
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = f.write_all(line.as_bytes());
        }
    }
}

fn parse(text: &str, source: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = format!("{source}:{}", i + 1);
        let mut cols = line.split_whitespace();
        let (Some(case), Some(dev), Some(issue)) = (cols.next(), cols.next(), cols.next()) else {
            panic!("{at}: expected `case device[@os] issue [note]`, got `{line}`");
        };
        assert!(
            issue.contains('#') && !issue.ends_with('#'),
            "{at}: issue `{issue}` must be `#N` or `owner/repo#N`"
        );
        let (device, os) = match dev.split_once('@') {
            Some((d, os)) => (d, Some(os.to_string())),
            None => (dev, None),
        };
        out.push(Entry {
            case: case.to_string(),
            device: device.to_ascii_lowercase(),
            os,
            issue: issue.to_string(),
            note: cols.collect::<Vec<_>>().join(" "),
            source: at,
        });
    }
    out
}

/// `#41` matches `owner/repo#41`.
fn issue_eq(listed: &str, wanted: &str) -> bool {
    listed == wanted
        || (wanted.starts_with('#') && listed.ends_with(wanted))
        || (listed.starts_with('#') && wanted.ends_with(listed))
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "<non-string panic payload>".to_string())
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = "\
# case   device      issue   note
a        cuda        #1      wrong at element 0
b        *           o/r#2
c        wgpu@linux  #3
d        wgpu@macos  #3
";

    #[test]
    fn parses_columns_and_qualifiers() {
        let k = KnownFailures::from_text("t", FILE);
        assert_eq!(k.entries().len(), 4);
        let a = &k.entries()[0];
        assert_eq!((a.case.as_str(), a.device.as_str()), ("a", "cuda"));
        assert_eq!(a.note, "wrong at element 0");
        assert_eq!(k.entries()[2].os.as_deref(), Some("linux"));
        assert!(k.entries()[0].source.ends_with(":2"));
    }

    #[test]
    fn lookup_honours_device_wildcard_and_os() {
        let k = KnownFailures::from_text("t", FILE);
        assert!(
            k.lookup("a", "Cuda").is_some(),
            "device match is case-insensitive"
        );
        assert!(k.lookup("a", "rocm").is_none());
        assert!(k.lookup("b", "anything").is_some());
        let here = std::env::consts::OS;
        assert_eq!(k.lookup("c", "wgpu").is_some(), here == "linux");
        assert_eq!(k.lookup("d", "wgpu").is_some(), here == "macos");
    }

    #[test]
    fn verdicts() {
        let k = KnownFailures::from_text("t", FILE);
        let fail = Err("boom".to_string());
        // Unlisted: pass is fine, failure fails.
        assert!(k.record("z", "cpu", &Ok(())).is_none());
        assert!(k.record("z", "cpu", &fail).unwrap().contains("boom"));
        // Listed: failure is expected, a pass is a stale entry.
        assert!(k.record("a", "cuda", &fail).is_none());
        let stale = k.record("a", "cuda", &Ok(())).unwrap();
        assert!(
            stale.contains("remove the entry") && stale.contains("#1"),
            "{stale}"
        );
    }

    #[test]
    fn check_catches_the_listed_panic_and_fails_the_unlisted_one() {
        let k = KnownFailures::from_text("t", FILE);
        k.check("a", "cuda", || panic!("expected"));
        let r = catch_unwind(AssertUnwindSafe(|| {
            k.check("z", "cuda", || panic!("new bug"))
        }));
        assert!(r.is_err());
        let r = catch_unwind(AssertUnwindSafe(|| k.check("a", "cuda", || {})));
        assert!(r.is_err(), "a listed case that passes must fail the test");
    }

    #[test]
    fn issue_matching_accepts_short_form() {
        assert!(issue_eq("o/r#41", "#41"));
        assert!(issue_eq("#41", "o/r#41"));
        assert!(!issue_eq("o/r#41", "#4"));
    }

    #[test]
    #[should_panic(expected = "must be `#N`")]
    fn issue_column_is_required() {
        KnownFailures::from_text("t", "a cuda wrong-at-zero\n");
    }

    #[test]
    fn report_lines_are_json() {
        assert_eq!(json_str("a\"b\\c\n"), "\"a\\\"b\\\\c\\n\"");
    }
}
