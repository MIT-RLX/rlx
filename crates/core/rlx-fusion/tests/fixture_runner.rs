// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! File-driven pass tests.
//!
//! Every `tests/fixtures/*.rlx` file is a pass test: input IR, the passes to
//! run, and the IR expected out. Adding a case is adding a file — no Rust, no
//! rebuild of a test harness, and the diff shows what the pass *does* rather
//! than which assertion moved.
//!
//! ```text
//! // RUN: lower_fma
//! graph @in {
//!   ...
//! }
//! // EXPECT
//! graph @out {
//!   ...
//! }
//! ```
//!
//! Directives are line comments so a fixture stays parseable as ordinary
//! textual IR. `RUN` takes a comma-separated list of [`pass_by_name`] names,
//! applied in order.
//!
//! Comparison is on IR under [`IgnoreConfig::SEMANTIC`], not on text, so
//! formatting and node naming are not under test. On mismatch both graphs are
//! printed.

use rlx_fusion::pass::{pass_by_name, run_passes};
use rlx_ir::{IgnoreConfig, text};

struct Fixture {
    passes: Vec<String>,
    input: String,
    expected: String,
}

fn parse_fixture(path: &std::path::Path, src: &str) -> Fixture {
    let mut passes = Vec::new();
    let mut input = String::new();
    let mut expected = String::new();
    let mut in_expected = false;

    for line in src.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("// RUN:") {
            passes.extend(
                rest.split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string),
            );
            continue;
        }
        if t == "// EXPECT" {
            in_expected = true;
            continue;
        }
        if in_expected {
            expected.push_str(line);
            expected.push('\n');
        } else {
            input.push_str(line);
            input.push('\n');
        }
    }

    assert!(
        !passes.is_empty(),
        "{}: no `// RUN:` directive",
        path.display()
    );
    assert!(in_expected, "{}: no `// EXPECT` section", path.display());
    Fixture {
        passes,
        input,
        expected,
    }
}

#[test]
fn fixtures_match() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "rlx"))
        .collect();
    files.sort();

    assert!(
        !files.is_empty(),
        "no fixtures found in {} — the runner would pass vacuously",
        dir.display()
    );

    for path in files {
        let src = std::fs::read_to_string(&path).unwrap();
        let fx = parse_fixture(&path, &src);
        let name = path.file_name().unwrap().to_string_lossy().to_string();

        let input =
            text::parse(&fx.input).unwrap_or_else(|e| panic!("{name}: input does not parse: {e}"));
        let want = text::parse(&fx.expected)
            .unwrap_or_else(|e| panic!("{name}: expected IR does not parse: {e}"));

        let passes: Vec<&dyn rlx_fusion::pass::Pass> = fx
            .passes
            .iter()
            .map(|p| pass_by_name(p).unwrap_or_else(|| panic!("{name}: unknown pass `{p}`")))
            .collect();

        let got = run_passes(input, &passes, false);
        assert!(
            got.structurally_eq(&want, IgnoreConfig::SEMANTIC),
            "{name}: `{}` produced unexpected IR\n\n--- got ---\n{}\n--- want ---\n{}",
            fx.passes.join(", "),
            text::print(&got),
            text::print(&want),
        );
    }
}
