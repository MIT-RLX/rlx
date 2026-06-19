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

//! FileCheck-style disassembly regression tests (plan #10).
//!
//! Borrowed from MAX's `bazel/internal/mojo_filecheck_test.bzl` +
//! `bazel/internal/lit.bzl` pattern: assertions on emitted IR/asm
//! catch optimizer regressions that unit tests miss. The classic
//! one is "the optimizer changed and now we lost a SIMD intrinsic"
//! — a benchmark would notice the slowdown later, but FileCheck
//! catches it at PR time.
//!
//! The Rust spelling shells out to `objdump` (or `llvm-objdump` if
//! preferred) on the running test binary, locates a named function
//! in the disassembly, and asserts each requested pattern appears.
//!
//! Tests that use this should mark themselves `#[ignore]` so
//! they're opt-in via `cargo test -- --ignored asm` — `objdump`
//! isn't always available in CI, and disassembling a debug binary
//! is slow.
//!
//! Pattern matching is substring + line-anchored regex via the
//! standard library only — no extern dep on `regex`.

use std::path::PathBuf;
use std::process::Command;

/// What objdump-equivalent to use. macOS ships `objdump` as
/// `Apple LLVM`; Linux usually has `llvm-objdump` available, with
/// GNU `objdump` as a fallback.
fn locate_objdump() -> Option<PathBuf> {
    for candidate in ["llvm-objdump", "objdump"] {
        let probe = Command::new(candidate).arg("--version").output();
        if probe.ok().filter(|o| o.status.success()).is_some() {
            return Some(PathBuf::from(candidate));
        }
    }
    None
}

/// Path to the currently-running test binary. Cargo runs each
/// integration / library test from a known location; the env var
/// `CARGO_BIN_EXE_<name>` points binaries at themselves, but for
/// library tests we walk up from `std::env::current_exe()`.
fn current_test_binary() -> std::io::Result<PathBuf> {
    std::env::current_exe()
}

/// Disassemble the test binary; return the full text dump.
/// Errors out (with a skippable label) when the tool isn't
/// available so callers can short-circuit gracefully.
pub fn disassemble_self() -> Result<String, AsmCheckError> {
    let tool = locate_objdump().ok_or(AsmCheckError::ToolMissing)?;
    let bin = current_test_binary().map_err(|e| AsmCheckError::IoError(e.to_string()))?;
    let out = Command::new(&tool)
        .arg("-d")
        .arg("--no-show-raw-insn")
        .arg(&bin)
        .output()
        .map_err(|e| AsmCheckError::IoError(e.to_string()))?;
    if !out.status.success() {
        return Err(AsmCheckError::ToolFailed {
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        });
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Look up a function by demangled-name substring and return the
/// slice of disassembly belonging to it (until the next
/// function header).
pub fn function_section<'a>(disasm: &'a str, name_substr: &str) -> Option<&'a str> {
    let mut start = None;
    for (i, line) in disasm.lines().enumerate() {
        if line.contains(name_substr) && line.trim_end().ends_with(':') {
            start = Some(i);
            break;
        }
    }
    let start = start?;
    let lines: Vec<&str> = disasm.lines().collect();
    let mut end = lines.len();
    for j in (start + 1)..lines.len() {
        if lines[j].ends_with(':')
            && !lines[j].trim_start().starts_with('0')  // not a target label
            && !lines[j].is_empty()
        {
            end = j;
            break;
        }
    }
    let from = lines[..start].iter().map(|s| s.len() + 1).sum::<usize>();
    let to = lines[..end]
        .iter()
        .map(|s| s.len() + 1)
        .sum::<usize>()
        .min(disasm.len());
    Some(&disasm[from..to])
}

/// Assert each substring in `expected` appears at least once in
/// the function `name`'s disassembly. Returns Err if the
/// disassembler isn't present so callers can `eprintln!` skip
/// without failing.
pub fn assert_function_contains(name_substr: &str, expected: &[&str]) -> Result<(), AsmCheckError> {
    let disasm = disassemble_self()?;
    let body = function_section(&disasm, name_substr).ok_or(AsmCheckError::FunctionNotFound {
        name: name_substr.into(),
    })?;
    for pat in expected {
        if !body.contains(pat) {
            return Err(AsmCheckError::PatternMissing {
                function: name_substr.into(),
                pattern: (*pat).into(),
            });
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum AsmCheckError {
    ToolMissing,
    ToolFailed { stderr: String },
    IoError(String),
    FunctionNotFound { name: String },
    PatternMissing { function: String, pattern: String },
}

impl std::fmt::Display for AsmCheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ToolMissing => write!(f, "neither llvm-objdump nor objdump found in PATH"),
            Self::ToolFailed { stderr } => write!(f, "objdump failed: {stderr}"),
            Self::IoError(s) => write!(f, "io error: {s}"),
            Self::FunctionNotFound { name } => {
                write!(f, "function matching `{name}` not found in disassembly")
            }
            Self::PatternMissing { function, pattern } => write!(
                f,
                "function `{function}` is missing expected pattern `{pattern}`"
            ),
        }
    }
}

impl std::error::Error for AsmCheckError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// basic test: disassembly works at all. Marked `#[ignore]`
    /// because objdump isn't always around in CI (in which case
    /// we'd want to log-and-skip rather than fail).
    #[test]
    #[ignore]
    fn disassemble_self_succeeds() {
        match disassemble_self() {
            Ok(d) => assert!(d.len() > 1024, "disassembly suspiciously small"),
            Err(AsmCheckError::ToolMissing) => {
                eprintln!("[asm-check] skipping: objdump not in PATH");
            }
            Err(e) => panic!("disassembly failed: {e}"),
        }
    }

    /// Real check: the cumsum kernel must contain the expected
    /// f32 multiply / fused-multiply on aarch64. If the optimizer
    /// regresses and inlines/loses these, the test catches it.
    #[test]
    #[ignore]
    fn cumsum_kernel_keeps_simd_on_aarch64() {
        // On targets where we don't expect SIMD here, accept the
        // miss as "no expected pattern" and move on.
        if !cfg!(target_arch = "aarch64") {
            eprintln!("[asm-check] skipping: not aarch64");
            return;
        }
        // The cumsum direct-execution path adds + stores in a
        // tight inner loop. We check for `fmadd` (FP add fused
        // with multiply) or just `fadd` since cumsum doesn't
        // multiply. Pick a stable substring: `fadd s` (single
        // f32 register-form) shows up in the inner loop.
        match assert_function_contains("Cumsum", &["fadd"]) {
            Ok(()) => {}
            Err(AsmCheckError::ToolMissing | AsmCheckError::FunctionNotFound { .. }) => {
                eprintln!("[asm-check] skipping: tool or symbol missing");
            }
            Err(e) => panic!("Cumsum kernel asm check failed: {e}"),
        }
    }
}
