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

//! Pure-Rust CSL (Cerebras Software Language) source writer.
//!
//! Deliberately tiny — one `String` buffer and an indent counter, with
//! `line` / `block` primitives that higher-level emitters in `codegen`
//! compose into whole `.csl` files. This mirrors `rlx_fpga::verilog::V`;
//! the only CSL-specific touch is `//`-style comments and a `func` helper
//! for `fn name() void { … }` blocks, which are the bulk of a PE program.

use std::fmt::Write;

/// CSL source buffer + indent state.
pub struct Csl {
    out: String,
    indent: usize,
}

impl Default for Csl {
    fn default() -> Self {
        Self::new()
    }
}

impl Csl {
    pub fn new() -> Self {
        Self {
            out: String::new(),
            indent: 0,
        }
    }

    pub fn into_string(self) -> String {
        self.out
    }

    pub fn as_str(&self) -> &str {
        &self.out
    }

    /// Write one line at the current indent.
    pub fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.out.push_str("  ");
        }
        self.out.push_str(s);
        self.out.push('\n');
    }

    pub fn lines(&mut self, ls: &[&str]) {
        for l in ls {
            self.line(l);
        }
    }

    pub fn blank(&mut self) {
        self.out.push('\n');
    }

    /// `// …` comment. Multi-line input is split per line.
    pub fn comment(&mut self, s: &str) {
        for ln in s.lines() {
            self.line(&format!("// {ln}"));
        }
    }

    /// Boxed comment banner.
    pub fn banner(&mut self, s: &str) {
        let bar = "// ".to_string() + &"─".repeat(s.chars().count() + 2);
        self.line(&bar);
        self.line(&format!("// {s}"));
        self.line(&bar);
    }

    /// Indented block: `f` runs at indent + 1.
    pub fn block(&mut self, f: impl FnOnce(&mut Self)) {
        self.indent += 1;
        f(self);
        self.indent -= 1;
    }

    /// `fn name() void { … }` — the common shape for a PE entry / helper.
    pub fn func(&mut self, name: &str, body: impl FnOnce(&mut Self)) {
        self.line(&format!("fn {name}() void {{"));
        self.block(body);
        self.line("}");
        self.blank();
    }

    /// `comptime { … }` block — symbol exports, fabric setup.
    pub fn comptime(&mut self, body: impl FnOnce(&mut Self)) {
        self.line("comptime {");
        self.block(body);
        self.line("}");
        self.blank();
    }

    /// Append a pre-formatted line verbatim (no auto-indent).
    pub fn raw(&mut self, s: &str) {
        let _ = writeln!(self.out, "{s}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn func_block_indents() {
        let mut c = Csl::new();
        c.func("gemv", |c| {
            c.line("var acc: f32 = 0.0;");
        });
        let s = c.into_string();
        assert!(s.contains("fn gemv() void {"));
        assert!(s.contains("  var acc: f32 = 0.0;"));
        assert!(s.contains("}"));
    }

    #[test]
    fn comptime_export_block() {
        let mut c = Csl::new();
        c.comptime(|c| {
            c.line("@export_symbol(Y_ptr, \"Y\");");
        });
        let s = c.into_string();
        assert!(s.contains("comptime {"));
        assert!(s.contains("  @export_symbol(Y_ptr, \"Y\");"));
    }
}
