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

//! Pure-Rust Verilog / SystemVerilog writer.
//!
//! Synthesizable subset that Yosys's default frontend handles cleanly:
//! `logic`, `always_ff @(posedge clk)`, `always_comb`, `parameter`,
//! `$readmemh`, packed arrays, generate blocks. Nothing exotic — no
//! interfaces, no assertions, no `program` blocks, no `unique case`.
//!
//! The writer is deliberately small. It owns one `String` buffer and an
//! indent counter; everything else is `writeln!` into it. Higher-level
//! modules in `codegen/` build entire SV files by calling these
//! primitives.

use std::fmt::Write;

/// Verilog source buffer + indent state.
pub struct V {
    out: String,
    indent: usize,
}

impl Default for V {
    fn default() -> Self {
        Self::new()
    }
}

impl V {
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

    /// Write one line of Verilog at the current indent.
    pub fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
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

    pub fn comment(&mut self, s: &str) {
        // Allow multi-line comments by splitting on newlines.
        for ln in s.lines() {
            self.line(&format!("// {ln}"));
        }
    }

    pub fn banner(&mut self, s: &str) {
        let bar = "// ".to_string() + &"─".repeat(s.chars().count() + 2);
        self.line(&bar);
        self.line(&format!("// {s}"));
        self.line(&bar);
    }

    /// Indented block: `f` is invoked at indent+1.
    pub fn block(&mut self, f: impl FnOnce(&mut Self)) {
        self.indent += 1;
        f(self);
        self.indent -= 1;
    }

    /// `module name(...) ... endmodule`. `ports` is the full port list,
    /// one entry per line as it would appear inside the parens.
    pub fn module(
        &mut self,
        name: &str,
        params: &[String],
        ports: &[String],
        body: impl FnOnce(&mut Self),
    ) {
        if params.is_empty() {
            self.line(&format!("module {name} ("));
        } else {
            self.line(&format!("module {name} #("));
            self.block(|v| {
                let n = params.len();
                for (i, p) in params.iter().enumerate() {
                    let sep = if i + 1 == n { "" } else { "," };
                    v.line(&format!("{p}{sep}"));
                }
            });
            self.line(") (");
        }
        self.block(|v| {
            let n = ports.len();
            for (i, p) in ports.iter().enumerate() {
                let sep = if i + 1 == n { "" } else { "," };
                v.line(&format!("{p}{sep}"));
            }
        });
        self.line(");");
        self.block(body);
        self.line(&format!("endmodule  // {name}"));
        self.blank();
    }

    /// `always_ff @(posedge clk) begin ... end`.
    pub fn always_ff(&mut self, body: impl FnOnce(&mut Self)) {
        self.line("always_ff @(posedge clk) begin");
        self.block(body);
        self.line("end");
    }

    /// `always_comb begin ... end`.
    pub fn always_comb(&mut self, body: impl FnOnce(&mut Self)) {
        self.line("always_comb begin");
        self.block(body);
        self.line("end");
    }

    /// `if (cond) begin ... end [else begin ... end]`. Pass `None` for
    /// the else clause.
    pub fn if_else(
        &mut self,
        cond: &str,
        then: impl FnOnce(&mut Self),
        els: Option<&dyn Fn(&mut Self)>,
    ) {
        self.line(&format!("if ({cond}) begin"));
        self.block(then);
        match els {
            Some(e) => {
                self.line("end else begin");
                self.block(|v| e(v));
                self.line("end");
            }
            None => self.line("end"),
        }
    }

    /// Append a raw line that already has indentation handled — useful
    /// for things like signal declarations grouped in a column-aligned
    /// table. Most callers should prefer `line`.
    pub fn raw(&mut self, s: &str) {
        let _ = writeln!(self.out, "{s}");
    }
}

/// Emit a hex `.mem` file for `$readmemh`. One byte per line as a
/// 2-digit hex value (lowercase). Suitable for `bit [7:0]` ROMs; for
/// wider words call this once per byte lane or use `mem_hex_words`.
pub fn mem_hex_bytes(bytes: &[i8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 3);
    for b in bytes {
        let _ = writeln!(s, "{:02x}", *b as u8);
    }
    s
}

/// Emit a hex `.mem` file with one `bits`-wide signed word per line.
/// Words are written as zero-padded two's-complement hex.
pub fn mem_hex_words_i32(words: &[i32], bits: u32) -> String {
    assert!((1..=32).contains(&bits));
    let nibbles = bits.div_ceil(4) as usize;
    let mask: u64 = if bits == 32 {
        0xFFFF_FFFF
    } else {
        (1u64 << bits) - 1
    };
    let mut s = String::with_capacity(words.len() * (nibbles + 1));
    for w in words {
        let u = (*w as u32 as u64) & mask;
        let _ = writeln!(s, "{:0width$x}", u, width = nibbles);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_skeleton_compiles_visually() {
        let mut v = V::new();
        v.module(
            "demo",
            &["parameter int W = 8".to_string()],
            &[
                "input  logic clk".to_string(),
                "output logic [W-1:0] q".to_string(),
            ],
            |v| {
                v.line("always_ff @(posedge clk) q <= q + 1'b1;");
            },
        );
        let s = v.into_string();
        assert!(s.contains("module demo #("));
        assert!(s.contains("parameter int W = 8"));
        assert!(s.contains("input  logic clk,"));
        assert!(s.contains("output logic [W-1:0] q"));
        assert!(s.contains("endmodule  // demo"));
    }

    #[test]
    fn mem_hex_bytes_signed() {
        // -1 → 0xff, 127 → 0x7f, -128 → 0x80
        let s = mem_hex_bytes(&[-1, 127, -128, 0]);
        assert_eq!(s, "ff\n7f\n80\n00\n");
    }

    #[test]
    fn mem_hex_words_pads_to_width() {
        let s = mem_hex_words_i32(&[1, -1, 256, -256], 16);
        // 16 bits = 4 hex nibbles
        assert_eq!(s, "0001\nffff\n0100\nff00\n");
    }
}
