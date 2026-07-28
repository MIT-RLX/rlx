// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure-Rust C/C++ source writer for the emitted `qnn_model.cpp`.
//!
//! Deliberately tiny — one `String` buffer and an indent counter, with
//! `line` / `block` / `braced` primitives that `codegen` composes into the
//! whole file. This mirrors `rlx_cerebras::csl::Csl`; the only QNN-specific
//! touch is [`Cpp::tensor_v1`], which emits the verbose `Qnn_Tensor_t`
//! designated-initializer literal that the `qnn_wrapper_api` surface (and the
//! `qnn-onnx-converter`) use for every graph tensor.

use std::fmt::Write;

/// C/C++ source buffer + indent state.
pub struct Cpp {
    out: String,
    indent: usize,
}

impl Default for Cpp {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpp {
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

    /// `header { … }` — emit `header {`, run `body` at indent + 1, then `}`.
    pub fn braced(&mut self, header: &str, body: impl FnOnce(&mut Self)) {
        self.line(&format!("{header} {{"));
        self.block(body);
        self.line("}");
    }

    /// Append a pre-formatted line verbatim (no auto-indent).
    pub fn raw(&mut self, s: &str) {
        let _ = writeln!(self.out, "{s}");
    }

    /// Emit a v1 `Qnn_Tensor_t` designated-initializer literal — an f32, rank-2,
    /// raw-buffer tensor with undefined quantization. `lead` is written before
    /// the literal (e.g. `VALIDATE(graph.addTensor("in0", `) and `trail` after
    /// it (e.g. `), err);`); pass empty strings for a bare array element.
    /// `dims_var` names the `uint32_t[]` of dimensions declared just above.
    ///
    /// This is the one QNN-aware helper — the bulk of any `qnn_wrapper_api`
    /// model file is these literals, so it earns its place the way
    /// `Csl::func` / `Csl::comptime` do for CSL.
    pub fn tensor_v1(&mut self, lead: &str, name: &str, ttype: &str, dims_var: &str, trail: &str) {
        self.line(&format!("{lead}(Qnn_Tensor_t){{"));
        self.block(|c| {
            c.line(".version = QNN_TENSOR_VERSION_1,");
            c.line(".v1 = {");
            c.block(|c| {
                c.line(".id             = 0,");
                c.line(&format!(".name           = \"{name}\","));
                c.line(&format!(".type           = {ttype},"));
                c.line(".dataFormat     = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,");
                c.line(".dataType       = QNN_DATATYPE_FLOAT_32,");
                c.line(".quantizeParams = {QNN_DEFINITION_UNDEFINED,");
                c.line("                   QNN_QUANTIZATION_ENCODING_UNDEFINED,");
                c.line(
                    "                   {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},",
                );
                c.line(".rank           = 2,");
                c.line(&format!(".dimensions     = {dims_var},"));
                c.line(".memType        = QNN_TENSORMEMTYPE_RAW,");
                // Fully-designated, matching the qnn-converter idiom: `.v1` and
                // `.clientBuf` are direct designators with no extra brace
                // nesting. Three `}` close clientBuf, `.v1`, and the
                // Qnn_Tensor_t, then the caller's `trail`.
                c.line(&format!(
                    ".clientBuf      = {{.data = nullptr, .dataSize = 0}}}}}}{trail}"
                ));
            });
        });
    }

    /// Emit a v1 `Qnn_Tensor_t` with an explicit `clientBuf` (STATIC weights).
    /// `data_expr` is a C expression for the buffer pointer (e.g. `w_data`);
    /// `size_expr` is the byte size (e.g. `sizeof(w_data)`).
    pub fn tensor_v1_buf(
        &mut self,
        lead: &str,
        name: &str,
        ttype: &str,
        dims_var: &str,
        data_expr: &str,
        size_expr: &str,
        trail: &str,
    ) {
        self.line(&format!("{lead}(Qnn_Tensor_t){{"));
        self.block(|c| {
            c.line(".version = QNN_TENSOR_VERSION_1,");
            c.line(".v1 = {");
            c.block(|c| {
                c.line(".id             = 0,");
                c.line(&format!(".name           = \"{name}\","));
                c.line(&format!(".type           = {ttype},"));
                c.line(".dataFormat     = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,");
                c.line(".dataType       = QNN_DATATYPE_FLOAT_32,");
                c.line(".quantizeParams = {QNN_DEFINITION_UNDEFINED,");
                c.line("                   QNN_QUANTIZATION_ENCODING_UNDEFINED,");
                c.line(
                    "                   {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}},",
                );
                c.line(".rank           = 2,");
                c.line(&format!(".dimensions     = {dims_var},"));
                c.line(".memType        = QNN_TENSORMEMTYPE_RAW,");
                // Build the line without nesting `}` inside a format string.
                let mut line = format!(
                    ".clientBuf      = {{.data = (void*){data_expr}, .dataSize = {size_expr}}}"
                );
                line.push('}'); // close .v1
                line.push('}'); // close Qnn_Tensor_t
                line.push_str(trail);
                c.line(&line);
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn braced_block_indents() {
        let mut c = Cpp::new();
        c.braced("extern \"C\"", |c| {
            c.line("int x = 0;");
        });
        let s = c.into_string();
        assert!(s.contains("extern \"C\" {"));
        assert!(s.contains("  int x = 0;"));
        assert!(s.contains("}"));
    }

    #[test]
    fn tensor_literal_carries_name_type_and_dims() {
        let mut c = Cpp::new();
        c.tensor_v1(
            "VALIDATE(g.addTensor(\"in0\", ",
            "in0",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in0",
            "), err);",
        );
        let s = c.into_string();
        assert!(s.contains("(Qnn_Tensor_t){"));
        assert!(s.contains(".v1 = {"));
        assert!(s.contains(".name           = \"in0\","));
        assert!(s.contains(".type           = QNN_TENSOR_TYPE_APP_WRITE,"));
        assert!(s.contains(".dimensions     = dimensions_in0,"));
        assert!(s.contains("QNN_DATATYPE_FLOAT_32"));
        assert!(s.trim_end().ends_with("), err);"));
    }
}
