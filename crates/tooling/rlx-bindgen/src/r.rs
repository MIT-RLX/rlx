// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The R binding (package `rlx`), from the lowered [`Ir`]:
//!
//! * `rlx-r/src/rlx_r.c` — a `.Call` entry per rlx-capi function, driven by
//!   C types alone: handles are external pointers (with a C finalizer on
//!   `rlx_*_free`), arrays R vectors (temporaries `R_alloc`'d, so R reclaims
//!   them even when an error unwinds), string lists character vectors. Each
//!   returns `list(status, results…)`.
//! * `rlx-r/R/generated.R` — the API: a method table per handle class,
//!   reached with `$` (`g$matmul(x, w)`); constructors and statics on a
//!   generator object (`Graph$new(name = "g")`, `Gguf$open(path)`); free
//!   functions prefixed `rlx_` (so `rlx_version` does not mask base R's
//!   `version`). Optional parameters are R defaults.
//! * `rlx-r/NAMESPACE`, and `rlx-r/src/rlx.h` (rlx-capi's header, so the
//!   package installs on its own).

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, doc_lines};
use crate::jni::c_params;

// ── the C glue ──────────────────────────────────────────────

fn class_of_out(ir: &Ir, c: &str) -> Option<String> {
    let name = c.strip_prefix("Rlx")?.trim_end_matches('*');
    ir.classes
        .iter()
        .find(|k| k.name == name)
        .map(|k| k.c.clone())
}

fn c_function(s: &mut String, ir: &Ir, f: &Func) {
    let params = c_params(ir, f);
    let ins = params.iter().filter(|(_, o)| !*o).count();
    let args_decl: Vec<String> = (0..ins).map(|i| format!("SEXP a{i}")).collect();
    let _ = writeln!(
        s,
        "SEXP R_{}({}) {{",
        f.c,
        if args_decl.is_empty() {
            "void".to_string()
        } else {
            args_decl.join(", ")
        }
    );
    let mut pre = Vec::new();
    let mut args = Vec::new();
    let mut rets = Vec::new();
    let mut arg = 0usize;
    for (i, (t, out)) in params.iter().enumerate() {
        if *out {
            match t.as_str() {
                "RlxF32s*" | "RlxBytes*" | "RlxU32s*" => {
                    let st = t.trim_end_matches('*');
                    pre.push(format!("{st} o{i} = {{0}};"));
                    args.push(format!("&o{i}"));
                    let take = match st {
                        "RlxF32s" => "rlx_r_from_f32s",
                        "RlxBytes" => "rlx_r_from_bytes",
                        _ => "rlx_r_from_u32s",
                    };
                    rets.push(format!("{take}(o{i})"));
                }
                "char**" => {
                    pre.push(format!("char* o{i} = NULL;"));
                    args.push(format!("&o{i}"));
                    rets.push(format!("rlx_r_from_string(o{i})"));
                }
                "bool*" => {
                    pre.push(format!("bool o{i} = false;"));
                    args.push(format!("&o{i}"));
                    rets.push(format!("ScalarLogical(o{i})"));
                }
                "float*" | "uint32_t*" | "size_t*" | "uint64_t*" => {
                    let ct = t.trim_end_matches('*');
                    pre.push(format!("{ct} o{i} = 0;"));
                    args.push(format!("&o{i}"));
                    rets.push(format!("ScalarReal((double)o{i})"));
                }
                ct => {
                    let base = ct.trim_end_matches('*');
                    pre.push(format!("{base}* o{i} = NULL;"));
                    args.push(format!("&o{i}"));
                    match class_of_out(ir, base) {
                        Some(c) => rets.push(format!("rlx_r_handle(o{i}, fin_{c})")),
                        None => rets.push(format!("rlx_r_handle(o{i}, NULL)")),
                    }
                }
            }
            continue;
        }
        let a = format!("a{arg}");
        arg += 1;
        match t.as_str() {
            "const char*" => args.push(format!("rlx_r_string({a})")),
            "bool" => args.push(format!("asLogical({a}) == TRUE")),
            "float" => args.push(format!("(float)asReal({a})")),
            "double" => args.push(format!("asReal({a})")),
            "int32_t" | "int64_t" | "uint32_t" | "uint64_t" | "size_t" => {
                args.push(format!("({t})rlx_r_number({a})"))
            }
            "const float*" => args.push(format!("rlx_r_floats({a})")),
            "const uint8_t*" => args.push(format!("rlx_r_bytes({a})")),
            "const uint32_t*" => args.push(format!("rlx_r_u32s({a})")),
            "const int64_t*" => args.push(format!("rlx_r_i64s({a})")),
            "const size_t*" | "const uint64_t*" => {
                let ct = t.trim_start_matches("const ").trim_end_matches('*');
                args.push(format!("({ct}*)rlx_r_u64s({a})"))
            }
            "const char* const*" => args.push(format!("rlx_r_strings({a})")),
            "const float* const*" => args.push(format!("rlx_r_float_arrays({a})")),
            "const uint8_t* const*" => args.push(format!("rlx_r_byte_arrays({a})")),
            handle => args.push(format!("({handle})rlx_r_ptr({a})")),
        }
    }
    for line in &pre {
        let _ = writeln!(s, "    {line}");
    }
    let call = format!("{}({})", f.c, args.join(", "));
    if f.direct {
        let wrap = match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => "mkString",
            _ => "ScalarLogical",
        };
        let _ = writeln!(s, "    return {wrap}({call});\n}}\n");
        return;
    }
    let _ = writeln!(s, "    int32_t rc = {call};");
    let _ = writeln!(
        s,
        "    SEXP out = PROTECT(allocVector(VECSXP, {}));",
        rets.len() + 1
    );
    let _ = writeln!(s, "    SET_VECTOR_ELT(out, 0, ScalarInteger(rc));");
    if !rets.is_empty() {
        let _ = writeln!(s, "    if (rc == RLX_OK) {{");
        for (k, r) in rets.iter().enumerate() {
            let _ = writeln!(s, "        SET_VECTOR_ELT(out, {}, {r});", k + 1);
        }
        let _ = writeln!(s, "    }}");
    }
    let _ = writeln!(s, "    UNPROTECT(1);\n    return out;\n}}\n");
}

/// `src/rlx_r.c`.
pub fn c_glue(ir: &Ir, defaults: &[String]) -> String {
    let mut s = String::from(
        "/* RLX — versatile ML compiler + runtime.\n\
         \x20* Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         \x20* SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \x20*\n\
         \x20* @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         \x20* do not edit; regenerate with `just gen-bindings`.\n\
         \x20*\n\
         \x20* The .Call entries of the rlx R package: one per rlx-capi function, each\n\
         \x20* returning list(status, results…). Conversions are in rlx_r_support.c.\n\
         \x20*/\n\n\
         #include \"rlx_r.h\"\n\n",
    );
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "static void fin_{c}(SEXP p) {{\n    void* h = R_ExternalPtrAddr(p);\n    if (h) {{ {free}((Rlx{n}*)h); R_ClearExternalPtr(p); }}\n}}\n",
            c = c.c,
            free = c.c_free(),
            n = c.name
        );
    }
    let mut entries = Vec::new();
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        let gate = f
            .feature
            .as_ref()
            .filter(|x| !defaults.contains(x))
            .map(|x| format!("RLX_FEATURE_{}", x.to_ascii_uppercase().replace('-', "_")));
        if let Some(g) = &gate {
            let _ = writeln!(s, "#ifdef {g}");
        }
        c_function(&mut s, ir, f);
        if gate.is_some() {
            s.push_str("#endif\n\n");
        }
        let n = c_params(ir, f).iter().filter(|(_, o)| !*o).count();
        entries.push((format!("R_{}", f.c), n, gate));
    }
    // Free now (`obj$close()`): the finalizer, run early.
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "SEXP R_{free}(SEXP p) {{\n    fin_{c}(p);\n    return R_NilValue;\n}}\n",
            free = c.c_free(),
            c = c.c
        );
        entries.push((format!("R_{}", c.c_free()), 1, None));
    }
    s.push_str("static const R_CallMethodDef CALLS[] = {\n");
    for (n, k, gate) in &entries {
        if let Some(g) = gate {
            let _ = writeln!(s, "#ifdef {g}");
        }
        let _ = writeln!(s, "    {{\"{n}\", (DL_FUNC)&{n}, {k}}},");
        if gate.is_some() {
            s.push_str("#endif\n");
        }
    }
    s.push_str(
        "    RLX_R_SUPPORT_CALLS\n    {NULL, NULL, 0},\n};\n\n\
         void R_init_rlx(DllInfo* dll) {\n\
         \x20   R_registerRoutines(dll, NULL, CALLS, NULL, NULL);\n\
         \x20   R_useDynamicSymbols(dll, FALSE);\n\
         }\n",
    );
    s
}

// ── the R API ───────────────────────────────────────────────

const RESERVED: &[&str] = &[
    "if", "else", "repeat", "while", "function", "for", "next", "break", "TRUE", "FALSE", "NULL",
    "Inf", "NaN", "NA", "in", "self",
];

fn local(name: &str) -> String {
    if RESERVED.contains(&name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}

/// The R name of a free function.
fn free_name(f: &Func) -> String {
    format!("rlx_{}", f.name)
}

fn literal(l: &Lit) -> String {
    match l {
        Lit::None => "NULL".into(),
        Lit::Bool(true) => "TRUE".into(),
        Lit::Bool(false) => "FALSE".into(),
        Lit::Num(n) => n.replace("inf", "Inf"),
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(literal).collect();
            format!("c({})", inner.join(", "))
        }
    }
}

/// Statements first, then the `.Call` arguments, for one parameter.
fn lower(p: &Param, v: &str) -> (Vec<String>, Vec<String>) {
    let t = format!("a_{}", p.name);
    match p.kind.as_str() {
        "json" | "value" => (vec![], vec![format!(".json({v})")]),
        "json?" | "value?" => (
            vec![],
            vec![format!("if (is.null({v})) NULL else .json({v})")],
        ),
        "node?" => (vec![], vec![format!("if (is.null({v})) NONE else {v}")]),
        "u64?" | "usize?" | "f32?" | "str?" => (vec![], vec![v.into()]),
        "f32s" | "u32s" | "nodes" | "usizes" | "usizes?" | "i64s" | "u8s" => {
            (vec![], vec![v.into(), format!("length({v})")])
        }
        "rows2" | "frows2" | "frows6" => {
            let (w, splat) = match p.kind.as_str() {
                "rows2" => (2, "TRUE"),
                "frows2" => (2, "FALSE"),
                _ => (6, "FALSE"),
            };
            (
                vec![format!("{t} <- .rows({v}, {w}, {splat})")],
                vec![t.clone(), format!("length({t}) / {w}")],
            )
        }
        "strs" | "strs?" => (
            vec![format!("{t} <- as.character({v})")],
            vec![t.clone(), format!("length({t})")],
        ),
        "inputs" => (
            vec![format!("{t} <- .inputs({v})")],
            vec![
                format!("{t}$names"),
                format!("{t}$data"),
                format!("{t}$lens"),
                format!("length({t}$names)"),
            ],
        ),
        "typed_inputs" => (
            vec![format!("{t} <- .typed_inputs({v})")],
            vec![
                format!("{t}$names"),
                format!("{t}$data"),
                format!("{t}$lens"),
                format!("{t}$dtypes"),
                format!("length({t}$names)"),
            ],
        ),
        _ => (vec![], vec![v.into()]),
    }
}

fn convert(kind: &str, v: &[String]) -> String {
    let x = &v[0];
    match kind {
        "json" | "object" | "rows" | "shapes" | "shape" | "strings" => format!(".decode({x})"),
        "outputs" => format!(".Call(R_support_take_outputs, {x})"),
        "typed_outputs" => format!(".Call(R_support_take_typed_outputs, {x})"),
        "tensors" => format!(".Call(R_support_take_tensors, {x})"),
        "f32?" => format!("if (isTRUE({})) {x} else NULL", v[1]),
        "graph" => format!(".wrap({x}, \"Graph\")"),
        k if k.starts_with(char::is_uppercase) => format!(".wrap({x}, \"{k}\")"),
        _ => x.clone(),
    }
}

fn width(kind: &str) -> usize {
    if kind == "f32?" { 2 } else { 1 }
}

/// `function(…) { … }` for one function (`self` first for a method).
fn function(ir: &Ir, f: &Func, ind: &str) -> String {
    let mut s = String::new();
    let mut params: Vec<String> = Vec::new();
    if f.role == Role::Method {
        params.push("self".into());
    }
    for p in &f.params {
        if p.optional() {
            let d = p.default.as_ref().map_or("NULL".into(), literal);
            params.push(format!("{} = {d}", local(&p.name)));
        } else {
            params.push(local(&p.name));
        }
    }
    let _ = writeln!(s, "function({}) {{", params.join(", "));
    let b = format!("{ind}  ");
    let mut args = vec![format!("R_{}", f.c)];
    if f.role == Role::Method {
        args.push("self".into());
    }
    for p in &f.params {
        let (stmts, a) = lower(p, &local(&p.name));
        for st in stmts {
            let _ = writeln!(s, "{b}{st}");
        }
        args.extend(a);
    }
    let call = format!(".Call({})", args.join(", "));
    if f.direct {
        let _ = write!(s, "{b}{call}\n{ind}}}");
        return s;
    }
    let _ = writeln!(s, "{b}r <- {call}");
    let _ = writeln!(s, "{b}.check(r[[1]])");
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), ir.class_of(f).unwrap().name.clone())],
        _ => f.rets.clone(),
    };
    let mut at = 2;
    let mut parts = Vec::new();
    for (n, k) in &rets {
        let vals: Vec<String> = (0..width(k)).map(|j| format!("r[[{}]]", at + j)).collect();
        at += width(k);
        parts.push((n.clone(), convert(k, &vals)));
    }
    match parts.len() {
        0 => {
            let _ = writeln!(s, "{b}invisible(NULL)");
        }
        1 => {
            let _ = writeln!(s, "{b}{}", parts[0].1);
        }
        _ => {
            let fields: Vec<String> = parts.iter().map(|(n, e)| format!("{n} = {e}")).collect();
            let _ = writeln!(s, "{b}list({})", fields.join(", "));
        }
    }
    let _ = write!(s, "{ind}}}");
    s
}

fn roxygen(doc: &str, ind: &str) -> String {
    doc_lines(doc, &format!("{ind}# "))
}

/// `R/generated.R`.
pub fn r_api(ir: &Ir) -> String {
    let mut s = String::from(
        "# RLX — versatile ML compiler + runtime.\n\
         # Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         # SPDX-License-Identifier: MIT OR Apache-2.0\n\
         #\n\
         # @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         # do not edit; regenerate with `just gen-bindings`.\n\n",
    );
    for f in ir.free_funcs().filter(|f| !f.manual) {
        s.push_str(&roxygen(&f.doc, ""));
        let _ = writeln!(s, "{} <- {}\n", free_name(f), function(ir, f, ""));
    }
    for c in &ir.classes {
        s.push_str(&roxygen(&c.doc, ""));
        let _ = writeln!(s, ".methods$rlx_{} <- list(", c.name);
        let methods: Vec<&Func> = ir.members(c).filter(|f| f.role == Role::Method).collect();
        let mut items = Vec::new();
        for f in &methods {
            let body = if f.manual {
                format!(
                    "function(self, ...) .{}_{}(self, ...)",
                    c.name.to_ascii_lowercase(),
                    f.name
                )
            } else {
                function(ir, f, "  ")
            };
            items.push(format!("{}  `{}` = {body}", roxygen(&f.doc, "  "), f.name));
        }
        s.push_str(&items.join(",\n"));
        s.push_str("\n)\n\n");
        let _ = writeln!(s, "{} <- .generator(\"{}\", list(", c.name, c.name);
        let mut statics = Vec::new();
        for f in ir
            .members(c)
            .filter(|f| matches!(f.role, Role::Constructor | Role::Static))
        {
            let name = if f.role == Role::Constructor {
                "new".to_string()
            } else {
                f.name.clone()
            };
            statics.push(format!(
                "{}  `{name}` = {}",
                roxygen(&f.doc, "  "),
                function(ir, f, "  ")
            ));
        }
        s.push_str(&statics.join(",\n"));
        s.push_str("\n))\n\n");
    }
    s.truncate(s.trim_end().len());
    s.push('\n');
    s
}

/// `NAMESPACE`.
pub fn namespace(ir: &Ir) -> String {
    let mut s = String::from(
        "# @generated by rlx-bindgen — do not edit; regenerate with `just gen-bindings`.\n\
         useDynLib(rlx, .registration = TRUE)\n\
         S3method(\"$\", rlx_handle)\n\
         S3method(print, rlx_handle)\n\
         S3method(\"$\", rlx_generator)\n\
         S3method(print, rlx_generator)\n\
         export(NONE)\n\
         export(rlx_flat)\n\
         export(rlx_unflat)\n",
    );
    for c in &ir.classes {
        let _ = writeln!(s, "export({})", c.name);
    }
    for f in ir.free_funcs().filter(|f| !f.manual) {
        let _ = writeln!(s, "export({})", free_name(f));
    }
    s
}
