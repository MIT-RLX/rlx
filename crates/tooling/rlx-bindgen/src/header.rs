// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-capi/include/rlx.h`, by parsing rlx-capi's own sources.
//!
//! Every `#[unsafe(no_mangle)] pub … extern "C" fn`, `pub const RLX_*`,
//! `#[repr(C)] pub struct`, opaque `pub struct Rlx*` and callback
//! `pub type Rlx*Fn = Option<unsafe extern "C" fn(…)>` becomes its C
//! declaration, with the Rust doc comment as the C comment. The header is
//! therefore whatever the library exports — it cannot list a function the
//! library lacks, or miss one it has.

use std::fmt::Write;

use quote::ToTokens;
use syn::{Attribute, FnArg, Item, ReturnType, Type};

/// One source file: its module doc title and its items.
pub struct Source {
    pub file: String,
    pub text: String,
}

enum Decl {
    Opaque(String, Vec<String>),
    Struct(String, Vec<(String, String)>, Vec<String>),
    Callback(String, String, Vec<String>, Vec<String>),
    Const(String, String, Vec<String>),
    Fn {
        name: String,
        ret: String,
        params: Vec<String>,
        doc: Vec<String>,
        feature: Option<String>,
    },
}

fn docs(attrs: &[Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            syn::Meta::NameValue(nv) => match &nv.value {
                syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) => Some(s.value()),
                _ => None,
            },
            _ => None,
        })
        .map(|line| line.strip_prefix(' ').unwrap_or(&line).to_string())
        .collect()
}

fn feature(attrs: &[Attribute]) -> Option<String> {
    attrs
        .iter()
        .find(|a| a.path().is_ident("cfg"))
        .and_then(|a| {
            let text = a.meta.to_token_stream().to_string();
            let start = text.find("feature = \"")? + "feature = \"".len();
            let end = text[start..].find('"')?;
            Some(text[start..start + end].to_string())
        })
}

fn is_no_mangle(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.meta.to_token_stream().to_string().contains("no_mangle"))
}

fn is_repr_c(attrs: &[Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path().is_ident("repr") && a.meta.to_token_stream().to_string().contains('C'))
}

/// A Rust FFI type as a C type.
fn c_type(t: &Type) -> String {
    match t {
        Type::Ptr(p) => {
            let inner = c_type(&p.elem);
            let inner_is_ptr = matches!(*p.elem, Type::Ptr(_));
            match (p.mutability.is_some(), inner_is_ptr) {
                (true, _) => format!("{inner}*"),
                (false, true) => format!("{inner} const*"),
                (false, false) => format!("const {inner}*"),
            }
        }
        Type::Path(tp) => {
            let last = tp
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            match last.as_str() {
                "u8" => "uint8_t",
                "i8" => "int8_t",
                "u16" => "uint16_t",
                "i16" => "int16_t",
                "u32" => "uint32_t",
                "i32" => "int32_t",
                "u64" => "uint64_t",
                "i64" => "int64_t",
                "usize" => "size_t",
                "isize" => "ptrdiff_t",
                "f32" => "float",
                "f64" => "double",
                "bool" => "bool",
                "c_char" => "char",
                "c_void" => "void",
                other => return other.to_string(),
            }
            .to_string()
        }
        Type::Tuple(t) if t.elems.is_empty() => "void".to_string(),
        other => panic!(
            "rlx-bindgen: no C spelling for `{}`",
            other.to_token_stream()
        ),
    }
}

fn c_param(arg: &FnArg) -> String {
    let FnArg::Typed(pt) = arg else {
        panic!("rlx-bindgen: methods are not part of a C ABI");
    };
    let name = pt.pat.to_token_stream().to_string();
    format!("{} {name}", c_type(&pt.ty))
}

fn c_ret(r: &ReturnType) -> String {
    match r {
        ReturnType::Default => "void".to_string(),
        ReturnType::Type(_, t) => c_type(t),
    }
}

fn walk(items: &[Item], inherited: Option<String>, out: &mut Vec<Decl>) {
    for item in items {
        match item {
            Item::Const(c) if matches!(c.vis, syn::Visibility::Public(_)) => {
                let name = c.ident.to_string();
                if !name.starts_with("RLX_") {
                    continue;
                }
                let expr = c.expr.to_token_stream().to_string().replace(' ', "");
                let value = match expr.as_str() {
                    "u32::MAX" => "0xFFFFFFFFu".to_string(),
                    other => other.to_string(),
                };
                out.push(Decl::Const(name, value, docs(&c.attrs)));
            }
            Item::Struct(s) if matches!(s.vis, syn::Visibility::Public(_)) => {
                let name = s.ident.to_string();
                if !name.starts_with("Rlx") {
                    continue;
                }
                if is_repr_c(&s.attrs) {
                    let fields = s
                        .fields
                        .iter()
                        .map(|f| (c_type(&f.ty), f.ident.as_ref().unwrap().to_string()))
                        .collect();
                    out.push(Decl::Struct(name, fields, docs(&s.attrs)));
                } else {
                    out.push(Decl::Opaque(name, docs(&s.attrs)));
                }
            }
            Item::Type(t) if matches!(t.vis, syn::Visibility::Public(_)) => {
                // `Option<unsafe extern "C" fn(…) -> R>`.
                let Type::Path(tp) = &*t.ty else { continue };
                let Some(seg) = tp.path.segments.last() else {
                    continue;
                };
                if seg.ident != "Option" {
                    continue;
                }
                let syn::PathArguments::AngleBracketed(ab) = &seg.arguments else {
                    continue;
                };
                let Some(syn::GenericArgument::Type(Type::BareFn(f))) = ab.args.first() else {
                    continue;
                };
                let params = f
                    .inputs
                    .iter()
                    .map(|a| {
                        let name = a
                            .name
                            .as_ref()
                            .map(|(n, _)| n.to_string())
                            .unwrap_or_default();
                        format!("{} {name}", c_type(&a.ty))
                    })
                    .collect();
                out.push(Decl::Callback(
                    t.ident.to_string(),
                    c_ret(&f.output),
                    params,
                    docs(&t.attrs),
                ));
            }
            Item::Fn(f) => {
                let is_c = f
                    .sig
                    .abi
                    .as_ref()
                    .is_some_and(|abi| abi.name.as_ref().is_some_and(|n| n.value() == "C"));
                if !is_c || !is_no_mangle(&f.attrs) || !matches!(f.vis, syn::Visibility::Public(_))
                {
                    continue;
                }
                out.push(Decl::Fn {
                    name: f.sig.ident.to_string(),
                    ret: c_ret(&f.sig.output),
                    params: f.sig.inputs.iter().map(c_param).collect(),
                    doc: docs(&f.attrs),
                    feature: feature(&f.attrs).or_else(|| inherited.clone()),
                });
            }
            Item::Mod(m) => {
                if let Some((_, inner)) = &m.content {
                    let f = feature(&m.attrs).or_else(|| inherited.clone());
                    walk(inner, f, out);
                }
            }
            _ => {}
        }
    }
}

fn comment(s: &mut String, lines: &[String], indent: &str) {
    if lines.is_empty() {
        return;
    }
    let _ = writeln!(s, "{indent}/*");
    for l in lines {
        let l = l.replace("*/", "* /");
        if l.is_empty() {
            let _ = writeln!(s, "{indent} *");
        } else {
            let _ = writeln!(s, "{indent} * {l}");
        }
    }
    let _ = writeln!(s, "{indent} */");
}

/// Module title: the first `//!` line of a source file.
fn title(text: &str) -> Option<String> {
    text.lines()
        .find_map(|l| l.strip_prefix("//!").map(str::trim))
        .filter(|l| !l.is_empty())
        .map(str::to_string)
}

/// One exported C function: name, C return type, `(C type, name)` per
/// parameter, and the feature it needs.
#[derive(Clone, Debug)]
pub struct CFn {
    pub name: String,
    pub ret: String,
    pub params: Vec<(String, String)>,
    pub feature: Option<String>,
}

/// Every `extern "C"` function the sources export.
pub fn functions(sources: &[Source]) -> Vec<CFn> {
    let mut out = Vec::new();
    for src in sources {
        let file = syn::parse_file(&src.text)
            .unwrap_or_else(|e| panic!("rlx-bindgen: cannot parse {}: {e}", src.file));
        let mut decls = Vec::new();
        walk(&file.items, None, &mut decls);
        for d in decls {
            if let Decl::Fn {
                name,
                ret,
                params,
                feature,
                ..
            } = d
            {
                let params = params
                    .iter()
                    .map(|p| {
                        let (ty, name) = p.rsplit_once(' ').expect("`type name`");
                        (ty.to_string(), name.to_string())
                    })
                    .collect();
                out.push(CFn {
                    name,
                    ret,
                    params,
                    feature,
                });
            }
        }
    }
    out
}

pub fn generate(sources: &[Source]) -> String {
    let mut sections: Vec<(String, Vec<Decl>)> = Vec::new();
    for src in sources {
        let file = syn::parse_file(&src.text)
            .unwrap_or_else(|e| panic!("rlx-bindgen: cannot parse {}: {e}", src.file));
        let mut decls = Vec::new();
        walk(&file.items, None, &mut decls);
        let t = title(&src.text).unwrap_or_else(|| src.file.clone());
        sections.push((t, decls));
    }

    let mut s = String::new();
    s.push_str(
        "/* RLX — versatile ML compiler + runtime.\n\
         * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         * SPDX-License-Identifier: MIT OR Apache-2.0\n\
         *\n\
         * The C ABI for rlx. @generated by rlx-bindgen from crates/bindings/rlx-capi —\n\
         * do not edit; regenerate with `just gen-bindings`.\n\
         *\n\
         * Every function returns RLX_OK (0) or a negative RLX_ERR_*; on failure\n\
         * rlx_last_error() holds the message. Handles are freed with their\n\
         * rlx_*_free; buffers and strings rlx returns with rlx_u32s_free /\n\
         * rlx_f32s_free / rlx_bytes_free / rlx_string_free. An optional node is\n\
         * RLX_NONE, an optional pointer NULL. Structured results are JSON text.\n\
         */\n\
         #ifndef RLX_H\n#define RLX_H\n\n\
         #include <stdbool.h>\n#include <stddef.h>\n#include <stdint.h>\n\n\
         #ifdef __cplusplus\nextern \"C\" {\n#endif\n",
    );

    // Types first, so every declaration after them can name any of them.
    s.push_str("\n/* ── types ─────────────────────────────────────────────── */\n\n");
    for (_, decls) in &sections {
        for d in decls {
            if let Decl::Opaque(name, doc) = d {
                comment(&mut s, doc, "");
                let _ = writeln!(s, "typedef struct {name} {name};\n");
            }
        }
    }
    for (_, decls) in &sections {
        for d in decls {
            if let Decl::Struct(name, fields, doc) = d {
                comment(&mut s, doc, "");
                let _ = writeln!(s, "typedef struct {name} {{");
                for (ty, f) in fields {
                    let _ = writeln!(s, "    {ty} {f};");
                }
                let _ = writeln!(s, "}} {name};\n");
            }
        }
    }
    for (_, decls) in &sections {
        for d in decls {
            if let Decl::Callback(name, ret, params, doc) = d {
                comment(&mut s, doc, "");
                let _ = writeln!(s, "typedef {ret} (*{name})({});\n", params.join(", "));
            }
        }
    }
    for (_, decls) in &sections {
        for d in decls {
            if let Decl::Const(name, value, doc) = d {
                comment(&mut s, doc, "");
                let _ = writeln!(s, "#define {name} ({value})\n");
            }
        }
    }

    for (t, decls) in &sections {
        let fns: Vec<&Decl> = decls
            .iter()
            .filter(|d| matches!(d, Decl::Fn { .. }))
            .collect();
        if fns.is_empty() {
            continue;
        }
        let _ = writeln!(s, "\n/* ── {t} */\n");
        for d in fns {
            let Decl::Fn {
                name,
                ret,
                params,
                doc,
                feature,
            } = d
            else {
                continue;
            };
            let mut lines = doc.clone();
            if let Some(f) = feature {
                if !lines.is_empty() {
                    lines.push(String::new());
                }
                lines.push(format!("Requires the `{f}` feature."));
            }
            comment(&mut s, &lines, "");
            if params.is_empty() {
                let _ = writeln!(s, "{ret} {name}(void);\n");
            } else if params.len() <= 3 && params.iter().map(String::len).sum::<usize>() < 70 {
                let _ = writeln!(s, "{ret} {name}({});\n", params.join(", "));
            } else {
                let _ = writeln!(s, "{ret} {name}(");
                for (i, p) in params.iter().enumerate() {
                    let sep = if i + 1 == params.len() { "" } else { "," };
                    let _ = writeln!(s, "    {p}{sep}");
                }
                s.push_str(");\n\n");
            }
        }
    }

    // Not rlx-capi's: a static library that links model plugins defines it.
    s.push_str(
        "\n/* ── linked model plugins */\n\n\
         /*\n \
         * Register the model plugins linked into this library and return how\n \
         * many providers they add. Only a static library built with\n \
         * rlx_runtime::link_model_plugins! defines it (iOS, Android, wasm: hosts\n \
         * that cannot load a plugin). Call it once, before the first rlx_model_*\n \
         * call.\n \
         */\n\
         int32_t rlx_link_model_plugins(void);\n\n",
    );
    s.push_str("#ifdef __cplusplus\n}\n#endif\n\n#endif /* RLX_H */\n");
    s
}
