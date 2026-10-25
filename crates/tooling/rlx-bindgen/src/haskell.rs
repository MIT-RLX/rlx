// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-haskell/src/RLX/Generated.hs`: the whole C ABI as Haskell, from the
//! lowered [`Ir`].
//!
//! Every C function is a `foreign import ccall` (feature-gated ones under CPP
//! `RLX_FEATURE_<NAME>`). A `newtype` over `ForeignPtr` per handle type,
//! finalized by its C free function. Functions are in `IO`; the graph's ops
//! take the graph first (`matmul g a b`), other classes' members carry the
//! class name (`compiledRun c inputs`, `ggufOpen path`). A function with
//! optional parameters also has a `…With` form taking a record whose
//! `def…` value holds the tables' defaults
//! (`softmaxWith defSoftmax { softmaxAxis = 1 } g x`). Options and reports
//! are aeson `Value`s; tensors `[Float]`.

use std::collections::HashSet;
use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, camel, doc_lines, pascal};
use crate::jni::c_params;

const KEYWORDS: &[&str] = &[
    "case", "class", "data", "default", "deriving", "do", "else", "foreign", "if", "import", "in",
    "infix", "infixl", "infixr", "instance", "let", "module", "newtype", "of", "then", "type",
    "where", "forall", "mdo", "rec", "proc", "pattern", "family", "role",
];

fn ident(s: String) -> String {
    if KEYWORDS.contains(&s.as_str()) {
        format!("{s}'")
    } else {
        s
    }
}

fn lower_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(h) => h.to_ascii_lowercase().to_string() + c.as_str(),
        None => String::new(),
    }
}

/// The Haskell name of every function (unique across the module).
fn names(ir: &Ir) -> Vec<String> {
    let free: HashSet<String> = ir.free_funcs().map(|f| camel(&f.name)).collect();
    ir.funcs
        .iter()
        .map(|f| {
            let class = ir.class_of(f).map(|c| c.name.clone());
            ident(match (f.role, class.as_deref()) {
                (Role::Constructor, Some(c)) => format!("new{c}"),
                (_, None) => camel(&f.name),
                // The graph's ops are the bulk of the API: unprefixed unless a
                // free function has the name.
                (_, Some("Graph")) if !free.contains(&camel(&f.name)) => camel(&f.name),
                (_, Some(c)) => format!("{}{}", lower_first(c), pascal(&f.name)),
            })
        })
        .collect()
}

fn var(name: &str) -> String {
    let c = camel(name);
    match c.as_str() {
        "h" | "o" | "rc" | "g" | "opts" => format!("{c}'"),
        _ => ident(c),
    }
}

fn cty(c: &str) -> &'static str {
    match c {
        "bool" => "CBool",
        "float" => "CFloat",
        "double" => "CDouble",
        "int32_t" => "Int32",
        "uint32_t" => "Word32",
        "int64_t" => "Int64",
        "uint64_t" => "Word64",
        "size_t" => "CSize",
        _ => "Ptr ()",
    }
}

fn ty(kind: &str) -> String {
    match kind {
        "str" => "String",
        "str?" => "Maybe String",
        "json" | "value" => "Value",
        "json?" | "value?" => "Maybe Value",
        "bool" => "Bool",
        "f32" => "Float",
        "f64" => "Double",
        "i32" | "i64" | "u64" | "usize" => "Int",
        "u32" | "node" => "Node",
        "node?" => "Maybe Node",
        "u64?" | "usize?" => "Maybe Int",
        "f32?" => "Maybe Float",
        "f32s" => "[Float]",
        "u8s" => "[Word8]",
        "u32s" | "nodes" => "[Node]",
        "usizes" | "i64s" => "[Int]",
        "usizes?" => "Maybe [Int]",
        "rows2" => "[[Int]]",
        "frows2" | "frows6" => "[[Float]]",
        "strs" => "[String]",
        "strs?" => "Maybe [String]",
        "inputs" => "[(String, [Float])]",
        "typed_inputs" => "[(String, TypedTensor)]",
        "graph" | "graph&" => "Graph",
        "graph?" => "Maybe Graph",
        other => return other.trim_end_matches('&').to_string(),
    }
    .to_string()
}

fn ret_ty(kind: &str) -> String {
    match kind {
        "str" => "String",
        "str?" => "Maybe String",
        "json" | "object" | "rows" => "Value",
        "shapes" => "[[Int]]",
        "shape" => "[Int]",
        "strings" => "[String]",
        "f32s" => "[Float]",
        "u8s" => "[Word8]",
        "u32s" | "nodes" => "[Node]",
        "outputs" => "[[Float]]",
        "typed_outputs" => "[TypedTensor]",
        "tensors" => "[(String, [Float])]",
        "bool" => "Bool",
        "usize" | "u64" => "Int",
        "u32" | "node" => "Node",
        "f32" => "Float",
        "f32?" => "Maybe Float",
        "graph" => "Graph",
        other => other,
    }
    .to_string()
}

fn literal(l: &Lit) -> String {
    match l {
        Lit::None => "Nothing".into(),
        Lit::Bool(true) => "True".into(),
        Lit::Bool(false) => "False".into(),
        Lit::Num(n) if n.contains("inf") => {
            if n.starts_with('-') {
                "(-1 / 0)".into()
            } else {
                "(1 / 0)".into()
            }
        }
        Lit::Num(n) if n.starts_with('-') => format!("({n})"),
        Lit::Num(n) => n.clone(),
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(literal).collect();
            format!("[{}]", inner.join(", "))
        }
    }
}

/// `(with-wrappers, C arguments)` for one parameter: each wrapper is a
/// `withX v $ \p ->` line.
fn lower(p: &Param, v: &str, i: usize) -> (Vec<String>, Vec<String>) {
    let a = format!("a{i}");
    let with = |f: &str, arg: String| (vec![format!("{f} {arg} $ \\{a} ->")], vec![a.clone()]);
    let len = |e: String| format!("(P.fromIntegral (P.length {e}))");
    let w = |f: &str| {
        let (x, mut y) = with(f, v.to_string());
        y.push(len(v.to_string()));
        (x, y)
    };
    match p.kind.as_str() {
        "str" => with("withStr", v.into()),
        "str?" => with("withStrMaybe", v.into()),
        "json" | "value" => with("withJson", format!("(Just {v})")),
        "json?" | "value?" => with("withJson", v.into()),
        "u64?" | "usize?" => with("withBoxU64", v.into()),
        "f32?" => with("withBoxF32", v.into()),
        "node?" => (vec![], vec![format!("(P.maybe none P.id {v})")]),
        "f32s" => w("withFloats"),
        "u8s" => w("withBytes"),
        "u32s" | "nodes" => w("withU32s"),
        "usizes" => w("withU64s"),
        "i64s" => w("withI64s"),
        "usizes?" => (
            vec![format!("withU64s (P.maybe [] P.id {v}) $ \\{a} ->")],
            vec![
                format!("(P.maybe nullPtr (P.const {a}) {v})"),
                len(format!("(P.maybe [] P.id {v})")),
            ],
        ),
        "rows2" => (
            vec![format!("withU64s (rows {v} 2 True) $ \\{a} ->")],
            vec![a.clone(), len(v.into())],
        ),
        "frows2" | "frows6" => {
            let n = if p.kind == "frows2" { 2 } else { 6 };
            (
                vec![format!("withFloats (frows {v} {n}) $ \\{a} ->")],
                vec![a.clone(), len(v.into())],
            )
        }
        "strs" => w("withStrs"),
        "strs?" => (
            vec![format!("withStrs (P.maybe [] P.id {v}) $ \\{a} ->")],
            vec![a.clone(), len(format!("(P.maybe [] P.id {v})"))],
        ),
        "inputs" => (
            vec![format!("withInputs {v} $ \\({a}n, {a}d, {a}l, {a}c) ->")],
            vec![
                format!("{a}n"),
                format!("{a}d"),
                format!("{a}l"),
                format!("{a}c"),
            ],
        ),
        "typed_inputs" => (
            vec![format!(
                "withTypedInputs {v} $ \\({a}n, {a}d, {a}l, {a}t, {a}c) ->"
            )],
            vec![
                format!("{a}n"),
                format!("{a}d"),
                format!("{a}l"),
                format!("{a}t"),
                format!("{a}c"),
            ],
        ),
        "graph?" => with("withHMaybe", v.into()),
        "bool" => (vec![], vec![format!("(fromBool {v})")]),
        "f32" => (vec![], vec![format!("(P.realToFrac {v})")]),
        "f64" => (vec![], vec![format!("(P.realToFrac {v})")]),
        "u32" | "node" => (vec![], vec![v.to_string()]),
        "i32" | "i64" | "u64" | "usize" => (vec![], vec![format!("(P.fromIntegral {v})")]),
        _ => with("withH", v.into()),
    }
}

/// `(alloca wrapper, C arguments, result expression)` for one result.
fn outs(kind: &str, i: usize) -> (String, Vec<String>, String) {
    let o = format!("o{i}");
    let (w, a) = match kind {
        "f32s" | "u8s" | "u32s" | "nodes" => (format!("allocaSlice $ \\{o} ->"), vec![o.clone()]),
        "f32?" => (
            format!("allocaOut $ \\{o} -> allocaOut $ \\{o}k ->"),
            vec![o.clone(), format!("{o}k")],
        ),
        _ => (format!("allocaOut $ \\{o} ->"), vec![o.clone()]),
    };
    let e = match kind {
        "str" => format!("takeString {o}"),
        "str?" => format!("takeStringMaybe {o}"),
        "json" | "object" | "rows" => format!("takeJson {o}"),
        "shapes" | "shape" | "strings" => format!("takeJsonAs {o}"),
        "f32s" => format!("takeF32s {o}"),
        "u8s" => format!("takeBytes {o}"),
        "u32s" | "nodes" => format!("takeU32s {o}"),
        "outputs" => format!("takeOutputs {o}"),
        "typed_outputs" => format!("takeTypedOutputs {o}"),
        "tensors" => format!("takeTensors {o}"),
        "bool" => format!("peekBool {o}"),
        "u32" | "node" => format!("peekU32 {o}"),
        "usize" | "u64" => format!("peekU64 {o}"),
        "f32" => format!("peekF32 {o}"),
        "f32?" => format!("peekF32Maybe {o} {o}k"),
        "graph" => format!("adopt Graph pFree_graph {o}"),
        class => format!("adopt {class} pFree_{} {o}", class_c(class)),
    };
    (w, a, e)
}

fn class_c(name: &str) -> String {
    // rlx_<c>_free: the snake_case of the class name.
    let mut s = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() && i > 0 {
            s.push('_');
        }
        s.push(ch.to_ascii_lowercase());
    }
    s
}

fn gate(f: &Func, defaults: &[String]) -> Option<String> {
    f.feature
        .as_ref()
        .filter(|x| !defaults.contains(x))
        .map(|x| format!("RLX_FEATURE_{}", x.to_ascii_uppercase().replace('-', "_")))
}

fn func(s: &mut String, ir: &Ir, f: &Func, name: &str, exports: &mut Vec<String>) {
    let class = ir.class_of(f).map(|c| c.name.clone());
    let required: Vec<&Param> = f.params.iter().filter(|p| !p.optional()).collect();
    let optional: Vec<&Param> = f.params.iter().filter(|p| p.optional()).collect();
    let stem = pascal(&lower_first(name).replace('\'', ""));
    let opts = format!("{stem}Opts");
    let field = |p: &Param| format!("{}{}", lower_first(&stem), pascal(&p.name));
    if !optional.is_empty() {
        let _ = writeln!(
            s,
            "-- | The optional parameters of '{name}'; 'def{stem}' holds the defaults."
        );
        let fields: Vec<String> = optional
            .iter()
            .map(|p| {
                let t = ty(&p.kind);
                let t = if matches!(p.default, None | Some(Lit::None)) && !t.starts_with("Maybe") {
                    format!(
                        "Maybe {}",
                        if t.contains(' ') { format!("({t})") } else { t }
                    )
                } else {
                    t
                };
                format!("{} :: {t}", field(p))
            })
            .collect();
        let _ = writeln!(
            s,
            "data {opts} = {opts}\n  {{ {}\n  }}\n",
            fields.join("\n  , ")
        );
        let defs: Vec<String> = optional
            .iter()
            .map(|p| {
                let d = match &p.default {
                    None | Some(Lit::None) => "Nothing".to_string(),
                    Some(l) if p.kind.ends_with('?') => format!("(Just {})", literal(l)),
                    Some(l) => literal(l),
                };
                format!("{} = {d}", field(p))
            })
            .collect();
        let _ = writeln!(
            s,
            "def{stem} :: {opts}\ndef{stem} = {opts}\n  {{ {}\n  }}\n",
            defs.join("\n  , ")
        );
        exports.push(format!("{opts} (..)"));
        exports.push(format!("def{stem}"));
    }
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), class.clone().unwrap())],
        _ => f.rets.clone(),
    };
    let ret = match rets.as_slice() {
        [] => "()".to_string(),
        [(_, k)] => ret_ty(k),
        _ => format!(
            "({})",
            rets.iter()
                .map(|(_, k)| ret_ty(k))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let wrap = |t: &str| {
        if t.contains(' ') && !t.starts_with('(') && !t.starts_with('[') {
            format!("({t})")
        } else {
            t.to_string()
        }
    };
    let mut sig_types: Vec<String> = Vec::new();
    let mut sig_vars: Vec<String> = Vec::new();
    if !optional.is_empty() {
        sig_types.push(opts.clone());
        sig_vars.push("opts'".into());
    }
    if f.role == Role::Method {
        sig_types.push(class.clone().unwrap());
        sig_vars.push("self'".into());
    }
    for p in &required {
        sig_types.push(wrap(&ty(&p.kind)));
        sig_vars.push(var(&p.name));
    }
    let full = if optional.is_empty() {
        name.to_string()
    } else {
        format!("{name}With")
    };
    let doc = doc_lines(&f.doc, "-- ");
    let doc = doc.replacen("-- ", "-- | ", 1);
    if !optional.is_empty() {
        // The short form: defaults.
        s.push_str(&doc);
        let short_types: Vec<String> = sig_types[1..].to_vec();
        let _ = writeln!(
            s,
            "{name} :: {}IO {}\n{name} = {full} def{stem}\n",
            short_types
                .iter()
                .map(|t| format!("{t} -> "))
                .collect::<String>(),
            wrap(&ret)
        );
        exports.push(name.to_string());
        let _ = writeln!(s, "-- | '{name}' with explicit optional parameters.");
    } else {
        s.push_str(&doc);
    }
    exports.push(full.clone());
    let _ = writeln!(
        s,
        "{full} :: {}IO {}",
        sig_types
            .iter()
            .map(|t| format!("{t} -> "))
            .collect::<String>(),
        wrap(&ret)
    );
    let _ = writeln!(s, "{full} {} =", sig_vars.join(" "));
    let mut lines: Vec<String> = Vec::new();
    let mut args = Vec::new();
    let mut i = 0;
    if f.role == Role::Method {
        lines.push(format!("withH self' $ \\a{i} ->"));
        args.push(format!("a{i}"));
        i += 1;
    }
    for p in &f.params {
        let v = if p.optional() {
            format!("({} opts')", field(p))
        } else {
            var(&p.name)
        };
        let (w, a) = lower(p, &v, i);
        i += 1;
        lines.extend(w);
        args.extend(a);
    }
    if f.direct {
        let call = format!("c_{} {}", f.c, args.join(" "));
        let e = match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => format!("{call} P.>>= peekCString"),
            _ => format!("P.fmap toBool ({call})"),
        };
        lines.push(e);
    } else {
        let mut exprs = Vec::new();
        for (k, (_, kind)) in rets.iter().enumerate() {
            let kind = if f.role == Role::Constructor {
                class.clone().unwrap()
            } else {
                kind.clone()
            };
            let (w, a, e) = outs(&kind, k);
            lines.push(w);
            args.extend(a);
            exprs.push(e);
        }
        lines.push("do".into());
        lines.push(format!("  checkRc P.=<< c_{} {}", f.c, args.join(" ")));
        match exprs.len() {
            0 => lines.push("  P.pure ()".into()),
            1 => lines.push(format!("  {}", exprs[0])),
            _ => {
                for (k, e) in exprs.iter().enumerate() {
                    lines.push(format!("  r{k} <- {e}"));
                }
                let rs: Vec<String> = (0..exprs.len()).map(|k| format!("r{k}")).collect();
                lines.push(format!("  P.pure ({})", rs.join(", ")));
            }
        }
    }
    for l in lines {
        let _ = writeln!(s, "  {l}");
    }
    s.push('\n');
}

pub fn generate(ir: &Ir, defaults: &[String]) -> String {
    let names = names(ir);
    let mut body = String::new();
    let mut exports: Vec<String> = Vec::new();
    // Foreign imports.
    for c in &ir.classes {
        let _ = writeln!(
            body,
            "foreign import ccall unsafe \"&{free}\" pFree_{c} :: FunPtr (Ptr () -> IO ())",
            free = c.c_free(),
            c = c.c
        );
    }
    for f in ir.funcs.iter().filter(|f| !f.manual) {
        let ps: Vec<&str> = c_params(ir, f).iter().map(|(t, _)| cty(t)).collect();
        let ret = if f.direct {
            match f.rets.first().map(|(_, k)| k.as_str()) {
                Some("str") => "CString",
                _ => "CBool",
            }
        } else {
            "Int32"
        };
        let g = gate(f, defaults);
        if let Some(g) = &g {
            let _ = writeln!(body, "#ifdef {g}");
        }
        // `safe`: a call may block (a mesh rank) or call back.
        let _ = writeln!(
            body,
            "foreign import ccall safe \"{c}\" c_{c} :: {}IO {ret}",
            ps.iter().map(|t| format!("{t} -> ")).collect::<String>(),
            c = f.c
        );
        if g.is_some() {
            let _ = writeln!(body, "#endif");
        }
    }
    body.push('\n');
    for c in &ir.classes {
        body.push_str(&doc_lines(&c.doc, "-- ").replacen("-- ", "-- | ", 1));
        let _ = writeln!(
            body,
            "newtype {n} = {n} (ForeignPtr ())\n\ninstance Handle {n} where\n  handlePtr ({n} p) = p\n",
            n = c.name
        );
        exports.push(c.name.clone());
    }
    for (f, name) in ir.funcs.iter().zip(&names) {
        if f.manual {
            continue;
        }
        let g = gate(f, defaults);
        if let Some(g) = &g {
            let _ = writeln!(body, "#ifdef {g}");
        }
        let mut ex = Vec::new();
        func(&mut body, ir, f, name, &mut ex);
        if let Some(g) = &g {
            let _ = writeln!(body, "#endif\n");
            // exports of a gated function are gated too
            for e in ex {
                exports.push(format!("#ifdef {g}\n  , {e}\n#endif"));
            }
        } else {
            exports.extend(ex);
        }
    }
    let mut s = String::from(
        "-- RLX — versatile ML compiler + runtime.\n\
         -- Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         -- SPDX-License-Identifier: MIT OR Apache-2.0\n\
         --\n\
         -- @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         -- do not edit; regenerate with `just gen-bindings`.\n\
         \n\
         {-# LANGUAGE CPP #-}\n\
         {-# LANGUAGE NoImplicitPrelude #-}\n\
         {-# OPTIONS_GHC -Wno-unused-imports #-}\n\
         \n\
         -- | The rlx API: every C function of rlx-capi (see \"RLX\").\n\
         module RLX.Generated\n  ( ",
    );
    let mut first = true;
    for e in &exports {
        if e.starts_with("#ifdef") {
            s.push('\n');
            s.push_str(e);
        } else {
            if !first {
                s.push_str("\n  , ");
            }
            s.push_str(e);
        }
        first = false;
    }
    s.push_str(
        "\n  ) where\n\n\
         import qualified Prelude as P\n\
         import Prelude (Bool (..), Double, Float, Int, IO, Maybe (..), String, ($))\n\
         import Data.Aeson (Value)\n\
         import Data.Int (Int32, Int64)\n\
         import Data.Word (Word8, Word32, Word64)\n\
         import Foreign.C.String (CString, peekCString)\n\
         import Foreign.C.Types (CBool (..), CDouble (..), CFloat (..), CSize (..))\n\
         import Foreign.ForeignPtr (ForeignPtr)\n\
         import Foreign.Marshal.Utils (fromBool, toBool)\n\
         import Foreign.Ptr (FunPtr, Ptr, nullPtr)\n\
         \n\
         import RLX.Support\n\n",
    );
    s.push_str(&body);
    s.truncate(s.trim_end().len());
    s.push('\n');
    s
}
