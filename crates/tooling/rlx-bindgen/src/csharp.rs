// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-csharp/src/Rlx/RLX.g.cs`: the whole C ABI as C# (P/Invoke), from the
//! lowered [`Ir`].
//!
//! `RLX.<Class>` per handle type (`IDisposable`, with a finalizer); free
//! functions are static methods of `RLX`. Optional parameters are C# optional
//! arguments (array defaults filled in the body); options go in as any object
//! (System.Text.Json), reports come back as `JsonNode`s. Errors are
//! `RlxException` with the `RLX_ERR_*` kind.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, camel, doc_lines, pascal};

const KEYWORDS: &[&str] = &[
    "abstract",
    "as",
    "base",
    "bool",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "checked",
    "class",
    "const",
    "continue",
    "decimal",
    "default",
    "delegate",
    "do",
    "double",
    "else",
    "enum",
    "event",
    "explicit",
    "extern",
    "false",
    "finally",
    "fixed",
    "float",
    "for",
    "foreach",
    "goto",
    "if",
    "implicit",
    "in",
    "int",
    "interface",
    "internal",
    "is",
    "lock",
    "long",
    "namespace",
    "new",
    "null",
    "object",
    "operator",
    "out",
    "override",
    "params",
    "private",
    "protected",
    "public",
    "readonly",
    "ref",
    "return",
    "sbyte",
    "sealed",
    "short",
    "sizeof",
    "stackalloc",
    "static",
    "string",
    "struct",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "uint",
    "ulong",
    "unchecked",
    "unsafe",
    "ushort",
    "using",
    "virtual",
    "void",
    "volatile",
    "while",
];

fn ident(name: &str) -> String {
    let c = camel(name);
    if KEYWORDS.contains(&c.as_str()) {
        format!("@{c}")
    } else {
        c
    }
}

fn method_name(name: &str) -> String {
    match pascal(name).as_str() {
        n
        @ ("ToString" | "Equals" | "GetHashCode" | "GetType" | "Finalize" | "MemberwiseClone") => {
            format!("{n}_")
        }
        n => n.to_string(),
    }
}

fn ty(kind: &str) -> String {
    match kind {
        "str" => "string",
        "str?" => "string?",
        "json" | "value" => "object",
        "json?" | "value?" => "object?",
        "bool" => "bool",
        "f32" => "float",
        "f64" => "double",
        "i32" => "int",
        "u32" | "node" => "uint",
        "i64" | "usize" => "long",
        "u64" => "ulong",
        "node?" => "uint?",
        "u64?" => "ulong?",
        "usize?" => "long?",
        "f32?" => "float?",
        "f32s" => "float[]",
        "u8s" => "byte[]",
        "u32s" | "nodes" => "uint[]",
        "usizes" | "i64s" => "long[]",
        "usizes?" => "long[]?",
        "rows2" => "long[][]",
        "frows2" | "frows6" => "float[][]",
        "strs" => "string[]",
        "strs?" => "string[]?",
        "inputs" => "IDictionary<string, float[]>",
        "typed_inputs" => "IDictionary<string, TypedTensor>",
        "graph" | "graph&" => "Graph",
        "graph?" => "Graph?",
        other => return other.trim_end_matches('&').to_string(),
    }
    .to_string()
}

fn ret_ty(kind: &str) -> String {
    match kind {
        "str" => "string",
        "str?" => "string?",
        "json" => "JsonNode?",
        "object" => "JsonObject",
        "rows" => "JsonArray",
        "shapes" => "long[][]",
        "shape" => "Shape",
        "strings" => "List<string>",
        "f32s" => "float[]",
        "u8s" => "byte[]",
        "u32s" | "nodes" => "uint[]",
        "outputs" => "float[][]",
        "typed_outputs" => "List<TypedTensor>",
        "tensors" => "Dictionary<string, float[]>",
        "bool" => "bool",
        "usize" => "long",
        "u32" | "node" => "uint",
        "u64" => "ulong",
        "f32" => "float",
        "f32?" => "float?",
        "graph" => "Graph",
        other => other,
    }
    .to_string()
}

/// The P/Invoke type of a C parameter type.
fn native(c: &str) -> &'static str {
    match c {
        "bool" => "byte",
        "float" => "float",
        "double" => "double",
        "int32_t" => "int",
        "uint32_t" => "uint",
        "int64_t" => "long",
        "uint64_t" => "ulong",
        "size_t" => "nuint",
        _ => "IntPtr",
    }
}

fn is_array_default(kind: &str) -> bool {
    ty(kind).contains('[')
}

fn literal(l: &Lit, kind: &str) -> String {
    match l {
        Lit::None => "null".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => {
            let n = n.replace("inf", "float.PositiveInfinity");
            match ty(kind).trim_end_matches("[]").trim_end_matches('?') {
                "float" if !n.contains("Infinity") => format!("{n}f"),
                "double" if !n.contains('.') && !n.contains('e') => format!("{n}.0"),
                "ulong" => format!("{n}UL"),
                "uint" => format!("{n}U"),
                "long" => format!("{n}L"),
                _ => n,
            }
        }
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let el = ty(kind).trim_end_matches('?').to_string();
            let inner: Vec<String> = items
                .iter()
                .map(|i| literal(i, el.trim_end_matches("[]")))
                .collect();
            format!("new {el} {{ {} }}", inner.join(", "))
        }
    }
}

/// `(fixed / using statements, native arguments)` for one parameter.
fn lower(p: &Param, v: &str, fixed: &mut Vec<String>, pre: &mut Vec<String>) -> Vec<String> {
    let n = format!("p_{}", p.name);
    match p.kind.as_str() {
        "str" | "str?" => {
            fixed.push(format!("byte* {n} = Support.Utf8({v})"));
            vec![format!("(IntPtr){n}")]
        }
        "json" | "json?" | "value" | "value?" => {
            fixed.push(format!("byte* {n} = Support.Json({v})"));
            vec![format!("(IntPtr){n}")]
        }
        "bool" => vec![format!("(byte)({v} ? 1 : 0)")],
        "f32" | "f64" | "i32" | "u32" | "node" | "u64" | "i64" => vec![v.to_string()],
        "usize" => vec![format!("(nuint){v}")],
        "node?" => vec![format!("({v} ?? None)")],
        "u64?" | "usize?" | "f32?" => {
            let t = match p.kind.as_str() {
                "u64?" => "ulong",
                "usize?" => "nuint",
                _ => "float",
            };
            pre.push(format!("{t} v_{0} = ({t})({v} ?? default);", p.name));
            vec![format!(
                "({v}.HasValue ? (IntPtr)(&v_{}) : IntPtr.Zero)",
                p.name
            )]
        }
        "f32s" | "u8s" | "u32s" | "nodes" | "usizes" | "usizes?" | "i64s" => {
            let el = match p.kind.as_str() {
                "f32s" => "float",
                "u8s" => "byte",
                "u32s" | "nodes" => "uint",
                _ => "long",
            };
            fixed.push(format!("{el}* {n} = {v}"));
            vec![format!("(IntPtr){n}"), format!("(nuint)({v}?.Length ?? 0)")]
        }
        "rows2" => {
            fixed.push(format!("long* {n} = Support.Rows({v}, 2, true)"));
            vec![format!("(IntPtr){n}"), format!("(nuint){v}.Length")]
        }
        "frows2" | "frows6" => {
            let w = if p.kind == "frows2" { 2 } else { 6 };
            fixed.push(format!("float* {n} = Support.FRows({v}, {w})"));
            vec![format!("(IntPtr){n}"), format!("(nuint){v}.Length")]
        }
        "strs" | "strs?" => {
            pre.push(format!("using var c_{} = new CStrings({v});", p.name));
            vec![format!("c_{}.Ptr", p.name), format!("c_{}.Count", p.name)]
        }
        "inputs" => {
            pre.push(format!("using var c_{} = new CInputs({v});", p.name));
            vec![
                format!("c_{}.Names", p.name),
                format!("c_{}.Data", p.name),
                format!("c_{}.Lens", p.name),
                format!("c_{}.Count", p.name),
            ]
        }
        "typed_inputs" => {
            pre.push(format!("using var c_{} = new CInputs({v});", p.name));
            vec![
                format!("c_{}.Names", p.name),
                format!("c_{}.Data", p.name),
                format!("c_{}.Lens", p.name),
                format!("c_{}.Dtypes", p.name),
                format!("c_{}.Count", p.name),
            ]
        }
        "graph?" => vec![format!("({v}?.H ?? IntPtr.Zero)")],
        _ => vec![format!("{v}.H")],
    }
}

/// `(declarations, native arguments, C# result expression)`.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("o{i}");
    let one = |t: &str| {
        (
            vec![format!("{t} {o} = default;")],
            vec![format!("(IntPtr)(&{o})")],
        )
    };
    let (d, a) = match kind {
        "f32s" | "u8s" | "u32s" | "nodes" => one("RlxArray"),
        "bool" => one("byte"),
        "usize" => one("nuint"),
        "u32" | "node" => one("uint"),
        "u64" => one("ulong"),
        "f32" => one("float"),
        "f32?" => (
            vec![format!("float {o} = 0;"), format!("byte {o}k = 0;")],
            vec![format!("(IntPtr)(&{o})"), format!("(IntPtr)(&{o}k)")],
        ),
        _ => one("IntPtr"),
    };
    let e = match kind {
        "str" => format!("Support.TakeString({o})"),
        "str?" => format!("Support.TakeStringOrNull({o})"),
        "json" => format!("JsonNode.Parse(Support.TakeString({o}))"),
        "object" => format!("JsonNode.Parse(Support.TakeString({o}))!.AsObject()"),
        "rows" => format!("JsonNode.Parse(Support.TakeString({o}))!.AsArray()"),
        "shapes" => format!("Support.Shapes(Support.TakeString({o}))"),
        "shape" => format!("Support.ShapeOf(Support.TakeString({o}))"),
        "strings" => format!("Support.Strings(Support.TakeString({o}))"),
        "f32s" => format!("Support.TakeFloats(ref {o})"),
        "u8s" => format!("Support.TakeBytes(ref {o})"),
        "u32s" | "nodes" => format!("Support.TakeUInts(ref {o})"),
        "outputs" => format!("Support.TakeOutputs({o})"),
        "typed_outputs" => format!("Support.TakeTypedOutputs({o})"),
        "tensors" => format!("Support.TakeTensors({o})"),
        "bool" => format!("{o} != 0"),
        "usize" => format!("(long){o}"),
        "u32" | "node" | "u64" | "f32" => o.clone(),
        "f32?" => format!("({o}k != 0 ? {o} : (float?)null)"),
        "graph" => format!("new Graph({o})"),
        class => format!("new {class}({o})"),
    };
    (d, a, e)
}

fn c_types(ir: &Ir, f: &Func) -> Vec<String> {
    let mut v = Vec::new();
    if f.role == Role::Method {
        v.push("ptr".to_string());
    }
    for p in &f.params {
        v.extend(crate::ir::c_args(&p.kind));
    }
    if !f.direct {
        let rets: Vec<String> = match f.role {
            Role::Constructor => vec![ir.class_of(f).unwrap().name.clone()],
            _ => f.rets.iter().map(|(_, k)| k.clone()).collect(),
        };
        for k in rets {
            v.extend(crate::ir::c_outs(&k));
        }
    }
    v
}

fn func(s: &mut String, natives: &mut Vec<String>, ir: &Ir, f: &Func, indent: &str) {
    let class = ir.class_of(f).map(|c| c.name.clone());
    let name = method_name(&f.name);
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![],
        _ => f.rets.clone(),
    };
    let result = format!("{}Result", pascal(&f.name));
    if rets.len() > 1 {
        let fields: Vec<String> = rets
            .iter()
            .map(|(n, k)| format!("{} {}", ret_ty(k), pascal(n)))
            .collect();
        let _ = writeln!(s, "{indent}/// <summary>What {name} returns.</summary>");
        let _ = writeln!(
            s,
            "{indent}public sealed record {result}({});\n",
            fields.join(", ")
        );
    }
    let ret = match rets.as_slice() {
        [] => "void".to_string(),
        [(_, k)] => ret_ty(k),
        _ => result.clone(),
    };
    let mut params: Vec<String> = f
        .params
        .iter()
        .filter(|p| !p.optional())
        .map(|p| format!("{} {}", ty(&p.kind), ident(&p.name)))
        .collect();
    for p in f.params.iter().filter(|p| p.optional()) {
        let t = ty(&p.kind);
        let (t, d) = match &p.default {
            Some(_) if is_array_default(&p.kind) => {
                (format!("{}?", t.trim_end_matches('?')), "null".to_string())
            }
            Some(l) => (t, literal(l, &p.kind)),
            None => {
                let t = if t.ends_with('?') { t } else { format!("{t}?") };
                (t, "null".to_string())
            }
        };
        params.push(format!("{t} {} = {d}", ident(&p.name)));
    }
    let doc = doc_lines(&f.doc, "");
    if !doc.trim().is_empty() {
        let esc = doc
            .trim_end()
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");
        let _ = writeln!(s, "{indent}/// <summary>");
        for line in esc.lines() {
            let _ = writeln!(s, "{indent}/// {line}");
        }
        let _ = writeln!(s, "{indent}/// </summary>");
    }
    let stat = matches!(f.role, Role::Function | Role::Static);
    let head = match f.role {
        Role::Constructor => format!(
            "public {}({}) : this(Make({}))",
            class.clone().unwrap(),
            params.join(", "),
            f.params
                .iter()
                .map(|p| ident(&p.name))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => format!(
            "public {}{ret} {name}({})",
            if stat { "static " } else { "" },
            params.join(", ")
        ),
    };
    if f.role == Role::Constructor {
        let _ = writeln!(s, "{indent}{head} {{ }}\n");
        let _ = writeln!(
            s,
            "{indent}private static unsafe IntPtr Make({}) {{",
            f.params
                .iter()
                .map(|p| {
                    let t = ty(&p.kind);
                    let t = if p.optional()
                        && !t.ends_with('?')
                        && (p.default.is_none() || is_array_default(&p.kind))
                    {
                        format!("{t}?")
                    } else {
                        t
                    };
                    format!("{t} {}", ident(&p.name))
                })
                .collect::<Vec<_>>()
                .join(", ")
        );
    } else {
        let head = head.replacen("public ", "public unsafe ", 1);
        let _ = writeln!(s, "{indent}{head} {{");
    }
    let b = format!("{indent}    ");
    for p in f.params.iter().filter(|p| p.optional()) {
        if let Some(l) = &p.default
            && is_array_default(&p.kind)
        {
            let _ = writeln!(s, "{b}{} ??= {};", ident(&p.name), literal(l, &p.kind));
        }
    }
    let mut fixed = Vec::new();
    let mut pre = Vec::new();
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("H".to_string());
    }
    for p in &f.params {
        args.extend(lower(p, &ident(&p.name), &mut fixed, &mut pre));
    }
    let all_rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), class.clone().unwrap())],
        _ => f.rets.clone(),
    };
    let mut decls = Vec::new();
    let mut exprs = Vec::new();
    if !f.direct {
        for (i, (_, k)) in all_rets.iter().enumerate() {
            let (d, a, e) = outs(k, i);
            decls.extend(d);
            args.extend(a);
            exprs.push(e);
        }
    }
    // The native declaration.
    let types = c_types(ir, f);
    let nparams: Vec<String> = types
        .iter()
        .enumerate()
        .map(|(i, t)| format!("{} a{i}", native(t)))
        .collect();
    let nret = if f.direct {
        match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => "IntPtr",
            _ => "byte",
        }
    } else {
        "int"
    };
    natives.push(format!(
        "        [DllImport(Lib, EntryPoint = \"{c}\")] internal static extern {nret} {c}({});",
        nparams.join(", "),
        c = f.c
    ));
    for line in pre.iter().chain(&decls) {
        let _ = writeln!(s, "{b}{line}");
    }
    let mut depth = String::new();
    for fx in &fixed {
        let _ = writeln!(s, "{b}{depth}fixed ({fx})");
        depth.push_str("");
    }
    let call = format!("Native.{}({})", f.c, args.join(", "));
    let open = if fixed.is_empty() { "" } else { "{\n" };
    if !fixed.is_empty() {
        let _ = writeln!(s, "{b}{{");
    }
    let _ = open;
    let ib = if fixed.is_empty() {
        b.clone()
    } else {
        format!("{b}    ")
    };
    if f.direct {
        match nret {
            "IntPtr" => {
                let _ = writeln!(s, "{ib}return Marshal.PtrToStringUTF8({call})!;");
            }
            _ => {
                let _ = writeln!(s, "{ib}return {call} != 0;");
            }
        }
    } else {
        let _ = writeln!(s, "{ib}Support.Check({call});");
        match (f.role, exprs.len()) {
            (Role::Constructor, _) => {
                let _ = writeln!(s, "{ib}return o0;");
            }
            (_, 0) => {}
            (_, 1) => {
                let _ = writeln!(s, "{ib}return {};", exprs[0]);
            }
            _ => {
                let _ = writeln!(s, "{ib}return new {result}({});", exprs.join(", "));
            }
        }
    }
    if !fixed.is_empty() {
        let _ = writeln!(s, "{b}}}");
    }
    let _ = writeln!(s, "{indent}}}\n");
}

pub fn generate(ir: &Ir) -> String {
    let mut natives = Vec::new();
    let mut body = String::new();
    for f in ir.free_funcs().filter(|f| !f.manual) {
        func(&mut body, &mut natives, ir, f, "    ");
    }
    for c in &ir.classes {
        let doc = doc_lines(&c.doc, "");
        let _ = writeln!(body, "    /// <summary>");
        for line in doc.trim_end().lines() {
            let _ = writeln!(
                body,
                "    /// {}",
                line.replace('<', "&lt;").replace('>', "&gt;")
            );
        }
        let _ = writeln!(body, "    /// </summary>");
        let _ = writeln!(
            body,
            "    public sealed partial class {n} : RlxHandle\n    {{\n        internal {n}(IntPtr h) : base(h) {{ }}\n        protected override void Free(IntPtr h) => Native.{free}(h);\n",
            n = c.name,
            free = c.c_free()
        );
        natives.push(format!(
            "        [DllImport(Lib, EntryPoint = \"{f}\")] internal static extern void {f}(IntPtr h);",
            f = c.c_free()
        ));
        for f in ir.members(c).filter(|f| !f.manual) {
            func(&mut body, &mut natives, ir, f, "        ");
        }
        body.push_str("    }\n\n");
    }
    let mut s = String::from(
        "// RLX — versatile ML compiler + runtime.\n\
         // Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         // SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // <auto-generated>\n\
         // Generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         // do not edit; regenerate with `just gen-bindings`.\n\
         // </auto-generated>\n\
         \n\
         #nullable enable\n\
         using System;\n\
         using System.Collections.Generic;\n\
         using System.Runtime.InteropServices;\n\
         using System.Text.Json.Nodes;\n\
         \n\
         namespace Rlx;\n\
         \n\
         /// <summary>\n\
         /// rlx for .NET: every C function of rlx-capi. Handle types are nested\n\
         /// classes (<c>RLX.Graph</c>, <c>RLX.Session</c>, …); free functions are static\n\
         /// methods. Optional parameters are optional arguments with the tables' defaults.\n\
         /// </summary>\n\
         public static partial class RLX\n\
         {\n\
         \x20   /// <summary>\"No node\", for an optional node argument.</summary>\n\
         \x20   public const uint None = uint.MaxValue;\n\n",
    );
    s.push_str(&body);
    s.push_str("    internal static partial class Native\n    {\n");
    for n in natives {
        s.push_str(&n);
        s.push('\n');
    }
    s.push_str("    }\n}\n");
    s
}
