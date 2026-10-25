// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-java/src/main/java/org/rlx/RLX.java`: the whole C ABI as Java (the
//! FFM API, Java 22+), from the lowered [`Ir`].
//!
//! `RLX.<Class>` per handle type (`AutoCloseable`, and freed by a `Cleaner`);
//! free functions are static methods of `RLX`. A function's optional
//! parameters are fields of its `…Opts` class, initialized to the tables'
//! defaults with fluent setters; each function also has an overload without
//! them. Errors are the unchecked `RLXException`.

use std::fmt::Write;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, camel, doc_lines, pascal};

const KEYWORDS: &[&str] = &[
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while",
    "true",
    "false",
    "null",
    "record",
    "yield",
    "var",
    "h",
    "opts",
    "a",
];

fn ident(name: &str) -> String {
    let c = camel(name);
    if KEYWORDS.contains(&c.as_str()) {
        format!("{c}_")
    } else {
        c
    }
}

fn method_name(name: &str) -> String {
    match camel(name).as_str() {
        // `Object.wait/notify/getClass/hashCode/equals/toString/finalize` stay Object's.
        n @ ("wait" | "notify" | "notifyAll" | "getClass" | "hashCode" | "equals" | "toString"
        | "finalize") => format!("{n}_"),
        n if KEYWORDS.contains(&n) => format!("{n}_"),
        n => n.to_string(),
    }
}

fn ty(kind: &str) -> String {
    match kind {
        "str" | "str?" => "String",
        "json" | "json?" => "Map<String, ?>",
        "value" | "value?" => "Object",
        "bool" => "boolean",
        "f32" => "float",
        "f64" => "double",
        "i32" | "u32" | "node" => "int",
        "i64" | "u64" | "usize" => "long",
        "node?" => "Integer",
        "u64?" | "usize?" => "Long",
        "f32?" => "Float",
        "f32s" => "float[]",
        "u8s" => "byte[]",
        "u32s" | "nodes" => "int[]",
        "usizes" | "usizes?" | "i64s" => "long[]",
        "rows2" => "long[][]",
        "frows2" | "frows6" => "float[][]",
        "strs" | "strs?" => "String[]",
        "inputs" => "Map<String, float[]>",
        "typed_inputs" => "Map<String, TypedTensor>",
        "graph" | "graph&" | "graph?" => "Graph",
        other => return other.trim_end_matches('&').to_string(),
    }
    .to_string()
}

fn ret_ty(kind: &str) -> String {
    match kind {
        "str" | "str?" => "String",
        "json" => "Object",
        "object" => "Map<String, Object>",
        "rows" => "List<Map<String, Object>>",
        "shapes" => "long[][]",
        "shape" => "Shape",
        "strings" => "List<String>",
        "f32s" => "float[]",
        "u8s" => "byte[]",
        "u32s" | "nodes" => "int[]",
        "outputs" => "float[][]",
        "typed_outputs" => "List<TypedTensor>",
        "tensors" => "Map<String, float[]>",
        "bool" => "boolean",
        "usize" | "u64" => "long",
        "u32" | "node" => "int",
        "f32" => "float",
        "f32?" => "Float",
        "graph" => "Graph",
        other => other,
    }
    .to_string()
}

/// The FFM layout of a C parameter type.
fn layout(c: &str) -> &'static str {
    match c {
        "bool" => "JAVA_BOOLEAN",
        "float" => "JAVA_FLOAT",
        "double" => "JAVA_DOUBLE",
        "int32_t" | "uint32_t" => "JAVA_INT",
        "int64_t" | "uint64_t" | "size_t" => "JAVA_LONG",
        _ => "ADDRESS",
    }
}

fn literal(l: &Lit, kind: &str) -> String {
    match l {
        Lit::None => "null".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => {
            let n = n.replace("inf", "Float.POSITIVE_INFINITY");
            match ty(kind).trim_end_matches("[]") {
                "float" | "Float" if !n.contains("INFINITY") => {
                    if n.contains('.') || n.contains('e') {
                        format!("{n}f")
                    } else {
                        format!("{n}.0f")
                    }
                }
                "long" | "Long" => format!("{n}L"),
                "double" if !n.contains('.') && !n.contains('e') => format!("{n}.0"),
                _ => n,
            }
        }
        Lit::Str(s) => format!("\"{s}\""),
        Lit::List(items) => {
            let el = ty(kind);
            let inner: Vec<String> = items
                .iter()
                .map(|i| literal(i, el.trim_end_matches("[]")))
                .collect();
            format!("new {el}{{{}}}", inner.join(", "))
        }
    }
}

/// Java expressions for the C arguments of one parameter, inside a call's
/// `try (Arena a = …)`.
fn lower(p: &Param, v: &str) -> Vec<String> {
    match p.kind.as_str() {
        "str" => vec![format!("a.allocateFrom({v})")],
        "str?" => vec![format!("Support.cstrOrNull(a, {v})")],
        "json" | "value" | "json?" | "value?" => vec![format!("Support.json(a, {v})")],
        "bool" | "f32" | "f64" | "i32" | "u32" | "node" | "i64" | "u64" | "usize" => {
            vec![v.to_string()]
        }
        "node?" => vec![format!("({v} == null ? NONE : {v})")],
        "u64?" | "usize?" => vec![format!("Support.boxLong(a, {v})")],
        "f32?" => vec![format!("Support.boxFloat(a, {v})")],
        "f32s" => vec![
            format!("Support.floats(a, {v})"),
            format!("(long) {v}.length"),
        ],
        "u8s" => vec![
            format!("Support.bytes(a, {v})"),
            format!("(long) {v}.length"),
        ],
        "u32s" | "nodes" => vec![
            format!("Support.ints(a, {v})"),
            format!("(long) {v}.length"),
        ],
        "usizes" | "i64s" => vec![
            format!("Support.longs(a, {v})"),
            format!("(long) {v}.length"),
        ],
        "usizes?" => vec![
            format!("({v} == null ? MemorySegment.NULL : Support.longs(a, {v}))"),
            format!("({v} == null ? 0L : (long) {v}.length)"),
        ],
        "rows2" => vec![
            format!("Support.longs(a, Support.rows({v}, 2, true))"),
            format!("(long) {v}.length"),
        ],
        "frows2" => vec![
            format!("Support.floats(a, Support.frows({v}, 2))"),
            format!("(long) {v}.length"),
        ],
        "frows6" => vec![
            format!("Support.floats(a, Support.frows({v}, 6))"),
            format!("(long) {v}.length"),
        ],
        "strs" | "strs?" => vec![
            format!("Support.cstrs(a, {v})"),
            format!("({v} == null ? 0L : (long) {v}.length)"),
        ],
        "inputs" => {
            let n = format!("in_{}", p.name);
            vec![
                format!("{n}.names()"),
                format!("{n}.data()"),
                format!("{n}.lens()"),
                format!("{n}.n()"),
            ]
        }
        "typed_inputs" => {
            let n = format!("in_{}", p.name);
            vec![
                format!("{n}.names()"),
                format!("{n}.data()"),
                format!("{n}.lens()"),
                format!("{n}.dtypes()"),
                format!("{n}.n()"),
            ]
        }
        "graph?" => vec![format!("({v} == null ? MemorySegment.NULL : {v}.h)")],
        _ => vec![format!("{v}.h")],
    }
}

/// `(declaration, C arguments, Java result expression)` for one result.
fn outs(kind: &str, i: usize) -> (Vec<String>, Vec<String>, String) {
    let o = format!("o{i}");
    let one = |size: &str| {
        (
            vec![format!("MemorySegment {o} = a.allocate({size});")],
            vec![o.clone()],
        )
    };
    let (d, args) = match kind {
        "f32s" | "u8s" | "u32s" | "nodes" => one("16"),
        "bool" => one("1"),
        "u32" | "node" | "f32" => one("4"),
        "f32?" => (
            vec![
                format!("MemorySegment {o} = a.allocate(4);"),
                format!("MemorySegment {o}k = a.allocate(1);"),
            ],
            vec![o.clone(), format!("{o}k")],
        ),
        _ => one("8"),
    };
    let e = match kind {
        "str" => format!("Support.takeString({o})"),
        "str?" => format!("Support.takeStringOrNull({o})"),
        "json" => format!("Support.parse(Support.takeString({o}))"),
        "object" => format!("Support.object(Support.takeString({o}))"),
        "rows" => format!("Support.rowsOf(Support.takeString({o}))"),
        "shapes" => format!("Support.shapes(Support.takeString({o}))"),
        "shape" => format!("Support.shape(Support.takeString({o}))"),
        "strings" => format!("Support.strings(Support.takeString({o}))"),
        "f32s" => format!("Support.takeFloats({o})"),
        "u8s" => format!("Support.takeBytes({o})"),
        "u32s" | "nodes" => format!("Support.takeInts({o})"),
        "outputs" => format!("Support.takeOutputs({o})"),
        "typed_outputs" => format!("Support.takeTypedOutputs({o})"),
        "tensors" => format!("Support.takeTensors({o})"),
        "bool" => format!("{o}.get(JAVA_BOOLEAN, 0)"),
        "usize" | "u64" => format!("{o}.get(JAVA_LONG, 0)"),
        "u32" | "node" => format!("{o}.get(JAVA_INT, 0)"),
        "f32" => format!("{o}.get(JAVA_FLOAT, 0)"),
        "f32?" => format!("({o}k.get(JAVA_BOOLEAN, 0) ? (Float) {o}.get(JAVA_FLOAT, 0) : null)"),
        "graph" => format!("new Graph({o}.get(ADDRESS, 0))"),
        class => format!("new {class}({o}.get(ADDRESS, 0))"),
    };
    (d, args, e)
}

/// The C parameter types of a function, for its `FunctionDescriptor`.
fn c_types(ir: &Ir, f: &Func) -> Vec<String> {
    let mut v = Vec::new();
    if f.role == Role::Method {
        v.push("ptr".into());
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

fn func(s: &mut String, ir: &Ir, f: &Func, indent: &str) {
    let class = ir.class_of(f).map(|c| c.name.clone());
    let name = method_name(&f.name);
    let optional: Vec<&Param> = f.params.iter().filter(|p| p.optional()).collect();
    let stem = match f.role {
        Role::Constructor => class.clone().unwrap(),
        _ => pascal(&f.name),
    };
    let opts = format!("{stem}Opts");
    if !optional.is_empty() {
        let _ = writeln!(
            s,
            "{indent}/** The optional parameters of {}, at their defaults. */",
            f.name
        );
        let _ = writeln!(s, "{indent}public static final class {opts} {{");
        for p in &optional {
            let t = ty(&p.kind);
            let t = match t.as_str() {
                // a value kind with no default needs a "none"
                "int" | "long" | "float" | "boolean" | "double" if p.default.is_none() => {
                    match t.as_str() {
                        "int" => "Integer".into(),
                        "long" => "Long".into(),
                        "float" => "Float".into(),
                        "double" => "Double".into(),
                        _ => "Boolean".into(),
                    }
                }
                _ => t,
            };
            let d = p
                .default
                .as_ref()
                .map(|l| literal(l, &p.kind))
                .unwrap_or_else(|| "null".into());
            let fld = ident(&p.name);
            let _ = writeln!(s, "{indent}    public {t} {fld} = {d};");
            let _ = writeln!(
                s,
                "{indent}    public {opts} {fld}({t} v) {{ this.{fld} = v; return this; }}"
            );
        }
        let _ = writeln!(s, "{indent}}}\n");
    }
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![],
        _ => f.rets.clone(),
    };
    let result = format!("{stem}Result");
    if rets.len() > 1 {
        let fields: Vec<String> = rets
            .iter()
            .map(|(n, k)| format!("{} {}", ret_ty(k), ident(n)))
            .collect();
        let _ = writeln!(s, "{indent}/** What {} returns. */", f.name);
        let _ = writeln!(
            s,
            "{indent}public record {result}({}) {{}}\n",
            fields.join(", ")
        );
    }
    let ret = match rets.as_slice() {
        [] => "void".to_string(),
        [(_, k)] => ret_ty(k),
        _ => result.clone(),
    };
    let required: Vec<String> = f
        .params
        .iter()
        .filter(|p| !p.optional())
        .map(|p| format!("{} {}", ty(&p.kind), ident(&p.name)))
        .collect();
    let req_names: Vec<String> = f
        .params
        .iter()
        .filter(|p| !p.optional())
        .map(|p| ident(&p.name))
        .collect();
    let javadoc = doc_lines(&f.doc, &format!("{indent} * "));
    let doc_block = if javadoc.trim().is_empty() {
        String::new()
    } else {
        format!("{indent}/**\n{}{indent} */\n", javadoc.replace("*/", "* /"))
    };
    let stat = matches!(f.role, Role::Function | Role::Static);
    let head = |params: &str| match f.role {
        Role::Constructor => format!("public {}({params})", class.clone().unwrap()),
        _ => format!(
            "public {}{ret} {name}({params})",
            if stat { "static " } else { "" }
        ),
    };
    // Overload without the options.
    if !optional.is_empty() {
        s.push_str(&doc_block);
        let call = {
            let mut a = req_names.clone();
            a.push(format!("({opts}) null"));
            a.join(", ")
        };
        let body = match f.role {
            Role::Constructor => format!("this({call});"),
            _ if ret == "void" => format!("{name}({call});"),
            _ => format!("return {name}({call});"),
        };
        let _ = writeln!(s, "{indent}{} {{ {body} }}\n", head(&required.join(", ")));
    }
    let mut params = required.clone();
    if !optional.is_empty() {
        params.push(format!("{opts} opts"));
    }
    s.push_str(&doc_block);
    let _ = writeln!(s, "{indent}{} {{", head(&params.join(", ")));
    let b = format!("{indent}    ");
    if f.role == Role::Constructor {
        let _ = writeln!(s, "{b}this(make{}({}));", class.clone().unwrap(), {
            let mut a = req_names.clone();
            if !optional.is_empty() {
                a.push("opts".into());
            }
            a.join(", ")
        });
        let _ = writeln!(s, "{indent}}}\n");
        // The C call lives in a static factory, since `this(…)` must come first.
        let _ = writeln!(
            s,
            "{indent}private static MemorySegment make{}({}) {{",
            class.clone().unwrap(),
            params.join(", ")
        );
    }
    if !optional.is_empty() {
        let _ = writeln!(s, "{b}if (opts == null) opts = new {opts}();");
    }
    let _ = writeln!(s, "{b}try (Arena a = Arena.ofConfined()) {{");
    let bb = format!("{b}    ");
    let mut args = Vec::new();
    if f.role == Role::Method {
        args.push("h".to_string());
    }
    for p in &f.params {
        let v = if p.optional() {
            format!("opts.{}", ident(&p.name))
        } else {
            ident(&p.name)
        };
        match p.kind.as_str() {
            "inputs" => {
                let _ = writeln!(s, "{bb}var in_{} = Support.inputs(a, {v});", p.name);
            }
            "typed_inputs" => {
                let _ = writeln!(s, "{bb}var in_{} = Support.typedInputs(a, {v});", p.name);
            }
            _ => {}
        }
        args.extend(lower(p, &v));
    }
    let types = c_types(ir, f);
    let lay: Vec<String> = types.iter().map(|t| layout(t).to_string()).collect();
    if f.direct {
        let r = match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => "ADDRESS",
            _ => "JAVA_BOOLEAN",
        };
        let call = format!(
            "Support.call(\"{}\", FunctionDescriptor.of({r}{}{}){}{})",
            f.c,
            if lay.is_empty() { "" } else { ", " },
            lay.join(", "),
            if args.is_empty() { "" } else { ", " },
            args.join(", ")
        );
        if r == "ADDRESS" {
            let _ = writeln!(
                s,
                "{bb}return ((MemorySegment) {call}).reinterpret(Long.MAX_VALUE).getString(0);"
            );
        } else {
            let _ = writeln!(s, "{bb}return (boolean) {call};");
        }
        let _ = writeln!(s, "{b}}}\n{indent}}}\n");
        return;
    }
    let all_rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), class.clone().unwrap())],
        _ => f.rets.clone(),
    };
    let mut exprs = Vec::new();
    for (i, (_, k)) in all_rets.iter().enumerate() {
        let (d, a, e) = outs(k, i);
        for line in d {
            let _ = writeln!(s, "{bb}{line}");
        }
        args.extend(a);
        exprs.push(e);
    }
    let _ = writeln!(
        s,
        "{bb}Support.check((int) Support.call(\"{}\", FunctionDescriptor.of(JAVA_INT{}{}){}{}));",
        f.c,
        if lay.is_empty() { "" } else { ", " },
        lay.join(", "),
        if args.is_empty() { "" } else { ", " },
        args.join(", ")
    );
    match (f.role, exprs.len()) {
        (Role::Constructor, _) => {
            let _ = writeln!(s, "{bb}return o0.get(ADDRESS, 0);");
        }
        (_, 0) => {}
        (_, 1) => {
            let _ = writeln!(s, "{bb}return {};", exprs[0]);
        }
        _ => {
            let _ = writeln!(s, "{bb}return new {result}({});", exprs.join(", "));
        }
    }
    let _ = writeln!(s, "{b}}}\n{indent}}}\n");
}

pub fn generate(ir: &Ir) -> String {
    let mut s = String::from(
        "// RLX — versatile ML compiler + runtime.\n\
         // Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         // SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \n\
         // @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         // do not edit; regenerate with `just gen-bindings`.\n\
         \n\
         package org.rlx;\n\
         \n\
         import java.lang.foreign.Arena;\n\
         import java.lang.foreign.FunctionDescriptor;\n\
         import java.lang.foreign.MemorySegment;\n\
         import java.util.List;\n\
         import java.util.Map;\n\
         \n\
         import static java.lang.foreign.ValueLayout.*;\n\
         \n\
         /**\n\
         \x20* rlx for Java: every C function of rlx-capi. Handle types are nested\n\
         \x20* classes ({@code RLX.Graph}, {@code RLX.Session}, …); free functions are\n\
         \x20* static methods. Optional parameters are the fields of each function's\n\
         \x20* {@code …Opts} class, at their defaults.\n\
         \x20*/\n\
         @SuppressWarnings({\"unused\", \"unchecked\"})\n\
         public final class RLX {\n\
         \x20   private RLX() {}\n\
         \n\
         \x20   /** \"No node\", for an optional node argument. */\n\
         \x20   public static final int NONE = -1;\n\n",
    );
    for f in ir.free_funcs().filter(|f| !f.manual) {
        func(&mut s, ir, f, "    ");
    }
    for c in &ir.classes {
        let doc = doc_lines(&c.doc, "     * ");
        let _ = writeln!(s, "    /**\n{doc}     */");
        let extra = if c.name == "Optimizer" || c.name == "Trainer" {
            format!(" implements Manual.{}Rows", c.name)
        } else {
            String::new()
        };
        let _ = writeln!(
            s,
            "    public static final class {n} extends Support.Handle{extra} {{\n        {n}(MemorySegment h) {{ super(h, \"{free}\"); }}\n",
            n = c.name,
            free = c.c_free()
        );
        for f in ir.members(c).filter(|f| !f.manual) {
            func(&mut s, ir, f, "        ");
        }
        s.push_str("    }\n\n");
    }
    s.push_str("}\n");
    s
}
