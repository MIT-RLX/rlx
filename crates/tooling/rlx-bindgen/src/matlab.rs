// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The MATLAB / GNU Octave binding, from the lowered [`Ir`]:
//!
//! * `rlx-matlab/src/rlx_mex.c` — the body of one MEX gateway, `rlx_mex`: a
//!   function per rlx-capi function, driven by C types alone, found by name
//!   (`rlx_mex('rlx_graph_matmul', h, a, b)`). Handles are `uint64` scalars,
//!   arrays numeric arrays, string lists cell arrays; the status comes back
//!   first and every out-parameter after it.
//! * `rlx-matlab/+rlx/*.m` — the API: a `classdef` handle class per handle
//!   type (freed by `delete`), and a function file per free function. Optional
//!   parameters are name-value pairs or a struct (`g.softmax(x, 'axis', 1)`);
//!   options and reports are structs (JSON).

use std::fmt::Write;
use std::path::PathBuf;

use rlx_bind::api::Role;

use crate::ir::{Func, Ir, Lit, Param, doc_lines};
use crate::jni::c_params;

// ── the MEX gateway ─────────────────────────────────────────

fn c_function(s: &mut String, ir: &Ir, f: &Func) {
    let params = c_params(ir, f);
    let _ = writeln!(
        s,
        "static void m_{}(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {{",
        f.c
    );
    let mut pre = Vec::new();
    let mut post = Vec::new();
    let mut args = Vec::new();
    let mut rets = Vec::new();
    let mut arg = 0usize; // index into prhs of the next input
    for (i, (t, out)) in params.iter().enumerate() {
        if *out {
            let r = rets.len() + 1;
            match t.as_str() {
                "RlxF32s*" | "RlxBytes*" | "RlxU32s*" => {
                    let st = t.trim_end_matches('*');
                    pre.push(format!("{st} o{i} = {{0}};"));
                    args.push(format!("&o{i}"));
                    let take = match st {
                        "RlxF32s" => "rlx_mx_from_f32s",
                        "RlxBytes" => "rlx_mx_from_bytes",
                        _ => "rlx_mx_from_u32s",
                    };
                    rets.push(format!("plhs[{r}] = {take}(o{i});"));
                }
                "char**" => {
                    pre.push(format!("char* o{i} = NULL;"));
                    args.push(format!("&o{i}"));
                    rets.push(format!("plhs[{r}] = rlx_mx_from_string(o{i});"));
                }
                "bool*" => {
                    pre.push(format!("bool o{i} = false;"));
                    args.push(format!("&o{i}"));
                    rets.push(format!("plhs[{r}] = mxCreateLogicalScalar(o{i});"));
                }
                "float*" | "uint32_t*" | "size_t*" | "uint64_t*" => {
                    let ct = t.trim_end_matches('*');
                    pre.push(format!("{ct} o{i} = 0;"));
                    args.push(format!("&o{i}"));
                    rets.push(format!("plhs[{r}] = mxCreateDoubleScalar((double)o{i});"));
                }
                ct => {
                    let base = ct.trim_end_matches('*');
                    pre.push(format!("{base}* o{i} = NULL;"));
                    args.push(format!("&o{i}"));
                    rets.push(format!("plhs[{r}] = rlx_mx_from_ptr(o{i});"));
                }
            }
            continue;
        }
        let a = arg;
        arg += 1;
        let m = format!("prhs[{a}]");
        match t.as_str() {
            "const char*" => {
                pre.push(format!("char* c{i} = rlx_mx_string({m});"));
                args.push(format!("c{i}"));
                post.push(format!("mxFree(c{i});"));
            }
            "bool" => args.push(format!("rlx_mx_double({m}) != 0")),
            "float" => args.push(format!("(float)rlx_mx_double({m})")),
            "double" => args.push(format!("rlx_mx_double({m})")),
            "int32_t" | "int64_t" => args.push(format!("({t})rlx_mx_double({m})")),
            "uint32_t" | "uint64_t" | "size_t" => args.push(format!("({t})rlx_mx_u64({m})")),
            "const float*" => {
                pre.push(format!("float* c{i} = rlx_mx_floats({m});"));
                args.push(format!("c{i}"));
                post.push(format!("mxFree(c{i});"));
            }
            "const uint8_t*" => {
                pre.push(format!("uint8_t* c{i} = rlx_mx_bytes({m});"));
                args.push(format!("c{i}"));
                post.push(format!("mxFree(c{i});"));
            }
            "const uint32_t*" => {
                pre.push(format!("uint32_t* c{i} = rlx_mx_u32s({m});"));
                args.push(format!("c{i}"));
                post.push(format!("mxFree(c{i});"));
            }
            "const int64_t*" => {
                pre.push(format!("int64_t* c{i} = rlx_mx_i64s({m});"));
                args.push(format!("c{i}"));
                post.push(format!("mxFree(c{i});"));
            }
            "const size_t*" | "const uint64_t*" => {
                let ct = t.trim_start_matches("const ").trim_end_matches('*');
                pre.push(format!("{ct}* c{i} = ({ct}*)rlx_mx_u64s({m});"));
                args.push(format!("c{i}"));
                post.push(format!("mxFree(c{i});"));
            }
            "const char* const*" => {
                pre.push(format!("RlxMxList c{i} = rlx_mx_strings({m});"));
                args.push(format!("(const char* const*)c{i}.ptrs"));
                post.push(format!("rlx_mx_list_free(&c{i});"));
            }
            "const float* const*" => {
                pre.push(format!("RlxMxList c{i} = rlx_mx_float_arrays({m});"));
                args.push(format!("(const float* const*)c{i}.ptrs"));
                post.push(format!("rlx_mx_list_free(&c{i});"));
            }
            "const uint8_t* const*" => {
                pre.push(format!("RlxMxList c{i} = rlx_mx_byte_arrays({m});"));
                args.push(format!("(const uint8_t* const*)c{i}.ptrs"));
                post.push(format!("rlx_mx_list_free(&c{i});"));
            }
            handle => args.push(format!("({handle})rlx_mx_ptr({m})")),
        }
    }
    let outs = if f.direct { 1 } else { rets.len() + 1 };
    let _ = writeln!(
        s,
        "    rlx_mx_arity(\"{}\", nlhs, {outs}, nrhs, {arg});",
        f.c
    );
    for line in &pre {
        let _ = writeln!(s, "    {line}");
    }
    let call = format!("{}({})", f.c, args.join(", "));
    if f.direct {
        let wrap = match f.rets.first().map(|(_, k)| k.as_str()) {
            Some("str") => "mxCreateString",
            _ => "mxCreateLogicalScalar",
        };
        let _ = writeln!(s, "    plhs[0] = {wrap}({call});");
        for line in post.iter().rev() {
            let _ = writeln!(s, "    {line}");
        }
        s.push_str("}\n\n");
        return;
    }
    let _ = writeln!(s, "    int32_t rc = {call};");
    for line in post.iter().rev() {
        let _ = writeln!(s, "    {line}");
    }
    let _ = writeln!(s, "    plhs[0] = mxCreateDoubleScalar(rc);");
    if !rets.is_empty() {
        let _ = writeln!(
            s,
            "    if (rc != RLX_OK) {{ rlx_mx_empty(plhs, {outs}); return; }}"
        );
        for r in &rets {
            let _ = writeln!(s, "    {r}");
        }
    }
    s.push_str("}\n\n");
}

/// `src/rlx_mex.c`.
pub fn mex(ir: &Ir, defaults: &[String]) -> String {
    let mut s = String::from(
        "/* RLX — versatile ML compiler + runtime.\n\
         \x20* Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
         \x20* SPDX-License-Identifier: MIT OR Apache-2.0\n\
         \x20*\n\
         \x20* @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
         \x20* do not edit; regenerate with `just gen-bindings`.\n\
         \x20*\n\
         \x20* The functions of the `rlx_mex` gateway: one per rlx-capi function, each\n\
         \x20* returning the status and then its results. The gateway itself\n\
         \x20* (mexFunction) is in rlx_mex_support.c.\n\
         \x20*/\n\n\
         #include \"rlx_mex.h\"\n\n",
    );
    let mut names = Vec::new();
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
        names.push((f.c.clone(), gate));
    }
    for c in &ir.classes {
        let _ = writeln!(
            s,
            "static void m_{free}(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {{\n\
             \x20   (void)plhs;\n\
             \x20   rlx_mx_arity(\"{free}\", nlhs, 0, nrhs, 1);\n\
             \x20   {free}((Rlx{n}*)rlx_mx_ptr(prhs[0]));\n\
             }}\n",
            free = c.c_free(),
            n = c.name
        );
        names.push((c.c_free(), None));
    }
    // Sorted by name: the gateway binary-searches it.
    names.sort();
    s.push_str("const RlxMexFn RLX_MEX_FNS[] = {\n");
    for (n, gate) in &names {
        if let Some(g) = gate {
            let _ = writeln!(s, "#ifdef {g}");
        }
        let _ = writeln!(s, "    {{\"{n}\", m_{n}}},");
        if gate.is_some() {
            s.push_str("#endif\n");
        }
    }
    s.push_str("};\nconst size_t RLX_MEX_NFNS = sizeof(RLX_MEX_FNS) / sizeof(RLX_MEX_FNS[0]);\n");
    s
}

// ── the MATLAB API ──────────────────────────────────────────

const KEYWORDS: &[&str] = &[
    "break",
    "case",
    "catch",
    "classdef",
    "continue",
    "else",
    "elseif",
    "end",
    "for",
    "function",
    "global",
    "if",
    "otherwise",
    "parfor",
    "persistent",
    "return",
    "spmd",
    "switch",
    "try",
    "while",
];

/// Methods that would overload an operator or an object built-in on the
/// handle classes (`==`, `.'`, `delete`, …): suffixed `_`, as pyrlx's `eq_`.
const RESERVED_METHODS: &[&str] = &[
    "eq",
    "ne",
    "lt",
    "gt",
    "le",
    "ge",
    "and",
    "or",
    "not",
    "plus",
    "minus",
    "times",
    "mtimes",
    "rdivide",
    "ldivide",
    "mrdivide",
    "mldivide",
    "power",
    "mpower",
    "uminus",
    "uplus",
    "transpose",
    "ctranspose",
    "colon",
    "horzcat",
    "vertcat",
    "cat",
    "subsref",
    "subsasgn",
    "subsindex",
    "numel",
    "size",
    "length",
    "ndims",
    "isempty",
    "end",
    "disp",
    "display",
    "delete",
    "isvalid",
    "class",
    "double",
    "single",
    "char",
    "logical",
    "isequal",
    "copy",
    "properties",
    "methods",
    "events",
    "notify",
    "addlistener",
];

fn local(name: &str) -> String {
    if KEYWORDS.contains(&name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}

/// The MATLAB name of a function: methods avoid the reserved set.
pub fn name(f: &Func) -> String {
    if f.class.is_some() && RESERVED_METHODS.contains(&f.name.as_str()) {
        format!("{}_", f.name)
    } else {
        local(&f.name)
    }
}

fn literal(l: &Lit) -> String {
    match l {
        Lit::None => "[]".into(),
        Lit::Bool(b) => b.to_string(),
        Lit::Num(n) => n.replace("inf", "Inf"),
        Lit::Str(s) => format!("'{s}'"),
        Lit::List(items) => {
            let inner: Vec<String> = items.iter().map(literal).collect();
            if items.iter().all(|i| matches!(i, Lit::Num(_))) {
                format!("[{}]", inner.join(", "))
            } else {
                format!("{{{}}}", inner.join(", "))
            }
        }
    }
}

/// Statements to run first, and the gateway arguments, for one parameter.
fn lower(p: &Param, v: &str) -> (Vec<String>, Vec<String>) {
    let t = format!("a_{}", p.name);
    match p.kind.as_str() {
        "json" | "value" => (vec![], vec![format!("rlx.internal.json({v})")]),
        "json?" | "value?" => (vec![], vec![format!("rlx.internal.json_or_empty({v})")]),
        "node?" => (vec![], vec![format!("rlx.internal.node({v})")]),
        "u64?" | "usize?" | "f32?" | "str?" => (vec![], vec![v.into()]),
        "f32s" | "u32s" | "nodes" | "usizes" | "usizes?" | "i64s" | "u8s" => {
            (vec![], vec![v.into(), format!("numel({v})")])
        }
        "rows2" | "frows2" | "frows6" => {
            let (w, splat) = match p.kind.as_str() {
                "rows2" => (2, "true"),
                "frows2" => (2, "false"),
                _ => (6, "false"),
            };
            (
                vec![format!(
                    "[{t}, n_{}] = rlx.internal.rows({v}, {w}, {splat});",
                    p.name
                )],
                vec![t, format!("n_{}", p.name)],
            )
        }
        "strs" | "strs?" => (
            vec![format!("{t} = rlx.internal.strs({v});")],
            vec![t.clone(), format!("numel({t})")],
        ),
        "inputs" => (
            vec![format!(
                "[{t}, d_{0}, l_{0}] = rlx.internal.inputs({v});",
                p.name
            )],
            vec![
                t.clone(),
                format!("d_{}", p.name),
                format!("l_{}", p.name),
                format!("numel({t})"),
            ],
        ),
        "typed_inputs" => (
            vec![format!(
                "[{t}, d_{0}, l_{0}, k_{0}] = rlx.internal.typed_inputs({v});",
                p.name
            )],
            vec![
                t.clone(),
                format!("d_{}", p.name),
                format!("l_{}", p.name),
                format!("k_{}", p.name),
                format!("numel({t})"),
            ],
        ),
        "graph?" => (vec![], vec![format!("rlx.internal.handle_arg({v})")]),
        "str" | "bool" | "f32" | "f64" | "i32" | "i64" | "u32" | "u64" | "usize" | "node" => {
            (vec![], vec![v.into()])
        }
        _ => (vec![], vec![format!("{v}.h")]),
    }
}

/// How many gateway results a result kind takes, and the conversion.
fn convert(kind: &str, vals: &[String]) -> String {
    let v = &vals[0];
    match kind {
        "json" | "object" | "rows" | "shapes" | "shape" | "strings" => {
            format!("rlx.internal.decode({v})")
        }
        "outputs" => format!("rlx_mex('support_take_outputs', {v})"),
        "typed_outputs" => format!("rlx_mex('support_take_typed_outputs', {v})"),
        "tensors" => format!("rlx.internal.tensors({v})"),
        "f32?" => format!("rlx.internal.optional({v}, {})", vals[1]),
        "graph" => format!("rlx.Graph('rlx:handle', {v})"),
        k if k.starts_with(char::is_uppercase) => format!("rlx.{k}('rlx:handle', {v})"),
        _ => v.clone(),
    }
}

fn width(kind: &str) -> usize {
    if kind == "f32?" { 2 } else { 1 }
}

/// The body of one function or method, indented by `ind`: the result is `r`
/// (a constructor's handle `h`).
fn body(s: &mut String, f: &Func, ind: &str) {
    let optional: Vec<&Param> = f.params.iter().filter(|p| p.optional()).collect();
    if !optional.is_empty() {
        let names: Vec<String> = optional.iter().map(|p| format!("'{}'", p.name)).collect();
        let defaults: Vec<String> = optional
            .iter()
            .map(|p| p.default.as_ref().map_or("[]".into(), literal))
            .collect();
        let _ = writeln!(
            s,
            "{ind}o = rlx.internal.opts(varargin, {{{}}}, {{{}}});",
            names.join(", "),
            defaults.join(", ")
        );
    }
    let mut args = vec![format!("'{}'", f.c)];
    if f.role == Role::Method {
        args.push("self.h".into());
    }
    for p in &f.params {
        let v = if p.optional() {
            format!("o.{}", p.name)
        } else {
            local(&p.name)
        };
        let (stmts, a) = lower(p, &v);
        for st in stmts {
            let _ = writeln!(s, "{ind}{st}");
        }
        args.extend(a);
    }
    let call = format!("rlx_mex({})", args.join(", "));
    if f.direct {
        let _ = writeln!(s, "{ind}r = {call};");
        return;
    }
    let rets: Vec<(String, String)> = match f.role {
        Role::Constructor => vec![("handle".into(), "handle".into())],
        _ => f.rets.clone(),
    };
    let mut names = vec!["rc".to_string()];
    let mut groups = Vec::new();
    for (i, (_, k)) in rets.iter().enumerate() {
        let vals: Vec<String> = (0..width(k)).map(|j| format!("r{i}_{j}")).collect();
        names.extend(vals.iter().cloned());
        groups.push((k.clone(), vals));
    }
    let lhs = if names.len() == 1 {
        "rc".to_string()
    } else {
        format!("[{}]", names.join(", "))
    };
    let _ = writeln!(s, "{ind}{lhs} = {call};");
    let _ = writeln!(s, "{ind}rlx.internal.check(rc);");
    match (f.role, groups.len()) {
        (Role::Constructor, _) => {
            let _ = writeln!(s, "{ind}h = r0_0;");
        }
        (_, 0) => {}
        (_, 1) => {
            let _ = writeln!(s, "{ind}r = {};", convert(&groups[0].0, &groups[0].1));
        }
        _ => {
            let _ = writeln!(s, "{ind}r = struct();");
            for ((n, _), (k, vals)) in rets.iter().zip(&groups) {
                let _ = writeln!(s, "{ind}r.{n} = {};", convert(k, vals));
            }
        }
    }
}

fn signature(f: &Func, with_self: bool) -> (String, bool) {
    let mut params: Vec<String> = Vec::new();
    if with_self {
        params.push("self".into());
    }
    params.extend(
        f.params
            .iter()
            .filter(|p| !p.optional())
            .map(|p| local(&p.name)),
    );
    if f.params.iter().any(|p| p.optional()) {
        params.push("varargin".into());
    }
    let returns = f.direct || (f.role != Role::Constructor && !f.rets.is_empty());
    (params.join(", "), returns)
}

fn help(doc: &str, ind: &str) -> String {
    doc_lines(doc, &format!("{ind}% "))
}

fn method(s: &mut String, ir: &Ir, f: &Func, with_self: bool) {
    let (mut params, returns) = signature(f, with_self);
    if f.manual {
        params = if with_self {
            "self, varargin"
        } else {
            "varargin"
        }
        .into();
    }
    let out = if returns || f.manual {
        "varargout = "
    } else {
        ""
    };
    let r = if returns && !f.manual { "r = " } else { out };
    let _ = writeln!(s, "        function {r}{}({params})", name(f));
    s.push_str(&help(&f.doc, "            "));
    if f.manual {
        // Hand-written: +rlx/+internal/<class>_<name>.m.
        let class = ir.class_of(f).unwrap().name.to_ascii_lowercase();
        let _ = writeln!(
            s,
            "            [varargout{{1:nargout}}] = rlx.internal.{class}_{}({});",
            f.name,
            params.replace("varargin", "varargin{:}")
        );
    } else {
        body(s, f, "            ");
    }
    s.push_str("        end\n\n");
}

const HEADER: &str = "% RLX — versatile ML compiler + runtime.\n\
                      % Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
                      % SPDX-License-Identifier: MIT OR Apache-2.0\n\
                      %\n\
                      % @generated by rlx-bindgen from rlx-bind's op table and runtime API table —\n\
                      % do not edit; regenerate with `just gen-bindings`.\n\n";

/// One `classdef` file.
fn class_file(ir: &Ir, ci: usize) -> String {
    let c = &ir.classes[ci];
    let mut s = String::new();
    let _ = writeln!(s, "classdef {} < rlx.internal.Handle", c.name);
    s.push_str(&help(&c.doc, "    "));
    s.push_str("    methods\n");
    let members: Vec<&Func> = ir.members(c).collect();
    let ctor = members.iter().find(|f| f.role == Role::Constructor);
    // The constructor: from its C function, or `rlx.X('rlx:handle', h)` to
    // adopt a handle rlx returned.
    let _ = writeln!(s, "        function self = {}(varargin)", c.name);
    if let Some(f) = ctor {
        s.push_str(&help(&f.doc, "            "));
    }
    s.push_str(
        "            if nargin == 2 && isequal(varargin{1}, 'rlx:handle')\n\
         \x20               h = varargin{2};\n\
         \x20           else\n",
    );
    match ctor {
        Some(f) => {
            let required: Vec<String> = f
                .params
                .iter()
                .filter(|p| !p.optional())
                .map(|p| local(&p.name))
                .collect();
            for (i, r) in required.iter().enumerate() {
                let _ = writeln!(s, "                {r} = varargin{{{}}};", i + 1);
            }
            if !required.is_empty() && f.params.iter().any(|p| p.optional()) {
                let _ = writeln!(
                    s,
                    "                varargin = varargin({}:end);",
                    required.len() + 1
                );
            }
            let mut b = String::new();
            body(&mut b, f, "                ");
            s.push_str(&b);
        }
        None => {
            let _ = writeln!(
                s,
                "                error('rlx:type', 'rlx.{} has no constructor; rlx returns one');",
                c.name
            );
        }
    }
    s.push_str("            end\n");
    let _ = writeln!(
        s,
        "            self@rlx.internal.Handle(h, '{}');",
        c.c_free()
    );
    s.push_str("        end\n\n");
    for f in members.iter().filter(|f| f.role == Role::Method) {
        method(&mut s, ir, f, true);
    }
    s.push_str("    end\n");
    let statics: Vec<&&Func> = members.iter().filter(|f| f.role == Role::Static).collect();
    if !statics.is_empty() {
        s.push_str("\n    methods (Static)\n");
        for f in statics {
            method(&mut s, ir, f, false);
        }
        s.push_str("    end\n");
    }
    s.push_str("end\n");
    format!("{HEADER}{s}")
}

/// One free function's file.
fn function_file(ir: &Ir, f: &Func) -> String {
    let mut s = String::new();
    method(&mut s, ir, f, false);
    // Function files are not indented.
    let text: String = s
        .lines()
        .map(|l| l.strip_prefix("        ").unwrap_or(l))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{HEADER}{}\n", text.trim_end())
}

/// Every generated `.m` file, relative to `rlx-matlab/`.
pub fn api(ir: &Ir) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    for (i, c) in ir.classes.iter().enumerate() {
        out.push((
            PathBuf::from(format!("+rlx/{}.m", c.name)),
            class_file(ir, i),
        ));
    }
    for f in ir.funcs.iter().filter(|f| f.class.is_none()) {
        out.push((
            PathBuf::from(format!("+rlx/{}.m", name(f))),
            function_file(ir, f),
        ));
    }
    // macOS and Windows file systems fold case: `+rlx/check.m` and a class
    // `Check` would be one file.
    let mut seen = std::collections::HashSet::new();
    for (p, _) in &out {
        let key = p.to_string_lossy().to_ascii_lowercase();
        assert!(
            seen.insert(key),
            "rlx-bindgen: MATLAB files collide by case: {}",
            p.display()
        );
    }
    out
}
